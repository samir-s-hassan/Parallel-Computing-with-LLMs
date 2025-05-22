#include <atomic>
#include <mutex>
#include <vector>
#include <random>
#include <thread>
#include <iostream>
#include <functional>
#include <chrono>

class HybridOpenAddressedSet {
private:
    static const size_t INITIAL_CAPACITY = 1000000; // 1 million
    static const size_t STRIPE_COUNT = 64; // Number of locks
    
    struct Bucket {
        int element;
        std::atomic<uint64_t> metadata;
        
        Bucket() : metadata(0) {}
        
        bool isEmpty() const {
            return (metadata.load(std::memory_order_acquire) & 1) == 0;
        }
        
        bool isTombstone() const {
            return (metadata.load(std::memory_order_acquire) & 2) == 2;
        }
        
        bool isOccupied() const {
            return (metadata.load(std::memory_order_acquire) & 1) == 1 && 
                   (metadata.load(std::memory_order_acquire) & 2) == 0;
        }
    };
    
    // Cache-line aligned bucket array to prevent false sharing
    alignas(64) Bucket* table;
    size_t capacity;
    
    // Striped locks for modifications
    alignas(64) std::vector<std::mutex> locks;
    
    // Atomic counter for size tracking
    std::atomic<size_t> elementCount;
    
    // Hash function to map keys to buckets
    size_t hash(const int& item) const {
        return static_cast<size_t>(std::hash<int>{}(item) % capacity);
    }
    
    // Get the lock index for a given hash
    size_t getLockIndex(size_t hash) const {
        return hash % STRIPE_COUNT;
    }
    
    // Helper to increment version while preserving state bits
    uint64_t incrementVersion(uint64_t metadata) const {
        return metadata + 4;
    }
    
    // Set bucket as occupied with item
    void markOccupied(size_t index, const int& item) {
        table[index].element = item;
        table[index].metadata.store(
            incrementVersion(table[index].metadata.load() & ~2) | 1, 
            std::memory_order_release
        );
    }
    
    // Mark bucket as tombstone (deleted)
    void markTombstone(size_t index) {
        table[index].metadata.store(
            incrementVersion(table[index].metadata.load()) | 3, 
            std::memory_order_release
        );
    }

public:
    explicit HybridOpenAddressedSet(size_t initialCapacity = INITIAL_CAPACITY) 
        : capacity(initialCapacity), 
          locks(STRIPE_COUNT),
          elementCount(0) {
        
        table = new Bucket[capacity];
    }
    
    ~HybridOpenAddressedSet() {
        delete[] table;
    }
    
    // Lock-free optimistic contains implementation
    bool contains(const int& item) const {
        size_t startIndex = hash(item);
        size_t index = startIndex;
        
        do {
            // Snapshot metadata
            uint64_t metadata = table[index].metadata.load(std::memory_order_acquire);
            
            // If bucket is occupied and has the item we're looking for
            if ((metadata & 1) && !(metadata & 2)) { // Occupied and not tombstone
                // If element matches
                if (table[index].element == item) {
                    // Validate our read was consistent by checking version hasn't changed
                    if (metadata == table[index].metadata.load(std::memory_order_acquire)) {
                        return true;
                    }
                    // Version changed, retry this bucket
                    continue;
                }
            } 
            // If we find an empty bucket (not tombstone), the item cannot be in the set
            else if ((metadata & 1) == 0) {
                return false;
            }
            
            // Continue linear probing
            index = (index + 1) % capacity;
        } while (index != startIndex);
        
        return false; // Full table traversal found nothing
    }
    
    // Lock-based add implementation
    bool add(const int& item) {
        size_t startIndex = hash(item);
        size_t index = startIndex;
        size_t firstTombstone = capacity; // Invalid index initially
        
        // First, optimistically check if item already exists without locking
        do {
            uint64_t metadata = table[index].metadata.load(std::memory_order_acquire);
            
            if ((metadata & 1) == 0) { // Empty bucket
                break; // We can potentially insert here
            } else if ((metadata & 2) && firstTombstone == capacity) { // Tombstone
                firstTombstone = index; // Remember first tombstone for potential reuse
            } else if ((metadata & 1) && !(metadata & 2)) { // Occupied, not tombstone
                if (table[index].element == item) {
                    // Item already exists, verify with a second read
                    if (metadata == table[index].metadata.load(std::memory_order_acquire)) {
                        return false;
                    }
                }
            }
            
            index = (index + 1) % capacity;
        } while (index != startIndex);
        
        // Acquire lock for the stripe containing our target bucket
        size_t lockIndex = getLockIndex(startIndex);
        std::lock_guard<std::mutex> guard(locks[lockIndex]);
        
        // After acquiring lock, we need to search again to ensure
        // item wasn't added by another thread
        index = startIndex;
        firstTombstone = capacity;
        
        do {
            if (table[index].isEmpty()) {
                // Found empty slot, insert here
                size_t insertIndex = (firstTombstone != capacity) ? firstTombstone : index;
                markOccupied(insertIndex, item);
                elementCount.fetch_add(1, std::memory_order_relaxed);
                return true;
            } else if (table[index].isTombstone() && firstTombstone == capacity) {
                // Track first tombstone for potential insert
                firstTombstone = index;
            } else if (table[index].isOccupied() && table[index].element == item) {
                // Item already exists
                return false;
            }
            
            index = (index + 1) % capacity;
        } while (index != startIndex);
        
        // If we get here, we either have a tombstone or the table is full
        if (firstTombstone != capacity) {
            markOccupied(firstTombstone, item);
            elementCount.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        
        // Table is full and no tombstones - would need resize in a production implementation
        return false;
    }
    
    // Lock-based remove implementation
    bool remove(const int& item) {
        size_t startIndex = hash(item);
        
        // Acquire lock for the stripe containing our target bucket
        size_t lockIndex = getLockIndex(startIndex);
        std::lock_guard<std::mutex> guard(locks[lockIndex]);
        
        size_t index = startIndex;
        
        do {
            if (table[index].isEmpty()) {
                // Found empty slot - item doesn't exist
                return false;
            } else if (table[index].isOccupied() && table[index].element == item) {
                // Found item, mark as tombstone
                markTombstone(index);
                elementCount.fetch_sub(1, std::memory_order_relaxed);
                return true;
            }
            
            index = (index + 1) % capacity;
        } while (index != startIndex);
        
        return false;
    }
    
    // Non-thread-safe size function as per requirements
    size_t size() const {
        return elementCount.load(std::memory_order_relaxed);
    }
    
    // Non-thread-safe population function as per requirements
    // FIXED: Ensure exactly count unique elements are added
    void populate(size_t count, int min = 0, int max = 1000000) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(min, max);
        
        size_t added = 0;
        while (added < count) {
            int value = dist(gen);
            if (add(value)) {
                added++;
            }
        }
    }
};

// Function to run benchmark with given threads and operations
void runBenchmark(int numThreads, int totalOps) {
    const int INITIAL_ELEMENTS = 500000;
    const int KEY_RANGE = 1000000;
    int opsPerThread = totalOps / numThreads;
    
    // Create the set
    HybridOpenAddressedSet set;
    
    // Populate with initial elements - ensure exactly 500,000 are added
    set.populate(INITIAL_ELEMENTS, 0, KEY_RANGE);
    
    // Verify initial size
    if (set.size() != INITIAL_ELEMENTS) {
        std::cerr << "Error: Initial population failed. Expected: " << INITIAL_ELEMENTS 
                  << ", Actual: " << set.size() << std::endl;
    }
    
    // Tracking expected size changes
    std::atomic<int> expectedSizeDelta(0);
    
    // Thread function to perform mixed operations
    auto threadFunc = [&](int threadId) {
        std::random_device rd;
        std::mt19937 gen(rd() + threadId); // Different seed for each thread
        std::uniform_int_distribution<int> opDist(1, 100);
        std::uniform_int_distribution<int> keyDist(0, KEY_RANGE);
        
        int localAdds = 0;
        int localRemoves = 0;
        
        for (int i = 0; i < opsPerThread; ++i) {
            int op = opDist(gen);
            int key = keyDist(gen);
            
            if (op <= 80) {
                // 80% contains operation
                set.contains(key);
            } else if (op <= 90) {
                // 10% add operation
                bool added = set.add(key);
                if (added) {
                    localAdds++;
                }
            } else {
                // 10% remove operation
                bool removed = set.remove(key);
                if (removed) {
                    localRemoves++;
                }
            }
        }
        
        // Update expected size delta
        expectedSizeDelta.fetch_add(localAdds - localRemoves, std::memory_order_relaxed);
    };
    
    // Launch threads
    std::vector<std::thread> threads;
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(threadFunc, i);
    }
    
    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    
    // Verify final size
    size_t expectedSize = INITIAL_ELEMENTS + expectedSizeDelta.load();
    size_t actualSize = set.size();
    
    if (expectedSize != actualSize) {
        std::cerr << "Size verification failed! Expected: " << expectedSize 
                  << ", Actual: " << actualSize << std::endl;
    }
    
    // Output just the duration for the benchmark script
    std::cout << duration;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <total_ops>" << std::endl;
        return 1;
    }
    
    int numThreads = std::stoi(argv[1]);
    int totalOps = std::stoi(argv[2]);
    
    runBenchmark(numThreads, totalOps);
    
    return 0;
}