#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <functional>
#include <memory>
#include <cassert>
#include <optional>

// Padding to prevent false sharing
struct alignas(64) PaddedMutex {
    std::mutex mutex;
    // Padding to fill a cache line (typically 64 bytes)
    char padding[64 - sizeof(std::mutex)];
    
    // Need to explicitly define special member functions
    PaddedMutex() = default;
    
    // Delete copy operations since std::mutex is not copyable
    PaddedMutex(const PaddedMutex&) = delete;
    PaddedMutex& operator=(const PaddedMutex&) = delete;
    
    // Allow move operations - we need these for std::vector
    PaddedMutex(PaddedMutex&&) noexcept {}
    PaddedMutex& operator=(PaddedMutex&&) noexcept { return *this; }
};

template <typename T>
class OpenAddressedHashSet {
protected:
    // Entry states
    enum class EntryState : uint8_t {
        EMPTY = 0,
        OCCUPIED = 1,
        DELETED = 2
    };

    struct Entry {
        T value;
        EntryState state;
        
        Entry() : state(EntryState::EMPTY) {}
    };

    // Hash table parameters
    std::vector<Entry> table;
    size_t capacity;
    static constexpr double MAX_LOAD_FACTOR = 0.7;

    // Hash function
    size_t hash(const T& value) const {
        return std::hash<T>{}(value) % capacity;
    }

    // Linear probing to find slot
    size_t findSlot(const T& value) const {
        size_t index = hash(value);
        size_t start = index;
        
        do {
            if (table[index].state == EntryState::EMPTY) {
                return index;
            }
            else if (table[index].state == EntryState::OCCUPIED && table[index].value == value) {
                return index;
            }
            index = (index + 1) % capacity;
        } while (index != start);
        
        return capacity; // Table is full
    }

public:
    OpenAddressedHashSet(size_t initialCapacity = 1024) 
        : capacity(initialCapacity) {
        table.resize(capacity);
    }

    virtual ~OpenAddressedHashSet() {}

    // The following core functions must be implemented by derived classes
    virtual bool add(const T& value) = 0;
    virtual bool remove(const T& value) = 0;
    virtual bool contains(const T& value) const = 0;

    // Non-thread safe size calculation
    virtual size_t size() const {
        size_t count = 0;
        for (const auto& entry : table) {
            if (entry.state == EntryState::OCCUPIED) {
                count++;
            }
        }
        return count;
    }

    // Non-thread safe population
    void populate(size_t numElements) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<T> dist(1, std::numeric_limits<T>::max() / 2);
        
        size_t added = 0;
        while (added < numElements) {
            T value = dist(gen);
            if (add(value)) {
                added++;
            }
        }
    }
};

// Sequential implementation
template <typename T>
class SequentialOpenAddressedHashSet : public OpenAddressedHashSet<T> {
public:
    using OpenAddressedHashSet<T>::OpenAddressedHashSet;
    using typename OpenAddressedHashSet<T>::EntryState;
    using OpenAddressedHashSet<T>::table;
    using OpenAddressedHashSet<T>::capacity;
    using OpenAddressedHashSet<T>::findSlot;

    bool add(const T& value) override {
        size_t index = findSlot(value);
        
        if (index == capacity) {
            return false; // Table is full
        }
        
        if (table[index].state == EntryState::OCCUPIED && table[index].value == value) {
            return false; // Value already exists
        }
        
        table[index].value = value;
        table[index].state = EntryState::OCCUPIED;
        return true;
    }

    bool remove(const T& value) override {
        size_t index = findSlot(value);
        
        if (index == capacity || table[index].state != EntryState::OCCUPIED) {
            return false; // Table is full or value not found
        }
        
        table[index].state = EntryState::DELETED;
        return true;
    }

    bool contains(const T& value) const override {
        size_t index = findSlot(value);
        
        return (index != capacity && 
                table[index].state == EntryState::OCCUPIED && 
                table[index].value == value);
    }
};

// Concurrent implementation with fine-grained locking
template <typename T>
class ConcurrentOpenAddressedHashSet : public OpenAddressedHashSet<T> {
private:
    static constexpr size_t NUM_LOCKS = 1024; // Increased number of locks to reduce contention
    mutable std::vector<PaddedMutex> locks;
    std::atomic<size_t> itemCount;

    // Map a hash value to a lock
    size_t getLockIndex(size_t hash) const {
        return hash % NUM_LOCKS;
    }

    // Acquire all needed locks for a linear probe sequence
    void acquireLocksForProbe(const T& value, std::vector<std::unique_lock<std::mutex>>& acquiredLocks) const {
        size_t index = this->hash(value);
        size_t start = index;
        std::vector<size_t> lockIndices;
        
        // First, collect all unique lock indices that we need to acquire
        do {
            size_t lockIndex = getLockIndex(index);
            if (std::find(lockIndices.begin(), lockIndices.end(), lockIndex) == lockIndices.end()) {
                lockIndices.push_back(lockIndex);
            }
            index = (index + 1) % this->capacity;
        } while (index != start && lockIndices.size() < 10); // Limit to prevent trying to lock too many
        
        // Sort lock indices to prevent deadlock
        std::sort(lockIndices.begin(), lockIndices.end());
        
        // Acquire locks in order
        for (size_t lockIndex : lockIndices) {
            acquiredLocks.emplace_back(locks[lockIndex].mutex);
        }
    }

public:
    using OpenAddressedHashSet<T>::OpenAddressedHashSet;
    using typename OpenAddressedHashSet<T>::EntryState;
    using OpenAddressedHashSet<T>::table;
    using OpenAddressedHashSet<T>::capacity;
    using OpenAddressedHashSet<T>::hash;

    ConcurrentOpenAddressedHashSet(size_t initialCapacity = 1024) 
        : OpenAddressedHashSet<T>(initialCapacity), itemCount(0) {
        locks.reserve(NUM_LOCKS);
        for (size_t i = 0; i < NUM_LOCKS; i++) {
            locks.emplace_back();
        }
    }

    bool add(const T& value) override {
        std::vector<std::unique_lock<std::mutex>> acquiredLocks;
        acquireLocksForProbe(value, acquiredLocks);
        
        size_t index = this->hash(value);
        size_t start = index;
        
        do {
            if (table[index].state == EntryState::EMPTY) {
                table[index].value = value;
                table[index].state = EntryState::OCCUPIED;
                itemCount.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            else if (table[index].state == EntryState::OCCUPIED && table[index].value == value) {
                return false; // Value already exists
            }
            else if (table[index].state == EntryState::DELETED) {
                // We can reuse this slot, but we have to check the rest of the probe sequence
                // to make sure the value isn't already in the table
                size_t deletedIndex = index;
                bool found = false;
                
                // Continue checking for duplicate
                size_t checkIndex = (index + 1) % capacity;
                while (checkIndex != start && !found) {
                    if (table[checkIndex].state == EntryState::OCCUPIED && 
                        table[checkIndex].value == value) {
                        found = true;
                    }
                    else if (table[checkIndex].state == EntryState::EMPTY) {
                        break;
                    }
                    checkIndex = (checkIndex + 1) % capacity;
                }
                
                if (!found) {
                    // We can safely add to the deleted slot
                    table[deletedIndex].value = value;
                    table[deletedIndex].state = EntryState::OCCUPIED;
                    itemCount.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
                return false;
            }
            index = (index + 1) % capacity;
        } while (index != start);
        
        return false; // Table is full
    }

    bool remove(const T& value) override {
        std::vector<std::unique_lock<std::mutex>> acquiredLocks;
        acquireLocksForProbe(value, acquiredLocks);
        
        size_t index = this->hash(value);
        size_t start = index;
        
        do {
            if (table[index].state == EntryState::EMPTY) {
                return false;
            }
            else if (table[index].state == EntryState::OCCUPIED && table[index].value == value) {
                table[index].state = EntryState::DELETED;
                itemCount.fetch_sub(1, std::memory_order_relaxed);
                return true;
            }
            index = (index + 1) % capacity;
        } while (index != start);
        
        return false;
    }

    bool contains(const T& value) const override {
        std::vector<std::unique_lock<std::mutex>> acquiredLocks;
        acquireLocksForProbe(value, acquiredLocks);
        
        size_t index = this->hash(value);
        size_t start = index;
        
        do {
            if (table[index].state == EntryState::EMPTY) {
                return false;
            }
            else if (table[index].state == EntryState::OCCUPIED && table[index].value == value) {
                return true;
            }
            index = (index + 1) % capacity;
        } while (index != start);
        
        return false;
    }
    
    // Override size method to use the atomic counter
    size_t size() const override {
        return itemCount.load(std::memory_order_relaxed);
    }
};

// Benchmark
template <typename SetType>
void runBenchmark(SetType& set, int numThreads, size_t totalOps, 
                  double containsPercent, double addPercent, 
                  bool showProgress = false) {
    // Calculate operations per thread
    size_t opsPerThread = totalOps / numThreads;
    
    // Populate first (not timed)
    const size_t initialSize = 100000;
    // std::cout << "Populating hash set with " << initialSize << " elements..." << std::endl;
    set.populate(initialSize);
    
    // std::cout << "Initial size: " << set.size() << std::endl;
    // std::cout << "Operations per thread: " << opsPerThread << " (Total: " << totalOps << ")" << std::endl;

    // Expected operations count
    std::atomic<int> addCount(0);
    std::atomic<int> removeCount(0);
    std::atomic<int> containsCount(0);
    std::atomic<int> successfulAdds(0);
    std::atomic<int> successfulRemoves(0);
    
    // Create threads
    std::vector<std::thread> threads;
    
    // Barrier to synchronize thread start
    std::atomic<int> startFlag(0);
    
    auto threadFunc = [&](int id) {
        // Wait for all threads to be ready
        while (startFlag.load() == 0) {
            std::this_thread::yield();
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        // Fixed: specify int instead of T
        std::uniform_int_distribution<int> valueDist(1, std::numeric_limits<int>::max());
        std::uniform_real_distribution<double> opTypeDist(0.0, 1.0);
        
        for (size_t i = 0; i < opsPerThread; i++) {
            int value = valueDist(gen);
            double opType = opTypeDist(gen);
            
            if (opType < containsPercent) {
                // Contains operation
                set.contains(value);
                containsCount++;
            } 
            else if (opType < containsPercent + addPercent) {
                // Add operation
                bool success = set.add(value);
                addCount++;
                if (success) successfulAdds++;
            } 
            else {
                // Remove operation
                bool success = set.remove(value);
                removeCount++;
                if (success) successfulRemoves++;
            }
            
            if (showProgress && id == 0 && i % (opsPerThread / 10) == 0) {
                std::cout << "Thread 0 progress: " << (i * 100 / opsPerThread) << "%" << std::endl;
            }
        }
    };
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Launch threads
    for (int i = 0; i < numThreads; i++) {
        threads.emplace_back(threadFunc, i);
    }
    
    // Start all threads simultaneously
    startFlag.store(1);
    
    // Wait for all threads to finish
    for (auto& t : threads) {
        t.join();
    }
    
    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    // Report results
    std::cout << "Benchmark completed in " << duration.count() << "um" << std::endl;
    std::cout << "Operations performed:" << std::endl;
    std::cout << "  Contains: " << containsCount.load() << std::endl;
    std::cout << "  Adds: " << addCount.load() << " (successful: " << successfulAdds.load() << ")" << std::endl;
    std::cout << "  Removes: " << removeCount.load() << " (successful: " << successfulRemoves.load() << ")" << std::endl;
    
    // Verify size
    size_t expectedSize = initialSize + successfulAdds.load() - successfulRemoves.load();
    size_t actualSize = set.size();
    std::cout << "Expected size: " << expectedSize << std::endl;
    std::cout << "Actual size: " << actualSize << std::endl;
    
    if (expectedSize == actualSize) {
        std::cout << "Size verification: SUCCESS" << std::endl;
    } else {
        std::cout << "Size verification: FAILED" << std::endl;
    }
    
    // Calculate throughput
    double opsPerSecond = totalOps / (duration.count() / 1000.0);
    std::cout << "Throughput: " << opsPerSecond << " ops/second" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <num_iterations>" << std::endl;
        return 1;
    }
    
    int numThreads = std::stoi(argv[1]);
    size_t totalOps = std::stoi(argv[2]);
    
    const size_t initialCapacity = 100000000;
    const double containsPercent = 0.8;
    const double addPercent = 0.1;
    
    // Only run with the specified thread count
    ConcurrentOpenAddressedHashSet<int> concSet(initialCapacity);
    auto startTime = std::chrono::high_resolution_clock::now();
    runBenchmark(concSet, numThreads, totalOps, containsPercent, addPercent, false);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    // Only output the duration in microseconds
    std::cout << duration.count() << std::endl;
    
    return 0;
}