#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <random>
#include <chrono>
#include <climits>
#include <cstring>

// Concurrent cuckoo hash set with striped locks
template<typename T>
class StripedCuckooHashSet {
private:
    // Private fields
    std::atomic<T>* table[2];  // Two tables for cuckoo hashing
    std::mutex** locks;        // 2D array of locks
    int capacity;
    int lockCapacity;
    std::hash<T> hashFunction;
    static const T EMPTY = 0;
    static const int PROBE_SIZE = 4;
    static const int THRESHOLD = 50;
    static const int LIMIT = 32;

    // Hash functions
    int hash0(T x) const {
        return std::hash<T>{}(x) % capacity;
    }

    int hash1(T x) const {
        return (std::hash<T>{}(x) / capacity) % capacity;
    }

    // Lock management
    void acquire(T x) {
        int h0 = hash0(x) % lockCapacity;
        int h1 = hash1(x) % lockCapacity;
        locks[0][h0].lock();
        locks[1][h1].lock();
    }

    void release(T x) {
        int h0 = hash0(x) % lockCapacity;
        int h1 = hash1(x) % lockCapacity;
        locks[0][h0].unlock();
        locks[1][h1].unlock();
    }

    bool relocate(int which, int index) {
        int route[LIMIT];
        int startLevel = 0;
        int i = 1 - which;
        int hi = index;
        
        for (int round = 0; round < LIMIT; round++) {
            T y = table[which][hi].load();
            if (y == EMPTY) {
                return false;
            }
            
            switch (which) {
                case 0: hi = hash1(y) % capacity; break;
                case 1: hi = hash0(y) % capacity; break;
            }
            
            T temp = table[i][hi].load();
            if (temp == EMPTY) {
                table[i][hi].store(y);
                table[which][index].store(EMPTY);
                return true;
            }
            
            table[i][hi].store(y);
            table[which][index].store(temp);
            index = hi;
            which = i;
            i = 1 - i;
        }
        
        return false;
    }

    bool resize() {
        int oldCapacity = capacity;
        if (capacity >= INT_MAX / 2) {
            return false;
        }
        
        // Save old tables
        std::atomic<T>* oldTable0 = table[0];
        std::atomic<T>* oldTable1 = table[1];
        
        // Double capacity
        capacity *= 2;
        table[0] = new std::atomic<T>[capacity];
        table[1] = new std::atomic<T>[capacity];
        
        // Initialize new tables
        for (int i = 0; i < capacity; i++) {
            table[0][i].store(EMPTY);
            table[1][i].store(EMPTY);
        }
        
        // Rehash all elements
        for (int i = 0; i < oldCapacity; i++) {
            T val = oldTable0[i].load();
            if (val != EMPTY) {
                int h0 = hash0(val) % capacity;
                table[0][h0].store(val);
            }
            
            val = oldTable1[i].load();
            if (val != EMPTY) {
                int h1 = hash1(val) % capacity;
                table[1][h1].store(val);
            }
        }
        
        // Clean up old tables
        delete[] oldTable0;
        delete[] oldTable1;
        
        return true;
    }

public:
    StripedCuckooHashSet(int initialCapacity) 
        : capacity(initialCapacity), lockCapacity(initialCapacity / 4) {
        
        // Initialize tables
        table[0] = new std::atomic<T>[capacity];
        table[1] = new std::atomic<T>[capacity];
        
        for (int i = 0; i < capacity; i++) {
            table[0][i].store(EMPTY);
            table[1][i].store(EMPTY);
        }
        
        // Initialize locks
        locks = new std::mutex*[2];
        locks[0] = new std::mutex[lockCapacity];
        locks[1] = new std::mutex[lockCapacity];
    }
    
    ~StripedCuckooHashSet() {
        delete[] table[0];
        delete[] table[1];
        delete[] locks[0];
        delete[] locks[1];
        delete[] locks;
    }

    bool add(T x) {
        if (x == EMPTY) return false;
        
        acquire(x);
        try {
            if (contains(x)) {
                release(x);
                return false;
            }
            
            int h0 = hash0(x) % capacity;
            int h1 = hash1(x) % capacity;
            
            // Try to add to table 0
            if (table[0][h0].load() == EMPTY) {
                table[0][h0].store(x);
                release(x);
                return true;
            }
            
            // Try to add to table 1
            if (table[1][h1].load() == EMPTY) {
                table[1][h1].store(x);
                release(x);
                return true;
            }
            
            // Must relocate
            if (!relocate(0, h0)) {
                release(x);
                if (resize()) {
                    return add(x);
                }
                return false;
            }
            
            table[0][h0].store(x);
            release(x);
            return true;
            
        } catch (...) {
            release(x);
            throw;
        }
    }

    bool remove(T x) {
        if (x == EMPTY) return false;
        
        acquire(x);
        try {
            int h0 = hash0(x) % capacity;
            int h1 = hash1(x) % capacity;
            
            if (table[0][h0].load() == x) {
                table[0][h0].store(EMPTY);
                release(x);
                return true;
            }
            
            if (table[1][h1].load() == x) {
                table[1][h1].store(EMPTY);
                release(x);
                return true;
            }
            
            release(x);
            return false;
            
        } catch (...) {
            release(x);
            throw;
        }
    }

    bool contains(T x) const {
        if (x == EMPTY) return false;
        
        int h0 = hash0(x) % capacity;
        int h1 = hash1(x) % capacity;
        
        return (table[0][h0].load() == x || table[1][h1].load() == x);
    }

    int size() const {  // Non-thread safe
        int count = 0;
        for (int i = 0; i < capacity; i++) {
            if (table[0][i].load() != EMPTY) count++;
            if (table[1][i].load() != EMPTY) count++;
        }
        return count;
    }

    void populate(int count) {  // Non-thread safe
        std::mt19937 gen(714);  // Fixed seed
        std::uniform_int_distribution<> dis(1, INT_MAX);
        
        int added = 0;
        int attempts = 0;
        while (added < count && attempts < count * 3) {
            int val = dis(gen);
            if (add(val)) {
                added++;
            }
            attempts++;
        }
    }

    int getCapacity() const {
        return capacity;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <operations> <threads>" << std::endl;
        return 1;
    }

    int numOperations = std::atoi(argv[1]);
    int numThreads = std::atoi(argv[2]);
    
    // Initialize with 1 million capacity
    StripedCuckooHashSet<int> hashSet(1000000);
    
    // Populate with 500,000 elements
    int initialPopulation = 500000;
    hashSet.populate(initialPopulation);
    
    int initialSize = hashSet.size();
    int initialCapacity = hashSet.getCapacity();
    
    std::vector<std::thread> threads;
    std::atomic<int> successfulAdds(0);
    std::atomic<int> successfulRemoves(0);
    
    // Print header
    std::cout << "- Running " << numOperations << " Operations w/ " << numThreads << " Threads -" << std::endl;
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch threads
    for (int i = 0; i < numThreads; i++) {
        threads.emplace_back([&, i]() {
            std::mt19937 gen(714 + i);  // Seed with offset for each thread
            std::uniform_int_distribution<> valueDis(1, INT_MAX);
            std::uniform_int_distribution<> opDis(1, 100);
            
            int localAdds = 0;
            int localRemoves = 0;
            
            for (int j = 0; j < numOperations / numThreads; j++) {
                int operation = opDis(gen);
                int value = valueDis(gen);
                
                if (operation <= 80) {  // 80% contains
                    hashSet.contains(value);
                } else if (operation <= 90) {  // 10% insert
                    if (hashSet.add(value)) {
                        localAdds++;
                    }
                } else {  // 10% remove
                    if (hashSet.remove(value)) {
                        localRemoves++;
                    }
                }
            }
            
            successfulAdds.fetch_add(localAdds);
            successfulRemoves.fetch_add(localRemoves);
        });
    }
    
    // Wait for threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Calculate results
    int finalSize = hashSet.size();
    int finalCapacity = hashSet.getCapacity();
    int expectedSize = initialSize + successfulAdds.load() - successfulRemoves.load();
    
    // Print results
    std::cout << "Total time: " << duration.count() << std::endl;
    std::cout << "Average time per operation: " << duration.count() / numOperations << std::endl;
    std::cout << "Hashset initial size: " << initialSize << std::endl;
    std::cout << "Hashset initial capacity: " << initialCapacity << std::endl;
    std::cout << "Expected size: " << expectedSize << std::endl;
    std::cout << "Final hashset size: " << finalSize << std::endl;
    std::cout << "Final hashset capacity: " << finalCapacity << std::endl;
    
    return 0;
}