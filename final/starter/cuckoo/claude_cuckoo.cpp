#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <random>
#include <functional>
#include <atomic>
#include <fstream>

#include <cstdlib> // for getenv
#include <iomanip> // for std::setprecision
#include <limits>

template<typename T>
class ConcurrentOpenAddressedSet {
private:
    static const size_t DEFAULT_CAPACITY = 1024;
    static const size_t DEFAULT_NUM_LOCKS = 16;
    static const size_t MAX_PROBE = 100; // Maximum probing attempts

    enum class BucketState {
        EMPTY,
        OCCUPIED,
        DELETED
    };

    struct Bucket {
        T data;
        BucketState state;

        Bucket() : state(BucketState::EMPTY) {}
    };

    std::vector<Bucket> table;
    std::vector<std::mutex> locks;
    size_t numLocks;

    size_t hash(const T& key) const {
        return std::hash<T>{}(key) % table.size();
    }

    // Map a table index to the corresponding lock index
    size_t lockIndex(size_t tableIndex) const {
        return tableIndex % numLocks;
    }

    // Find the position of an element or an empty slot for insertion
    // Returns true if found or suitable empty position found, and updates pos
    bool find(const T& item, size_t& pos) {
        size_t startPos = hash(item);
        size_t i = 0;

        while (i < MAX_PROBE) {
            pos = (startPos + i) % table.size();

            if (table[pos].state == BucketState::EMPTY) {
                return false; // Item not found, but empty slot available at pos
            }

            if (table[pos].state == BucketState::OCCUPIED && table[pos].data == item) {
                return true; // Item found at pos
            }

            i++;
        }

        return false; // Table might be full or item not found within MAX_PROBE attempts
    }

public:
    ConcurrentOpenAddressedSet(size_t capacity = DEFAULT_CAPACITY, size_t numLocks = DEFAULT_NUM_LOCKS)
        : table(capacity), locks(numLocks), numLocks(numLocks) {}

    bool add(const T& item) {
        size_t pos;
        size_t startPos = hash(item);
        size_t lockIdx = lockIndex(startPos);
        bool result = false;

        // Acquire the lock for this bucket
        std::lock_guard<std::mutex> guard(locks[lockIdx]);

        // Check if item already exists
        if (find(item, pos)) {
            return false; // Item already exists
        }

        // If we reached here, item doesn't exist
        // Try to find an empty or deleted slot
        size_t i = 0;
        while (i < MAX_PROBE) {
            pos = (startPos + i) % table.size();

            if (table[pos].state != BucketState::OCCUPIED) {
                // Found a suitable slot
                table[pos].data = item;
                table[pos].state = BucketState::OCCUPIED;
                return true;
            }

            i++;
        }

        return false; // Table is full or couldn't find a slot
    }

    bool remove(const T& item) {
        size_t pos;
        size_t startPos = hash(item);
        size_t lockIdx = lockIndex(startPos);

        // Acquire the lock for this bucket
        std::lock_guard<std::mutex> guard(locks[lockIdx]);

        if (find(item, pos) && table[pos].state == BucketState::OCCUPIED) {
            table[pos].state = BucketState::DELETED;
            return true;
        }

        return false; // Item not found
    }

    bool contains(const T& item) {
        size_t pos;
        size_t startPos = hash(item);
        size_t lockIdx = lockIndex(startPos);

        // Acquire the lock for this bucket
        std::lock_guard<std::mutex> guard(locks[lockIdx]);

        return find(item, pos) && table[pos].state == BucketState::OCCUPIED;
    }

    // Not thread-safe, use only when no other thread is accessing the set
    size_t size() const {
        size_t count = 0;
        for (const auto& bucket : table) {
            if (bucket.state == BucketState::OCCUPIED) {
                count++;
            }
        }
        return count;
    }

    // Not thread-safe, use only when no other thread is accessing the set
    void populate(size_t count) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<T> dist(1, std::numeric_limits<int>::max());

        for (size_t i = 0; i < count; i++) {
            add(static_cast<T>(dist(gen)));
        }
    }
};

void runBenchmark() {
    // Parameters for benchmarking
    int numThreads = 4;
    int numOperations = 100000;
    int initialSize = 5000000;
    const char* env_p = getenv("NUM_OPERATIONS");
    if (env_p) {
        numOperations = atoi(env_p);
    }
    env_p = getenv("THREADS");
    if (env_p) {
        numThreads = atoi(env_p);
    }

    // Create the concurrent hash set
    ConcurrentOpenAddressedSet<int> set(initialSize * 2);

    // Populate the set
    set.populate(initialSize);

    // Vector to store the expected size change
    std::atomic<int> expectedSizeChange(0);

    // Timing start
    auto start = std::chrono::high_resolution_clock::now();

    // Launch threads
    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; t++) {
        threads.push_back(std::thread([&set, numOperations, &expectedSizeChange]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> opDist(1, 100);
            std::uniform_int_distribution<int> valDist(1, std::numeric_limits<int>::max());

            for (int i = 0; i < numOperations; i++) {
                int op = opDist(gen);
                int value = valDist(gen);

                if (op <= 80) {
                    set.contains(value);
                } else if (op <= 90) {
                    bool added = set.add(value);
                    if (added) expectedSizeChange++;
                } else {
                    bool removed = set.remove(value);
                    if (removed) expectedSizeChange--;
                }
            }
        }));
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Timing end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double elapsed_time = elapsed.count(); // in seconds

    // Get size info
    size_t actualSize = set.size();
    int expectedSize = initialSize + expectedSizeChange.load();

    // Write to CSV
    std::ofstream outfile("results.csv", std::ios::app); // open in append mode

    if (outfile.is_open()) {
        outfile << numThreads << ","
                << numOperations << ","
                << std::fixed << std::setprecision(9) << elapsed_time << ","
                << std::scientific << (numOperations * numThreads) / elapsed_time << ","
                << std::fixed << actualSize << ","
                << expectedSize << "\n";

        outfile.close();
    } else {
        std::cerr << "Error opening file for writing!" << std::endl;
    }
}

int main() {
    runBenchmark();
    return 0;
}