#include <iostream>
#include <vector>
#include <optional>
#include <functional> // for std::hash
#include <mutex>
// #include <shared_mutex>
#include <thread>
#include <atomic>
#include <random>
#include <chrono>
#include <stdexcept> // For std::runtime_error
#include <numeric>   // For std::iota
#include <algorithm> // For std::shuffle, std::min, std::max
#include <new>       // For std::hardware_destructive_interference_size

// --- Optimization: Padded Mutex ---
#ifdef __cpp_lib_hardware_interference_size
    constexpr size_t cache_line_size = std::hardware_destructive_interference_size;
#else
    constexpr size_t cache_line_size = 64; // Fallback
#endif

struct PaddedMutex {
    std::mutex mtx;
    char padding[cache_line_size > sizeof(std::mutex) ? cache_line_size - sizeof(std::mutex) : 1];
};
// --- End Optimization ---


// Configuration Constants
const size_t NUM_TABLES = 2;
const size_t NUM_LOCKS = 32;
const int MAX_KICK_LIMIT = 100;

// --- Use the specified global generator and mutex ---
// This generator is used for population and seeding thread-local generators.
std::mt19937 gen(714);
std::mutex gen_mutex; // Protects the global 'gen' during seeding/population
// --- End Change ---


template <typename T>
class StripedCuckooHashSet {
private:
    std::vector<std::vector<std::optional<T>>> table;
    std::atomic<size_t> capacity;
    std::atomic<size_t> current_size;
    std::vector<PaddedMutex> locks; // Use Padded Mutexes


    // Hash functions (unchanged)
    size_t h1(const T& x, size_t current_capacity) const {
        if (current_capacity == 0) return 0;
        return (std::hash<T>{}(x)) % current_capacity;
    }

    size_t h2(const T& x, size_t current_capacity) const {
        if (current_capacity == 0) return 0;
        size_t hash_val = std::hash<T>{}(x);
        return ((hash_val ^ (hash_val >> 16)) * 0x85ebca6b) % current_capacity;
    }

    // Locking (uses PaddedMutex, logic unchanged)
    std::pair<size_t, size_t> get_lock_indices(const T& x, size_t current_capacity) const {
        if (current_capacity == 0) return {0, 0};
        size_t pos1 = h1(x, current_capacity);
        size_t pos2 = h2(x, current_capacity);
        size_t lock1_idx = pos1 % NUM_LOCKS;
        size_t lock2_idx = pos2 % NUM_LOCKS;
        return {std::min(lock1_idx, lock2_idx), std::max(lock1_idx, lock2_idx)};
    }

    std::unique_lock<std::mutex> acquire_lock1(size_t lock_idx) {
        if (lock_idx >= locks.size()) throw std::out_of_range("Lock index 1 out of bounds");
        return std::unique_lock<std::mutex>(locks[lock_idx].mtx);
    }

    std::unique_lock<std::mutex> acquire_lock2(size_t lock_idx1, size_t lock_idx2) {
         if (lock_idx1 == lock_idx2) return std::unique_lock<std::mutex>();
         if (lock_idx2 >= locks.size()) throw std::out_of_range("Lock index 2 out of bounds");
         return std::unique_lock<std::mutex>(locks[lock_idx2].mtx);
    }


    // Unsafe Operations (unchanged, with bounds checks)
    bool contains_unsafe(const T& x, size_t current_capacity) const {
       if (current_capacity == 0) return false;
        size_t pos1 = h1(x, current_capacity);
        if (pos1 < current_capacity && table.size() > 0 && table[0].size() > pos1 && table[0][pos1].has_value() && table[0][pos1].value() == x) {
            return true;
        }
        size_t pos2 = h2(x, current_capacity);
        if (pos2 < current_capacity && table.size() > 1 && table[1].size() > pos2 && table[1][pos2].has_value() && table[1][pos2].value() == x) {
            return true;
        }
        return false;
    }

    bool add_cuckoo_unsafe(T item, std::vector<std::vector<std::optional<T>>>& target_table, size_t target_capacity) {
        if (target_capacity == 0) return false;
        T current_item = std::move(item); // Use move constructor
        size_t current_table_idx = 0;

        for (int count = 0; count < MAX_KICK_LIMIT; ++count) {
            size_t pos = (current_table_idx == 0) ? h1(current_item, target_capacity) : h2(current_item, target_capacity);
            if (pos >= target_capacity) return false; // Safety check
            if (target_table.size() <= current_table_idx || target_table[current_table_idx].size() <= pos) return false; // Safety check

            if (!target_table[current_table_idx][pos].has_value()) {
                target_table[current_table_idx][pos] = std::move(current_item);
                return true;
            }
            // Swap logic using move
            T victim = std::move(target_table[current_table_idx][pos].value());
            target_table[current_table_idx][pos] = std::move(current_item);
            current_item = std::move(victim);
            current_table_idx = 1 - current_table_idx; // Switch table
        }
        return false; // Kick limit reached
    }

    // Resize (unchanged, uses PaddedMutex)
    void resize() {
        size_t old_capacity = capacity.load(std::memory_order_acquire);

        std::vector<std::unique_lock<std::mutex>> acquired_locks;
        acquired_locks.reserve(NUM_LOCKS);
        for (size_t i = 0; i < NUM_LOCKS; ++i) {
            acquired_locks.emplace_back(locks[i].mtx); // Lock the padded mutex
        }

        if (capacity.load(std::memory_order_relaxed) != old_capacity) return; // Double check
        if (old_capacity > (1 << 29)) { std::cerr << "Max capacity\n"; return; } // Limit check

        size_t new_capacity = (old_capacity == 0) ? 16 : old_capacity * 2;
        std::vector<std::vector<std::optional<T>>> new_table(NUM_TABLES);
         try {
             new_table[0].resize(new_capacity); new_table[1].resize(new_capacity);
         } catch (const std::bad_alloc&) { std::cerr << "Resize alloc fail\n"; return; }

        size_t rehash_failures = 0;
        size_t current_elements_found = 0;
        for (size_t i = 0; i < table.size() && i < NUM_TABLES; ++i) {
            for (size_t j = 0; j < old_capacity && j < table[i].size(); ++j) {
                if (table[i][j].has_value()) {
                    current_elements_found++;
                    if (!add_cuckoo_unsafe(std::move(table[i][j].value()), new_table, new_capacity)) {
                        rehash_failures++; std::cerr << "Rehash fail\n";
                    }
                    table[i][j].reset();
                }
            }
        }

        if (rehash_failures == 0) {
            table = std::move(new_table);
            capacity.store(new_capacity, std::memory_order_release);
            current_size.store(current_elements_found, std::memory_order_relaxed);
        } else { std::cerr << "Resize aborted\n"; }
        // Locks released by RAII
    }

public:
    // Constructor (uses PaddedMutex)
    explicit StripedCuckooHashSet(size_t initial_capacity = 16) :
        capacity(initial_capacity == 0 ? 16 : initial_capacity),
        current_size(0),
        locks(NUM_LOCKS) // Initialize vector of PaddedMutex
    {
        size_t cap = capacity.load(std::memory_order_relaxed);
        table.resize(NUM_TABLES);
         try {
            table[0].resize(cap); table[1].resize(cap);
         } catch (const std::bad_alloc& e) {
             std::cerr << "Initial alloc fail: " << e.what() << std::endl;
             capacity.store(0); throw;
         }
    }

    // Add with Cuckoo Kicks (uses PaddedMutex)
    bool add(const T& x) {
        T item_to_insert = x;
        int attempts = 0;
        const int max_resize_attempts = 2;

        while(attempts < max_resize_attempts) {
            size_t current_cap = capacity.load(std::memory_order_acquire);
            if (current_cap == 0) return false;

            // Lock relevant stripes
            auto [lock_idx1, lock_idx2] = get_lock_indices(item_to_insert, current_cap);
            std::unique_lock<std::mutex> lk1 = acquire_lock1(lock_idx1);
            std::unique_lock<std::mutex> lk2 = acquire_lock2(lock_idx1, lock_idx2);

            if (capacity.load(std::memory_order_relaxed) != current_cap) continue; // Cap changed, retry

            if (contains_unsafe(item_to_insert, current_cap)) return false; // Already present


            // Cuckoo Kick Loop
            T current_item = std::move(item_to_insert); // Item to potentially insert/kick
            size_t current_table_idx = 0;

            for (int kick = 0; kick < MAX_KICK_LIMIT; ++kick) {
                size_t pos = (current_table_idx == 0) ? h1(current_item, current_cap) : h2(current_item, current_cap);
                if (pos >= current_cap || table.size() <= current_table_idx || table[current_table_idx].size() <= pos) {
                     std::cerr << "Bounds err add kick\n"; goto resize_check; // Error, try resize
                }

                if (!table[current_table_idx][pos].has_value()) { // Empty slot found
                    table[current_table_idx][pos] = std::move(current_item);
                    current_size.fetch_add(1, std::memory_order_relaxed);
                    return true; // Success
                }
                // Occupied: Swap item with victim, continue kicking victim
                T victim = std::move(table[current_table_idx][pos].value());
                table[current_table_idx][pos] = std::move(current_item);
                current_item = std::move(victim);
                current_table_idx = 1 - current_table_idx;
            }
            // Kick limit reached, 'current_item' is the one that failed
            item_to_insert = std::move(current_item); // Restore item that needs inserting after resize


        resize_check: // Needs resize or error occurred
            lk1.unlock(); if (lk2.owns_lock()) lk2.unlock(); // Release locks *before* resize

            attempts++;
            if (attempts >= max_resize_attempts) return false; // Failed even after resize attempts

            resize(); // Perform resize
            // Outer while loop continues -> retry insert with item_to_insert
        }
        return false; // Should not be reached
    }


    // Remove (uses PaddedMutex)
    bool remove(const T& x) {
        size_t current_cap = capacity.load(std::memory_order_acquire);
         if (current_cap == 0) return false;
         auto [lock_idx1, lock_idx2] = get_lock_indices(x, current_cap);
        std::unique_lock<std::mutex> lk1 = acquire_lock1(lock_idx1);
        std::unique_lock<std::mutex> lk2 = acquire_lock2(lock_idx1, lock_idx2);

         if (capacity.load(std::memory_order_relaxed) != current_cap) return false; // Resized concurrently

        size_t pos1 = h1(x, current_cap);
         if (pos1 < current_cap && table.size() > 0 && table[0].size() > pos1 && table[0][pos1].has_value() && table[0][pos1].value() == x) {
            table[0][pos1].reset(); current_size.fetch_sub(1, std::memory_order_relaxed); return true;
        }
        size_t pos2 = h2(x, current_cap);
         if (pos2 < current_cap && table.size() > 1 && table[1].size() > pos2 && table[1][pos2].has_value() && table[1][pos2].value() == x) {
            table[1][pos2].reset(); current_size.fetch_sub(1, std::memory_order_relaxed); return true;
        }
        return false; // Not found
    }

    // Contains (uses PaddedMutex)
    bool contains(const T& x) {
        size_t current_cap = capacity.load(std::memory_order_acquire);
         if (current_cap == 0) return false;
        auto [lock_idx1, lock_idx2] = get_lock_indices(x, current_cap);
        std::unique_lock<std::mutex> lk1 = acquire_lock1(lock_idx1);
        std::unique_lock<std::mutex> lk2 = acquire_lock2(lock_idx1, lock_idx2);
         if (capacity.load(std::memory_order_relaxed) != current_cap) return false; // Resized concurrently
        return contains_unsafe(x, current_cap); // Checks bounds
    }

    // --- Non-Thread-Safe Methods ---
    size_t size() const { return current_size.load(std::memory_order_relaxed); }
    size_t get_capacity() const { return capacity.load(std::memory_order_relaxed); }

    // Populate uses the optimized thread-safe add, but needs the global gen
    void populate(size_t num_elements_to_add) {
        if (capacity.load(std::memory_order_relaxed) == 0) { std::cerr << "Populate invalid\n"; return; }
        size_t added_count = 0;
        std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());
        size_t attempt_limit = num_elements_to_add * 4;
        size_t attempts = 0;

        while (added_count < num_elements_to_add && attempts < attempt_limit) {
             attempts++;
             T val;
             // --- Use global gen for population ---
             {
                 std::lock_guard<std::mutex> lock(gen_mutex); // Lock global gen
                 val = dist(gen);
             }
             // --- End Change ---
             if (add(val)) { // Use the optimized add (which handles its own locking)
                 added_count++;
             }
        }
        if (added_count < num_elements_to_add) {
            std::cerr << "Warn: Populate added " << added_count << "/" << num_elements_to_add << "\n";
        }
        // std::cout << "Population added " << added_count << " elements." << std::endl;
    }
};


// --- Test Harness ---
std::atomic<long long> successful_adds = 0;
std::atomic<long long> successful_removes = 0;

// Worker uses thread-local RNG seeded from global gen
void worker_thread(StripedCuckooHashSet<int>& hash_set, int num_ops_for_this_thread, unsigned int seed) {
    thread_local std::mt19937 thread_gen(seed); // Thread-local generator
    std::uniform_int_distribution<int> op_dist(0, 99);
    std::uniform_int_distribution<int> val_dist(0, std::numeric_limits<int>::max());

    for (int i = 0; i < num_ops_for_this_thread; ++i) {
        // Use thread-local generator - NO LOCK NEEDED HERE
        int operation_type = op_dist(thread_gen);
        int value = val_dist(thread_gen);

        if (operation_type < 80) { hash_set.contains(value); }
        else if (operation_type < 90) { if (hash_set.add(value)) { successful_adds.fetch_add(1, std::memory_order_relaxed); } }
        else { if (hash_set.remove(value)) { successful_removes.fetch_add(1, std::memory_order_relaxed); } }
    }
}


int main(int argc, char* argv[]) {
    if (argc != 3) { std::cerr << "Usage: " << argv[0] << " <ops> <threads>\n"; return 1; }

    int total_operations = 0; int num_threads = 0;
    try { total_operations = std::stoi(argv[1]); num_threads = std::stoi(argv[2]); }
    catch (const std::exception& e) { std::cerr << "Arg parse err: " << e.what() << "\n"; return 1; }
    if (total_operations < 0 || num_threads <= 0) { std::cerr << "Ops>=0, threads>0\n"; return 1; }

    // --- Setup ---
    size_t initial_capacity = 1000000; // 1 Million
    size_t populate_count = 500000;   // 500 Thousand
    StripedCuckooHashSet<int> hash_set(initial_capacity);

    // --- Populate (uses global 'gen' with lock internally) ---
    hash_set.populate(populate_count);

    size_t initial_size = hash_set.size();
    size_t initial_cap_after_populate = hash_set.get_capacity();

    // --- Concurrent Operations ---
    std::vector<std::thread> threads;
    int ops_per_thread = (num_threads == 0) ? 0 : total_operations / num_threads;
    int remaining_ops = (num_threads == 0) ? 0 : total_operations % num_threads;
    std::cout << "– Running " << total_operations << " Operations w/ " << num_threads << " Threads –" << std::endl;
    successful_adds = 0; successful_removes = 0; // Reset counters
    auto start_time = std::chrono::high_resolution_clock::now();

    // --- Seed thread-local RNGs using global 'gen' ---
    std::vector<unsigned int> seeds(num_threads);
    {
        std::lock_guard<std::mutex> lock(gen_mutex); // Lock global gen ONLY for seeding
        std::uniform_int_distribution<unsigned int> seed_dist;
        for(int i=0; i<num_threads; ++i) {
            seeds[i] = seed_dist(gen); // Generate seeds from global gen
        }
    }
    // --- End Seeding ---

    for (int i = 0; i < num_threads; ++i) {
        int thread_ops = ops_per_thread + (i < remaining_ops ? 1 : 0);
        if (thread_ops > 0) {
             threads.emplace_back(worker_thread, std::ref(hash_set), thread_ops, seeds[i]); // Pass seed
        }
    }
    for (auto& t : threads) { if (t.joinable()) { t.join(); } } // Join threads
    auto end_time = std::chrono::high_resolution_clock::now();
    long long total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    // --- Verification and Output ---
    size_t final_size = hash_set.size();
    size_t final_capacity = hash_set.get_capacity();
    long long current_adds = successful_adds.load(std::memory_order_seq_cst);
    long long current_removes = successful_removes.load(std::memory_order_seq_cst);
    long long expected_size = static_cast<long long>(initial_size) + current_adds - current_removes;
    long long avg_time_us = (total_operations == 0) ? 0 : total_time_us / total_operations;

    // --- Print Results ---
    std::cout << "Total time: " << total_time_us << std::endl;
    std::cout << "Average time per operation: " << avg_time_us << std::endl;
    std::cout << "Hashset initial size: " << initial_size << std::endl;
    std::cout << "Hashset initial capacity: " << initial_cap_after_populate << std::endl;
    std::cout << "Successful Adds: " << current_adds << std::endl;
    std::cout << "Successful Removes: " << current_removes << std::endl;
    std::cout << "Expected size: " << expected_size << std::endl;
    std::cout << "Final hashset size: " << final_size << std::endl;
    std::cout << "Final hashset capacity: " << final_capacity << std::endl;

    // Final check
    if (static_cast<long long>(final_size) != expected_size) {
        std::cerr << "[Error] Mismatch final(" << final_size << ") vs expected(" << expected_size << ")!\n";
    }

    return 0;
}