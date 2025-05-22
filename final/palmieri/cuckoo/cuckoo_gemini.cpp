#include <iostream>
#include <vector>
#include <optional>
#include <functional> // for std::hash
#include <atomic>
#include <mutex>
#include <thread>
#include <random>
#include <chrono>
#include <stdexcept> // for exceptions, stoi
#include <cmath>     // for std::ceil
#include <memory>    // for std::unique_ptr
#include <string>    // for std::string, std::stoi

// Configuration Constants - Default values, can be overridden
const size_t DEFAULT_INITIAL_CAPACITY = 16; // Default if not specified
const double MAX_LOAD_FACTOR = 0.75;

// Forward declaration for the test function
template <typename T> class OptimisticFineGrainedHashSet;

// *** (Keep the OptimisticFineGrainedHashSet class definition exactly as in the previous response) ***
// START PASTE OptimisticFineGrainedHashSet class HERE
// ... (Full class code from previous response) ...
template <typename T>
class OptimisticFineGrainedHashSet {
private:
    enum class State : uint8_t {
        EMPTY,
        OCCUPIED,
        DELETED // Tombstone
    };

    struct Bucket {
        std::optional<T> value;
        std::atomic<State> state;
        std::atomic<uint64_t> version; // Version counter for optimistic reads
        std::mutex lock;               // Fine-grained lock per bucket

        Bucket() : value(), state(State::EMPTY), version(0) {}

        // Need copy/move semantics that handle mutex (or prevent them)
        Bucket(const Bucket&) = delete;
        Bucket& operator=(const Bucket&) = delete;

        // Allow move construction for resizing (lock is not moved)
        Bucket(Bucket&& other) noexcept :
            value(std::move(other.value)),
            state(other.state.load(std::memory_order_relaxed)), // Relaxed ok during resize setup
            version(other.version.load(std::memory_order_relaxed))
        {
             // Each new bucket gets its own mutex; don't move the old one's lock state.
        }
        // Allow move assignment for resizing
         Bucket& operator=(Bucket&& other) noexcept {
            if (this != &other) {
                // Acquire both locks before moving to prevent race conditions during resize
                // Note: This simple lock order works because resize holds a global lock.
                // If resize were concurrent, we'd need a more robust lock ordering.
                // UPDATE: No locks needed here IF resize_mutex is held during the entire swap/rehash
                // std::lock_guard<std::mutex> guard_this(lock);
                // std::lock_guard<std::mutex> guard_other(other.lock);

                value = std::move(other.value);
                state.store(other.state.load(std::memory_order_relaxed), std::memory_order_relaxed);
                version.store(other.version.load(std::memory_order_relaxed), std::memory_order_relaxed);
                 other.state.store(State::EMPTY, std::memory_order_relaxed); // Clear source
                 other.value.reset();
            }
            return *this;
        }

    };

    std::vector<Bucket> table;
    std::atomic<size_t> element_count;
    size_t capacity; // Current capacity (size of table vector)
    std::hash<T> hasher;

    // Global mutex specifically to serialize resize operations
    std::mutex resize_mutex;

    // --- Helper Functions ---

    // Gets the index using modulo (capacity must be power of 2)
    size_t get_index(const T& key) const {
        // Check if capacity is a power of 2 - simplifies modulo
         if ((capacity & (capacity - 1)) != 0) {
             // Fallback if not power of 2 (less efficient)
             return hasher(key) % capacity;
         }
        // Optimization for power-of-2 capacity
        return hasher(key) & (capacity - 1);
    }

     // Linear probing
    size_t probe_next(size_t index) const {
         // Check if capacity is a power of 2
         if ((capacity & (capacity - 1)) != 0) {
             // Fallback if not power of 2
              return (index + 1) % capacity;
         }
         // Optimization for power-of-2 capacity
        return (index + 1) & (capacity - 1); // Same as (index + 1) % capacity
    }

    // Non-concurrent resize - acquires resize_mutex
    void resize() {
        std::lock_guard<std::mutex> resize_guard(resize_mutex);

        // Double-check condition inside the lock
        if (static_cast<double>(element_count.load(std::memory_order_acquire)) / capacity <= MAX_LOAD_FACTOR) {
            return; // Another thread already resized
        }

        // std::cerr << "--- Resizing from " << capacity << " to " << capacity * 2 << " ---" << std::endl; // Use cerr


        size_t old_capacity = capacity;
        size_t new_capacity = capacity * 2;
        // Optimization: Use unique_ptr for old_table to avoid large stack allocation
        auto old_table_owner = std::make_unique<std::vector<Bucket>>(new_capacity);
        old_table_owner->swap(table); // Swap old data into old_table_owner, table now has default buckets
        std::vector<Bucket>& old_table_ref = *old_table_owner; // Use reference for convenience


        // Reallocate the main table - ensures new mutexes are constructed
        table = std::vector<Bucket>(new_capacity); // New buckets with fresh state/mutexes
        capacity = new_capacity;
        element_count.store(0, std::memory_order_relaxed); // Reset count, will be updated by add

        // Rehash elements from old_table to the new table
        // Note: This part is NOT thread-safe by itself; relies on resize_mutex
        for (size_t i = 0; i < old_capacity; ++i) {
             State current_state = old_table_ref[i].state.load(std::memory_order_relaxed);

            if (current_state == State::OCCUPIED && old_table_ref[i].value.has_value()) {
                 // Must use the internal add function that doesn't trigger resize
                internal_add(old_table_ref[i].value.value());
            }
        }
         // std::cerr << "--- Resize complete. New capacity: " << capacity << " New element count: " << element_count.load() << " ---" << std::endl; // Use cerr
         // old_table_owner goes out of scope here, freeing the old table memory
    }


    // Internal add used ONLY during resize - assumes resize lock is held, no concurrent access
    bool internal_add(const T& key) {
        size_t start_index = get_index(key);
        size_t current_index = start_index;

        do {
            // Direct access, no atomics/locks needed as resize lock is held
            if (table[current_index].state.load(std::memory_order_relaxed) == State::EMPTY ||
                table[current_index].state.load(std::memory_order_relaxed) == State::DELETED)
            {
                table[current_index].value = key;
                table[current_index].state.store(State::OCCUPIED, std::memory_order_relaxed);
                 table[current_index].version.store(1, std::memory_order_relaxed); // Start version at 1
                element_count.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
             if (table[current_index].state.load(std::memory_order_relaxed) == State::OCCUPIED &&
                table[current_index].value.has_value() &&
                table[current_index].value.value() == key)
             {
                return false; // Key already exists
            }
            current_index = probe_next(current_index);
        } while (current_index != start_index);

        // Should not happen if resize logic is correct (table should not be full)
        throw std::runtime_error("Internal add failed: table unexpectedly full during resize.");
         return false; // Should be unreachable
    }


public:
    // MODIFIED: Constructor takes desired initial capacity
    explicit OptimisticFineGrainedHashSet(size_t initial_cap = DEFAULT_INITIAL_CAPACITY) :
        element_count(0),
        capacity(initial_cap) // Store requested capacity first
    {
         // Ensure capacity is power of 2 for modulo optimization
         if (capacity == 0) capacity = 1;
         size_t rounded_capacity = 1;
         // Round up to the nearest power of 2
         while(rounded_capacity < initial_cap) rounded_capacity <<= 1;
         capacity = rounded_capacity; // Use the rounded-up power of 2

        table.resize(capacity); // Vector of Buckets
        // std::cerr << "Initialized HashSet with capacity: " << capacity << " (requested " << initial_cap << ")" << std::endl; // Use cerr
    }

    // --- Public Concurrent API --- (Keep contains, add, remove methods as before) ---
     bool contains(const T& key) {
        size_t start_index = get_index(key);
        size_t current_index = start_index;
        uint64_t observed_version;

        for (int attempt = 0; attempt < 2 * capacity; ++attempt) { // Limit attempts to prevent infinite loops
            current_index = start_index;
            bool potential_found = false;
            size_t potential_index = 0;
            uint64_t potential_version = 0;

            // 1. Optimistic Read Phase
            do {
                // Acquire semantics ensure we see prior writes relevant to state/version
                State current_state = table[current_index].state.load(std::memory_order_acquire);
                observed_version = table[current_index].version.load(std::memory_order_acquire);

                if (current_state == State::EMPTY) {
                     potential_found = false; // Definitely not here
                     break; // End optimistic probe
                }

                if (current_state == State::OCCUPIED) {
                    // Optimization: Check hash first before accessing value (potentially expensive)
                    // Requires value.has_value() check AND access under potential race condition.
                    // Safer: Check value equality only under lock.
                    // Potential match - needs validation under lock.
                     if (table[current_index].value.has_value() && table[current_index].value.value() == key) {
                           potential_found = true;
                           potential_index = current_index;
                           potential_version = observed_version;
                           break; // Found potential match, proceed to lock/validate
                     }
                }
                // If DELETED, continue probing
                current_index = probe_next(current_index);
            } while (current_index != start_index);


            // 2. Lock and Validate Phase
            if (potential_found) {
                std::lock_guard<std::mutex> guard(table[potential_index].lock);
                // Re-read state and version under lock
                State locked_state = table[potential_index].state.load(std::memory_order_acquire);
                uint64_t locked_version = table[potential_index].version.load(std::memory_order_acquire);

                // Validate version AND state AND value
                if (locked_version == potential_version && locked_state == State::OCCUPIED) {
                     // Check value again under lock for certainty
                     if (table[potential_index].value.has_value() && table[potential_index].value.value() == key) {
                           return true; // Validated successfully
                     }
                }
                // Validation failed - version changed, state changed, or value didn't match
                // Fall through to retry the whole operation
            } else {
                 // Optimistic probe finished without finding a potential match (hit EMPTY or looped)
                 return false;
            }

            // If validation failed, loop will continue (retry)
             // Optional: Add a small delay/backoff here if contention is high
             // std::this_thread::yield();

        }
         // If we retry too many times, assume high contention or a bug
         // For production, might throw or log. For assignment, maybe return false.
         // std::cerr << "Warning: contains exceeded max attempts for key." << std::endl; // Use cerr
         return false; // Fallback after too many retries

    }


    bool add(const T& key) {
         if (static_cast<double>(element_count.load(std::memory_order_acquire)) / capacity > MAX_LOAD_FACTOR) {
              try {
                resize();
              } catch (const std::runtime_error& e) {
                 std::cerr << "Resize failed during add: " << e.what() << std::endl; // Use cerr
                 return false; // Indicate add failure due to resize issue
              }
         }


        size_t start_index = get_index(key);
        size_t current_index = start_index;
        size_t insert_index = capacity; // Invalid index initially
        uint64_t insert_version = 0;
        bool found_existing = false;


        for (int attempt = 0; attempt < 2 * capacity; ++attempt) { // Limit attempts
            current_index = start_index;
            insert_index = capacity;
            found_existing = false;


            // 1. Optimistic Read Phase - Find existing OR first EMPTY/DELETED slot
            do {
                State current_state = table[current_index].state.load(std::memory_order_acquire);
                uint64_t current_version = table[current_index].version.load(std::memory_order_acquire);

                if (current_state == State::OCCUPIED) {
                     // Optimistic check - re-validated under lock later
                     if (table[current_index].value.has_value() && table[current_index].value.value() == key) {
                        found_existing = true;
                        insert_index = current_index; // Use this index for validation
                        insert_version = current_version;
                        break; // Found existing, move to validation
                    }
                } else if (current_state == State::EMPTY || current_state == State::DELETED) {
                     if (insert_index == capacity) { // Record first available slot
                         insert_index = current_index;
                         insert_version = current_version;
                     }
                     if (current_state == State::EMPTY) {
                         break; // Stop probing if EMPTY is found
                     }
                }
                current_index = probe_next(current_index);
            } while (current_index != start_index);

            // Check if table might be full (probed all slots, didn't find empty/deleted, didn't find key)
             if (insert_index == capacity && !found_existing) {
                 // std::cerr << "Warning: add could not find insert slot." << std::endl; // Use cerr
                 continue; // Retry the operation
             }


            // 2. Lock and Validate Phase
            std::lock_guard<std::mutex> guard(table[insert_index].lock);
            State locked_state = table[insert_index].state.load(std::memory_order_acquire);
            uint64_t locked_version = table[insert_index].version.load(std::memory_order_acquire);

            if (found_existing) {
                // Validate if the key is *still* present
                 if (locked_version == insert_version && locked_state == State::OCCUPIED) {
                     if(table[insert_index].value.has_value() && table[insert_index].value.value() == key) {
                         return false; // Key still exists, validation successful, add fails
                     }
                 }
                // Validation failed (state/version changed, or key changed) - retry
            } else {
                // Validate if the slot is *still* available (EMPTY or DELETED)
                if ((locked_state == State::EMPTY || locked_state == State::DELETED)) {
                     // Stricter check: `locked_version == insert_version`
                     if (locked_version == insert_version) {
                           // Perform the insert
                           table[insert_index].value = key;
                           // Increment version *before* setting state to OCCUPIED? Or after? Let's do version then state.
                           table[insert_index].version.fetch_add(1, std::memory_order_release); // Make change visible
                           table[insert_index].state.store(State::OCCUPIED, std::memory_order_release);
                           element_count.fetch_add(1, std::memory_order_relaxed); // Relaxed ok for approximate count
                           return true; // Insert successful
                     }
                }
                // Validation failed (slot taken or version changed) - retry
            }
            // If validation failed, loop continues (retry)
             // Optional: Backoff
             // std::this_thread::yield();
        }

         // std::cerr << "Warning: add exceeded max attempts for key." << std::endl; // Use cerr
         return false; // Fallback after too many retries

    }


    bool remove(const T& key) {
        size_t start_index = get_index(key);
        size_t current_index = start_index;
        size_t remove_index = capacity; // Invalid index initially
        uint64_t remove_version = 0;

        for (int attempt = 0; attempt < 2 * capacity; ++attempt) { // Limit attempts
            current_index = start_index;
            remove_index = capacity;


            // 1. Optimistic Read Phase - Find the key
             do {
                State current_state = table[current_index].state.load(std::memory_order_acquire);
                 uint64_t current_version = table[current_index].version.load(std::memory_order_acquire);


                if (current_state == State::OCCUPIED) {
                    // Optimistic check - re-validated under lock
                     if (table[current_index].value.has_value() && table[current_index].value.value() == key) {
                        remove_index = current_index;
                        remove_version = current_version;
                        break; // Found potential match
                    }
                } else if (current_state == State::EMPTY) {
                    break; // Key cannot be further down the probe sequence
                }
                // If DELETED, continue probing
                current_index = probe_next(current_index);
            } while (current_index != start_index);

            if (remove_index == capacity) {
                // Optimistic probe completed without finding the key
                return false;
            }

            // 2. Lock and Validate Phase
             std::lock_guard<std::mutex> guard(table[remove_index].lock);
             State locked_state = table[remove_index].state.load(std::memory_order_acquire);
             uint64_t locked_version = table[remove_index].version.load(std::memory_order_acquire);


             // Validate version, state, and key match
             if (locked_version == remove_version && locked_state == State::OCCUPIED) {
                  if (table[remove_index].value.has_value() && table[remove_index].value.value() == key) {
                       // Perform the remove (mark as DELETED)
                       table[remove_index].state.store(State::DELETED, std::memory_order_release);
                       table[remove_index].version.fetch_add(1, std::memory_order_release); // Increment version
                       table[remove_index].value.reset(); // Clear the value
                       element_count.fetch_sub(1, std::memory_order_relaxed); // Relaxed ok
                       return true; // Remove successful
                   }
             }
            // Validation failed (state/version changed, or key mismatch) - retry

             // Optional: Backoff
             // std::this_thread::yield();
        }

        // std::cerr << "Warning: remove exceeded max attempts for key." << std::endl; // Use cerr
        return false; // Fallback after too many retries
    }


    // --- Non-Thread-Safe Functions ---

    // Non-thread-safe: Counts elements precisely. Requires external synchronization.
    size_t size() const {
        size_t count = 0;
        for (size_t i = 0; i < capacity; ++i) {
            // No lock needed, assumes no concurrent modifications
            if (table[i].state.load(std::memory_order_relaxed) == State::OCCUPIED) {
                count++;
            }
        }
        return count;
    }

    // Non-thread-safe: Populates the set. Call before concurrent access.
    void populate(size_t num_elements_to_add, int seed = 0) {
        std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
        // Use a large range for integers to reduce collisions during populate
        std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()); // Use full range for T

        std::cerr << "Populating with " << num_elements_to_add << " elements..." << std::endl; // Use cerr
        size_t added_count = 0;
        size_t attempted_count = 0;
        const size_t max_attempts = num_elements_to_add * 5; // Limit attempts to prevent infinite loop if keyspace is small

        while(added_count < num_elements_to_add && attempted_count < max_attempts) {
             attempted_count++;
             // Ensure enough capacity during population phase
             if (static_cast<double>(element_count.load(std::memory_order_relaxed)) / capacity > MAX_LOAD_FACTOR * 0.9) // Resize proactively during populate
             {
                  // Use internal resize (requires locking, but populate is non-concurrent)
                  try {
                     resize();
                  } catch (const std::runtime_error& e) {
                      std::cerr << "Resize failed during populate: " << e.what() << std::endl; // Use cerr
                      return; // Stop populating if resize fails
                  }
             }

            T key = dist(rng);
            if (add(key)) { // Use the public add to ensure resize triggers correctly
                added_count++;
                 if(added_count % (num_elements_to_add / 10 + 1) == 0) { // Progress indicator
                     std::cerr << "  Added " << added_count << "/" << num_elements_to_add << std::endl; // Use cerr
                 }
            }
        }
        if(attempted_count >= max_attempts && added_count < num_elements_to_add) {
            std::cerr << "Warning: Population stopped early after " << max_attempts << " attempts. Added " << added_count << " elements." << std::endl;
        }
        std::cerr << "Population complete. Initial size reported by element_count: " << element_count.load() << std::endl; // Use cerr
        // We don't call size() here as it might be slow for large tables and populate isn't the benchmarked part.
    }


     // Simple function to print the state (for debugging, non-thread-safe)
     void print_state() const {
         std::cerr << "--- Hash Set State (Capacity: " << capacity << ", Elements: " << element_count.load() << ") ---" << std::endl; // Use cerr
         for (size_t i = 0; i < capacity; ++i) {
             State s = table[i].state.load(std::memory_order_relaxed);
             uint64_t v = table[i].version.load(std::memory_order_relaxed);
             std::cerr << "[" << i << "]: "; // Use cerr
             switch(s) {
                 case State::EMPTY:   std::cerr << "EMPTY   (v" << v << ")"; break; // Use cerr
                 case State::OCCUPIED: std::cerr << "OCCUPIED(v" << v << ", val=" << table[i].value.value() << ")"; break; // Use cerr
                 case State::DELETED:  std::cerr << "DELETED (v" << v << ")"; break; // Use cerr
             }
             std::cerr << std::endl; // Use cerr
         }
         std::cerr << "-----------------------------------------" << std::endl; // Use cerr
     }
};
// END PASTE OptimisticFineGrainedHashSet class HERE



// --- Testing Framework ---

template <typename T>
struct ThreadResult {
    // Stats primarily for debugging, not needed for script output
    int adds_attempted = 0;
    int adds_succeeded = 0;
    int removes_attempted = 0;
    int removes_succeeded = 0;
    int contains_attempted = 0;
    int contains_found = 0;
    int contains_notfound = 0;
    long long size_delta = 0; // Net change in size from this thread's successful ops
};


template <typename T>
void worker_thread(OptimisticFineGrainedHashSet<T>& set,
                   int num_operations, // Now ops *per thread*
                   std::mt19937& rng, // Pass generator by ref
                   std::atomic<long long>& global_size_delta, // Use atomic for global delta
                   ThreadResult<T>& result)
{
    // Use thread-specific distribution if needed, or share if appropriate
     std::uniform_int_distribution<T> key_dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()); // Use full range
     std::uniform_int_distribution<int> op_dist(1, 100);   // For operation type %

    long long local_delta = 0;

    for (int i = 0; i < num_operations; ++i) {
        T key = key_dist(rng);
        int op_type = op_dist(rng);

        if (op_type <= 80) { // 80% Contains
            result.contains_attempted++;
            // We don't strictly need to record contains success for the benchmark result
            [[maybe_unused]] bool found = set.contains(key); // Call the function
            // if (found) { result.contains_found++; } else { result.contains_notfound++; }
        } else if (op_type <= 90) { // 10% Add
            result.adds_attempted++;
             if (set.add(key)) {
                 result.adds_succeeded++;
                 local_delta++;
             }
        } else { // 10% Remove
             result.removes_attempted++;
             if (set.remove(key)) {
                 result.removes_succeeded++;
                 local_delta--;
             }
        }
    }
    result.size_delta = local_delta;
    global_size_delta.fetch_add(local_delta, std::memory_order_relaxed); // Atomically update global delta
}

// MODIFIED: run_test now returns the duration in microseconds
template <typename T>
long long run_test(OptimisticFineGrainedHashSet<T>& set, int num_threads, int total_operations) {
    std::vector<std::thread> threads;
    std::vector<ThreadResult<T>> results(num_threads);
    std::atomic<long long> global_size_delta(0); // Shared atomic delta

    // Calculate operations per thread
    if (num_threads <= 0) {
        std::cerr << "Error: Number of threads must be positive." << std::endl;
        return -1;
    }
    int ops_per_thread = total_operations / num_threads;
    int remainder_ops = total_operations % num_threads; // Distribute remainder ops
     if (ops_per_thread == 0 && remainder_ops > 0) { // Ensure at least some ops if total < threads
          ops_per_thread = 1;
          remainder_ops = 0; // Handled by giving first few threads 1 op
          std::cerr << "Warning: Total operations less than threads, running 1 op per thread where possible." << std::endl;
     } else if (ops_per_thread == 0 && remainder_ops == 0){
          std::cerr << "Warning: Zero operations requested." << std::endl;
          return 0; // No work to do
     }


    // Seed RNGs uniquely for each thread
    std::random_device rd;
    std::vector<std::mt19937> rngs;
    for(int i=0; i<num_threads; ++i) {
        rngs.emplace_back(rd() + i); // Add thread index for better seed uniqueness
    }


    std::cerr << "\n--- Starting Benchmark ---" << std::endl; // Use cerr
    std::cerr << "Threads: " << num_threads << std::endl; // Use cerr
    std::cerr << "Total Operations: " << total_operations << std::endl; // Use cerr
    // size_t initial_size = set.size(); // Getting size might be slow, skip for benchmark focus
    // std::cerr << "Initial size (approx pre-benchmark): " << set.element_count.load() << std::endl; // Use cerr


    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        int current_ops = ops_per_thread + (i < remainder_ops ? 1 : 0); // Distribute remainder
         if (current_ops > 0) { // Only start threads if they have work
             threads.emplace_back(worker_thread<T>,
                                  std::ref(set),
                                  current_ops,
                                  std::ref(rngs[i]), // Pass thread-local RNG
                                  std::ref(global_size_delta),
                                  std::ref(results[i]));
         }
    }

    for (auto& t : threads) {
        if (t.joinable()) { // Check if thread was actually created
           t.join();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    // MODIFIED: Calculate duration in microseconds
    long long total_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    std::cerr << "--- Benchmark Complete ---" << std::endl; // Use cerr
    std::cerr << "Total wall clock time: " << total_duration_us << " us (" << total_duration_us / 1000.0 << " ms)" << std::endl; // Use cerr


    // Verification (optional, keep using cerr)
    /*
    std::cerr << "\n--- Verification ---" << std::endl;
    size_t final_size = set.size(); // Non-thread-safe call after join
    long long expected_final_size = static_cast<long long>(set.element_count.load()) ; // Approx expected based on atomic counter START value + delta ? No, initial size needed.
    // Rerun size or trust delta: Let's just report delta for info.
    std::cerr << "Global Size Delta Recorded: " << global_size_delta.load() << std::endl;
    std::cerr << "Final Approx Size (element_count): " << set.element_count.load() << std::endl;
    // Precise verification is hard without knowing exact initial size reliably and quickly.
    // size_t precise_final_size = set.size(); // May be slow
    // std::cerr << "Final Precise Size (slow count): " << precise_final_size << std::endl;
    */

    return total_duration_us; // Return duration for the script
}


int main(int argc, char* argv[]) {
    // --- Configuration ---
    size_t initial_capacity = 1000000; // Requested initial capacity
    size_t populate_size = 500000;    // Requested initial fill size

    int num_threads;
    int total_operations;

    // --- Argument Parsing ---
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <total_operations>" << std::endl;
        return 1;
    }

    try {
        num_threads = std::stoi(argv[1]);
        total_operations = std::stoi(argv[2]);
        if (num_threads <= 0 || total_operations < 0) {
             throw std::invalid_argument("Threads must be positive, operations non-negative.");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return 1;
    }

    // --- Setup ---
    // Pass desired initial capacity to constructor
    OptimisticFineGrainedHashSet<int> my_set(initial_capacity);

    // --- Population (Non-Thread-Safe) ---
    my_set.populate(populate_size);

    // --- Run Concurrent Test ---
    long long duration_us = run_test(my_set, num_threads, total_operations);

    // --- Output for Script ---
    // Print ONLY the duration in microseconds to stdout
    if (duration_us >= 0) {
       std::cout << duration_us << std::endl;
    } else {
       return 1; // Indicate error if benchmark failed
    }


    return 0;
}