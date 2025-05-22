// hash_set.h

#ifndef HASH_SET_H
#define HASH_SET_H

#include <vector>
#include <functional>
#include <optional>
#include <stdexcept>
#include <mutex>
#include <atomic>
#include <thread>
#include <random>
#include <iostream>
#include <numeric>
#include <chrono>
#include <memory> // For std::unique_ptr in main

// --- Common Components ---

// Define Cache Line Size (adjust if needed, 64 is common for x86_64)
// Used for padding to prevent false sharing between locks.
constexpr size_t CACHE_LINE_SIZE = 64;

enum class BucketState { EMPTY, OCCUPIED, DELETED };

// Bucket structure holding the value and its state
template<typename T>
struct Bucket {
    std::optional<T> value;
    BucketState state = BucketState::EMPTY;
    // Optional: If sizeof(Bucket<T>) is small, multiple might fit in a cache line.
    // If elements are frequently updated and cause false sharing between adjacent buckets,
    // padding might be needed *here* too, but start without it.
    // alignas(CACHE_LINE_SIZE) char padding_for_bucket_if_needed_[...];
};

// --- Sequential Hash Set ---
template<typename T, typename Hash = std::hash<T>>
class SequentialHashSet {
public:
    explicit SequentialHashSet(size_t capacity) : capacity_(find_next_power_of_2(capacity)), table_(capacity_), count_(0) {
        if (capacity == 0) throw std::invalid_argument("Capacity must be positive.");
        // Using power-of-2 capacity allows using bitwise AND for modulo, which can be faster.
        // capacity_ = capacity; // Alternatively, use the provided capacity directly
        // table_.resize(capacity_);
    }

    // Non-thread-safe populate
    void populate(size_t initial_count, int value_range = 1000000) {
        if (initial_count > capacity_ * 0.75) { // Keep load factor reasonable (e.g., < 0.75)
             std::cerr << "Warning: High initial population count requested (" << initial_count
                       << " for capacity " << capacity_ << "). Target load factor > 0.75. May lead to poor performance." << std::endl;
        }
        count_ = 0; // Reset count
        std::fill(table_.begin(), table_.end(), Bucket<T>{}); // Clear table

        std::random_device rd;
        std::mt19937 gen(rd());
        // Ensure T can be generated from int; handle other types if necessary
        if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
            std::uniform_int_distribution<int> distrib(0, value_range);
            size_t added = 0;
            size_t attempts = 0;
            size_t max_attempts = initial_count * 3 + 100; // Limit attempts to avoid infinite loop if values collide too much

            while (added < initial_count && attempts < max_attempts) {
                T item = static_cast<T>(distrib(gen));
                if (add(item)) {
                    added++;
                }
                attempts++;
            }
            if (added != initial_count) {
                 std::cerr << "Warning: Could only populate " << added << " elements out of requested " << initial_count
                           << " (after " << attempts << " attempts)." << std::endl;
            }
        } else {
            // Provide a way to generate random T or throw error
            throw std::logic_error("Populate currently only supports integral/floating-point types for random generation.");
        }
    }

    bool add(const T& item) {
        size_t index = find_slot_for_add(item);

        if (table_[index].state == BucketState::OCCUPIED) {
            return false; // Already exists
        }

        // Found an EMPTY or DELETED slot
        // Since we don't resize, check if adding exceeds a logical capacity limit (optional)
        // if (count_ >= capacity_) return false; // Or based on load factor

        table_[index].value = item;
        table_[index].state = BucketState::OCCUPIED;
        count_++; // Increment count of occupied slots
        return true;
    }

    bool remove(const T& item) {
        size_t index = find_slot_for_contains_remove(item);
        if (table_[index].state == BucketState::OCCUPIED && table_[index].value.value() == item) {
            table_[index].state = BucketState::DELETED;
            table_[index].value.reset(); // Clear the value
            count_--;
            return true;
        }
        return false;
    }

    bool contains(const T& item) const {
        size_t index = find_slot_for_contains_remove(item);
        // Check state AND value equality
        return table_[index].state == BucketState::OCCUPIED && table_[index].value.value() == item;
    }

    // Non-thread-safe size. Returns the internally tracked count.
    size_t size() const {
        // This is fast but relies on add/remove correctly maintaining count_.
        return count_;
        /* // Alternative: Recalculate (slower, but verifies count_)
        size_t current_count = 0;
        for(const auto& bucket : table_) {
           if (bucket.state == BucketState::OCCUPIED) {
               current_count++;
           }
        }
        return current_count;
        */
    }

    size_t get_capacity() const { return capacity_; }


private:
    size_t capacity_;
    std::vector<Bucket<T>> table_;
    size_t count_; // Number of OCCUPIED slots
    Hash hash_fn_;

    // Helper to find the next power of 2 >= n (optional, for modulo optimization)
    size_t find_next_power_of_2(size_t n) {
        if (n == 0) return 1;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        if constexpr (sizeof(size_t) > 4) { // Support 64-bit size_t
             n |= n >> 32;
        }
        n++;
        return n;
    }

    // Hash function application with modulo
    size_t get_index(const T& item) const {
        // If capacity is power of 2: return hash_fn_(item) & (capacity_ - 1);
        return hash_fn_(item) % capacity_;
    }

    // Finds the index where the item IS or SHOULD BE inserted.
    // Stops at the first EMPTY slot or the item itself if found.
    // Can insert into the first DELETED slot encountered *if* the item isn't found later.
    size_t find_slot_for_add(const T& item) {
        size_t h = get_index(item);
        size_t index = h;
        size_t first_deleted = -1; // Sentinel value indicating not found yet
        size_t probe_count = 0;

        while (probe_count < capacity_) {
            const auto& bucket = table_[index];

            if (bucket.state == BucketState::EMPTY) {
                // Found empty slot. If we saw a deleted slot earlier, use that. Otherwise use this empty one.
                return (first_deleted != (size_t)-1) ? first_deleted : index;
            }

            if (bucket.state == BucketState::OCCUPIED && bucket.value.value() == item) {
                return index; // Found the item itself
            }

            if (bucket.state == BucketState::DELETED && first_deleted == (size_t)-1) {
                first_deleted = index; // Remember the first deleted slot
            }

            // Continue probing (Linear Probing)
            probe_count++;
            index = (h + probe_count) % capacity_;
             // If capacity is power of 2: index = (h + probe_count) & (capacity_ - 1);

             if (index == h) break; // Should not happen if probe_count < capacity_ unless capacity is 1
        }

        // Table is full or only contains DELETED slots in the probe path.
        // If we found a deleted slot, return it for potential insertion. Otherwise, indicates failure/full.
        // Since we don't resize, this path means we might fail to add if the table is logically full of occupied/deleted items.
        if (first_deleted != (size_t)-1) {
            return first_deleted;
        }
        // If no empty or deleted slot found, and item not present, return the starting index
        // The caller (add) must check the state at the returned index.
        // Or maybe throw an exception? Let's return the starting index, add will see it's occupied/fail.
        return h; // Indicates probe failed to find suitable slot or item
    }

     // Finds the index where the item IS for contains/remove operations.
     // MUST probe past DELETED slots. Stops only at EMPTY or the item itself.
     size_t find_slot_for_contains_remove(const T& item) const {
        size_t h = get_index(item);
        size_t index = h;
        size_t probe_count = 0;

        while (probe_count < capacity_) {
            const auto& bucket = table_[index];

            if (bucket.state == BucketState::EMPTY) {
                return index; // Item not found (hit an empty slot)
            }

            if (bucket.state == BucketState::OCCUPIED && bucket.value.value() == item) {
                return index; // Found the item
            }

            // Continue probing if DELETED or OCCUPIED with a different item
            probe_count++;
            index = (h + probe_count) % capacity_;
             // If capacity is power of 2: index = (h + probe_count) & (capacity_ - 1);

             if (index == h) break; // Wrapped around
        }
        // Item not found after checking all possible slots in probe sequence
        return h; // Return original index; caller must check state.
    }
};


// --- Concurrent Hash Set (Striped Locking with Padding) ---

// Padded mutex to prevent false sharing between adjacent locks
struct alignas(CACHE_LINE_SIZE) PaddedMutex {
    std::mutex mtx;
    // Padding is implicit due to alignas ensuring the start of the next PaddedMutex
    // is at least CACHE_LINE_SIZE bytes away.
};


template<typename T, typename Hash = std::hash<T>>
class ConcurrentHashSet {
public:
    explicit ConcurrentHashSet(size_t capacity, size_t num_stripes = 0)
        : capacity_(find_next_power_of_2(capacity)), table_(capacity_)
        // Note: Internal count_ is not maintained thread-safely per requirements.
        // The test harness tracks size changes externally.
    {
        if (capacity == 0) throw std::invalid_argument("Capacity must be positive.");

        // Determine number of stripes
        if (num_stripes == 0) {
            size_t hardware_threads = std::thread::hardware_concurrency();
            // Heuristic: More stripes than cores can help reduce contention further. Needs tuning.
            num_stripes_ = std::max(1UL, (hardware_threads > 0) ? hardware_threads * 4 : 64);
        } else {
            num_stripes_ = num_stripes;
        }
        // Clamp stripes to avoid excessive numbers or zero
        num_stripes_ = std::max(1UL, std::min(num_stripes_, capacity_));

        // Allocate the padded locks
        locks_ = std::make_unique<PaddedMutex[]>(num_stripes_);
    }

    // Non-thread-safe populate (assumes called before threads start)
    void populate(size_t initial_count, int value_range = 1000000) {
         if (initial_count > capacity_ * 0.75) { // Keep load factor reasonable
              std::cerr << "Warning: High initial population count requested (" << initial_count
                        << " for capacity " << capacity_ << "). Target load factor > 0.75. May lead to poor performance." << std::endl;
         }
         std::fill(table_.begin(), table_.end(), Bucket<T>{}); // Clear table

         std::random_device rd;
         std::mt19937 gen(rd());
         if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
             std::uniform_int_distribution<int> distrib(0, value_range);
             size_t added = 0;
             size_t attempts = 0;
             size_t max_attempts = initial_count * 3 + 100;

             // Use internal add logic *without* locking for population phase
             while (added < initial_count && attempts < max_attempts) {
                 T item = static_cast<T>(distrib(gen));
                 size_t index = find_slot_for_add_internal(item); // Use internal unlocked version
                 if (table_[index].state != BucketState::OCCUPIED) {
                     table_[index].value = item;
                     table_[index].state = BucketState::OCCUPIED;
                     added++;
                 }
                 attempts++;
             }
             if (added != initial_count) {
                  std::cerr << "Warning: Could only populate " << added << " elements out of requested " << initial_count
                            << " (after " << attempts << " attempts)." << std::endl;
             }
         } else {
             throw std::logic_error("Populate currently only supports integral/floating-point types.");
         }
    }

    bool add(const T& item) {
        size_t h = get_index(item);
        std::lock_guard<std::mutex> lock(get_lock(h)); // Lock the specific stripe

        size_t index = find_slot_for_add_internal(item, h); // Search within the lock

        if (table_[index].state == BucketState::OCCUPIED) {
            return false; // Already exists
        }

        // Found EMPTY or DELETED slot suitable for insertion
        table_[index].value = item;
        table_[index].state = BucketState::OCCUPIED;
        // Do NOT update shared count_ here. Rely on caller tracking success.
        return true;
    }

    bool remove(const T& item) {
        size_t h = get_index(item);
        std::lock_guard<std::mutex> lock(get_lock(h)); // Lock the specific stripe

        size_t index = find_slot_for_contains_remove_internal(item, h); // Search within the lock

        if (table_[index].state == BucketState::OCCUPIED && table_[index].value.value() == item) {
            table_[index].state = BucketState::DELETED;
            table_[index].value.reset();
            // Do NOT update shared count_ here. Rely on caller tracking success.
            return true;
        }
        return false;
    }

    // Note: contains is non-const because std::mutex::lock() is non-const.
    // If read performance is critical and writes are relatively rare,
    // std::shared_mutex could be used, making contains const and allowing concurrent reads.
    // But for simplicity and given 20% writes, std::mutex is reasonable.
    bool contains(const T& item) /* const */ { // Cannot be const with std::mutex easily
        size_t h = get_index(item);
        std::lock_guard<std::mutex> lock(get_lock(h)); // Lock the specific stripe

        size_t index = find_slot_for_contains_remove_internal(item, h); // Search within the lock

        return table_[index].state == BucketState::OCCUPIED && table_[index].value.value() == item;
    }

    // Non-thread-safe size. Iterates the whole table without locks.
    // Assumes no other threads are operating on the set.
    size_t size() const {
        size_t current_count = 0;
        for (size_t i = 0; i < capacity_; ++i) {
            // No lock needed here per non-thread-safe requirement
            if (table_[i].state == BucketState::OCCUPIED) {
                current_count++;
            }
        }
        return current_count;
    }

    size_t get_capacity() const { return capacity_; }
    size_t get_num_stripes() const { return num_stripes_; }


private:
    size_t capacity_;
    std::vector<Bucket<T>> table_;
    // size_t count_; // No thread-safe count maintained internally
    Hash hash_fn_;

    size_t num_stripes_;
    std::unique_ptr<PaddedMutex[]> locks_; // Use unique_ptr for dynamic array of padded locks

    // Helper to find the next power of 2 (same as sequential)
    size_t find_next_power_of_2(size_t n) {
        if (n == 0) return 1; n--;
        n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
        if constexpr (sizeof(size_t) > 4) n |= n >> 32;
        n++; return n;
    }

    // Get index using hash and modulo/mask
    size_t get_index(const T& item) const {
         // If capacity is power of 2: return hash_fn_(item) & (capacity_ - 1);
         return hash_fn_(item) % capacity_;
    }

    // Get the lock for the stripe corresponding to a hash value `h`
    std::mutex& get_lock(size_t h) const {
        // Map the bucket index (or hash value) to a lock stripe index
        return locks_[h % num_stripes_].mtx;
    }


    // --- Internal helper functions used *after* lock is acquired ---
    // These perform the actual probing logic, assuming the correct stripe lock is held.
    // They mirror the sequential logic but operate under the lock.

    size_t find_slot_for_add_internal(const T& item, size_t h) {
        size_t index = h; // Start search at the initial hash index 'h'
        size_t first_deleted = -1;
        size_t probe_count = 0;
        while (probe_count < capacity_) {
            const auto& bucket = table_[index];
            if (bucket.state == BucketState::EMPTY) {
                return (first_deleted != (size_t)-1) ? first_deleted : index;
            }
            if (bucket.state == BucketState::OCCUPIED && bucket.value.value() == item) {
                return index;
            }
            if (bucket.state == BucketState::DELETED && first_deleted == (size_t)-1) {
                first_deleted = index;
            }
            probe_count++;
            index = (h + probe_count) % capacity_;
            // Power of 2: index = (h + probe_count) & (capacity_ - 1);
            if (index == h) break;
        }
        return (first_deleted != (size_t)-1) ? first_deleted : h; // Return deleted or original index if full/failed
    }

     size_t find_slot_for_contains_remove_internal(const T& item, size_t h) const {
        size_t index = h;
        size_t probe_count = 0;
        while (probe_count < capacity_) {
            const auto& bucket = table_[index];
            if (bucket.state == BucketState::EMPTY) {
                return index;
            }
            if (bucket.state == BucketState::OCCUPIED && bucket.value.value() == item) {
                return index;
            }
            probe_count++;
            index = (h + probe_count) % capacity_;
            // Power of 2: index = (h + probe_count) & (capacity_ - 1);
            if (index == h) break;
        }
        return h; // Return original index if not found
    }

    // Overloads used by non-thread-safe populate (calculate hash internally)
    size_t find_slot_for_add_internal(const T& item) { return find_slot_for_add_internal(item, get_index(item)); }
    size_t find_slot_for_contains_remove_internal(const T& item) const { return find_slot_for_contains_remove_internal(item, get_index(item)); }


};

#endif // HASH_SET_H