/*
 * parallel_hash_set.cpp
 *
 * Lock-free open-addressed hash set implementation
 * - Sequential and concurrent versions
 * - Concurrent version uses per-slot atomics and CAS for add/remove
 * - Slot struct aligned to 64 bytes to avoid false sharing
 * - No resizing to avoid complex synchronization
 * - Contains operations are read-only and lock-free
 * - Thread-local RNG and thread-local counters with 64-byte alignment to avoid false sharing
 * - Parallel operations spawn N threads, each performing M iterations
 * - Avoids shared resource contention and unnecessary synchronization
 * - Uses power-of-two capacity for efficient masking
 * - Suggestion: further optimize contains with SIMD to probe multiple states at once
 */

#include <atomic>
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

// Utility: next power-of-two
static size_t nextPow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Sequential open-addressed hash set (linear probing)
template<typename T>
class SeqHashSet {
public:
    explicit SeqHashSet(size_t capacity)
        : capacity_(capacity), table_(capacity_) {}

    bool add(const T& x) {
        size_t h = hasher_(x) & (capacity_ - 1);
        for (size_t i = 0; i < capacity_; ++i) {
            size_t idx = (h + i) & (capacity_ - 1);
            auto &e = table_[idx];
            if (e.state == EMPTY || e.state == DELETED) {
                e.value = x;
                e.state = OCCUPIED;
                return true;
            }
            if (e.state == OCCUPIED && e.value == x) {
                return false;
            }
        }
        return false;
    }

    bool remove(const T& x) {
        size_t h = hasher_(x) & (capacity_ - 1);
        for (size_t i = 0; i < capacity_; ++i) {
            size_t idx = (h + i) & (capacity_ - 1);
            auto &e = table_[idx];
            if (e.state == EMPTY) return false;
            if (e.state == OCCUPIED && e.value == x) {
                e.state = DELETED;
                return true;
            }
        }
        return false;
    }

    bool contains(const T& x) const {
        size_t h = hasher_(x) & (capacity_ - 1);
        for (size_t i = 0; i < capacity_; ++i) {
            size_t idx = (h + i) & (capacity_ - 1);
            const auto &e = table_[idx];
            if (e.state == EMPTY) return false;
            if (e.state == OCCUPIED && e.value == x) return true;
        }
        return false;
    }

    size_t size() const {
        size_t cnt = 0;
        for (const auto &e : table_) if (e.state == OCCUPIED) ++cnt;
        return cnt;
    }

    void populate(size_t n) {
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<T> dist;
        for (size_t i = 0; i < n; ++i) {
            add(dist(rng) & (capacity_ - 1));
        }
    }

private:
    enum State : uint8_t { EMPTY, OCCUPIED, DELETED };
    struct Entry { T value; State state = EMPTY; };

    size_t capacity_;
    std::vector<Entry> table_;
    std::hash<T> hasher_;
};

// Concurrent open-addressed hash set (lock-free, linear probing)
template<typename T>
class ConcHashSet {
public:
    explicit ConcHashSet(size_t capacity)
        : capacity_(capacity), table_(capacity) {
        for (size_t i = 0; i < capacity_; ++i)
            table_[i].state.store(EMPTY, std::memory_order_relaxed);
    }

    bool add(const T& x) {
        size_t h = hasher_(x) & (capacity_ - 1);
        for (size_t i = 0; i < capacity_; ++i) {
            size_t idx = (h + i) & (capacity_ - 1);
            auto &slot = table_[idx];
            uint8_t st = slot.state.load(std::memory_order_acquire);
            if (st == OCCUPIED) {
                if (slot.value.load(std::memory_order_relaxed) == x) return false;
                continue;
            }
            if (st == EMPTY || st == DELETED) {
                uint8_t expected = st;
                if (slot.state.compare_exchange_strong(
                        expected, BUSY,
                        std::memory_order_acq_rel, std::memory_order_acquire)) {
                    slot.value.store(x, std::memory_order_relaxed);
                    slot.state.store(OCCUPIED, std::memory_order_release);
                    return true;
                }
            }
            // if BUSY or CAS failed, retry
        }
        return false;
    }

    bool remove(const T& x) {
        size_t h = hasher_(x) & (capacity_ - 1);
        for (size_t i = 0; i < capacity_; ++i) {
            size_t idx = (h + i) & (capacity_ - 1);
            auto &slot = table_[idx];
            uint8_t st = slot.state.load(std::memory_order_acquire);
            if (st == EMPTY) return false;
            if (st == OCCUPIED && slot.value.load(std::memory_order_relaxed) == x) {
                if (slot.state.compare_exchange_strong(
                        st, DELETED,
                        std::memory_order_acq_rel, std::memory_order_acquire)) {
                    return true;
                }
                return false;
            }
        }
        return false;
    }

    bool contains(const T& x) const {
        size_t h = hasher_(x) & (capacity_ - 1);
        for (size_t i = 0; i < capacity_; ++i) {
            size_t idx = (h + i) & (capacity_ - 1);
            const auto &slot = table_[idx];
            uint8_t st = slot.state.load(std::memory_order_acquire);
            if (st == EMPTY) return false;
            if (st == OCCUPIED && slot.value.load(std::memory_order_relaxed) == x)
                return true;
        }
        return false;
    }

    // Non-thread-safe size (safe after all threads joined)
    size_t size() const {
        size_t cnt = 0;
        for (const auto &slot : table_)
            if (slot.state.load(std::memory_order_relaxed) == OCCUPIED)
                ++cnt;
        return cnt;
    }

    // Non-thread-safe populate (before concurrent access)
    void populate(size_t n) {
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<T> dist;
        for (size_t i = 0; i < n; ++i) {
            add(dist(rng) & (capacity_ - 1));
        }
    }

private:
    static constexpr uint8_t EMPTY = 0;
    static constexpr uint8_t OCCUPIED = 1;
    static constexpr uint8_t DELETED = 2;
    static constexpr uint8_t BUSY = 3;

    struct alignas(64) Slot {
        std::atomic<T> value;
        std::atomic<uint8_t> state;
    };

    size_t capacity_;
    std::vector<Slot> table_;
    std::hash<T> hasher_;
};

// Thread-local counters with padding to avoid false sharing
struct alignas(64) ThreadCount { size_t adds = 0, removes = 0; };

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <threads> <iterations>\n";
        return 1;
    }
    size_t num_threads = std::stoul(argv[1]);
    size_t iters_per_thread = std::stoul(argv[2]);

    // Capacity = next power-of-two >= 2 * initial population size
    size_t capacity = nextPow2(iters_per_thread * 2);

    // Choose version based on threads
    if (num_threads <= 1) {
        SeqHashSet<size_t> set(capacity);
        set.populate(iters_per_thread);
        size_t init_size = set.size();

        std::mt19937_64 rng(12345);
        std::uniform_real_distribution<double> op_dist(0.0, 1.0);
        std::uniform_int_distribution<size_t> key_dist(0, capacity - 1);

        size_t adds = 0, removes = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iters_per_thread; ++i) {
            double op = op_dist(rng);
            size_t key = key_dist(rng);
            if (op < 0.8) {
                set.contains(key);
            } else if (op < 0.9) {
                if (set.add(key)) ++adds;
            } else {
                if (set.remove(key)) ++removes;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        size_t expected = init_size + adds - removes;
        assert(set.size() == expected);
        double dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << dur;

    } else {
        ConcHashSet<size_t> set(capacity);
        set.populate(iters_per_thread);
        size_t init_size = set.size();

        std::vector<ThreadCount> counters(num_threads);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                std::mt19937_64 rng(t ^ 0xDEADBEEF);
                std::uniform_real_distribution<double> op_dist(0.0, 1.0);
                std::uniform_int_distribution<size_t> key_dist(0, capacity - 1);
                for (size_t i = 0; i < iters_per_thread; ++i) {
                    double op = op_dist(rng);
                    size_t key = key_dist(rng);
                    if (op < 0.8) {
                        set.contains(key);
                    } else if (op < 0.9) {
                        if (set.add(key)) ++counters[t].adds;
                    } else {
                        if (set.remove(key)) ++counters[t].removes;
                    }
                }
            });
        }
        for (auto &th : threads) th.join();
        auto end = std::chrono::high_resolution_clock::now();

        size_t total_adds = 0, total_removes = 0;
        for (auto &c : counters) {
            total_adds += c.adds;
            total_removes += c.removes;
        }
        size_t expected = init_size + total_adds - total_removes;
        assert(set.size() == expected);

        double dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << dur;
    }
    return 0;
}
