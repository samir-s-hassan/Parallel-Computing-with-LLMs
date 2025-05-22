#include <vector>
#include <atomic>
#include <thread>
#include <shared_mutex>
#include <mutex>
#include <optional>
#include <random>
#include <iostream>
#include <chrono>

//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Constants and sentinels
//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
constexpr int EMPTY   = std::numeric_limits<int>::min();
constexpr int DELETED = std::numeric_limits<int>::min() + 1;

//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Simple 64-bit mix hash for ints
//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
static inline size_t mix_hash(size_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Base template for open-address set (linear probing), no resizing at runtime.
//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class OpenAddressBase {
protected:
    const size_t capacity;
    std::vector<int> table;              // EMPTY / DELETED / key
    std::atomic<size_t> set_size{0};     // current number of elements

    OpenAddressBase(size_t cap)
      : capacity(cap), table(cap, EMPTY) {}

    // find slot for x; returns index where x is or should be inserted; if table is full, returns npos
    size_t find_slot(int x) const {
        size_t mask = capacity - 1;
        size_t idx  = mix_hash((size_t)x) & mask;
        for (size_t i = 0; i < capacity; ++i) {
            int v = table[(idx + i) & mask];
            if (v == EMPTY || v == x)
                return (idx + i) & mask;
        }
        return SIZE_MAX; // table full
    }

public:
    // Returns true if inserted (i.e. x was not already present)
    virtual bool add(int x) = 0;
    // Returns true if removed (i.e. x was present)
    virtual bool remove(int x) = 0;
    // Returns true if present
    virtual bool contains(int x) = 0;

    // Number of elements
    size_t size() const { return set_size.load(std::memory_order_relaxed); }

    // Fill with 1..n
    void populate(int n) {
        for (int i = 1; i <= n; ++i)
            add(i);
    }

    virtual ~OpenAddressBase() = default;
};

//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// 1) Sequential version – all operations single‐threaded
//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class SequentialHashSet : public OpenAddressBase {
public:
    SequentialHashSet(size_t cap) : OpenAddressBase(cap) {}

    bool add(int x) override {
        size_t slot = find_slot(x);
        if (slot == SIZE_MAX) return false;               // full
        if (table[slot] == x) return false;               // already present
        table[slot] = x;
        set_size.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool remove(int x) override {
        size_t slot = find_slot(x);
        if (slot == SIZE_MAX || table[slot] != x) return false;
        table[slot] = DELETED;
        set_size.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    bool contains(int x) override {
        size_t slot = find_slot(x);
        return slot != SIZE_MAX && table[slot] == x;
    }
};

//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// 2) Locked version – stripe locks with shared_mutex for read‐heavy workloads
//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class LockedHashSet : public OpenAddressBase {
    static constexpr size_t NUM_STRIPES = 256;
    alignas(64) std::shared_mutex stripes[NUM_STRIPES];

    std::shared_mutex& stripe_for(int x) {
        return stripes[mix_hash((size_t)x) & (NUM_STRIPES - 1)];
    }

public:
    LockedHashSet(size_t cap) : OpenAddressBase(cap) {}

    bool add(int x) override {
        auto& m = stripe_for(x);
        std::unique_lock lock(m);
        size_t slot = find_slot(x);
        if (slot == SIZE_MAX || table[slot] == x) return false;
        table[slot] = x;
        set_size.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool remove(int x) override {
        auto& m = stripe_for(x);
        std::unique_lock lock(m);
        size_t slot = find_slot(x);
        if (slot == SIZE_MAX || table[slot] != x) return false;
        table[slot] = DELETED;
        set_size.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    bool contains(int x) override {
        auto& m = stripe_for(x);
        std::shared_lock lock(m);
        size_t slot = find_slot(x);
        return slot != SIZE_MAX && table[slot] == x;
    }
};

//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// 3) Transactional version – wraps operations in HW TM blocks (GCC -fgnu-tm)
//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class TransactionalHashSet : public OpenAddressBase {
public:
    TransactionalHashSet(size_t cap) : OpenAddressBase(cap) {}

    bool add(int x) override {
        bool result = false;
        __transaction_relaxed {
            size_t slot = find_slot(x);
            if (slot != SIZE_MAX && table[slot] != x) {
                table[slot] = x;
                set_size.fetch_add(1, std::memory_order_relaxed);
                result = true;
            }
        }
        return result;
    }

    bool remove(int x) override {
        bool result = false;
        __transaction_relaxed {
            size_t slot = find_slot(x);
            if (slot != SIZE_MAX && table[slot] == x) {
                table[slot] = DELETED;
                set_size.fetch_sub(1, std::memory_order_relaxed);
                result = true;
            }
        }
        return result;
    }

    bool contains(int x) override {
        bool result = false;
        __transaction_relaxed {
            size_t slot = find_slot(x);
            result = (slot != SIZE_MAX && table[slot] == x);
        }
        return result;
    }
};

//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Test harness
//–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
int main(int argc, char* argv[]) {
    constexpr size_t INITIAL_CAPACITY = 1 << 21;  // ~2M slots (power of two)
    constexpr int POPULATE_N      = INITIAL_CAPACITY / 2;
    const unsigned NUM_THREADS   = std::stoi(argv[1]);
    int OPS_PER_THREAD = std::stoi(argv[2]);

    // Pre-generate per-thread operation sequences (char op; int key)
    struct Op { char type; int key; };
    std::vector<std::vector<Op>> thread_ops(NUM_THREADS);
    {
        std::mt19937_64 rng(123456);
        std::uniform_int_distribution<int> dist_key(1, INITIAL_CAPACITY);
        std::uniform_real_distribution<double> dist_op(0.0, 1.0);
        for (unsigned t = 0; t < NUM_THREADS; ++t) {
            thread_ops[t].reserve(OPS_PER_THREAD);
            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                double p = dist_op(rng);
                char  ty = (p < 0.80 ? 'C' : (p < 0.90 ? 'A' : 'R'));
                thread_ops[t].push_back({ty, dist_key(rng)});
            }
        }
    }

    auto run_test = [&](OpenAddressBase& set) -> long long {
        set.populate(POPULATE_N);
        std::atomic<int> succ_add{0}, succ_rem{0};

        auto work = [&](unsigned tid){
            for (auto& op : thread_ops[tid]) {
                if (op.type == 'C') {
                    set.contains(op.key);
                } else if (op.type == 'A') {
                    if (set.add(op.key)) succ_add.fetch_add(1, std::memory_order_relaxed);
                } else {
                    if (set.remove(op.key)) succ_rem.fetch_add(1, std::memory_order_relaxed);
                }
            }
        };

        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> thr;
        thr.reserve(NUM_THREADS);
        for (unsigned t = 0; t < NUM_THREADS; ++t)
            thr.emplace_back(work, t);
        for (auto& th : thr) th.join();
        auto t1 = std::chrono::steady_clock::now();

        size_t expected = POPULATE_N + succ_add.load() - succ_rem.load();
        size_t actual   = set.size();
        if (expected != actual) {
            throw std::runtime_error("Final size mismatch!");
        }

        return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    };


    // 1) Sequential (single‐threaded; still ran with threads but they do no real work in parallel)
    // SequentialHashSet seq(INITIAL_CAPACITY);
    // run_test(seq, "Sequential");

    // 2) Locked
    LockedHashSet locked(INITIAL_CAPACITY);
    long long lock_time = run_test(locked);

    // 3) Transactional
    TransactionalHashSet tx(INITIAL_CAPACITY);
    long long tran_time = run_test(tx);

    std::cout << lock_time << " " << tran_time << std::endl;
    return 0;
}
