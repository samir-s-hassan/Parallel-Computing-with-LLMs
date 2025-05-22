#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// A concurrent open-addressed hash set of ints with striping (fixed number of locks).
class StripedHashSet {
public:
    // state values: EMPTY=0, DELETED=-1, otherwise the stored key (>0)
    static constexpr int EMPTY   =  0;
    static constexpr int DELETED = -1;

    StripedHashSet(size_t table_size, size_t stripes)
      : _size(table_size),
        _stripes(stripes),
        table(table_size),
        locks(stripes)
    {
        // initialize all slots to EMPTY
        for (size_t i = 0; i < _size; i++) {
            table[i].store(EMPTY, std::memory_order_relaxed);
        }
    }

    // Non-thread-safe bulk initialization: insert keys [1..count]
    void populate(int count) {
        assert(count <= (int)_size);
        for (int x = 1; x <= count; x++) {
            // We ignore return value here; assume count <= load capacity
            add(x);
        }
    }

    // Thread-safe contains (lock-free)
    bool contains(int key) const {
        size_t h = hash(key);
        for (size_t i = 0; i < _size; i++) {
            size_t idx = (h + i) % _size;
            int v = table[idx].load(std::memory_order_acquire);
            if (v == EMPTY)       return false;
            if (v == key)         return true;
            // otherwise keep probing
        }
        return false;
    }

    // Thread-safe add
    bool add(int key) {
        size_t h = hash(key);
        for (size_t i = 0; i < _size; i++) {
            size_t idx = (h + i) % _size;
            std::mutex& m = locks[idx % _stripes];
            std::lock_guard<std::mutex> guard(m);

            int v = table[idx].load(std::memory_order_relaxed);
            if (v == key) {
                return false;
            }
            if (v == EMPTY || v == DELETED) {
                table[idx].store(key, std::memory_order_release);
                return true;
            }
            // else occupied by other key → continue probing under lock
        }
        // table is full
        return false;
    }

    // Thread-safe remove
    bool remove(int key) {
        size_t h = hash(key);
        for (size_t i = 0; i < _size; i++) {
            size_t idx = (h + i) % _size;
            std::mutex& m = locks[idx % _stripes];
            std::lock_guard<std::mutex> guard(m);

            int v = table[idx].load(std::memory_order_relaxed);
            if (v == EMPTY) {
                return false;
            }
            if (v == key) {
                table[idx].store(DELETED, std::memory_order_release);
                return true;
            }
            // else keep probing
        }
        return false;
    }

    // Non-thread-safe full scan; only call after all threads have joined
    size_t size() const {
        size_t cnt = 0;
        for (size_t i = 0; i < _size; i++) {
            int v = table[i].load(std::memory_order_relaxed);
            if (v != EMPTY && v != DELETED) ++cnt;
        }
        return cnt;
    }

private:
    size_t _size, _stripes;
    std::vector<std::atomic<int>> table;
    std::vector<std::mutex>       locks;

    static size_t hash(int x) {
        // simple std::hash plus ensure positive
        return std::hash<int>()(x);
    }
};

int main(int argc, char* argv[]) {
    // configuration
    constexpr size_t TABLE_SIZE = 1'000'003;     // prime near 1e6
    constexpr size_t STRIPES    = 128;
    constexpr int    INIT_FILL  = 500'000;       // 50% of TABLE_SIZE
    int    OPS_PER_THREAD = std::stoi(argv[2]);
    int    THREAD_COUNT   = std::stoi(argv[1]);
    constexpr int    KEY_RANGE      = 1'000'000; // keys in [1..1e6]
    constexpr double P_CONTAINS     = 0.80;
    constexpr double P_ADD          = 0.10;
    // P_REMOVE = 0.10

    StripedHashSet set(TABLE_SIZE, STRIPES);
    set.populate(INIT_FILL);

    // Pre-generate all operations
    struct Op { char type; int key; };
    std::vector<Op> ops;
    ops.reserve(THREAD_COUNT * OPS_PER_THREAD);

    std::mt19937_64 rng(123456);
    std::uniform_real_distribution<double> distOp(0.0, 1.0);
    std::uniform_int_distribution<int>    distKey(1, KEY_RANGE);

    for (int t = 0; t < THREAD_COUNT; t++) {
        for (int i = 0; i < OPS_PER_THREAD; i++) {
            double p = distOp(rng);
            char typ = (p < P_CONTAINS      ? 'c'
                        : p < P_CONTAINS+P_ADD ? 'a'
                                                : 'r');
            ops.push_back({ typ, distKey(rng) });
        }
    }

    std::atomic<int> expected_delta{0};

    auto worker = [&](int tid){
        int offset = tid * OPS_PER_THREAD;
        for (int i = 0; i < OPS_PER_THREAD; i++) {
            const Op& op = ops[offset + i];
            switch (op.type) {
                case 'c':
                    set.contains(op.key);
                    break;
                case 'a':
                    if (set.add(op.key)) {
                        expected_delta.fetch_add(1, std::memory_order_relaxed);
                    }
                    break;
                case 'r':
                    if (set.remove(op.key)) {
                        expected_delta.fetch_sub(1, std::memory_order_relaxed);
                    }
                    break;
            }
        }
    };

    // run benchmark
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);
    for (int t = 0; t < THREAD_COUNT; t++) {
        threads.emplace_back(worker, t);
    }
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::steady_clock::now();

    // verify final size

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << "\n";
    return 0;
}
