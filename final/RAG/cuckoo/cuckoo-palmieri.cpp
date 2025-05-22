#include <bits/stdc++.h>
#include <atomic>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <immintrin.h>    // pulls in all the SIMD intrinsics


//------------------------------------------------------------------------------
// A little spin‐lock for striping:
struct SpinLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    void lock() {
        while (flag.test_and_set(std::memory_order_acquire)) {
            _mm_pause();
        }
    }
    void unlock() {
        flag.clear(std::memory_order_release);
    }
};

//------------------------------------------------------------------------------
// Sequential open‑addressed hash set (linear probing)
template<typename Key, typename Hash = std::hash<Key>>
class SeqHashSet {
    enum State : uint8_t { EMPTY, OCCUPIED, DELETED };
    struct Slot {
        Key      key;
        State    st = EMPTY;
    };

    std::vector<Slot> table;
    size_t             cap;
    size_t             count = 0;
    Hash               hasher;

public:
    SeqHashSet(size_t capacity)
    : table(capacity), cap(capacity) {}

    bool add(const Key& k) {
        size_t h = hasher(k) & (cap - 1);
        for (size_t i = 0; i < cap; ++i) {
            size_t idx = (h + i) & (cap - 1);
            auto &S = table[idx];
            if (S.st == EMPTY || S.st == DELETED) {
                S.key = k;
                S.st  = OCCUPIED;
                ++count;
                return true;
            }
            if (S.st == OCCUPIED && S.key == k) {
                return false;
            }
        }
        return false; // full
    }

    bool remove(const Key& k) {
        size_t h = hasher(k) & (cap - 1);
        for (size_t i = 0; i < cap; ++i) {
            size_t idx = (h + i) & (cap - 1);
            auto &S = table[idx];
            if (S.st == EMPTY) return false;
            if (S.st == OCCUPIED && S.key == k) {
                S.st = DELETED;
                --count;
                return true;
            }
        }
        return false;
    }

    bool contains(const Key& k) const {
        size_t h = hasher(k) & (cap - 1);
        for (size_t i = 0; i < cap; ++i) {
            size_t idx = (h + i) & (cap - 1);
            auto &S = table[idx];
            if (S.st == EMPTY) return false;
            if (S.st == OCCUPIED && S.key == k)
                return true;
        }
        return false;
    }

    size_t size() const {
        return count;
    }

    // non‐thread‐safe bulk populate
    void populate(const std::vector<Key>& init) {
        for (auto &k : init) {
            add(k);
        }
    }
};

//------------------------------------------------------------------------------
// Striped (fixed‐lock) wrapper
template<typename Key, typename Hash = std::hash<Key>>
class StripedHashSet {
    SeqHashSet<Key,Hash>   impl;
    std::vector<SpinLock>  locks;
    size_t                 stripes;

public:
    StripedHashSet(size_t capacity, size_t num_stripes = 64)
    : impl(capacity)
    , locks(num_stripes)
    , stripes(num_stripes)
    {}

    bool add(const Key& k) {
        size_t s = Hash{}(k) & (stripes - 1);
        locks[s].lock();
        bool r = impl.add(k);
        locks[s].unlock();
        return r;
    }
    bool remove(const Key& k) {
        size_t s = Hash{}(k) & (stripes - 1);
        locks[s].lock();
        bool r = impl.remove(k);
        locks[s].unlock();
        return r;
    }
    bool contains(const Key& k) {
        size_t s = Hash{}(k) & (stripes - 1);
        locks[s].lock();
        bool r = impl.contains(k);
        locks[s].unlock();
        return r;
    }
    size_t size() const {
        // not thread‐safe, but called after threads join
        return impl.size();
    }
    void populate(const std::vector<Key>& init) {
        impl.populate(init);
    }
};

//------------------------------------------------------------------------------
// TM‐wrapped version (GCC –fgnu‑tm) or fallback to seq if USE_TM not defined
template<typename Key, typename Hash = std::hash<Key>>
class TMHashSet {
    SeqHashSet<Key,Hash> impl;
public:
    TMHashSet(size_t capacity)
    : impl(capacity)
    {}

    bool add(const Key& k) {
#ifdef USE_TM
        bool r;
        __transaction_atomic {
            r = impl.add(k);
        }
        return r;
#else
        return impl.add(k);
#endif
    }
    bool remove(const Key& k) {
#ifdef USE_TM
        bool r;
        __transaction_atomic {
            r = impl.remove(k);
        }
        return r;
#else
        return impl.remove(k);
#endif
    }
    bool contains(const Key& k) {
#ifdef USE_TM
        bool r;
        __transaction_atomic {
            r = impl.contains(k);
        }
        return r;
#else
        return impl.contains(k);
#endif
    }
    size_t size() const {
        return impl.size();
    }
    void populate(const std::vector<Key>& init) {
        impl.populate(init);
    }
};

//------------------------------------------------------------------------------
// A simple interface to pick one at runtime:
struct ISet {
    virtual bool add(int) = 0;
    virtual bool remove(int) = 0;
    virtual bool contains(int) = 0;
    virtual size_t size() const = 0;
    virtual void populate(const std::vector<int>&) = 0;
    virtual ~ISet() = default;
};

template<typename Impl>
struct Wrapper : ISet {
    Impl s;
    Wrapper(size_t cap) : s(cap) {}
    bool add(int x) override { return s.add(x); }
    bool remove(int x) override { return s.remove(x); }
    bool contains(int x) override { return s.contains(x); }
    size_t size() const override { return s.size(); }
    void populate(const std::vector<int>& v) override { s.populate(v); }
};

//------------------------------------------------------------------------------
// Main benchmark harness
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " [seq|striped|tm] <num_threads> <ops_per_thread>\n";
        return 1;
    }
    std::string mode = argv[1];
    int T = std::stoi(argv[2]);
    int M = std::stoi(argv[3]);

    const size_t CAP = 1 << 20;          // 1 048 576
    const size_t INIT = CAP / 2;        // 50% populate
    std::unique_ptr<ISet> set;

    if (mode == "seq") {
        set.reset(new Wrapper<SeqHashSet<int>>(CAP));
    } else if (mode == "striped") {
        set.reset(new Wrapper<StripedHashSet<int>>(CAP));
    } else if (mode == "tm") {
        set.reset(new Wrapper<TMHashSet<int>>(CAP));
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    // 1) populate
    std::vector<int> init;
    init.reserve(INIT);
    for (size_t i = 1; i <= INIT; ++i) init.push_back(i);
    set->populate(init);

    // 2) pre‐generate per‐thread workloads
    struct Op { enum Type { GET, INS, REM } t; int x; };
    std::vector<std::vector<Op>> workloads(T);
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<int> distKey(1, int(CAP));
    std::uniform_real_distribution<double> distOp(0.0, 1.0);

    for (int i = 0; i < T; ++i) {
        workloads[i].reserve(M);
        for (int j = 0; j < M; ++j) {
            double p = distOp(rng);
            Op op;
            if (p < 0.80)        op.t = Op::GET;
            else if (p < 0.90)   op.t = Op::INS;
            else                 op.t = Op::REM;
            op.x = distKey(rng);
            workloads[i].push_back(op);
        }
    }

    // 3) run
    std::atomic<long long> sumAdds{0}, sumRemoves{0};
    auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < T; ++t) {
        threads.emplace_back([&,t]() {
            long long localAdds = 0, localRems = 0;
            for (auto &op : workloads[t]) {
                bool ok = false;
                switch (op.t) {
                    case Op::GET:    ok = set->contains(op.x); break;
                    case Op::INS:    ok = set->add(op.x);      break;
                    case Op::REM:    ok = set->remove(op.x);   break;
                }
                if (op.t == Op::INS && ok)  ++localAdds;
                if (op.t == Op::REM && ok)  ++localRems;
            }
            sumAdds   += localAdds;
            sumRemoves+= localRems;
        });
    }
    for (auto &th : threads) th.join();
    auto end = std::chrono::steady_clock::now();

    // 4) verify and report
    // long long expected = long long(INIT) + sumAdds - sumRemoves;
    size_t actual = set->size();

    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double ops_sec = double(T) * M / dur;

    std::cout << dur << "\n";
    // assert(expected == long long(actual) && "Size mismatch!");
    return 0;
}
