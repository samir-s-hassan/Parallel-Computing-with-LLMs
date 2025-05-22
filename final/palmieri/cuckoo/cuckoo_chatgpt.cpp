// fpc_chat.cpp
// g++ -O3 -std=c++20 fpc_chat.cpp -pthread -o fpc_chat

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <new>
#include <random>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>

// ---------------------------------
// lock-free open-addressed set
template <typename Key>
class LockFreeHashSet
{
    static_assert(std::is_integral_v<Key>, "Key must be integral");
    static_assert(sizeof(Key) * 8 <= 64, "Key must fit in 64 bits");

    enum State : uint64_t
    {
        Empty = 0,
        Occupied = 1,
        Tombstone = 2
    };

    struct alignas(8) Cell
    {
        std::atomic<uint64_t> word;
        Cell() : word(0) {}
    };

    size_t N, mask;
    Cell *table;
    std::hash<Key> hash_fn;

    static size_t next_power_of_two(size_t n)
    {
        size_t p = 1;
        while (p < n)
            p <<= 1;
        return p;
    }

    static uint64_t pack(Key k, State s)
    {
        // mask off top 2 bits so (v<<2) never overflows
        uint64_t v = uint64_t(k) & ((uint64_t(1) << 62) - 1);
        return (v << 2) | uint64_t(s);
    }
    static Key unpack_key(uint64_t w) { return Key(w >> 2); }
    static State unpack_state(uint64_t w) { return State(w & 0x3); }

public:
    explicit LockFreeHashSet(size_t capacity)
        : N(next_power_of_two(capacity)), mask(N - 1)
    {
        table = static_cast<Cell *>(
            ::operator new[](N * sizeof(Cell), std::align_val_t(alignof(Cell))));
        for (size_t i = 0; i < N; ++i)
            new (&table[i]) Cell();
    }
    ~LockFreeHashSet()
    {
        for (size_t i = 0; i < N; ++i)
            table[i].~Cell();
        ::operator delete[](table, std::align_val_t(alignof(Cell)));
    }

    bool add(const Key &key)
    {
        uint64_t packed = pack(key, Occupied);
        size_t idx = hash_fn(key) & mask;
        size_t backoff = 1;
        for (size_t probes = 0; probes < N; ++probes)
        {
            uint64_t w = table[idx].word.load(std::memory_order_acquire);
            State s = unpack_state(w);
            if (s == Empty || s == Tombstone)
            {
                uint64_t expected = w;
                if (table[idx].word.compare_exchange_weak(
                        expected, packed,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire))
                    return true;
            }
            else if (s == Occupied && unpack_key(w) == key)
            {
                return false;
            }
            for (size_t k = 0; k < backoff; ++k)
                __asm__ volatile("pause");
            if (backoff < 64)
                backoff <<= 1;
            idx = (idx + 1) & mask;
        }
        throw std::runtime_error("HashSet full");
    }

    bool remove(const Key &key)
    {
        uint64_t tomb = pack(key, Tombstone);
        size_t idx = hash_fn(key) & mask;
        size_t backoff = 1;
        for (size_t probes = 0; probes < N; ++probes)
        {
            uint64_t w = table[idx].word.load(std::memory_order_acquire);
            if (unpack_state(w) == Occupied && unpack_key(w) == key)
            {
                uint64_t expected = w;
                if (table[idx].word.compare_exchange_weak(
                        expected, tomb,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire))
                    return true;
            }
            else if (unpack_state(w) == Empty)
            {
                return false;
            }
            for (size_t k = 0; k < backoff; ++k)
                __asm__ volatile("pause");
            if (backoff < 64)
                backoff <<= 1;
            idx = (idx + 1) & mask;
        }
        return false;
    }

    bool contains(const Key &key) const
    {
        size_t idx = hash_fn(key) & mask;
        for (size_t probes = 0; probes < N; ++probes)
        {
            uint64_t w = table[idx].word.load(std::memory_order_acquire);
            State s = unpack_state(w);
            if (s == Occupied && unpack_key(w) == key)
                return true;
            if (s == Empty)
                return false;
            idx = (idx + 1) & mask;
        }
        return false;
    }

    size_t size() const
    {
        size_t cnt = 0;
        for (size_t i = 0; i < N; ++i)
            if (unpack_state(table[i].word.load(std::memory_order_acquire)) == Occupied)
                ++cnt;
        return cnt;
    }

    void populate(size_t count, Key max_key = Key(0))
    {
        std::mt19937_64 gen(std::random_device{}());
        uint64_t upper = max_key ? uint64_t(max_key) : uint64_t(N * 4 - 1);
        std::uniform_int_distribution<uint64_t> dist(0, upper);
        for (size_t i = 0; i < count; ++i)
            add(Key(dist(gen)));
    }
};
// ---------------------------------

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <threads> <total_ops>\n";
        return 1;
    }
    int T = std::atoi(argv[1]);
    long total_ops = std::atol(argv[2]);
    if (T <= 0 || total_ops <= 0)
    {
        std::cerr << "threads and total_ops must be > 0\n";
        return 1;
    }

    const size_t INITIAL_CAP = 1'000'000;
    const size_t INITIAL_POP = 500'000;

    LockFreeHashSet<uint64_t> set(INITIAL_CAP);
    set.populate(INITIAL_POP);

    size_t init_sz = set.size();
    std::cerr << "Initial size: " << init_sz << "\n";

    // split total_ops evenly among threads
    std::vector<long> ops_per_thread(T, total_ops / T);
    long rem = total_ops % T;
    for (int i = 0; i < rem; ++i)
        ops_per_thread[i]++;

    std::atomic<size_t> total_adds{0}, total_rems{0};

    auto worker = [&](int id, long ops)
    {
        std::mt19937_64 gen(std::random_device{}() ^ (uint64_t(id) << 32));
        std::uniform_int_distribution<uint64_t> key_dist(0, INITIAL_CAP * 4 - 1);
        std::uniform_int_distribution<int> op_dist(1, 100);
        size_t local_add = 0, local_rem = 0;

        for (long i = 0; i < ops; ++i)
        {
            uint64_t k = key_dist(gen);
            int op = op_dist(gen);
            if (op <= 80)
            {
                set.contains(k);
            }
            else if (op <= 90)
            {
                if (set.add(k))
                    local_add++;
            }
            else
            {
                if (set.remove(k))
                    local_rem++;
            }
        }
        total_adds.fetch_add(local_add, std::memory_order_relaxed);
        total_rems.fetch_add(local_rem, std::memory_order_relaxed);
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int i = 0; i < T; ++i)
        threads.emplace_back(worker, i, ops_per_thread[i]);
    for (auto &th : threads)
        th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
            .count();

    // validate
    size_t expected = init_sz + total_adds - total_rems;
    size_t actual = set.size();
    if (actual != expected)
    {
        std::cerr << "Validation failed: expected=" << expected
                  << " actual=" << actual << "\n";
        return 2;
    }

    // print duration for benchmark script
    std::cout << std::fixed
              << std::setprecision(3)
              << duration_us
              << "\n";
    return 0;
}
