 // bank-parallel.cpp
#include <iostream>
#include <memory>
#include <atomic>
#include <shared_mutex>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <algorithm>

// —— global state —— 
static std::unique_ptr<std::atomic<long long>[]> accounts;  // balances in cents
static size_t gNumAccounts = 0;
static std::shared_mutex globalMutex;                       // excludes balance vs deposits

// —— Step 2: init so total = $100 000 —— 
void initAccounts(size_t numAccounts) {
    gNumAccounts = numAccounts;
    accounts = std::make_unique<std::atomic<long long>[]>(numAccounts);

    long long total = 100000LL * 100;           // 100 000 dollars → cents
    long long base  = total / (long long)numAccounts;
    for (size_t i = 0; i < numAccounts; ++i)
        accounts[i].store(base, std::memory_order_relaxed);

    // fix any remainder on the last account
    accounts[numAccounts - 1]
      .fetch_add(total - base * (long long)numAccounts,
                 std::memory_order_relaxed);
}

// —— Step 3: lock‑free atomic transfer under shared lock —— 
inline void deposit(size_t id1, size_t id2, long long V) {
    std::shared_lock<std::shared_mutex> gl(globalMutex);
    accounts[id1].fetch_sub(V, std::memory_order_relaxed);
    accounts[id2].fetch_add(V, std::memory_order_relaxed);
}

// —— Step 4: atomic snapshot under exclusive lock —— 
long long balance() {
    std::unique_lock<std::shared_mutex> ul(globalMutex);
    long long sum = 0;
    for (size_t i = 0; i < gNumAccounts; ++i)
        sum += accounts[i].load(std::memory_order_relaxed);
    return sum;
}

// —— Step 5: worker loop & timing —— 
void worker(size_t /*tid*/, size_t iterations,
            std::promise<long long> prom)
{
    std::mt19937_64                    rng(std::random_device{}());
    std::uniform_real_distribution<>  coin(0.0, 1.0);
    std::uniform_int_distribution<>   pickAcct(0, gNumAccounts - 1);
    std::uniform_int_distribution<long long>
                                      pickAmt(1, 100);

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        if (coin(rng) < 0.95) {
            size_t a = pickAcct(rng), b = pickAcct(rng);
            if (b == a) b = (b + 1) % gNumAccounts;
            deposit(a, b, pickAmt(rng));
        } else {
            (void)balance();
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    prom.set_value(ms);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <iterations> <numAccounts> <numThreads>\n";
        return 1;
    }
    size_t iterations = std::stoull(argv[1]);
    size_t numAccounts = std::stoull(argv[2]);
    size_t numThreads  = std::stoull(argv[3]);

    initAccounts(numAccounts);

    // spawn threads & futures
    std::vector<std::thread> threads;
    std::vector<std::future<long long>> futures;
    threads.reserve(numThreads);
    futures.reserve(numThreads);

    for (size_t t = 0; t < numThreads; ++t) {
        std::promise<long long> p;
        futures.push_back(p.get_future());
        threads.emplace_back(worker, t, iterations, std::move(p));
    }

    for (auto &th : threads) th.join();

    // gather per-thread times
    std::vector<long long> times(numThreads);
    for (size_t t = 0; t < numThreads; ++t)
        times[t] = futures[t].get();

    // find the slowest thread
    auto itMax = std::max_element(times.begin(), times.end());
    size_t slowestTid    = std::distance(times.begin(), itMax);
    long long slowestTime = *itMax;

    // final consistency check
    if (balance() != 100000LL * 100) {
        std::cerr << "ERROR: balance invariant violated\n";
        return 2;
    }

    // —— print exactly as requested —— 
    std::cout << "Thread " << slowestTid
              << " had the longest execution time: "
              << slowestTime << " ms.\n";
    for (size_t t = 0; t < numThreads; ++t) {
        std::cout << "Thread " << t
                  << " execution time (ms): "
                  << times[t] << "\n";
    }
    return 0;
}
