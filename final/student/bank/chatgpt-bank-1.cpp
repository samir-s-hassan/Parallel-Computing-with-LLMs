// bank-parallel.cpp
#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <algorithm>

// --- globals ---
static std::vector<long long> accounts;               // balances in cents
static std::unique_ptr<std::mutex[]> acctMutex;       // per-account mutex array
static std::shared_mutex globalMutex;                 // for deposit vs. balance exclusion

// --- initialize so total = $100 000 ---
void initAccounts(size_t numAccounts) {
    accounts.assign(numAccounts, 0LL);
    long long totalCents = 100'000LL * 100;
    long long base       = totalCents / (long long)numAccounts;
    for (size_t i = 0; i < numAccounts; ++i)
        accounts[i] = base;
    accounts.back() += (totalCents - base * (long long)numAccounts);

    acctMutex = std::make_unique<std::mutex[]>(numAccounts);
}

// --- atomic transfer of V cents from id1→id2 ---
void deposit(size_t id1, size_t id2, long long V) {
    std::shared_lock<std::shared_mutex> gLock(globalMutex);
    size_t lo = std::min(id1, id2), hi = std::max(id1, id2);
    std::unique_lock<std::mutex> l1(acctMutex[lo], std::defer_lock);
    std::unique_lock<std::mutex> l2(acctMutex[hi], std::defer_lock);
    std::lock(l1, l2);
    accounts[id1] -= V;
    accounts[id2] += V;
}

// --- exclusive sum of all balances ---
long long balance() {
    std::unique_lock<std::shared_mutex> gLock(globalMutex);
    long long sum = 0;
    for (auto v : accounts) sum += v;
    return sum;
}

// --- worker: N ops, measure ms, send via promise ---
void worker(size_t /*tid*/,
            size_t iterations,
            size_t numAccounts,
            std::promise<long long> prom)
{
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::uniform_int_distribution<size_t> pickAcct(0, numAccounts - 1);
    std::uniform_int_distribution<long long> pickAmt(1, 100);

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        if (coin(rng) < 0.95) {
            size_t a = pickAcct(rng);
            size_t b = pickAcct(rng);
            if (b == a) b = (b + 1) % numAccounts;
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
    size_t numAccounts= std::stoull(argv[2]);
    size_t numThreads = std::stoull(argv[3]);

    initAccounts(numAccounts);

    // --- start total timer ---
    auto main_t0 = std::chrono::steady_clock::now();

    // spawn threads + futures
    std::vector<std::thread> threads;
    std::vector<std::future<long long>> futures;
    threads.reserve(numThreads);
    futures.reserve(numThreads);
    for (size_t t = 0; t < numThreads; ++t) {
        std::promise<long long> p;
        futures.push_back(p.get_future());
        threads.emplace_back(worker, t, iterations, numAccounts, std::move(p));
    }

    // join all threads
    for (auto &th : threads) th.join();

    // collect per-thread times
    std::vector<long long> times(numThreads);
    for (size_t t = 0; t < numThreads; ++t)
        times[t] = futures[t].get();

    // final sanity-check and balance
    long long finalBal = balance();

    // --- stop total timer ---
    auto main_t1 = std::chrono::steady_clock::now();
    long long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(main_t1 - main_t0).count();

    // --- print results ---
    std::cout << "Total time: " << total_ms << "\n";
    std::cout << "Final balance: " << finalBal << "\n";
    for (size_t t = 0; t < numThreads; ++t) {
        std::cout << "Thread " << t
                << " execution time (ms): " << times[t] << "\n";
    }


    return 0;
}
