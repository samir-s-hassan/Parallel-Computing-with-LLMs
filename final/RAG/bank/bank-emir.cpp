// main.cpp
#include <iostream>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <shared_mutex>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <string>
#include <iostream>



// total money in cents
static constexpr long long TOTAL_CENTS = 100000LL * 100;
// number of accounts
static int NUM_ACCOUNTS = 1000;
// number of iterations per thread
static long long ITERS_PER_THREAD = 1000000;
// probability of deposit
static constexpr double P_DEPOSIT = 0.95;

// ------------------------------------------------------------------
// SEQUENTIAL VERSION (no locks, no atomics)
// ------------------------------------------------------------------
long long do_work_seq(std::unordered_map<int,long long>& accounts, int seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_int_distribution<int>  acct(0, NUM_ACCOUNTS-1);
    std::uniform_int_distribution<long long> amount(1, TOTAL_CENTS);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (long long i = 0; i < ITERS_PER_THREAD; ++i) {
        if (prob(rng) < P_DEPOSIT) {
            int a = acct(rng), b = acct(rng);
            long long v = amount(rng);
            accounts[a] -= v;
            accounts[b] += v;
        } else {
            long long sum = 0;
            for (auto &kv : accounts)
                sum += kv.second;
            (void)sum;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// ------------------------------------------------------------------
// MULTI‑THREADED VERSION (per‑account atomics + shared_mutex barrier)
// ------------------------------------------------------------------
std::unordered_map<int, std::atomic<long long>> accounts_mt;
std::shared_mutex                     guard;  // to make balance exclusive

void deposit_mt(int a, int b, long long v) {
    std::shared_lock<std::shared_mutex> lk(guard);
    accounts_mt[a].fetch_sub(v, std::memory_order_relaxed);
    accounts_mt[b].fetch_add(v, std::memory_order_relaxed);
}

long long balance_mt() {
    std::unique_lock<std::shared_mutex> lk(guard);
    long long sum = 0;
    for (auto &kv : accounts_mt)
        sum += kv.second.load(std::memory_order_relaxed);
    return sum;
}

long long do_work_mt(int seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_int_distribution<int>  acct(0, NUM_ACCOUNTS-1);
    std::uniform_int_distribution<long long> amount(1, TOTAL_CENTS);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (long long i = 0; i < ITERS_PER_THREAD; ++i) {
        if (prob(rng) < P_DEPOSIT) {
            int a = acct(rng), b = acct(rng);
            long long v = amount(rng);
            deposit_mt(a,b,v);
        } else {
            long long sum = balance_mt();
            (void)sum;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// ------------------------------------------------------------------
// HELPERS
// ------------------------------------------------------------------
void init_seq(std::unordered_map<int,long long>& accounts) {
    accounts.clear();
    long long each = TOTAL_CENTS / NUM_ACCOUNTS;
    for (int i = 0; i < NUM_ACCOUNTS; ++i)
        accounts[i] = each;
}

void init_mt() {
    accounts_mt.clear();
    long long each = TOTAL_CENTS / NUM_ACCOUNTS;
    for (int i = 0; i < NUM_ACCOUNTS; ++i)
        accounts_mt.emplace(i, each);
}

// ------------------------------------------------------------------
// MAIN
// ------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // std::vector<int> threads_list = {1,2,4,8};
    // std::cout << "threads,time_seconds\n";
    int T = std::stoi(argv[1]);
    NUM_ACCOUNTS = std::stoi(argv[2]);
    ITERS_PER_THREAD = std::stoll(argv[3]) / T;
    // for (int T : threads_list) {
        long long duration = 0.0;
        if (T == 1) {
            // pure sequential
            std::unordered_map<int,long long> accounts;
            init_seq(accounts);
            long long t = do_work_seq(accounts, /*seed=*/42);
            // final balance check
            duration = t;
        } else {
            // multi-threaded
            init_mt();
            std::vector<std::future<long long>> futs;
            for (int i = 0; i < T; ++i) {
                futs.emplace_back(
                  std::async(std::launch::async, do_work_mt, /*seed=*/i+100)
                );
            }
            long long sum = 0;
            for (auto &f : futs) {
                sum += f.get();
            }
            duration = sum;

        }
        std::cout << duration << "\n";
    // }
    return 0;
}
