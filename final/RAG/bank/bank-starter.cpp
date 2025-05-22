#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <chrono>
#include <thread>
#include <future>
#include <vector>
#include <cmath>

using namespace std;

// Global mutex to protect account operations
mutex accounts_mtx;

// 1) Initialize N accounts splitting TOTAL evenly
void init_accounts(map<int, float>& accounts) {
    const int N = 10;
    const float TOTAL = 100000.0f;
    const float BASE = TOTAL / N;

    accounts.clear();
    for (int i = 0; i < N; ++i) {
        accounts.emplace(i, BASE);
    }

    // Correct any floating-point rounding error
    float sum = 0.0f;
    for (auto const& kv : accounts) sum += kv.second;
    float diff = TOTAL - sum;
    if (fabs(diff) > 1e-3f) {
        accounts.begin()->second += diff;
    }
}

// 2) Deposit: move a random amount between two distinct accounts atomically
void deposit(map<int, float>& accounts,
             mt19937& gen,
             int num_accounts,
             float max_transfer)
{
    uniform_int_distribution<int> acc_dist(0, num_accounts - 1);
    int a = acc_dist(gen), b;
    do { b = acc_dist(gen); } while (b == a);

    uniform_real_distribution<float> amt_dist(0.0f, max_transfer);
    float amount = amt_dist(gen);

    // Atomic transfer via lock_guard
    {
        lock_guard<mutex> lock(accounts_mtx);
        if (accounts[a] >= amount) {
            accounts[a] -= amount;
            accounts[b] += amount;
        }
    }
}

// 3) Balance: sum all accounts under lock to get a consistent snapshot
float balance(const map<int, float>& accounts) {
    lock_guard<mutex> lock(accounts_mtx);
    float sum = 0.0f;
    for (auto const& kv : accounts) sum += kv.second;
    return sum;
}

// 4) do_work: perform iterations of deposit(95%) or balance(5%), timing the loop
void do_work(int iterations,
             map<int, float>& accounts,
             int num_accounts,
             float max_transfer,
             promise<long long> time_promise)
{
    mt19937 gen(random_device{}());
    uniform_real_distribution<float> prob(0.0f, 1.0f);

    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (prob(gen) < 0.95f) {
            deposit(accounts, gen, num_accounts, max_transfer);
        } else {
            balance(accounts);
        }
    }
    auto t1 = chrono::high_resolution_clock::now();

    long long elapsed =std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    time_promise.set_value(elapsed);
}

int main(int argc, char* argv[]) {
    const int thread_counts = stoi(argv[1]);
    const int NUM_ACCOUNTS   = stoi(argv[2]);
    const float MAX_TRANSFER = 50.0f;
    const int iters_per    = stoi(argv[3]) / thread_counts;

    // Store MT timings per thread count
    map<int, long long> timings;

    // Multithreaded runs
    // for (int tcount : thread_counts) {
        map<int, float> accounts;
        init_accounts(accounts);

        vector<thread> threads;
        vector<future<long long>> futures;
        // int iters_per = TOTAL_ITERS / tcount;

        for (int i = 0; i < thread_counts; ++i) {
            promise<long long> p;
            futures.push_back(p.get_future());
            threads.emplace_back(do_work,
                                 iters_per,
                                 ref(accounts),
                                 NUM_ACCOUNTS,
                                 MAX_TRANSFER,
                                 move(p));
        }

        for (auto& th : threads) th.join();

        long long sum = 0;
        for (auto& f : futures) sum += f.get();
        // timings[tcount] = total_time;
        cout << sum << endl;
        // float final_bal = balance(accounts);
        // cout << "[MT] threads=" << tcount
        //      << " time=" << total_time
        //      << "s balance=" << final_bal << "\n";
    // }

    // Single-threaded run (no locking)
    // {
    //     map<int, float> accounts;
    //     init_accounts(accounts);

    //     mt19937 gen(random_device{}());
    //     uniform_real_distribution<float> prob(0.0f, 1.0f);
    //     uniform_int_distribution<int> acc_dist(0, NUM_ACCOUNTS - 1);
    //     uniform_real_distribution<float> amt_dist(0.0f, MAX_TRANSFER);

    //     auto t0 = chrono::high_resolution_clock::now();
    //     for (int i = 0; i < TOTAL_ITERS; ++i) {
    //         if (prob(gen) < 0.95f) {
    //             int a = acc_dist(gen), b;
    //             do { b = acc_dist(gen); } while (b == a);
    //             float amount = amt_dist(gen);
    //             if (accounts[a] >= amount) {
    //                 accounts[a] -= amount;
    //                 accounts[b] += amount;
    //             }
    //         } else {
    //             float s = 0.0f;
    //             for (auto const& kv : accounts) s += kv.second;
    //         }
    //     }
    //     auto t1 = chrono::high_resolution_clock::now();

    //     double single_time = chrono::duration<double>(t1 - t0).count();
    //     timings[0] = single_time;

    //     float final_bal = 0.0f;
    //     for (auto const& kv : accounts) final_bal += kv.second;
    //     cout << "[ST] time=" << single_time
    //          << "s balance=" << final_bal << "\n";
    // }

    return 0;
}
