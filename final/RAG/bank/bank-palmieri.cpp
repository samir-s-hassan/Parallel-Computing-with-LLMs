#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <future>
#include <random>
#include <vector>
#include <chrono>
#include <cassert>

using namespace std;
using Clock = chrono::steady_clock;

// ‑‑ Global data structures
map<int,double> accounts;
map<int,mutex>   acct_mutex;
shared_mutex     balance_mutex;   // reader‐writer lock: deposits take shared, balance takes exclusive

vector<int>      account_ids;     // for fast random pick

// Initialize N accounts summing to TOTAL_FUNDS
void init_accounts(int N, double TOTAL_FUNDS) {
    double per_acct = TOTAL_FUNDS / N;
    for(int i = 0; i < N; ++i) {
        accounts[i] = per_acct;

        // Fix: Default-construct mutex with subscript
        acct_mutex[i];  // This creates a new mutex at key i
        account_ids.push_back(i);
    }
}


// Atomically transfer V from acct A to acct B
void deposit(int a, int b, double v) {
    // 1) Prevent concurrent balance()
    shared_lock<shared_mutex> bal_lk(balance_mutex);
    // 2) Lock the two accounts (in canonical order to avoid deadlock)
    if(a < b) {
        lock(acct_mutex[a], acct_mutex[b]);
        lock_guard<mutex> lk1(acct_mutex[a], adopt_lock);
        lock_guard<mutex> lk2(acct_mutex[b], adopt_lock);
        accounts[a] -= v;
        accounts[b] += v;
    } else {
        lock(acct_mutex[b], acct_mutex[a]);
        lock_guard<mutex> lk1(acct_mutex[b], adopt_lock);
        lock_guard<mutex> lk2(acct_mutex[a], adopt_lock);
        accounts[a] -= v;
        accounts[b] += v;
    }
    // locks release automatically
}

// Atomically compute total balance (must exclude deposits)
double balance() {
    // exclusive lock: no deposits may run concurrently
    unique_lock<shared_mutex> bal_lk(balance_mutex);
    double sum = 0;
    for(auto &kv : accounts) {
        sum += kv.second;
    }
    return sum;
}

// Worker: do ITER ops, 95% deposit, 5% balance; return elapsed seconds
long long do_work(size_t ITER) {
    // Per‐thread RNG
    thread_local mt19937_64 rng{random_device{}()};
    uniform_real_distribution<double> coin(0.0,1.0);
    uniform_int_distribution<size_t>  pick(0, account_ids.size()-1);
    uniform_real_distribution<double> amount(1.0, 100.0);

    auto start = Clock::now();
    for(size_t i = 0; i < ITER; ++i) {
        if(coin(rng) < 0.95) {
            int a = account_ids[pick(rng)];
            int b;
            do { b = account_ids[pick(rng)]; }
            while(b == a);
            deposit(a, b, amount(rng));
        } else {
            volatile double tmp = balance(); 
            (void)tmp;  // drop result
        }
    }
    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

int main(int argc, char* argv[]) {
    const int     NUM_ACCOUNTS  = std::stoi(argv[2]);
    const double  TOTAL_FUNDS   = 100000.0;
    const int     NUM_THREADS   = stoi(argv[1]);
    const long long  ITER_PER_THREAD = std::stoll(argv[3]) / NUM_THREADS;

    // Step 1–2: init
    init_accounts(NUM_ACCOUNTS, TOTAL_FUNDS);

    // Step 3–6: spawn workers via futures
    vector< future<long long> > futures;
    futures.reserve(NUM_THREADS);

    for(int t = 0; t < NUM_THREADS; ++t) {
        futures.emplace_back( async(launch::async,
            [=] { return do_work(ITER_PER_THREAD); }
        ));
    }

    // Collect per‐thread times
    long long sum = 0;

    for(auto &f : futures) {
        sum += f.get();
    }

    // Step 7: report

    cout << sum << endl;

    return 0;
}
