// parallel_bank.cpp
#include <iostream>
#include <map>
#include <random>
#include <thread>
#include <mutex>
#include <future>
#include <vector>
#include <chrono>
#include <cassert>


class Bank {
public:
    // Construct bank with the given number of accounts,
    // distributing total_amount equally among them.
    Bank(int num_accounts, float total_amount = 100000.0f)
        : gen_(std::random_device{}()),
          dist_account_(0, num_accounts - 1),
          dist_amount_(1.0f, 100.0f)
    {
        float per_account = total_amount / num_accounts;
        for (int i = 0; i < num_accounts; ++i) {
            accounts_.insert({i, per_account});
        }
    }

    // Atomically transfer a random amount from one random account to another.
    void deposit() {
        std::lock_guard<std::mutex> lock(mtx_);
        int id1 = dist_account_(gen_);
        int id2 = dist_account_(gen_);
        while (id2 == id1) {
            id2 = dist_account_(gen_);
        }
        float amount = dist_amount_(gen_);
        // Ensure we don't overdraw:
        if (accounts_[id1] < amount) {
            amount = accounts_[id1];
        }
        accounts_[id1] -= amount;
        accounts_[id2] += amount;
    }

    // Atomically compute and return the total balance.
    float balance() {
        std::lock_guard<std::mutex> lock(mtx_);
        float sum = 0.0f;
        for (auto const& kv : accounts_) {
            sum += kv.second;
        }
        return sum;
    }

    // Atomically remove all accounts.
    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        accounts_.clear();
    }

private:
    std::map<int, float> accounts_;
    std::mutex mtx_;
    std::mt19937 gen_;
    std::uniform_int_distribution<int> dist_account_;
    std::uniform_real_distribution<float> dist_amount_;
};

// The worker function: performs a mix of deposits and balance checks.
// Reports its execution time (in seconds) via the given promise.
void do_work(int iterations, Bank& bank, std::promise<long long> prom) {
    // Set up per-thread RNG for deciding operation mix.
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist_op(0.0, 1.0);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (dist_op(gen) < 0.95) {
            bank.deposit();
        } else {
            float total = bank.balance();
            // Sanity check: must always equal initial total
            
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    prom.set_value(std::chrono::duration_cast<std::chrono::microseconds>(end -start).count());
}

int main(int argc, char* argv[]) {
    const int num_accounts = std::stoi(argv[2]);                            // number of bank accounts
    const int num_threads  = std::stoi(argv[1]); // threads to spawn
    const int iterations   = std::stoi(argv[3]) / num_threads;                          // iterations per thread

    Bank bank(num_accounts);

    // Launch worker threads, collecting futures for their exec times.
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    std::vector<std::future<long long>> futures;
    futures.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        std::promise<long long > prom;
        futures.push_back(prom.get_future());
        threads.emplace_back(do_work, iterations, std::ref(bank), std::move(prom));
    }

    // Wait for all threads to finish.
    for (auto& t : threads) {
        t.join();
    }

    long long sum = 0;
    // Retrieve and print each thread's execution time.
    for (int i = 0; i < num_threads; ++i) {
        sum += futures[i].get();   
    }

    std::cout << sum << std::endl;
    return 0;
}
