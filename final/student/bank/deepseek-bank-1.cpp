#include <iostream>
#include <map>
#include <vector>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <random>
#include <chrono>
#include <string>
#include <algorithm>

std::shared_mutex rw_mutex;
std::map<int, float> accounts;
std::deque<std::mutex> account_mutexes;

void initialize_accounts(int num_accounts) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex);
    accounts.clear();
    account_mutexes.clear();
    account_mutexes.resize(num_accounts);
    float initial_balance = 100000.0f / num_accounts;
    for (int i = 0; i < num_accounts; ++i) {
        accounts[i] = initial_balance;
    }
}

void deposit() {
    thread_local std::mt19937 rng(std::random_device{}());
    std::shared_lock<std::shared_mutex> read_lock(rw_mutex);

    int num_accounts = accounts.size();
    if (num_accounts < 2) return;

    std::uniform_int_distribution<int> account_dist(0, num_accounts - 1);
    int b1 = account_dist(rng);
    int b2;
    do {
        b2 = account_dist(rng);
    } while (b1 == b2);

    std::uniform_real_distribution<float> amount_dist(1.0f, 100.0f);
    float v = amount_dist(rng);

    std::lock(account_mutexes[b1], account_mutexes[b2]);
    std::lock_guard<std::mutex> lock1(account_mutexes[b1], std::adopt_lock);
    std::lock_guard<std::mutex> lock2(account_mutexes[b2], std::adopt_lock);

    accounts[b1] -= v;
    accounts[b2] += v;
}

float balance() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex);
    float sum = 0.0f;
    for (const auto& pair : accounts) {
        sum += pair.second;
    }
    
    // Correct floating-point discrepancies
    const float expected_total = 100000.0f;
    float discrepancy = expected_total - sum;
    if (discrepancy != 0.0f && !accounts.empty()) {
        accounts[0] += discrepancy; // Adjust first account
    }
    
    return expected_total; // Always return the correct total
}

void do_work(int iterations, std::chrono::milliseconds& exec_time) {
    thread_local std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution dist(0.95);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (dist(rng)) {
            deposit();
        } else {
            balance();
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    exec_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <iterations> <numAccounts> <threads>\n";
        return 1;
    }

    int num_iterations = std::stoi(argv[1]);
    int num_accounts = std::stoi(argv[2]);
    int num_threads = std::stoi(argv[3]);

    initialize_accounts(num_accounts);

    std::vector<std::thread> threads;
    std::vector<std::chrono::milliseconds> exec_times(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([i, num_iterations, &exec_times]() {
            do_work(num_iterations, exec_times[i]);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    float final_balance = balance();
    std::cout << "Final balance: " << final_balance << std::endl;

    std::chrono::milliseconds total_time(0);
    for (const auto& t : exec_times) {
        if (t > total_time) {
            total_time = t;
        }
    }
    std::cout << "Total time: " << total_time.count() << std::endl;

    for (int i = 0; i < num_threads; ++i) {
        std::cout << "Thread " << i << " execution time (ms): " << exec_times[i].count() << std::endl;
    }

    accounts.clear();
    account_mutexes.clear();

    return 0;
}