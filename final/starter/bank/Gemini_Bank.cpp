#include <iostream>
#include <map>
#include <random>
#include <thread>
#include <vector>
#include <numeric>
#include <chrono>
#include <future>
#include <mutex>

using namespace std;

// Global mutex to ensure atomic operations on the bank accounts map
mutex accounts_mutex;

// Global random number generator
random_device rd;
mt19937 gen(rd());

// Type alias for the bank account map
using BankAccounts = map<int, int>;

// Step 3: Deposit function
void deposit(BankAccounts& accounts, int amount) {
    lock_guard<mutex> lock(accounts_mutex); // Acquire lock for atomic operation

    if (accounts.size() < 2) {
        return; // Need at least two accounts to perform a transfer
    }

    // Select two distinct random accounts
    uniform_int_distribution<> distrib1(0, accounts.size() - 1);
    int index1 = distrib1(gen);
    int index2 = distrib1(gen);
    while (index2 == index1) {
        index2 = distrib1(gen);
    }

    auto it1 = accounts.begin();
    advance(it1, index1);
    auto it2 = accounts.begin();
    advance(it2, index2);

    int account_id1 = it1->first;
    int account_id2 = it2->first;

    // Ensure the first account has enough balance
    if (accounts[account_id1] >= amount) {
        accounts[account_id1] -= amount;
        accounts[account_id2] += amount;
    }
}

// Step 4: Balance function
int balance(const BankAccounts& accounts) {
    lock_guard<mutex> lock(accounts_mutex); // Acquire lock for atomic operation
    int sum = 0;
    for (const auto& pair : accounts) {
        sum += pair.second;
    }
    return sum;
}

// Step 5: do_work function
long long do_work(int iterations, BankAccounts& accounts) {
    auto start_time = chrono::high_resolution_clock::now();

    uniform_real_distribution<> prob_distrib(0.0, 1.0);

    for (int i = 0; i < iterations; ++i) {
        if (prob_distrib(gen) < 0.95) {
            // Deposit with 95% probability
            uniform_int_distribution<> amount_distrib(1, 100); // Random deposit amount
            deposit(accounts, amount_distrib(gen));
        } else {
            // Calculate balance with 5% probability
            balance(accounts); // Just call it, the result isn't directly used in the loop
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    return duration.count();
}

int main() {
    // Step 1 & 2: Initialize the map
    std::vector<int> accountCounts = {250, 1000, 10000};
    std::vector<int> iterationCounts = {1000, 10000, 100000, 1000000, 10000000};
    BankAccounts accounts;


    // Step 6: Multi-threaded execution
    vector<int> num_threads_to_test = {2, 4, 8,16};
    map<int, long long> multi_thread_times;
    for (int NUM_ACCOUNTS : accountCounts) {
        accounts.clear();
        //initialize accounts with a balance of 100000.0
        int initial_balance = 100000.0 / NUM_ACCOUNTS;
        for (int i = 0; i < NUM_ACCOUNTS; ++i) {
            accounts.insert({i, initial_balance});
        }

        for (int ITERATIONS_PER_THREAD : iterationCounts) {
            for (int num_threads : num_threads_to_test) {
                vector<thread> threads;
                vector<future<long long>> futures;
                BankAccounts shared_accounts = accounts; // Create a copy for each test run
                int iterations_per_thread = ITERATIONS_PER_THREAD / num_threads;

                auto start_time = chrono::high_resolution_clock::now();

                for (int i = 0; i < num_threads; ++i) {
                    packaged_task<long long(int, BankAccounts&)> task(do_work);
                    futures.push_back(task.get_future());
                    threads.emplace_back(move(task), iterations_per_thread, ref(shared_accounts));
                }

                long long total_exec_time = 0;
                for (auto& t : threads) {
                    t.join();
                }
                for (auto& f : futures) {
                    total_exec_time += f.get();
                }

                auto end_time = chrono::high_resolution_clock::now();
                auto overall_duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
                multi_thread_times[num_threads] = overall_duration.count();

                cout <<NUM_ACCOUNTS << "," << ITERATIONS_PER_THREAD <<"," <<num_threads << "," << overall_duration.count() << endl;
            }
        }
    }

    cout << "\n--- Multi-threaded Execution Times ---" << endl;
    for (const auto& pair : multi_thread_times) {
        cout << "Number of threads: " << pair.first << ", Execution time: " << pair.second << " ms" << endl;
    }


    return 0;
}