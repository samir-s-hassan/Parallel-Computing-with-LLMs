#include <iostream>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <random>
#include <thread>
#include <vector>
#include <future>
#include <chrono>
#include <iomanip>
#include <atomic>

class BankSystem {
private:
    // Step 1: Define a map of types <int,float> for bank accounts
    std::unordered_map<int, float> accounts;
    mutable std::shared_mutex mutex; // Reader-writer lock for the accounts map
    std::mt19937 rng; // Random number generator
    std::uniform_int_distribution<int> account_dist; // Account distribution

public:
    BankSystem(int num_accounts = 1000) : rng(std::random_device{}()), account_dist(0, num_accounts - 1) {
        // Step 2: Populate the map in a way that sum of amounts is 100000
        float initial_amount = 100000.0f / num_accounts;

        std::unique_lock<std::shared_mutex> lock(mutex);
        for (int i = 0; i < num_accounts; ++i) {
            accounts[i] = initial_amount;
        }
    }

    // Step 3: Define a function "deposit" that transfers an amount between two accounts atomically
    void deposit() {
        // Generate random account numbers and amount BEFORE acquiring the lock
        int account1, account2;
        float amount;

        // Select two different random accounts
        account1 = account_dist(rng);
        do {
            account2 = account_dist(rng);
        } while (account2 == account1);

        // Generate a random amount to transfer (between 0 and 100)
        std::uniform_real_distribution<float> amount_dist(0.0f, 100.0f);
        amount = amount_dist(rng);

        // Lock the entire map to ensure atomicity
        std::unique_lock<std::shared_mutex> lock(mutex);

        // Verify accounts exist (they might not if we're using a sparse map)
        if (accounts.find(account1) == accounts.end() || accounts.find(account2) == accounts.end()) {
            return;
        }

        // Check if account1 has enough funds
        if (accounts[account1] >= amount) {
            accounts[account1] -= amount;
            accounts[account2] += amount;
        }
    }

    // Step 4: Define a function "balance" that sums all account amounts atomically
    float balance() const {
        // Lock the map for reading to ensure consistency
        std::shared_lock<std::shared_mutex> lock(mutex);

        float total = 0.0f;
        for (const auto& [id, amount] : accounts) {
            total += amount;
        }

        return total;
    }

    // Clear all accounts
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex);
        accounts.clear();
    }

    // Re-initialize the accounts
    void initialize(int num_accounts) {
        float initial_amount = 100000.0f / num_accounts;

        std::unique_lock<std::shared_mutex> lock(mutex);
        accounts.clear();
        for (int i = 0; i < num_accounts; ++i) {
            accounts[i] = initial_amount;
        }

        // Update the account distribution
        account_dist = std::uniform_int_distribution<int>(0, num_accounts - 1);
    }
};

// Step 5: Define a function 'do_work' that calls deposit or balance based on probability
std::chrono::microseconds do_work(BankSystem& bank, int iterations) {
    // Create thread-local random number generator to avoid contention
    thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        if (dist(gen) < 0.95f) {
            // 95% probability: call deposit
            bank.deposit();
        } else {
            // 5% probability: call balance
            float total = bank.balance();
            // Verify the total is still 100000
            if (std::abs(total - 100000.0f) > 0.01f) {
                //std::cerr << "Warning: Balance check failed! Current balance: " << total << std::endl;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
}

int main() {
    std::vector<int> accountCounts = {250, 1000, 10000};
    std::vector<int> iterationCounts = {1'000'000, 10'000'000};
    std::vector<int> thread_counts = {2, 4, 8, 16};

    std::cout << "Bank Account Simulation" << std::endl;
    std::cout << "----------------------" << std::endl;

    // Results for plotting
    std::vector<std::tuple<int, int, int, double>> results; // num_accounts, iterations, threads, time

    // Run with different combinations
    for (int NUM_ACCOUNTS : accountCounts) {
        BankSystem bank(NUM_ACCOUNTS);

        for (int ITERATIONS_PER_THREAD : iterationCounts) {
            for (int num_threads : thread_counts) {
                // Re-initialize the bank system for each run
                bank.initialize(NUM_ACCOUNTS);

                int iterations_per_thread = ITERATIONS_PER_THREAD / num_threads;

                std::cout << "Running with " << num_threads << " threads..." << std::endl;
                std::cout << "Running with " << iterations_per_thread << " iterations per thread..." << std::endl;
                std::cout << "Running with " << NUM_ACCOUNTS << " accounts..." << std::endl;

                // Step 6: Create threads and collect execution times
                std::vector<std::future<std::chrono::microseconds>> futures;

                auto start_time = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < num_threads; ++i) {
                    futures.push_back(
                        std::async(std::launch::async, do_work, std::ref(bank), iterations_per_thread)
                    );
                }

                // Wait for all threads to complete and collect their execution times
                std::vector<std::chrono::microseconds> thread_times;
                for (auto& future : futures) {
                    thread_times.push_back(future.get());
                }

                auto end_time = std::chrono::high_resolution_clock::now();
                auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

                // Verify final balance
                float final_balance = bank.balance();
                std::cout << "Final balance: " << final_balance << std::endl;

                // Calculate average thread execution time
                std::chrono::microseconds avg_thread_time(0);
                for (const auto& time : thread_times) {
                    avg_thread_time += time;
                }
                avg_thread_time /= thread_times.size();

                std::cout << "Total execution time: " << total_time.count() << " ms" << std::endl;
                std::cout << "Average thread execution time: " << avg_thread_time.count() / 1000.0 << " ms" << std::endl;
                std::cout << std::endl;

                results.emplace_back(NUM_ACCOUNTS, ITERATIONS_PER_THREAD, num_threads, total_time.count());
            }
        }
    }

    // Step 8: Print results for plotting
    std::cout << "Results for plotting:" << std::endl;
    std::cout << "Accounts,Iterations,Threads,Execution Time (ms)" << std::endl;
    for (const auto& [accounts, iterations, threads, time] : results) {
        std::cout << accounts << "," << iterations << "," << threads << "," << time << std::endl;
    }

    return 0;
}