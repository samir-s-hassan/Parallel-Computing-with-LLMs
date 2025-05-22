#include <iostream>
#include <unordered_map>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <chrono>
#include <shared_mutex>
#include <future>
#include <atomic>
#include <iomanip>
#include <cmath>

class BankSystem {
private:
    std::unordered_map<int, double> accounts; // Using double for better precision
    std::shared_mutex map_mutex; // Reader-writer lock for better concurrency
    std::mt19937 rng; // Random number generator
    const double TOTAL_BALANCE = 100000.0; // Constant expected balance
    const double EPSILON = 1e-6; // Tolerance for floating point comparison

public:
    BankSystem(int num_accounts) : rng(std::random_device{}()) {
        // Step 1 & 2: Define and populate the map with accounts
        double initial_amount = TOTAL_BALANCE / num_accounts;
        
        std::unique_lock<std::shared_mutex> lock(map_mutex);
        for (int i = 0; i < num_accounts; ++i) {
            accounts.insert({i, initial_amount});
        }
        
        // Verify initial balance is exact
        double total = 0.0;
        for (const auto& account : accounts) {
            total += account.second;
        }
        
        // Adjust the last account if there's any rounding error in the initial setup
        if (std::abs(total - TOTAL_BALANCE) > EPSILON) {
            int last_id = num_accounts - 1;
            accounts[last_id] += (TOTAL_BALANCE - total);
        }
    }

    // Step 3: Deposit function (transfer funds between accounts)
    void deposit() {
        // Select two random accounts
        std::uniform_int_distribution<int> account_dist(0, accounts.size() - 1);
        int acc1_id, acc2_id;
        
        do {
            acc1_id = account_dist(rng);
            acc2_id = account_dist(rng);
        } while (acc1_id == acc2_id); // Ensure different accounts
        
        // Random amount to transfer (between 0 and 10)
        std::uniform_real_distribution<double> amount_dist(0.0, 10.0);
        double amount = amount_dist(rng);
        
        // Atomic operation using exclusive lock
        std::unique_lock<std::shared_mutex> lock(map_mutex);
        
        // Ensure the source account has enough funds
        if (accounts[acc1_id] >= amount) {
            // Use exact operations to avoid precision loss
            accounts[acc1_id] -= amount;
            accounts[acc2_id] += amount;
        }
    }

    // Step 4: Balance function
    double balance() {
        // Use shared lock for read-only operations
        std::shared_lock<std::shared_mutex> lock(map_mutex);
        
        double total = 0.0;
        for (const auto& account : accounts) {
            total += account.second;
        }
        
        return total;
    }

    // Step 5: Worker function
    static std::chrono::milliseconds do_work(BankSystem& bank, int iterations) {
        std::mt19937 local_rng(std::random_device{}());
        std::uniform_real_distribution<double> probability_dist(0.0, 1.0);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            double probability = probability_dist(local_rng);
            
            if (probability < 0.95) { // 95% probability
                bank.deposit();
            } else { // 5% probability
                double current_balance = bank.balance();
                // Check if balance is within acceptable tolerance
                if (std::abs(current_balance - bank.TOTAL_BALANCE) > bank.EPSILON) {
                    std::cerr << "Balance error detected: " << std::fixed << std::setprecision(6) << current_balance 
                              << " (diff: " << (current_balance - bank.TOTAL_BALANCE) << ")" << std::endl;
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    }

    // Step 8: Clean up resources
    void cleanup() {
        std::unique_lock<std::shared_mutex> lock(map_mutex);
        accounts.clear();
    }
};

int main(int argc, char* argv[]) {
    // Parse command line arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <iterations> <accounts> <threads>" << std::endl;
        return 1;
    }

    int iterations = std::stoi(argv[1]);
    int num_accounts = std::stoi(argv[2]);
    int num_threads = std::stoi(argv[3]);
    
    // Adjust iterations per thread
    int iterations_per_thread = iterations / num_threads;

    // Create the bank system
    BankSystem bank(num_accounts);
    
    // Step 6: Create threads and collect execution times
    std::vector<std::thread> threads;
    std::vector<std::future<std::chrono::milliseconds>> futures;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_threads; ++i) {
        std::promise<std::chrono::milliseconds> promise;
        futures.push_back(promise.get_future());
        
        threads.emplace_back([&bank, iterations_per_thread, p = std::move(promise)]() mutable {
            auto exec_time = BankSystem::do_work(bank, iterations_per_thread);
            p.set_value(exec_time);
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Collect execution times
    std::vector<std::chrono::milliseconds> exec_times;
    for (auto& future : futures) {
        exec_times.push_back(future.get());
    }
    
    // Check final balance - should be exactly 100000.0 without any corrections
    double final_balance = bank.balance();
    
    // Print results in the required format
    std::cout << "Total time: " << total_time.count() << std::endl;
    std::cout << "Final balance: " << std::fixed << std::setprecision(1) << final_balance << std::endl;
    
    for (int i = 0; i < num_threads; ++i) {
        std::cout << "Thread " << i << " execution time (ms): " << exec_times[i].count() << std::endl;
    }
    
    // Step 8: Clean up
    bank.cleanup();
    
    return 0;
}