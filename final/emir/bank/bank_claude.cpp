#include <iostream>
#include <unordered_map>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <future>
#include <shared_mutex>
#include <atomic>
#include <iomanip>

class BankAccountSystem {
private:
    // Use shared_mutex for reader-writer lock pattern
    mutable std::shared_mutex bank_mutex;
    std::unordered_map<int, int> accounts;
    std::mt19937 rng;

public:
    BankAccountSystem(size_t num_accounts = 250) : rng(std::random_device{}()) {
        // Step 2: Initialize accounts
        std::uniform_int_distribution<int> dist(1, 500); // Random initial balances
        
        int total_balance = 0;
        for (int i = 0; i < num_accounts - 1; ++i) {
            int balance = dist(rng);
            accounts[i] = balance;
            total_balance += balance;
        }
        
        // Make sure the last account balances to the total of 100000
        accounts[num_accounts - 1] = 100000 - total_balance;
    }

    // Step 3: Atomic deposit function
    bool deposit(int from_id, int to_id, int amount) {
        // Exclusive lock for writing
        std::unique_lock<std::shared_mutex> lock(bank_mutex);
        
        auto from_iter = accounts.find(from_id);
        auto to_iter = accounts.find(to_id);
        
        if (from_iter == accounts.end() || to_iter == accounts.end()) {
            return false;
        }
        
        if (from_iter->second < amount) {
            return false; // Insufficient funds
        }
        
        from_iter->second -= amount;
        to_iter->second += amount;
        
        return true;
    }

    // Step 4: Atomic balance function
    int balance() const {
        // Shared lock for reading (allows multiple concurrent readers)
        std::shared_lock<std::shared_mutex> lock(bank_mutex);
        
        int total = 0;
        // Use a basic accumulation - this could be parallelized in a more complex implementation
        // for very large maps, but would need to handle the shared lock carefully
        for (const auto& [id, amount] : accounts) {
            total += amount;
        }
        
        return total;
    }

    // Helper function to get random account IDs
    std::pair<int, int> getRandomAccounts() {
        std::vector<int> ids;
        {
            // Shared lock for reading the account IDs
            std::shared_lock<std::shared_mutex> lock(bank_mutex);
            ids.reserve(accounts.size());
            for (const auto& [id, _] : accounts) {
                ids.push_back(id);
            }
        }
        
        std::uniform_int_distribution<size_t> id_dist(0, ids.size() - 1);
        
        int first_idx = id_dist(rng);
        int second_idx;
        do {
            second_idx = id_dist(rng);
        } while (second_idx == first_idx);
        
        return {ids[first_idx], ids[second_idx]};
    }

    // Helper function to get random amount
    int getRandomAmount() {
        std::uniform_int_distribution<int> amount_dist(1, 100);
        return amount_dist(rng);
    }
};

// Step 5: Worker function
std::chrono::milliseconds do_work(BankAccountSystem& bank, int num_iterations, int thread_id) {
    std::random_device rd;
    std::mt19937 gen(rd() + thread_id); // Adding thread_id helps avoid same seed across threads
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Thread-local RNGs for high-performance parallel execution
    thread_local std::mt19937 local_gen(rd() + thread_id * 1000);
    
    for (int i = 0; i < num_iterations; ++i) {
        double chance = dist(local_gen);
        
        if (chance < 0.95) { // 95% chance for deposit
            auto [from_id, to_id] = bank.getRandomAccounts();
            int amount = bank.getRandomAmount();
            bank.deposit(from_id, to_id, amount);
        } else { // 5% chance for balance check
            int total = bank.balance();
            // Verify balance is correct, but don't output to avoid I/O contention
            if (total != 100000) {
                std::cerr << "Balance error detected: " << total << std::endl;
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
}

int main(int argc, char* argv[]) {
    int iterations_per_thread = 250000;
    int num_accounts = 250;
    int num_threads = 8;
    
    // Parse command line arguments if provided
    if (argc > 1) iterations_per_thread = std::atoi(argv[1]);
    if (argc > 2) num_accounts = std::atoi(argv[2]);
    if (argc > 3) num_threads = std::atoi(argv[3]);
    
    BankAccountSystem bank(num_accounts); // Use the provided number of accounts
    
    std::cout << "Starting bank system with " << num_threads << " threads" << std::endl;
    std::cout << "Number of accounts: " << num_accounts << std::endl;
    std::cout << "Iterations per thread: " << iterations_per_thread << std::endl;
    std::cout << "Initial balance: " << bank.balance() << std::endl;
    
    // Step 6: Create threads and collect execution times
    std::vector<std::future<std::chrono::milliseconds>> futures;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        futures.push_back(
            std::async(std::launch::async, do_work, 
                      std::ref(bank), iterations_per_thread, i)
        );
    }
    
    // Wait for all threads and collect times
    std::vector<std::chrono::milliseconds> thread_times;
    for (auto& future : futures) {
        thread_times.push_back(future.get());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Verify final balance
    int final_balance = bank.balance();
    std::cout << "Final balance: " << final_balance << std::endl;
    std::cout << "Total execution time: " << total_time.count() << " ms" << std::endl;
    
    // Report per-thread statistics
    std::cout << "\nThread execution times (milliseconds):" << std::endl;
    for (int i = 0; i < thread_times.size(); ++i) {
        std::cout << "Thread " << i << ": " << thread_times[i].count() << " ms" << std::endl;
    }
    
    // Calculate metrics
    double avg_time = 0.0;
    for (const auto& time : thread_times) {
        avg_time += time.count();
    }
    avg_time /= thread_times.size();
    
    std::cout << "\nAverage thread execution time: " << avg_time << " ms" << std::endl;
    std::cout << "Transactions per second: " 
              << static_cast<double>(num_threads * iterations_per_thread) / (total_time.count() / 1000.0)
              << std::endl;
    
    return final_balance == 100000 ? 0 : 1;
}