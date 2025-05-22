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

class BankSystem {
private:
    std::unordered_map<int, float> accounts;
    std::shared_mutex map_mutex; // Reader-writer lock for better concurrency
    std::mt19937 rng; // Random number generator

public:
    BankSystem(int num_accounts) : rng(std::random_device{}()) {
        // Step 1 & 2: Define and populate the map with accounts
        float initial_amount = 100000.0f / num_accounts;
        
        for (int i = 0; i < num_accounts; ++i) {
            accounts.insert({i, initial_amount});
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
        std::uniform_real_distribution<float> amount_dist(0.0f, 10.0f);
        float amount = amount_dist(rng);
        
        // Atomic operation using exclusive lock
        std::unique_lock<std::shared_mutex> lock(map_mutex);
        
        // Ensure the source account has enough funds
        if (accounts[acc1_id] >= amount) {
            accounts[acc1_id] -= amount;
            accounts[acc2_id] += amount;
        }
    }

    // Step 4: Balance function
    float balance() {
        // Use shared lock for read-only operations
        std::shared_lock<std::shared_mutex> lock(map_mutex);
        
        float total = 0.0f;
        for (const auto& account : accounts) {
            total += account.second;
        }
        
        return total;
    }

    // Step 5: Worker function
    static std::chrono::milliseconds do_work(BankSystem& bank, int iterations) {
        std::mt19937 local_rng(std::random_device{}());
        std::uniform_real_distribution<float> probability_dist(0.0f, 1.0f);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            float probability = probability_dist(local_rng);
            
            if (probability < 0.95f) { // 95% probability
                bank.deposit();
            } else { // 5% probability
                float current_balance = bank.balance();
                // Uncomment for debugging: 
                // std::cout << "Thread " << std::this_thread::get_id() << " balance check: " << current_balance << std::endl;
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
    
    // Check final balance
    float final_balance = bank.balance();
    
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