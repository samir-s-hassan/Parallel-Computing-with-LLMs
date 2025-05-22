#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <iomanip>

class BankSystem {
private:
    // Custom account structure with its own mutex
    struct Account {
        float balance;
        mutable std::mutex mutex;
        
        Account(float initial = 0.0f) : balance(initial) {}

        // Delete copy and move operations since mutex is not copyable/movable
        Account(const Account&) = delete;
        Account& operator=(const Account&) = delete;
        Account(Account&&) = delete;
        Account& operator=(Account&&) = delete;
    };
    
    // The map of accounts
    std::unordered_map<int, Account> accounts;
    
    // A shared mutex to protect the map structure itself
    mutable std::shared_mutex map_mutex;
    
    // Thread-local random number generators to avoid contention
    static thread_local std::mt19937 gen;
    static thread_local std::uniform_real_distribution<float> op_type_dist;
    int num_accounts;

public:
    // Initialize the map with accounts totaling 100,000
    void initialize(int n_accounts) {
        std::unique_lock lock(map_mutex);
        num_accounts = n_accounts;
        
        float initial_balance = 100000.0f / num_accounts;
        for (int i = 0; i < num_accounts; ++i) {
            // Use try_emplace to construct the account in-place
            accounts.try_emplace(i, initial_balance);
        }
    }
    
    // Deposit function that transfers money between two random accounts
    void deposit() {
        // Create thread-local distributions based on account count
        thread_local std::uniform_int_distribution<int> account_dist(0, num_accounts - 1);
        thread_local std::uniform_real_distribution<float> amount_dist(0.1f, 10.0f);
        
        // Select two different random accounts
        int account1_id, account2_id;
        do {
            account1_id = account_dist(gen);
            account2_id = account_dist(gen);
        } while (account1_id == account2_id);
        
        // Ensure we always lock in the same order to prevent deadlocks
        if (account1_id > account2_id) {
            std::swap(account1_id, account2_id);
        }
        
        float transfer_amount = amount_dist(gen);
        
        // Shared lock on the map to prevent structural changes
        std::shared_lock map_read_lock(map_mutex);
        
        // Find the accounts
        auto it1 = accounts.find(account1_id);
        auto it2 = accounts.find(account2_id);
        
        if (it1 == accounts.end() || it2 == accounts.end()) {
            return; // One of the accounts doesn't exist
        }
        
        // Lock both accounts in order
        std::lock_guard<std::mutex> lock1(it1->second.mutex);
        std::lock_guard<std::mutex> lock2(it2->second.mutex);
        
        // Map read lock can be released once we have the account locks
        map_read_lock.unlock();
        
        // Perform the transfer
        it1->second.balance -= transfer_amount;
        it2->second.balance += transfer_amount;
    }
    
    // Balance function that sums all account balances
    float balance() {
        // Take a shared lock for reading the map structure
        std::shared_lock lock(map_mutex);
        
        float total = 0.0f;
        
        // For correctness, we need to lock each account while reading its balance
        for (auto& [id, account] : accounts) {
            std::lock_guard<std::mutex> acc_lock(account.mutex);
            total += account.balance;
        }
        
        return total;
    }
    
    // Remove all accounts
    void clear() {
        std::unique_lock lock(map_mutex);
        accounts.clear();
    }
    
    // Function that will be executed by worker threads
    std::chrono::milliseconds do_work(int iterations) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            if (op_type_dist(gen) < 0.95f) {
                deposit();
            } else {
                float total = balance();
                // Optional validation during execution
                // if (std::abs(total - 100000.0f) > 0.01f) {
                //     std::cerr << "Balance inconsistency detected: " << total << std::endl;
                // }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
};

// Initialize thread-local random number generators
thread_local std::mt19937 BankSystem::gen(std::random_device{}());
thread_local std::uniform_real_distribution<float> BankSystem::op_type_dist(0.0f, 1.0f);

int main(int argc, char* argv[]) {
    // Parse command line arguments for number of threads, accounts, and iterations
    int num_threads = (argc > 1) ? std::stoi(argv[1]) : 4;
    int num_accounts = (argc > 2) ? std::stoi(argv[2]) : 1000;
    int total_iterations = (argc > 3) ? std::stoi(argv[3]) : 10000;
    
    // Calculate iterations per thread
    int iterations_per_thread = total_iterations / num_threads;
    
    // std::cout << "Running with " << num_threads << " threads, " 
    //           << num_accounts << " accounts, "
    //           << total_iterations << " total iterations ("
    //           << iterations_per_thread << " per thread)" << std::endl;
    
    BankSystem bank;
    bank.initialize(num_accounts);
    
    // Verify initial balance
    float initial_balance = bank.balance();
    // std::cout << "Initial balance: " << std::fixed << std::setprecision(2) << initial_balance << std::endl;
    
    // Prepare threads and futures for execution times
    std::vector<std::thread> threads;
    std::vector<std::future<std::chrono::milliseconds>> futures(num_threads);
    
    // Start measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Launch worker threads
    for (int i = 0; i < num_threads; ++i) {
        std::promise<std::chrono::milliseconds> promise;
        futures[i] = promise.get_future();
        
        threads.emplace_back([&bank, iterations_per_thread, promise = std::move(promise)]() mutable {
            auto exec_time = bank.do_work(iterations_per_thread);
            promise.set_value(exec_time);
        });
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Verify final balance
    float final_balance = bank.balance();
    // std::cout << "Final balance: " << std::fixed << std::setprecision(2) << final_balance << std::endl;
    
    // Collect and report thread execution times
    std::vector<std::chrono::milliseconds> thread_times;
    std::chrono::milliseconds longest_time(0);
    size_t longest_thread_id = 0;
    
    for (size_t i = 0; i < num_threads; ++i) {
        auto time = futures[i].get();
        thread_times.push_back(time);
        
        if (time > longest_time) {
            longest_time = time;
            longest_thread_id = i;
        }
        
        std::cout << "Thread " << i << " execution time: " << time.count() << " ms" << std::endl;
    }
    
    std::cout << "Thread with ID " << longest_thread_id
              << " had the longest execution time: " << longest_time.count() << " ms" << std::endl;
    
    // std::cout << "Total wall clock time: " << total_time.count() << " ms" << std::endl;
    
    // Clean up
    bank.clear();
    
    return 0;
}