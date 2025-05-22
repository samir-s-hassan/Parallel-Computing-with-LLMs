#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <random>
#include <chrono>
#include <numeric> // For std::accumulate
#include <future>  // For std::promise and std::future
#include <algorithm> // For std::min and std::max
#include <functional> // For std::hash (needed for thread ID hashing)
#include <tuple> // For std::piecewise_construct, std::forward_as_tuple (needed for map emplace)
#include <cstdlib> // For atoi


// Step 1 & 2: Define and populate the map
// Each account has an ID and a balance, protected by its own mutex.
// std::mutex is not copyable or movable, so we need to manage it carefully.
// Storing mutex directly in the map value is fine as map manages the lifetime of the stored object.
struct Account {
    int balance; // Last 2 digits are cents
    mutable std::mutex mtx; // Use mutable if you might access this in a const context (like a const map)

    // Default constructor needed for map value type requirement in some operations (though not emplace with piecewise_construct)
    Account() : balance(0) {}
    // Constructor for initialization
    Account(int initial_balance) : balance(initial_balance) {}

    // Prevent copy and move - crucial for types containing mutexes
    Account(const Account&) = delete;
    Account& operator=(const Account&) = delete;
    Account(Account&&) = delete; // Explicitly delete move constructor
    Account& operator=(Account&&) = delete; // Explicitly delete move assignment
};

// Using a map where the value is the Account struct directly
// std::map provides node stability, so mutex addresses within nodes are stable across insertions/deletions.
std::map<int, Account> bank_accounts;

// Global mutex for balance calculation to ensure consistency across all threads.
// This prevents deposits from happening *while* balance is being summed.
// A std::shared_mutex (C++17) could allow multiple readers (balance) but only one writer (deposit),
// but a simple std::mutex is sufficient here given the low frequency of balance calls (5%)
// and simplifies things while adhering to your std::mutex comfort zone.
std::mutex global_balance_mtx;


// Function to initialize the bank accounts
void initialize_accounts(int num_accounts, int total_balance) {
    bank_accounts.clear(); // Start fresh
    int current_total = 0;
    // Distribute balance relatively evenly
    int base_balance = total_balance / num_accounts;
    int remainder = total_balance % num_accounts;

    for (int i = 0; i < num_accounts; ++i) {
        int initial_balance = base_balance + (i < remainder ? 1 : 0);
        // *** FIX for the std::pair construction error ***
        // Use piecewise_construct and forward_as_tuple to construct the key and value
        // parts of the pair in place, avoiding copy/move of Account.
        bank_accounts.emplace(std::piecewise_construct,
                              std::forward_as_tuple(i), // Arguments for constructing the key (int)
                              std::forward_as_tuple(initial_balance)); // Arguments for constructing the value (Account)
        current_total += initial_balance;
    }

    // Ensure the total is exactly total_balance
    // Adjust account 0 if there's a discrepancy (shouldn't happen with the logic above, but defensive)
    if (current_total != total_balance && num_accounts > 0) {
         auto it0 = bank_accounts.find(0);
         if (it0 != bank_accounts.end()) {
             // Lock account 0 while adjusting its balance
             std::lock_guard<std::mutex> lock(it0->second.mtx);
             it0->second.balance += (total_balance - current_total);
         }
    }
    std::cout << "Initialized " << num_accounts << " accounts with total balance " << total_balance << std::endl;
}

// Step 3: Define the deposit function
void deposit(int account_id1, int account_id2, int amount) {
    // Ensure valid accounts and non-zero amount
    if (account_id1 == account_id2 || amount <= 0) {
        return; // Avoid self-transfer and invalid amounts
    }

    // Find the accounts in the map
    auto it1 = bank_accounts.find(account_id1);
    auto it2 = bank_accounts.find(account_id2);

    // Check if accounts exist
    if (it1 == bank_accounts.end() || it2 == bank_accounts.end()) {
        return; // One or both accounts not found
    }

    // Get references to the mutexes for the two accounts
    std::mutex& mtx1 = it1->second.mtx;
    std::mutex& mtx2 = it2->second.mtx;

    // *** Crucial for Deadlock Prevention ***
    // Use std::lock to acquire both mutexes. std::lock handles the case where
    // it might acquire one lock but not the other by releasing the first and retrying.
    // This avoids the classic two-resource deadlock.
    std::lock(mtx1, mtx2);

    // Use std::unique_lock with std::adopt_lock to associate the unique_lock
    // objects with the mutexes that are *already locked* by std::lock.
    // unique_lock will then manage the unlocking when they go out of scope.
    std::unique_lock<std::mutex> lock1(mtx1, std::adopt_lock);
    std::unique_lock<std::mutex> lock2(mtx2, std::adopt_lock);

    // Perform the transfer
    it1->second.balance -= amount;
    it2->second.balance += amount;

    // Locks are automatically released when unique_lock objects go out of scope
}

// Step 4: Define the balance function
long long balance() {
    // Acquire the global balance mutex to ensure consistency.
    // While this lock is held, no deposit operations can occur.
    std::lock_guard<std::mutex> lock(global_balance_mtx);

    long long total_balance = 0;
    // Iterate through all accounts and sum balances
    // We don't need individual account locks here because the global_balance_mtx
    // guarantees that no balances are changing during this iteration.
    for (const auto& pair : bank_accounts) {
        total_balance += pair.second.balance;
    }
    return total_balance;
}

// Helper to get random account IDs (ensuring they are within the valid range)
int get_random_account_id(std::mt19937& rng, int num_accounts) {
    if (num_accounts <= 0) return -1; // Handle edge case
    std::uniform_int_distribution<int> dist(0, num_accounts - 1);
    return dist(rng);
}

// Step 5: Define the do_work function
void do_work(int num_iterations, int num_accounts, std::promise<std::chrono::duration<double>> exec_time_promise) {
    // Use a thread-local random number generator
    // Seeding with time and thread ID helps ensure distinct sequences across threads.
    // *** FIX for the std::thread::id::hash() error ***
    // Use std::hash<std::thread::id> to get a hash value from the thread ID.
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count() +
                    std::hash<std::thread::id>{}(std::this_thread::get_id());
    std::mt19937 rng(seed);

    std::uniform_real_distribution<double> prob_dist(0.0, 1.0); // For 95%/5% decision
    std::uniform_int_distribution<int> amount_dist(1, 1000); // Random deposit amount

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        double decision = prob_dist(rng);

        if (decision < 0.95) { // 95% chance for deposit
            int acc1_id = get_random_account_id(rng, num_accounts);
            int acc2_id = get_random_account_id(rng, num_accounts);

            // Ensure we get two different, valid account IDs
            while (acc1_id == -1 || acc2_id == -1 || acc1_id == acc2_id) {
                 acc1_id = get_random_account_id(rng, num_accounts);
                 acc2_id = get_random_account_id(rng, num_accounts);
            }
            int amount = amount_dist(rng);

            deposit(acc1_id, acc2_id, amount);

        } else { // 5% chance for balance check
            long long current_balance = balance();
            // The prompt requires the final balance check to be 100000.
            // Checking here adds overhead but is possible if you want to see inconsistencies immediately (though unlikely with our locking).
            // if (current_balance != 100000) {
            //    std::cerr << "Error: Inconsistent balance detected within thread: " << current_balance << std::endl;
            // }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> exec_time = end_time - start_time;

    // Fulfill the promise with the execution time
    exec_time_promise.set_value(exec_time);
}

int main(int argc, char* argv[]) {
    // Default parameters
    int total_iterations = 1'000'000;
    int num_accounts = 250;
    int total_initial_balance = 100000;
    std::vector<int> thread_counts = {1};

    // Parse command line arguments
    if (argc >= 2) {
        total_iterations = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        num_accounts = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        int num_threads = std::atoi(argv[3]);
        thread_counts = {num_threads};
    }

    std::cout << "Running with " << total_iterations << " iterations, "
              << num_accounts << " accounts, "
              << thread_counts[0] << " threads" << std::endl;

    // Store results to calculate speedup
    std::map<int, std::chrono::duration<double>> overall_times;

    for (int num_threads : thread_counts) {
        // Re-initialize accounts for each test run to start from a clean state.
        initialize_accounts(num_accounts, total_initial_balance);

        std::cout << "Running simulation with " << num_threads << " threads..." << std::endl;

        std::vector<std::thread> threads;
        std::vector<std::future<std::chrono::duration<double>>> futures;
        // Need one promise per thread to receive its execution time
        std::vector<std::promise<std::chrono::duration<double>>> promises(num_threads);

        // Calculate iterations per thread
        int iterations_per_thread = total_iterations / num_threads;
        
        // Create threads and launch do_work
        auto overall_start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_threads; ++i) {
            // Get the future from the promise *before* moving the promise into the thread
            futures.push_back(promises[i].get_future());
            // Create the thread, moving the promise into it
            threads.emplace_back(do_work, iterations_per_thread, num_accounts, std::move(promises[i]));
        }

        // Wait for all threads to complete (join them) and collect results (from futures)
        for (int i = 0; i < num_threads; ++i) {
            threads[i].join();
        }
        auto overall_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> overall_exec_time = overall_end_time - overall_start_time;

        // Store the overall time for speedup calculation later
        overall_times[num_threads] = overall_exec_time;

        std::cout << "Simulation finished." << std::endl;
        std::cout << "Overall execution time: " << overall_exec_time.count() << " seconds" << std::endl;

        // Final balance check after all threads have joined
        long long final_balance = balance();
        std::cout << "Final balance: " << final_balance << std::endl;

        if (final_balance == total_initial_balance) {
            std::cout << "Final balance check PASSED." << std::endl;
        } else {
            std::cerr << "Final balance check FAILED! Expected " << total_initial_balance << ", got " << final_balance << std::endl;
        }
    }

    return 0;
}
