#include <iostream>      // For console input/output
#include <vector>        // For dynamic arrays (threads, futures, locks)
#include <thread>        // For creating and managing threads
#include <mutex>         // For mutexes (locks)
#include <unordered_map> // For the account map (hash table)
#include <random>        // For random number generation
#include <chrono>        // For timing execution
#include <future>        // For retrieving results from threads (exec_time_i)
#include <numeric>       // For std::accumulate (optional, for balance)
#include <stdexcept>     // For exceptions (e.g., invalid arguments)
#include <iomanip>       // For formatting output (e.g., setprecision)
#include <algorithm>     // For std::max/min
#include <functional>    // For std::ref (optional)

// --- Constants and Configuration ---
const double INITIAL_TOTAL_BALANCE = 100000.0;
const double DEPOSIT_PROBABILITY = 0.95; // 95% chance to deposit, 5% to balance

// --- Bank Class ---
class Bank
{
private:
    // Step 1: Define the map of accounts <int accountID, double balance>
    // Using double for better precision than float.
    std::unordered_map<int, double> accounts;

    // Lock Striping specific members
    size_t num_stripes;                   // Number of locks/stripes
    std::vector<std::mutex> stripe_locks; // The locks for each stripe

    // Random number generation (for deposit/balance choice, account selection, amount)
    // Note: For true high-performance randomness, each thread should ideally have its own
    // seeded generator. For simplicity here, we create one per 'do_work' call.
    std::mt19937 rng_engine{std::random_device{}()}; // Mersenne Twister engine seeded with random_device

    // Helper function to determine the stripe for an account ID
    size_t get_stripe_index(int account_id) const
    {
        // Simple modulo hashing. Ensure non-negative result.
        // std::hash<int> could also be used but modulo is often sufficient.
        return static_cast<size_t>(account_id) % num_stripes;
    }

public:
    // Constructor: Initializes accounts and locks
    // Constructor: Initializes accounts and locks
    Bank(int num_accounts, size_t num_threads) : num_stripes(std::max(1u, static_cast<unsigned int>(num_threads * 4))),
                                                 stripe_locks(num_stripes) // Default-constructs num_stripes mutexes
    {
        if (num_accounts <= 0)
        {
            throw std::invalid_argument("Number of accounts must be positive.");
        }

        // std::cout << "Initializing bank with " << num_accounts << " accounts and " << num_stripes << " lock stripes." << std::endl;

        // Step 2: Populate the map and initialize total balance
        double initial_amount_per_account = INITIAL_TOTAL_BALANCE / num_accounts;
        double current_total = 0.0;
        for (int i = 0; i < num_accounts; ++i)
        {
            accounts[i] = initial_amount_per_account;
            current_total += initial_amount_per_account;
        }

        // Adjust the last account slightly to ensure the sum is *exactly* INITIAL_TOTAL_BALANCE
        double remainder = INITIAL_TOTAL_BALANCE - current_total;
        if (num_accounts > 0)
        {
            accounts[num_accounts - 1] += remainder;
        }

        // Optional: Verify initial balance immediately (requires locking)
        // double verified_initial_balance = balance();
        // std::cout << "Initial balance verified: " << std::fixed << std::setprecision(2) << verified_initial_balance << std::endl;
        // if (std::abs(verified_initial_balance - INITIAL_TOTAL_BALANCE) > 1e-6) {
        //     throw std::runtime_error("Initial balance verification failed!");
        // }
    } // Step 3: Define the 'deposit' function (atomic transfer)
    void deposit(int account1_id, int account2_id, double amount)
    {
        if (account1_id == account2_id)
        {
            // Transferring to the same account is a no-op or could be an error.
            // Let's treat as no-op for simplicity based on the prompt's B1-=V, B2+=V logic.
            return;
        }
        if (amount <= 0)
        {
            // Transferring non-positive amount doesn't make sense.
            return;
        }

        // Determine the stripes for both accounts
        size_t stripe1 = get_stripe_index(account1_id);
        size_t stripe2 = get_stripe_index(account2_id);

        // Acquire locks using std::scoped_lock for automatic deadlock avoidance and RAII
        if (stripe1 == stripe2)
        {
            // Both accounts are in the same stripe, lock only once
            std::scoped_lock lock(stripe_locks[stripe1]);

            // Check if accounts exist (optional, depends on requirements)
            // if (accounts.find(account1_id) == accounts.end() || accounts.find(account2_id) == accounts.end()) {
            //     std::cerr << "Warning: Attempting deposit with non-existent account ID." << std::endl;
            //     return;
            // }

            // Perform the transfer
            // Note: Could add check if account1 has sufficient funds, but prompt doesn't require it.
            accounts[account1_id] -= amount;
            accounts[account2_id] += amount;
        }
        else
        {
            // Accounts are in different stripes, lock both in a fixed order (implicitly handled by scoped_lock)
            std::scoped_lock lock(stripe_locks[stripe1], stripe_locks[stripe2]);

            // Perform the transfer
            accounts[account1_id] -= amount;
            accounts[account2_id] += amount;
        }
        // Locks are automatically released when 'lock' goes out of scope (RAII)
    }

    // Step 4: Define the 'balance' function (atomic sum)
    double balance()
    {
        // Acquire all stripe locks to get a consistent snapshot
        // Create a vector of lock references for scoped_lock
        std::vector<std::mutex *> all_locks_ptrs;
        all_locks_ptrs.reserve(num_stripes);
        for (size_t i = 0; i < num_stripes; ++i)
        {
            all_locks_ptrs.push_back(&stripe_locks[i]);
        }
        // Use variadic std::scoped_lock (needs C++17) - simpler way requires manual locking or helper
        // Let's lock manually in order for broader compatibility/clarity here
        for (size_t i = 0; i < num_stripes; ++i)
        {
            stripe_locks[i].lock();
        }

        // Calculate the total balance
        double total_balance = 0.0;
        for (const auto &pair : accounts)
        {
            total_balance += pair.second;
        }

        // Unlock in reverse order
        for (int i = static_cast<int>(num_stripes) - 1; i >= 0; --i)
        {
            stripe_locks[i].unlock();
        }

        return total_balance;
    }

    // Step 4 Variant: Balance using std::scoped_lock (C++17) - More concise but less explicit locking order
    double balance_scoped()
    {
        // Acquire all stripe locks using scoped_lock for RAII and deadlock safety
        // Note: This requires C++17 and might be less clear about the locking order than manual locking.
        std::scoped_lock lock(stripe_locks[0]); // Lock the first one
        if (num_stripes > 1)
        {
            // Lock remaining ones - this is NOT how scoped_lock works for multiple mutexes from a vector easily.
            // The manual lock/unlock above or a recursive helper is needed for dynamic numbers of mutexes.
            // Let's stick to the manual lock/unlock version for clarity and correctness with vectors.
            // The code below is incorrect for locking all mutexes in a vector dynamically.
            // std::apply([&](auto&... m){ std::scoped_lock lock(m...); }, stripe_locks); // Incorrect usage
        }
        // --> Reverting to the manual lock/unlock version implemented above in `balance()` <--

        double total_balance = 0.0;
        // This part would be inside the locked section
        // for (const auto& pair : accounts) {
        //     total_balance += pair.second;
        // }
        // return total_balance;
        return balance(); // Call the correct version
    }

    // Step 5: Define the 'do_work' function executed by each thread
    long long do_work(int iterations, int num_accounts)
    {
        // Each thread gets its own random number generator, seeded differently
        // Using thread ID + time might be better, but this is simpler for now.
        std::mt19937 thread_rng(std::random_device{}() ^ std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        std::uniform_int_distribution<int> account_dist(0, num_accounts - 1);
        // Define a reasonable range for transfer amounts
        std::uniform_real_distribution<double> amount_dist(0.01, 100.0);

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i)
        {
            double choice = prob_dist(thread_rng);

            if (choice < DEPOSIT_PROBABILITY)
            {
                // Perform deposit
                int id1 = account_dist(thread_rng);
                int id2 = account_dist(thread_rng);
                // Ensure accounts are distinct for the transfer
                while (id1 == id2)
                {
                    id2 = account_dist(thread_rng);
                }
                double amount = amount_dist(thread_rng);
                deposit(id1, id2, amount);
            }
            else
            {
                // Perform balance check
                // We don't need to store/use the result here, just call it.
                balance(); // Call the correctly implemented balance function
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        return duration.count(); // Return execution time in milliseconds
    }
}; // End of Bank class

// --- Main Function ---
int main(int argc, char *argv[])
{
    // Step 6: Parse arguments, create threads, run work, collect results
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <num_accounts> <num_iterations_per_thread>" << std::endl;
        return 1;
    }

    int num_threads;
    int num_accounts;
    int num_iterations;
    try
    {
        num_threads = std::stoi(argv[1]);
        num_accounts = std::stoi(argv[2]);
        num_iterations = std::stoi(argv[3]);
        num_iterations = num_iterations / num_threads; // Adjust iterations per thread

        if (num_threads <= 0 || num_accounts <= 0 || num_iterations <= 0)
        {
            throw std::invalid_argument("Arguments must be positive integers.");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return 1;
    }

    // std::cout << "Configuration: Threads=" << num_threads
    //           << ", Accounts=" << num_accounts
    //           << ", Iterations/Thread=" << num_iterations << std::endl;

    try
    {
        // Create the Bank object
        Bank bank(num_accounts, num_threads);

        // Vector to hold futures for thread execution times
        std::vector<std::future<long long>> futures;
        // Vector to hold thread objects
        std::vector<std::thread> threads;

        // std::cout << "Starting " << num_threads << " worker threads..." << std::endl;

        // Launch threads
        for (int i = 0; i < num_threads; ++i)
        {
            // Use std::packaged_task to wrap the member function call
            std::packaged_task<long long(int, int)> task(
                // Need to bind 'this' pointer for member function calls
                [&bank](int iter, int acc)
                { return bank.do_work(iter, acc); });
            // Get the future before moving the task to the thread
            futures.push_back(task.get_future());
            // Launch the thread, moving the task into it
            threads.emplace_back(std::move(task), num_iterations, num_accounts);
        }

        // Wait for all threads to complete and collect execution times
        std::vector<long long> exec_times;
        long long max_time = 0;
        int slowest_thread_idx = -1;

        for (int i = 0; i < num_threads; ++i)
        {
            if (threads[i].joinable())
            {
                threads[i].join(); // Wait for thread completion
            }
            long long time_ms = futures[i].get(); // Get result from future
            exec_times.push_back(time_ms);
            std::cout << "Thread " << i << " finished in " << time_ms << " ms." << std::endl;
            if (time_ms > max_time)
            {
                max_time = time_ms;
                slowest_thread_idx = i;
            }
        }

        // std::cout << "All threads finished." << std::endl;
        // Output the longest time in the format expected by the script
        if (slowest_thread_idx != -1)
        {
            std::cout << "Thread " << slowest_thread_idx << " had the longest execution time: " << max_time << std::endl;
        }
        else
        {
            // std::cout << "No threads were run or times could not be determined." << std::endl;
        }

        // Step 6 (cont.): Final balance check after all threads joined
        // std::cout << "Performing final balance check..." << std::endl;
        double final_balance = bank.balance(); // Use the correct balance function
        // std::cout << "Final balance: " << std::fixed << std::setprecision(2) << final_balance << std::endl;

        // Verify the final balance
        if (std::abs(final_balance - INITIAL_TOTAL_BALANCE) > 1e-6)
        { // Allow for tiny floating point differences
            std::cerr << "Error: Final balance is incorrect!" << std::endl;
            return 1; // Indicate error
        }
        else
        {
            std::cout << "Final balance is correct." << std::endl;
        }

        // Step 8: Cleanup - Handled automatically by RAII (destructors for map, vectors, mutexes)
    }
    catch (const std::exception &e)
    {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    // std::cout << "Execution terminates happily!" << std::endl;
    return 0; // Success
}
