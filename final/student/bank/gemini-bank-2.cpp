#include <iostream>
#include <vector>
#include <unordered_map> // USE HASH MAP
#include <thread>
#include <mutex>
#include <random>
#include <chrono>
#include <numeric> // For std::accumulate
#include <future>  // For std::future and std::packaged_task
#include <stdexcept> // For argument error handling
#include <cstdlib>   // For std::atoi
#include <iomanip> // For std::fixed, std::setprecision
#include <string> // For std::stoi
#include <cmath>   // For std::abs
#include <functional> // For std::hash
#include <atomic> // Required for std::atomic_flag if we implement spinlocks later (not used currently)

// --- Configuration ---
// Number of stripes (locks) to use.
// This is a trade-off:
// - More stripes -> Lower deposit contention, Higher balance locking overhead.
// - Fewer stripes -> Higher deposit contention, Lower balance locking overhead.
// Max threads = 16 in tests. Let's use 64 (4 * 16) as a balance between
// reducing deposit contention and keeping balance overhead manageable.
// Previous value was 256. Tuning this is key for performance.
const size_t NUM_STRIPES = 64; // *** TUNED VALUE ***

// --- Global Shared Data ---
// Use unordered_map for potentially faster (O(1) average) access
std::unordered_map<int, double> accountMap;

// Lock Striping mutexes
std::vector<std::mutex> stripeLocks(NUM_STRIPES);

// Per-Stripe Partial Sums for faster balance calculation
std::vector<double> stripeSums(NUM_STRIPES, 0.0); // Initialize sums to zero

// --- Random Number Generation ---
// Use thread_local for better performance and less contention
thread_local std::mt19937 rng; // Mersenne Twister engine

// Function to seed the thread-local RNG uniquely
void seed_rng(int threadId) {
    // Seed with a combination of time and thread ID for better randomness across threads
    // Using high_resolution_clock ensures varying seeds even for threads starting close together.
    unsigned int seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^ (threadId << 16);
    rng.seed(seed);
}

// --- Helper Functions ---

// Function to determine the stripe index for a given account ID
size_t getStripeIndex(int accountId) {
    // Simple modulo is efficient for sequential IDs 0 to N-1
    // Using unsigned cast to avoid issues with negative accountId if that were possible
    return static_cast<size_t>(accountId) % NUM_STRIPES;
    // Alternative using std::hash (might distribute better for non-sequential IDs)
    // return std::hash<int>{}(accountId) % NUM_STRIPES;
}

// Initializes the map and partial sums (No locking needed here as it runs before threads start)
void initializeAccounts(int numAccounts, double totalBalance = 100000.0) {
    accountMap.clear();
    // Reset stripe sums as well - essential for correctness
    std::fill(stripeSums.begin(), stripeSums.end(), 0.0);

    if (numAccounts <= 0) {
         // Handle the case of zero or negative accounts gracefully
         if (totalBalance != 0.0) {
              std::cerr << "Warning: Cannot distribute non-zero balance (" << totalBalance
                        << ") among zero accounts." << std::endl;
         }
        return; // Nothing to populate
    }

    double balancePerAccount = totalBalance / static_cast<double>(numAccounts);
    accountMap.reserve(numAccounts); // Pre-allocate buckets for unordered_map efficiency

    for (int i = 0; i < numAccounts; ++i) {
        accountMap.insert({i, balancePerAccount});
        // Initialize partial sums during account creation
        size_t stripe = getStripeIndex(i);
        stripeSums[stripe] += balancePerAccount;
    }

    // Optional: Initial balance verification using partial sums (more reliable check now)
    double current_total_sums = std::accumulate(stripeSums.begin(), stripeSums.end(), 0.0);
     // Use a small epsilon relative to the expected total for robust floating point comparison
     if (std::abs(current_total_sums - totalBalance) > 1e-9 * std::abs(totalBalance)) {
          std::cerr << "[WARN] Initial balance verification (Sums) failed! Sum: " << std::fixed << std::setprecision(6) << current_total_sums
                    << ", Expected: " << totalBalance << std::endl;
     }
}

// --- Core Functions ---

// Deposit function using Lock Striping, Unordered_Map, and Partial Sums
void deposit(int numAccounts) {
    // Ensure there are at least two accounts to perform a transfer
    if (numAccounts < 2) return;

    // Use thread-local rng seeded previously
    std::uniform_int_distribution<int> account_dist(0, numAccounts - 1);
    std::uniform_real_distribution<double> amount_dist(1.0, 100.0); // Example random amount range

    int id1 = account_dist(rng);
    int id2 = account_dist(rng);
    // Ensure the two selected accounts are distinct
    while (id1 == id2) {
        id2 = account_dist(rng);
    }
    double amount = amount_dist(rng); // Use double for amount

    // Determine the stripes for the two accounts
    size_t stripe1 = getStripeIndex(id1);
    size_t stripe2 = getStripeIndex(id2);

    // Acquire locks based on stripe indices
    if (stripe1 == stripe2) {
        // Both accounts are in the same stripe, lock only one mutex
        std::lock_guard<std::mutex> lock(stripeLocks[stripe1]);
        // Check accounts exist using find for efficiency (avoids potential default construction with [])
        auto it1 = accountMap.find(id1);
        auto it2 = accountMap.find(id2);
        if (it1 != accountMap.end() && it2 != accountMap.end()) {
            // Update map entries
            it1->second -= amount;
            it2->second += amount;
            // Update partial sum (only one stripe involved)
            // The net change to stripeSums[stripe1] is zero ( -= amount and += amount)
            // So, no update needed for stripeSums here.
        } else {
             // This state should ideally not be reached if numAccounts is correct
             std::cerr << "[ERROR] Invalid account ID in deposit (same stripe)! ID1: " << id1 << " ID2: " << id2 << std::endl;
        }
    } else {
        // Accounts are in different stripes. Lock both mutexes using C++17 std::scoped_lock
        // It acquires locks in a standardized order to prevent deadlock.
        std::scoped_lock lock(stripeLocks[stripe1], stripeLocks[stripe2]);
        auto it1 = accountMap.find(id1);
        auto it2 = accountMap.find(id2);
        if (it1 != accountMap.end() && it2 != accountMap.end()) {
            // Update map entries
            it1->second -= amount;
            it2->second += amount;
            // Update partial sums for their respective stripes atomically with map update
            stripeSums[stripe1] -= amount;
            stripeSums[stripe2] += amount;
        } else {
             // This state should ideally not be reached
             std::cerr << "[ERROR] Invalid account ID in deposit (diff stripe)! ID1: " << id1 << " ID2: " << id2 << std::endl;
        }
    }
    // Locks are automatically released when lock_guard/scoped_lock goes out of scope (RAII)
}

// Balance function using Lock Striping and Partial Sums (Much Faster)
double balance() {
    // Lock ALL stripes to ensure consistent read of partial sums while deposits might be happening.
    // Create a vector of unique_lock objects, one for each stripe mutex, initially deferred.
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(NUM_STRIPES);
    for (size_t i = 0; i < NUM_STRIPES; ++i) {
        // Create unique_lock associated with the mutex but don't lock yet
        locks.emplace_back(stripeLocks[i], std::defer_lock);
    }

    // Lock all the unique_lock objects sequentially.
    // This order (0 to N-1) is fixed and consistent, avoiding deadlock
    // with the deposit function's ordered pair locking.
    for (auto& lock : locks) {
        lock.lock(); // Acquire the lock for this unique_lock object
    }

    // --- Critical Section Start (All locks acquired) ---
    // Calculate total balance by summing the pre-calculated partial sums.
    // This is O(NUM_STRIPES) instead of O(numAccounts).
    double total_balance = std::accumulate(stripeSums.begin(), stripeSums.end(), 0.0);
    // --- Critical Section End ---

    // Verification check (important for correctness)
    const double expected_balance = 100000.0;
    // Use a small absolute tolerance suitable for double precision after many operations
    if (std::abs(total_balance - expected_balance) > 1e-6) {
         std::cerr << "[ALERT] Balance check returned: " << std::fixed << std::setprecision(6)
                   << total_balance << " (Deviation detected! Expected: " << expected_balance << ")" << std::endl;
    }

    // unique_locks automatically release their mutexes upon destruction (RAII).
    return total_balance;
}

// Worker function executed by each thread
long long do_work(int iterations, int numAccounts, int threadId) {
    // Seed this thread's random number generator
    seed_rng(threadId);

    // Define the probability distribution for actions
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f); // Probability check is fine with float
    const float deposit_probability = 0.95f;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        if (prob_dist(rng) < deposit_probability) {
            // 95% chance to call deposit
            deposit(numAccounts);
        } else {
            // 5% chance to call balance
            balance(); // Call the optimized balance function
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Return execution time for this thread
    return duration.count();
}

// --- Main Execution Logic ---
int main(int argc, char* argv[]) {
    // Step 6: Parse command-line arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <iterations> <numAccounts> <numThreads>" << std::endl;
        return 1;
    }

    int n; // Number of iterations per thread
    int numAccounts; // Number of bank accounts
    int numThreads; // Number of threads to create

    try {
        n = std::stoi(argv[1]);
        numAccounts = std::stoi(argv[2]);
        numThreads = std::stoi(argv[3]);
        // Basic validation
        if (n <= 0 || numAccounts < 0 || numThreads <= 0) {
             throw std::invalid_argument("Iterations and threads must be positive, numAccounts non-negative.");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
         std::cerr << "Usage: " << argv[0] << " <iterations> <numAccounts> <numThreads>" << std::endl;
        return 1;
    }

    // Initialize the bank accounts and partial sums
    const double initialTotalBalance = 100000.0;
    initializeAccounts(numAccounts, initialTotalBalance);

    // Step 6 (cont.): Create threads and manage futures
    std::vector<std::thread> threads;
    // Store futures to retrieve execution time from each thread
    std::vector<std::future<long long>> futures;

    auto overall_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numThreads; ++i) {
        // Create a packaged_task to wrap the do_work function.
        std::packaged_task<long long(int, int, int)> task(do_work);
        // Get the future associated with the task's result.
        futures.push_back(task.get_future());
        // Create and launch the thread, moving the task into it.
        threads.emplace_back(std::move(task), n, numAccounts, i);
    }

    // Step 6 (cont.): Wait for all threads to complete execution
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto overall_end_time = std::chrono::high_resolution_clock::now();
    // Calculate the total wall-clock time
    auto overall_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time);

    // Step 6 (cont.): Collect execution times from futures
    std::vector<long long> execution_times;
    execution_times.reserve(numThreads);
    for (auto& f : futures) {
        try {
            execution_times.push_back(f.get()); // Retrieve the result (time)
        } catch (const std::future_error& e) {
            std::cerr << "Error retrieving future result: " << e.what() << " (" << e.code() << ")" << std::endl;
            execution_times.push_back(-1); // Indicate an error
        }
    }

    // Step 6 (cont.): Perform the final balance check after all threads joined
    double final_balance = balance();

    // --- Output Results ---
    // Print results in the EXACT format required by the test script

    // Total time (overall wall-clock time)
    std::cout << "Total time: " << overall_duration.count() << std::endl;

    // Final balance (rounded to nearest integer for output consistency)
    long long final_balance_int = static_cast<long long>(std::round(final_balance));
    std::cout << "Final balance: " << final_balance_int << std::endl;

    // Individual thread execution times
    for (int i = 0; i < numThreads; ++i) {
        if (i < execution_times.size()) {
            std::cout << "Thread " << i << " execution time (ms): " << execution_times[i] << std::endl;
        } else {
             // Should not happen if futures vector is populated correctly
             std::cout << "Thread " << i << " execution time (ms): [Error retrieving]" << std::endl;
        }
    }

    // Step 8: Clean up resources (optional as OS cleans up, but good practice)
    accountMap.clear();
    // Vectors (stripeLocks, stripeSums) and their contents are automatically destroyed.

    // Execution terminates successfully
    return 0;
}