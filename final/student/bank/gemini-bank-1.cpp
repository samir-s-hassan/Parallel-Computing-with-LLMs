#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <random>
#include <chrono>
#include <numeric> // For std::accumulate
#include <future>  // For std::future and std::packaged_task
#include <stdexcept> // For argument error handling
#include <cstdlib>   // For std::atoi, std::atof
#include <iomanip> // For std::fixed, std::setprecision
#include <string> // For std::stoi
#include <cmath>   // For std::abs

// --- Global Shared Data ---
// Step 1: Define the map using double for higher precision
std::map<int, double> accountMap; // USE DOUBLE
// Mutex to protect access to the accountMap for atomic operations
std::mutex mapMutex;

// --- Random Number Generation ---
thread_local std::mt19937 rng; // Mersenne Twister engine

void seed_rng(int threadId) {
    unsigned int seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^ (threadId << 16);
    rng.seed(seed);
}

// --- Helper Function ---
// Initializes the map with a specified number of accounts and total balance.
void initializeAccounts(int numAccounts, double totalBalance = 100000.0) { // USE DOUBLE
    accountMap.clear();
    if (numAccounts <= 0) {
         accountMap.clear();
         if (totalBalance != 0.0) {
              std::cerr << "Warning: Cannot distribute non-zero balance (" << totalBalance
                        << ") among zero accounts." << std::endl;
         }
        return;
    }

    // Step 2: Populate the map ensuring the total sum is totalBalance
    double balancePerAccount = totalBalance / static_cast<double>(numAccounts); // USE DOUBLE
    for (int i = 0; i < numAccounts; ++i) {
        accountMap.insert({i, balancePerAccount});
    }

    // Verification (optional, good for debugging initial state)
    double current_total = 0.0; // USE DOUBLE
    { // Scope for lock guard during verification if needed (though map isn't shared yet)
      // std::lock_guard<std::mutex> lock(mapMutex); // Not strictly needed here before threads start
      current_total = std::accumulate(accountMap.begin(), accountMap.end(), 0.0, // Use 0.0 double literal
                                    [](double sum, const auto& pair) { // USE DOUBLE in lambda
                                        return sum + pair.second;
                                    });
    }
     // Use a small epsilon for floating point comparison
     if (std::abs(current_total - totalBalance) > 1e-9 * std::abs(totalBalance)) { // Tighter tolerance for double init check
          std::cerr << "[WARN] Initial balance verification failed! Sum: " << std::fixed << std::setprecision(6) << current_total
                    << ", Expected: " << totalBalance << std::endl;
     }
}


// --- Core Functions ---

// Step 3: Define the 'deposit' function (atomic transfer)
void deposit(int numAccounts) {
    if (numAccounts < 2) return;

    std::uniform_int_distribution<int> account_dist(0, numAccounts - 1);
    // Use double for amount as well for consistency
    std::uniform_real_distribution<double> amount_dist(1.0, 100.0); // USE DOUBLE distribution
    int id1 = account_dist(rng);
    int id2 = account_dist(rng);
    while (id1 == id2) {
        id2 = account_dist(rng);
    }

    double amount = amount_dist(rng); // USE DOUBLE

    std::lock_guard<std::mutex> lock(mapMutex);

    if (accountMap.count(id1) && accountMap.count(id2)) {
        // Perform the transfer using double precision
        accountMap[id1] -= amount;
        accountMap[id2] += amount;
    } else {
        std::cerr << "[ERROR] Invalid account ID (" << id1 << " or " << id2
                  << ") encountered during deposit! NumAccounts: " << numAccounts << std::endl;
    }
}

// Step 4: Define the 'balance' function (atomic sum)
double balance() { // Return DOUBLE
    double total_balance = 0.0; // USE DOUBLE

    std::lock_guard<std::mutex> lock(mapMutex);

    total_balance = std::accumulate(accountMap.begin(), accountMap.end(), 0.0, // Use 0.0 double literal
                                    [](double sum, const auto& pair) { // USE DOUBLE in lambda
                                        return sum + pair.second;
                                    });

    // Verification with adjusted tolerance for double
    // 1e-6 is usually safe for double after many operations, adjust if needed
    const double expected_balance = 100000.0;
    if (std::abs(total_balance - expected_balance) > 1e-6) { // ADJUSTED TOLERANCE
         std::cerr << "[ALERT] Balance check returned: " << std::fixed << std::setprecision(6) // More precision in output
                   << total_balance << " (Deviation detected! Expected: " << expected_balance << ")" << std::endl;
    }

    return total_balance;
}

// Step 5: Define the 'do_work' function executed by each thread
long long do_work(int iterations, int numAccounts, int threadId) {
    seed_rng(threadId);

    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f); // Probability check is fine with float
    const float deposit_probability = 0.95f;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        if (prob_dist(rng) < deposit_probability) {
            deposit(numAccounts);
        } else {
            // Calling balance still returns double, but we don't store/use it here often
            balance();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    return duration.count();
}

// --- Main Execution Logic ---
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <iterations> <numAccounts> <numThreads>" << std::endl;
        return 1;
    }

    int n;
    int numAccounts;
    int numThreads;

    try {
        n = std::stoi(argv[1]);
        numAccounts = std::stoi(argv[2]);
        numThreads = std::stoi(argv[3]);
        if (n <= 0 || numAccounts < 0 || numThreads <= 0) {
             throw std::invalid_argument("Arguments must be positive integers (numAccounts can be 0).");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
         std::cerr << "Usage: " << argv[0] << " <iterations> <numAccounts> <numThreads>" << std::endl;
        return 1;
    }

    // Use double for initial balance
    const double initialTotalBalance = 100000.0; // USE DOUBLE
    initializeAccounts(numAccounts, initialTotalBalance);

    std::vector<std::thread> threads;
    std::vector<std::future<long long>> futures;

    auto overall_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numThreads; ++i) {
        std::packaged_task<long long(int, int, int)> task(do_work);
        futures.push_back(task.get_future());
        threads.emplace_back(std::move(task), n, numAccounts, i);
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto overall_end_time = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time);

    std::vector<long long> execution_times;
    for (auto& f : futures) {
        try {
            execution_times.push_back(f.get());
        } catch (const std::future_error& e) {
            std::cerr << "Error retrieving future result: " << e.what() << " (" << e.code() << ")" << std::endl;
            execution_times.push_back(-1);
        }
    }

    // Perform the final balance check after all threads joined
    double final_balance = balance(); // USE DOUBLE

    // --- Output Results ---
    std::cout << "Total time: " << overall_duration.count() << std::endl;
    // Ensure final balance output matches the required format (likely integer representation)
    // Cast to long long after checking it's close to the integer value for safety/clarity
    long long final_balance_int = static_cast<long long>(std::round(final_balance));
     std::cout << "Final balance: " << final_balance_int << std::endl;
    // If exact float/double output is needed, adjust formatting:
    // std::cout << "Final balance: " << std::fixed << std::setprecision(0) << final_balance << std::endl;


    for (int i = 0; i < numThreads; ++i) {
        if (i < execution_times.size()) {
            std::cout << "Thread " << i << " execution time (ms): " << execution_times[i] << std::endl;
        } else {
             std::cout << "Thread " << i << " execution time (ms): [Error retrieving]" << std::endl;
        }
    }

    // Step 8: Clean up resources
    accountMap.clear();

    return 0;
}