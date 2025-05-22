#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <random>
#include <chrono>
#include <vector>
#include <future>
#include <fstream>
#include <algorithm>

// Global bank accounts map and a mutex to protect its operations.
std::map<int, int> bankAccounts;
std::mutex bankMutex;

//------------------------------------------------------------
// Function: initializeBankAccounts
// Purpose:  Create and initialize the bank accounts such that
//           the total funds equal 100000. For simplicity, we
//           create 10 accounts, each with 10000.
//------------------------------------------------------------
void initializeBankAccounts(int numAccounts) {
    int initialBalance = 100000 / numAccounts;  // Each account gets 10000
    for (int i = 0; i < numAccounts; i++) {
        bankAccounts[i] = initialBalance;
    }
}

//------------------------------------------------------------
// Function: deposit
// Purpose:  Atomically transfer a random amount from one bank
//           account to another. The entire operation is guarded
//           by a mutex to ensure thread safety.
//------------------------------------------------------------
void deposit() {
    // Lock the mutex to ensure exclusive access to bankAccounts.
    std::lock_guard<std::mutex> lock(bankMutex);

    // Set up random generators. Using thread_local ensures each thread
    // has its own generator instance.
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> accountDist(0, bankAccounts.size() - 1);
    std::uniform_int_distribution<int> amountDist(1, 100);
  // Random amount between 1 and 100

    // Randomly select two distinct accounts.
    int acc1 = accountDist(generator);
    int acc2 = accountDist(generator);
    while (acc2 == acc1) {
        acc2 = accountDist(generator);
    }

    // Determine a random transfer amount.
    int transferAmount = amountDist(generator);

    // Execute the atomic transfer.
    bankAccounts[acc1] -= transferAmount;
    bankAccounts[acc2] += transferAmount;
}

//------------------------------------------------------------
// Function: balance
// Purpose:  Atomically compute the sum of all bank account balances.
//           The mutex is locked so no deposit operations can occur
//           concurrently during the summing.
//------------------------------------------------------------
int balance() {
    std::lock_guard<std::mutex> lock(bankMutex);
    int total = 0;
    for (const auto &account : bankAccounts) {
        total += account.second;
    }
    return total;
}

//------------------------------------------------------------
// Function: do_work
// Purpose:  Runs a loop for a specified number of iterations. In
//           each iteration, it randomly chooses between performing
//           a deposit (95% chance) or calculating the balance (5%).
//           It measures the total execution time for the loop.
// Returns:  The execution time in seconds.
//------------------------------------------------------------
double do_work(int iterations) {
    auto start = std::chrono::high_resolution_clock::now();

    // Set up a random generator for operation selection.
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> probDist(0.0, 1.0);

    for (int i = 0; i < iterations; i++) {
        double prob = probDist(generator);
        if (prob < 0.95) {
            deposit(); // Perform deposit 95% of the time.
        } else {
            // Calculate balance 5% of the time.
            int total = balance();
            // Optionally, one could check that total equals 100000.
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> exec_time = end - start;
    return exec_time.count();
}

//------------------------------------------------------------
// Function: main
// Purpose:  Spawns threads to execute do_work concurrently with
//           different thread counts and also executes a single-
//           threaded version for performance comparison.
//           Timing data is exported to a CSV file for later plotting.
//------------------------------------------------------------
int main() {
    std::vector<int> accountCounts = {250, 1000, 10000};
    std::vector<int> iterationCounts = {1000, 10000, 100000, 1000000, 10000000};
    std::vector<int> threadCounts = {2, 4, 8, 16};

    std::ofstream csv("benchmark_results.csv");
    csv << "Accounts,Iterations,Threads,TimeSeconds\n";

    for (int accounts : accountCounts) {
        for (int iterations : iterationCounts) {
            // Multithreaded runs
            for (int numThreads : threadCounts) {
                {
                    std::lock_guard<std::mutex> lock(bankMutex);
                    bankAccounts.clear();
                    initializeBankAccounts(accounts);
                }

                std::vector<std::future<double>> futures;

                int itersPerThread = iterations / numThreads;
                for (int i = 0; i < numThreads; ++i) {
                    futures.push_back(std::async(std::launch::async, do_work, itersPerThread));
                }

                std::vector<double> times;
                for (auto& fut : futures) {
                    times.push_back(fut.get());
                }

                double max_time = *max_element(times.begin(), times.end());
                int finalBalance = balance();

                std::cout << "MT | Accounts: " << accounts
                          << ", Iterations: " << iterations
                          << ", Threads: " << numThreads
                          << ", MaxTime: " << max_time << " s"
                          << ", Final Balance: " << finalBalance << "\n";

                csv << accounts << "," << iterations << "," << numThreads << "," << max_time << "\n";
            }


        }
    }

    csv.close();
    std::cout << "\nResults written to benchmark_results03.csv\n";
    return 0;
}
