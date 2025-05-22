#include <iostream>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include <random>
#include <algorithm>
#include <sstream>

// Account storage with atomic balances
std::unordered_map<int, std::atomic<int>> accounts;
std::unordered_map<int, std::mutex> account_mutexes;
int NUM_ACCOUNTS = 100; // Now a variable, not const

// Transfer money between two accounts
void deposit() {
    // Use thread-local random number generator for better performance
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    
    // Pick two different accounts
    std::uniform_int_distribution<int> account_dist(0, NUM_ACCOUNTS - 1);
    int id1 = account_dist(gen);
    int id2;
    do {
        id2 = account_dist(gen);
    } while (id2 == id1);

    // Lock accounts in consistent order to prevent deadlock
    if (id1 > id2) std::swap(id1, id2);
    std::scoped_lock lock(account_mutexes[id1], account_mutexes[id2]);

    // Determine maximum transfer amount
    int max_transfer = std::min(accounts[id1].load(), 100); // limit max transfer
    if (max_transfer > 0) {
        std::uniform_int_distribution<int> amount_dist(0, max_transfer);
        int v = amount_dist(gen);
        accounts[id1] -= v;
        accounts[id2] += v;
    }
}

// Calculate the total balance across all accounts
int balance() {
    // Lock all accounts to ensure a consistent snapshot
    // We can't use a vector of std::scoped_lock, so we'll lock them manually
    std::vector<std::mutex*> mutex_ptrs;
    for (int i = 0; i < NUM_ACCOUNTS; ++i) {
        mutex_ptrs.push_back(&account_mutexes[i]);
    }
    
    // Sort mutex pointers to avoid deadlock
    std::sort(mutex_ptrs.begin(), mutex_ptrs.end());
    
    // Lock all mutexes
    for (auto mutex_ptr : mutex_ptrs) {
        mutex_ptr->lock();
    }
    
    // Calculate the balance
    int total = 0;
    for (int i = 0; i < NUM_ACCOUNTS; ++i) {
        total += accounts[i].load();
    }
    
    // Unlock all mutexes in reverse order
    for (auto it = mutex_ptrs.rbegin(); it != mutex_ptrs.rend(); ++it) {
        (*it)->unlock();
    }

    return total;
}

// Worker function for each thread
std::chrono::milliseconds do_work(int iterations, int initial_balance) {
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<int> op_dist(0, 99);
    
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        if (op_dist(gen) < 95) {
            deposit();
        } else {
            int b = balance();
            if (b != initial_balance) {
                std::cerr << "ERROR! Balance mismatch: " << b << " should be " << initial_balance << std::endl;
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    int iterations = 1000000;  // Default value
    int num_threads = 8;       // Default value
    
    if (argc >= 2) {
        iterations = std::stoi(argv[1]);
    }
    
    if (argc >= 3) {
        num_threads = std::stoi(argv[2]);
    }
    
    if (argc >= 4) {
        NUM_ACCOUNTS = std::stoi(argv[3]);
    }
    
    // Initialize accounts
    int total = 100000;  // Total balance stays the same
    int per_account = total / NUM_ACCOUNTS;
    int remainder = total % NUM_ACCOUNTS;
    
    for (int i = 0; i < NUM_ACCOUNTS; ++i) {
        accounts[i] = per_account;
        if (i < remainder) {
            accounts[i]++;
        }
    }
    
    // Initialize account mutexes
    for (int i = 0; i < NUM_ACCOUNTS; ++i) {
        account_mutexes[i];  // Default initialize the mutex
    }

    // Calculate iterations per thread
    int iterations_per_thread = iterations / num_threads;
    
    // Start the worker threads
    std::vector<std::thread> threads;
    std::vector<std::future<std::chrono::milliseconds>> futures;

    for (int i = 0; i < num_threads; ++i) {
        // Create a promise and get its future
        std::promise<std::chrono::milliseconds> prom;
        futures.push_back(prom.get_future());
        
        // Move the promise into the thread by value (not by reference)
        threads.emplace_back([prom = std::move(prom), iterations_per_thread, total]() mutable {
            auto t = do_work(iterations_per_thread, total);
            prom.set_value(t);
        });
    }

    // Wait for all threads to complete
    for (auto& th : threads) {
        th.join();
    }
    
    // Calculate average execution time
    long long total_time = 0;
    for (auto& fut : futures) {
        total_time += fut.get().count();
    }
    long long avg_time = total_time / num_threads;
    
    // Output in CSV format for the script to parse
    std::cout << avg_time << std::endl;
    
    return 0;
} 