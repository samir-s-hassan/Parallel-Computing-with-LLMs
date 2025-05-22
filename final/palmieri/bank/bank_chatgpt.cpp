#include <iostream>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <string>

// Define the Account struct with a proper constructor.
// We disable copy and move operations because std::mutex is neither copyable nor movable.
struct Account
{
    float balance;
    std::mutex mtx;

    Account(float initial) : balance(initial) {}
    Account(const Account &) = delete;
    Account &operator=(const Account &) = delete;
};

// The Bank class encapsulates the collection of accounts, a global shared mutex,
// and methods to perform deposit and balance operations.
class Bank
{
public:
    // Constructor: evenly distribute total_funds among num_accounts.
    Bank(int num_accounts, float total_funds)
    {
        // Reserve capacity to avoid rehashing which would require moving the Account.
        accounts_.reserve(num_accounts);
        float initial_balance = total_funds / num_accounts;
        for (int i = 0; i < num_accounts; ++i)
        {
            // Use piecewise_construct to properly construct std::pair<const int, Account>
            accounts_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(i),
                std::forward_as_tuple(initial_balance));
        }
        expected_total_ = total_funds;
    }

    // Transfers 'amount' from account id1 to account id2.
    // Uses per-account locks (ordered by account id) plus a shared lock to
    // block balance computations from interleaving.
    void deposit(int id1, int id2, float amount)
    {
        // Acquire global shared lock so that no balance() computation is running.
        std::shared_lock<std::shared_mutex> global_lock(global_mutex_);

        if (id1 == id2)
            return; // Skip if the same account is chosen.

        // Lock the two involved accounts in a fixed order to avoid deadlock.
        int first = std::min(id1, id2);
        int second = std::max(id1, id2);

        std::unique_lock<std::mutex> lock_first(accounts_.at(first).mtx);
        std::unique_lock<std::mutex> lock_second(accounts_.at(second).mtx);

        // Perform the transaction:
        // Subtract amount from id1 and add to id2.
        accounts_.at(id1).balance -= amount;
        accounts_.at(id2).balance += amount;
    }

    // Returns the total sum of all account balances.
    // Acquires a unique lock on the global mutex to prevent any deposits from interleaving.
    float balance()
    {
        std::unique_lock<std::shared_mutex> global_lock(global_mutex_);
        float sum = 0.0f;
        for (auto &pair : accounts_)
        {
            sum += pair.second.balance;
        }
        return sum;
    }

    // Clear all accounts.
    void clear()
    {
        std::unique_lock<std::shared_mutex> global_lock(global_mutex_);
        accounts_.clear();
    }

    float expected_total() const { return expected_total_; }
    int num_accounts() const { return static_cast<int>(accounts_.size()); }

private:
    std::unordered_map<int, Account> accounts_;
    // Global shared mutex: deposits use shared_lock; balance() requires a unique_lock.
    std::shared_mutex global_mutex_;
    float expected_total_;
};

// Helper: returns a random integer in range [low, high].
int random_int(std::mt19937 &rng, int low, int high)
{
    std::uniform_int_distribution<int> dist(low, high);
    return dist(rng);
}

// Helper: returns a random float value in range [low, high].
float random_float(std::mt19937 &rng, float low, float high)
{
    std::uniform_real_distribution<float> dist(low, high);
    return dist(rng);
}

// Worker function executed by each thread.
// It performs 'iterations' number of iterations; on each iteration it randomly
// chooses (with 95% probability) to execute a deposit, and with 5% probability to compute
// the balance. Execution time is measured and returned (in milliseconds).
long long do_work(Bank *bank, int iterations, int thread_id)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(std::random_device{}());

    int num_accounts = bank->num_accounts();

    for (int i = 0; i < iterations; ++i)
    {
        // 95% chance for deposit, 5% for balance.
        int op = random_int(rng, 1, 100);
        if (op <= 95)
        { // deposit operation
            // Select two distinct random accounts.
            int id1 = random_int(rng, 0, num_accounts - 1);
            int id2 = random_int(rng, 0, num_accounts - 1);
            while (id2 == id1)
            {
                id2 = random_int(rng, 0, num_accounts - 1);
            }
            // Choose a random deposit amount (e.g., between 1 and 10).
            float amount = random_float(rng, 1.0f, 10.0f);
            bank->deposit(id1, id2, amount);
        }
        else
        { // balance operation
            float total = bank->balance();
            // // Verify invariant: the total should always equal the initial total.
            // if (std::abs(total - bank->expected_total()) > .1f) {
            //     std::cerr << "Invariant violation: total balance = " << total
            //               << ", expected " << bank->expected_total() << std::endl;
            // }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <threads> <accounts> <total_iterations>" << std::endl;
        return 1;
    }

    // Parse command-line arguments.
    int num_threads = std::stoi(argv[1]);
    int num_accounts = std::stoi(argv[2]);
    int total_iterations = std::stoi(argv[3]);

    if (num_accounts <= 0 || num_threads <= 0 || total_iterations <= 0)
    {
        std::cerr << "All arguments must be positive integers." << std::endl;
        return 1;
    }

    std::cout << "[DEBUG] Running with: "
              << num_threads << " threads, "
              << num_accounts << " accounts, "
              << total_iterations << " iterations" << std::endl;

    // Determine how many iterations each thread should perform.
    // We distribute the iterations roughly evenly; the first few threads get one extra if needed.
    int base_iterations = total_iterations / num_threads;
    int remainder = total_iterations % num_threads;

    const float total_funds = 100000.0f;

    Bank bank(num_accounts, total_funds);
    // std::cout << "Bank initialized with " << num_accounts << " accounts, "
    //           << total_funds << " funds." << std::endl;

    std::vector<std::thread> threads;
    std::vector<long long> exec_times(num_threads, 0);

    // Spawn threads; each thread receives its share of iterations.
    for (int i = 0; i < num_threads; ++i)
    {
        int iterations = base_iterations + (i < remainder ? 1 : 0);
        threads.emplace_back([&bank, &exec_times, i, iterations]()
                             {
            long long time_ms = do_work(&bank, iterations, i);
            exec_times[i] = time_ms; });
    }

    // Join all threads.
    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    // Compute the thread with the longest execution time.
    long long max_time = 0;
    int max_thread = -1;
    for (int i = 0; i < num_threads; ++i)
    {
        if (exec_times[i] > max_time)
        {
            max_time = exec_times[i];
            max_thread = i;
        }
    }

    // Final balance computation.
    float final_balance = bank.balance();
    // std::cout << "Final balance: " << final_balance << std::endl;

    // Output execution times.
    std::cout << "Thread " << max_thread << " had the longest execution time: "
              << max_time << " ms." << std::endl;
    for (int i = 0; i < num_threads; ++i)
    {
        std::cout << "Thread " << i << " execution time (ms): "
                  << exec_times[i] << std::endl;
    }

    // Clean up.
    bank.clear();
    // std::cout << "All accounts removed, bank cleared." << std::endl;

    return 0;
}
