#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <functional>
#include <unordered_set>
#include <memory> // Include for std::unique_ptr
#include <fstream>

// Use a large prime number for the hash table size
size_t TABLE_SIZE = 1000000; // Changed to a prime number.  Start with a reasonable size.
const int MAX_CUCKOO_ITERATIONS = 100;
using namespace std;

// Define a simple, thread-safe random number generator.
class ThreadSafeRandom {
public:
    ThreadSafeRandom() : gen(rd()) {}

    size_t get_random_size_t() {
        std::lock_guard<std::mutex> lock(m);
        return dist(gen);
    }

    double get_random_double() {
        std::lock_guard<std::mutex> lock(m);
        return double_dist(gen);
    }

private:
    std::random_device rd;
    std::mt19937_64 gen;
    std::uniform_int_distribution<size_t> dist{ 0, std::numeric_limits<size_t>::max() };
    std::uniform_real_distribution<double> double_dist{ 0.0, 1.0 };
    std::mutex m;
};

ThreadSafeRandom random_generator;

// Use a more robust hash function (FNV-1a)
size_t fnv1a_hash(const void* data, size_t size) {
    size_t hash = 0xcbf29ce484222325;
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 0x100000001b3;
    }
    return hash;
}

size_t custom_hash(size_t key, size_t i) {
    // Combine FNV-1a with a secondary hash function
    size_t primary_hash = fnv1a_hash(&key, sizeof(key));
    return (primary_hash + i * (primary_hash >> 16 | 1)) % TABLE_SIZE;
}

// Concurrent Cuckoo Hash Set Implementation
class ConcurrentCuckooHashSet {
public:
    ConcurrentCuckooHashSet() : table1(TABLE_SIZE), table2(TABLE_SIZE), num_elements(0), rehash_count(0) {
        // Initialize the mutexes using unique_ptr
        locks1.resize(TABLE_SIZE);
        locks2.resize(TABLE_SIZE);
        for (size_t i = 0; i < TABLE_SIZE; ++i) {
            locks1[i] = make_unique<mutex>();
            locks2[i] = make_unique<mutex>();
        }
    }

    ~ConcurrentCuckooHashSet() {}

    bool add(size_t key) {
        rehash_count = 0; // Reset rehash count for each add operation.
        size_t current_key = key;
        for (int i = 0; i < MAX_CUCKOO_ITERATIONS + 1; ++i) { // Allow one extra iteration
            size_t index1 = custom_hash(current_key, 0);
            size_t index2 = custom_hash(current_key, 1);

            {
                // Use lock_guard with the raw mutex from the unique_ptr
                std::unique_lock<std::mutex> lock1(*(locks1[index1]));
                std::unique_lock<std::mutex> lock2(*(locks2[index2]));


                if (table1[index1] == 0) {
                    table1[index1] = current_key;
                    num_elements++;
                    return true;
                }
                else if (table2[index2] == 0) {
                    table2[index2] = current_key;
                    num_elements++;
                    return true;
                }
                else {
                    size_t evicted_key = (i % 2 == 0) ? table1[index1] : table2[index2];
                    if (i % 2 == 0) {
                        table1[index1] = current_key;
                    }
                    else {
                        table2[index2] = current_key;
                    }
                    current_key = evicted_key;
                }
            }
        }
        //If it reaches the max iteration, rehash the entire table.
        if (rehash_count < MAX_REHASH_ATTEMPTS)
        {
            rehash_count++;
            rehash();
            return add(key); // Use original key here
        }
        else
        {
             return false; // Indicate failure to add after max rehashes.
        }
    }

    bool remove(size_t key) {
        size_t index1 = custom_hash(key, 0);
        size_t index2 = custom_hash(key, 1);

        {
            // Use lock_guard with the raw mutex from the unique_ptr
            std::unique_lock<std::mutex> lock1(*(locks1[index1]));
            std::unique_lock<std::mutex> lock2(*(locks2[index2]));

            if (table1[index1] == key) {
                table1[index1] = 0;
                num_elements--;
                return true;
            }
            else if (table2[index2] == key) {
                table2[index2] = 0;
                num_elements--;
                return true;
            }
        }
        return false;
    }

    bool contains(size_t key)  {
        size_t index1 = custom_hash(key, 0);
        size_t index2 = custom_hash(key, 1);

        {
            // Use lock_guard with the raw mutex from the unique_ptr
            std::lock_guard<std::mutex> lock1(*(locks1[index1]));
            std::lock_guard<std::mutex> lock2(*(locks2[index2]));
            return (table1[index1] == key || table2[index2] == key);
        }
    }

    size_t size() const {
        return num_elements.load();
    }

    void populate(size_t num_elements_to_populate) {
        for (size_t i = 0; i < num_elements_to_populate; ++i) {
            size_t key = random_generator.get_random_size_t();
            if(!add(key))
            {
                cout << "Failed to add key during populate: " << key << endl;
            }
        }
    }

private:
    std::vector<size_t> table1;
    std::vector<size_t> table2;
    // Use vectors of unique_ptr to mutexes
    std::vector<unique_ptr<mutex>> locks1;
    std::vector<unique_ptr<mutex>> locks2;
    std::atomic<size_t> num_elements;
    std::atomic<int> rehash_count;
    const int MAX_REHASH_ATTEMPTS = 5;
    const size_t MAX_TABLE_SIZE = 1 << 24;

    void rehash() {
        size_t new_table_size = TABLE_SIZE * 2; // Double the table size.
        if (new_table_size > MAX_TABLE_SIZE)
        {
            cout << "Maximum table size reached.  Rehashing failed." << endl;
            return;
        }

        std::vector<size_t> new_table1(new_table_size, 0);
        std::vector<size_t> new_table2(new_table_size, 0);
        // Create new vectors of unique_ptr for the new mutexes
        std::vector<unique_ptr<mutex>> new_locks1(new_table_size);
        std::vector<unique_ptr<mutex>> new_locks2(new_table_size);

        // Initialize the new mutexes
        for (size_t i = 0; i < new_table_size; ++i) {
            new_locks1[i] = make_unique<mutex>();
            new_locks2[i] = make_unique<mutex>();
        }

        std::vector<size_t> all_keys;
        all_keys.reserve(num_elements);

        // 1. Collect all keys
        for (size_t i = 0; i < TABLE_SIZE; ++i) {
             // Use lock_guard with the raw mutex.
            std::unique_lock<std::mutex> lock1(*(locks1[i]));
            if (table1[i] != 0) {
                all_keys.push_back(table1[i]);
            }
        }
        for (size_t i = 0; i < TABLE_SIZE; ++i) {
             // Use lock_guard with the raw mutex.
            std::unique_lock<std::mutex> lock2(*(locks2[i]));
            if (table2[i] != 0) {
                all_keys.push_back(table2[i]);
            }
        }

        // 2. Update the tables and locks
        table1 = new_table1;
        table2 = new_table2;
        locks1 = std::move(new_locks1); // Use std::move to transfer ownership
        locks2 = std::move(new_locks2); // Use std::move to transfer ownership
        TABLE_SIZE = new_table_size; // VERY IMPORTANT:  Update the global table size!

        num_elements = 0; // Reset the element count
        rehash_count = 0;
        // 3. Re-add all keys
        for (size_t key : all_keys) {
            add(key); // Use add() to re-insert, handles cuckoo logic
        }
    }
};

void do_work(ConcurrentCuckooHashSet &hashTable, int num_operations, int thread_id, int expected_size) {
    mt19937 mt(random_device{}() + thread_id);
    uniform_int_distribution<int> ranDeposit(1, expected_size * 2);
    uniform_real_distribution<double> op_dist(0.0, 1.0);

    for (int i = 0; i < num_operations; ++i) {
        double p = op_dist(mt);
        int key = ranDeposit(mt);

        if (p < .8) {
            hashTable.contains(key);
        } else if (p < .9) {
            int first_table = i % 2;
            if (hashTable.add(key))
                hashTable.remove(key);
        } else {
            if (hashTable.remove(key))
                hashTable.add(key);
        }
    }
}

int main() {
    int expected_size = 1000000;
    ConcurrentCuckooHashSet hashTable;
    hashTable.populate(expected_size); // Populate with expected size

    mt19937 mt{};
    uniform_int_distribution<int> ranDeposit{1, expected_size * 10};
    cout << "Populating" << endl;



    int initial_size = hashTable.size();
    cout << "Initial population done. Size: " << initial_size << endl;

    auto start = chrono::high_resolution_clock::now();

    int num_operations = 1000000;
    const char* env_p = getenv("NUM_OPERATIONS");
    if (env_p) {
        num_operations = atoi(env_p);
    }

    int num_threads = 16;
    env_p = getenv("THREADS");
    if (env_p) {
        num_threads = atoi(env_p);
    }
    num_operations /= num_threads;
    vector<thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(do_work, ref(hashTable), num_operations, i, expected_size);
    }
    for (auto& t : threads) t.join();

    auto end = chrono::high_resolution_clock::now();
    double elapsed_time = chrono::duration<double>(end - start).count();


    cout << "Threads: " << num_threads << "\n";
    cout << "Operations per thread: " << num_operations << "\n";
    cout << "Execution Time: " << elapsed_time << " seconds\n";
    cout << "Throughput: " << (num_operations * num_threads) / elapsed_time << " ops/sec\n";
    cout << "Final Size: " << hashTable.size()  << "\n";
    cout << "Expected Size: " << expected_size << "\n";

    std::ofstream outfile("results.csv", std::ios::app);

    // If file is opened successfully
    if (outfile.is_open()) {
        outfile << num_threads << ","
                << num_operations << ","
                << elapsed_time << ","
                << (num_operations * num_threads) / elapsed_time << ","
                << hashTable.size() << ","
                << expected_size << "\n";

        outfile.close();  // Close the file after writing
    } else {
        std::cerr << "Error opening file for writing!" << std::endl;
    }

    return 0;
}
