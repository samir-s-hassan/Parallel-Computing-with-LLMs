#include <iostream>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <optional>
#include <algorithm>
#include <functional>
#include <random>
#include <chrono>
#include <unordered_set>
#include <cstdlib>
#include <fstream>
#include <cstdlib> // for getenv
#include <iomanip> // for std::setprecision
#include <limits>


using namespace std;

const int MAX_KICKS = 100;

class CuckooHashConcurrent {
    public:
        CuckooHashConcurrent(size_t size = 5000000)
            : table_size(size), table1(size), table2(size), locks1(size), locks2(size) {}

        bool addKey(int key) {
            for (int kick = 0; kick < MAX_KICKS; ++kick) {
                shared_lock<shared_mutex> lock(resize_mutex);

                size_t i1 = hash1(key);
                {
                    lock_guard<mutex> l1(locks1[i1]);
                    if (!table1[i1].has_value()) {
                        table1[i1] = key;
                        return true;
                    }
                    swap(key, table1[i1].value());
                }

                size_t i2 = hash2(key);
                {
                    lock_guard<mutex> l2(locks2[i2]);
                    if (!table2[i2].has_value()) {
                        table2[i2] = key;
                        return true;
                    }
                    swap(key, table2[i2].value());
                }
            }

            // After MAX_KICKS, force resize
            unique_lock<shared_mutex> ulock(resize_mutex);
            if (!resize()) return false;
            return addKey(key);
        }

        bool contains(int key) const {
            size_t i1 = hash1(key);
            if (table1[i1].has_value() && table1[i1].value() == key) return true;

            size_t i2 = hash2(key);
            if (table2[i2].has_value() && table2[i2].value() == key) return true;

            return false;
        }

        bool removeKey(int key) {
            shared_lock<shared_mutex> lock(resize_mutex);

            size_t i1 = hash1(key);
            {
                lock_guard<mutex> l1(locks1[i1]);
                if (table1[i1].has_value() && table1[i1].value() == key) {
                    table1[i1].reset();
                    return true;
                }
            }

            size_t i2 = hash2(key);
            {
                lock_guard<mutex> l2(locks2[i2]);
                if (table2[i2].has_value() && table2[i2].value() == key) {
                    table2[i2].reset();
                    return true;
                }
            }
            return false;
        }

        size_t size() const { return table_size; }

        size_t value_size() const {
            size_t count = 0;
            for (const auto& v : table1) if (v.has_value()) ++count;
            for (const auto& v : table2) if (v.has_value()) ++count;
            return count;
        }

    private:
        size_t table_size;
        vector<optional<int>> table1;
        vector<optional<int>> table2;
        vector<mutex> locks1;
        vector<mutex> locks2;
        mutable shared_mutex resize_mutex;

        size_t hash1(int key) const { return std::hash<int>{}(key) % table_size; }
        size_t hash2(int key) const { return std::hash<int>{}(key * 31) % table_size; }

        bool resize() {
            if (value_size() * 2 < table_size) {
                // No need to resize
                return false;
            }

            size_t new_size = table_size * 2 + 1;
            vector<optional<int>> old_table1 = std::move(table1);
            vector<optional<int>> old_table2 = std::move(table2);

            table1 = vector<optional<int>>(new_size);
            table2 = vector<optional<int>>(new_size);
            locks1 = vector<mutex>(new_size);
            locks2 = vector<mutex>(new_size);
            table_size = new_size;

            for (const auto& v : old_table1) {
                if (v.has_value()) {
                    size_t i1 = hash1(v.value());
                    if (!table1[i1].has_value()) {
                        table1[i1] = v.value();
                    } else {
                        size_t i2 = hash2(v.value());
                        table2[i2] = v.value();
                    }
                }
            }

            for (const auto& v : old_table2) {
                if (v.has_value()) {
                    size_t i1 = hash1(v.value());
                    if (!table1[i1].has_value()) {
                        table1[i1] = v.value();
                    } else {
                        size_t i2 = hash2(v.value());
                        table2[i2] = v.value();
                    }
                }
            }

            return true;
        }

    };


void do_work(CuckooHashConcurrent &hashTable, int num_operations, int thread_id, int expected_size) {
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
            if (hashTable.addKey(key))
                hashTable.removeKey(key);
        } else {
            if (hashTable.removeKey(key))
                hashTable.addKey(key);
        }
    }
}

int main() {
    int expected_size = 5000000;
    CuckooHashConcurrent hashTable(expected_size);

    mt19937 mt{};
    uniform_int_distribution<int> ranDeposit{1, expected_size * 10};
    cout << "Populating" << endl;

    unordered_set<int> initial_keys;
    while (initial_keys.size() < expected_size * 4) {
        initial_keys.insert(ranDeposit(mt));
    }

    int inserted = 0;
    for (const int& key : initial_keys) {
        if (hashTable.addKey(key)) {
            inserted++;
        }
    }

    int initial_size = hashTable.size();
    cout << "Initial population done. Size: " << initial_size << endl;

    auto start = chrono::high_resolution_clock::now();

    int num_operations = 1000000;
    const char* env_p = getenv("NUM_OPERATIONS");
    if (env_p) {
        num_operations = atoi(env_p);
    }

    int num_threads = 16;
    num_operations /= num_threads;
    vector<thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(do_work, ref(hashTable), num_operations, i, expected_size);
    }
    for (auto& t : threads) t.join();

    auto end = chrono::high_resolution_clock::now();
    double elapsed_time = chrono::duration<double>(end - start).count();

    int missing = 0;
    for (const int& key : initial_keys) {
        if (!hashTable.contains(key)) {
            missing++;
        }
    }

    float final_size = static_cast<float>(hashTable.value_size()) / hashTable.size();
    cout << "Execution Time: " << elapsed_time << " seconds\n";
    cout << "Number of operations: " << num_operations * num_threads << "\n";
    cout << "Throughput: " << (num_operations * num_threads) / elapsed_time << " ops/sec\n";
    cout << "Final Size: " << hashTable.size() << " Percent: " << final_size << "\n";
    cout << "Expected Size: " << inserted << "\n";
    cout << "Missing Keys: " << missing << "\n";

    std::ofstream outfile("results.csv", std::ios::app); // open in append mode

    if (outfile.is_open()) {
        outfile << num_threads << ","
                << num_operations << ","
                << std::fixed << std::setprecision(9) << elapsed_time << ","
                << std::scientific << (num_operations * num_threads) / elapsed_time << "\n";

        outfile.close();
    } else {
        std::cerr << "Error opening file for writing!" << std::endl;
    }
    return 0;
}
