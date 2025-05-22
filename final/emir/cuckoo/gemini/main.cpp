// main.cpp

#include "hash_set.h" // Include the header with both set implementations
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <numeric>
#include <atomic>
#include <string>
#include <iomanip> // For std::setw, std::fixed, std::setprecision
#include <memory> // For std::unique_ptr and std::make_unique for set instance
#include <cmath> // For abs checking percentages

// --- Configuration ---
struct Config {
    size_t num_threads = std::thread::hardware_concurrency(); // Default to number of cores
    size_t capacity = 100'000'000;        // Capacity of the hash set
    size_t initial_elements = 50'000'000;  // Number of elements to populate initially (aim for ~50% load factor)
    size_t ops_per_thread = 1'000'000 / num_threads; // Distribute total ops among threads
    int value_range = 100'000'000;        // Range of random values [0, value_range] (larger range reduces conflicts)
    double contains_perc = 0.80;
    double add_perc = 0.10;
    double remove_perc = 0.10;
    bool use_concurrent = true; // Choose which implementation to test
    size_t num_stripes = 0;     // For ConcurrentHashSet (0 = auto)
    std::string mode = "conc";  // "seq" or "conc"
};

// --- Thread Worker Function ---
// Templated to work with either SequentialHashSet or ConcurrentHashSet
template<typename HashSetType>
void worker_thread(
    int thread_id,
    HashSetType& hash_set,
    const Config& config,
    std::atomic<long long>& expected_size_delta // Tracks net change from adds/removes
) {
    // Per-thread random number generator. Seeding with thread_id improves randomness across threads.
    std::mt19937 gen(std::random_device{}() ^ (thread_id + 1)); // XOR with thread_id+1 for better seed diversity
    std::uniform_int_distribution<int> value_dist(0, config.value_range);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);

    long long local_delta = 0; // Track size changes locally first

    for (size_t i = 0; i < config.ops_per_thread; ++i) {
        int value = value_dist(gen); // Generate random value for the operation
        double op_choice = op_dist(gen); // Choose operation type

        if (op_choice < config.contains_perc) {
            // Contains operation (80% chance)
            // We call it but don't *need* the result for the benchmark itself
            [[maybe_unused]] volatile bool result = hash_set.contains(value);
        } else if (op_choice < config.contains_perc + config.add_perc) {
            // Add operation (10% chance)
            if (hash_set.add(value)) {
                local_delta++; // Increment delta only if add succeeded
            }
        } else {
            // Remove operation (10% chance)
            if (hash_set.remove(value)) {
                local_delta--; // Decrement delta only if remove succeeded
            }
        }
    }
    // Atomically add the local delta to the shared total delta
    // memory_order_relaxed is sufficient because we only need the final sum after all threads join.
    expected_size_delta.fetch_add(local_delta, std::memory_order_relaxed);
}

// --- Main Function ---
int main(int argc, char* argv[]) {
    // Simplified config for run.sh script
    Config config;
    
    // Expected usage: ./main <num_threads> <iterations>
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <iterations>" << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    config.num_threads = std::stoul(argv[1]);
    config.ops_per_thread = std::stoul(argv[2]) / config.num_threads;
    
    // Always use concurrent mode with the provided thread count
    config.use_concurrent = true;
    config.mode = "conc";
    
    // Initialize Hash Set
    std::unique_ptr<ConcurrentHashSet<int>> conc_set = nullptr;
    size_t actual_capacity = 0;
    size_t actual_num_stripes = 0;

    try {
        conc_set = std::make_unique<ConcurrentHashSet<int>>(config.capacity, config.num_stripes);
        actual_capacity = conc_set->get_capacity();
        actual_num_stripes = conc_set->get_num_stripes();
        
        // Populate the hash set
        conc_set->populate(config.initial_elements, config.value_range);
    } catch (const std::exception& e) {
        std::cerr << "Error during initialization or population: " << e.what() << std::endl;
        return 1;
    }

    // Get initial size before benchmark threads start
    size_t initial_size_after_populate = conc_set->size();
    
    // Benchmark
    std::vector<std::thread> threads;
    std::atomic<long long> total_expected_size_delta(0);

    // Start timing
    auto benchmark_start = std::chrono::high_resolution_clock::now();

    // Launch threads
    for (size_t i = 0; i < config.num_threads; ++i) {
        threads.emplace_back(worker_thread<ConcurrentHashSet<int>>, i, std::ref(*conc_set), std::cref(config), std::ref(total_expected_size_delta));
    }

    // Join threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // End timing
    auto benchmark_end = std::chrono::high_resolution_clock::now();
    auto benchmark_duration = std::chrono::duration_cast<std::chrono::microseconds>(benchmark_end - benchmark_start);

    // Output only the duration in microseconds for the run.sh script
    std::cout << benchmark_duration.count() << std::endl;

    return 0;
}