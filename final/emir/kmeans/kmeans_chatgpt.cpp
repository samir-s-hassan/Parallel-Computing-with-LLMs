#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <limits>
#include <iomanip>
#include <map>

// Enable fast-math and strict aliasing for better compiler optimizations
#pragma GCC optimize("Ofast", "unroll-loops", "omit-frame-pointer", "inline")
#pragma GCC option("arch=native", "tune=native", "no-zero-upper")
#pragma GCC target("avx2", "fma")

// Alignment for cache-line size to prevent false sharing
constexpr size_t CACHE_LINE_SIZE = 64;

// Flat data structure for better cache locality and SIMD
class KMeans {
private:
    int K, total_points, total_values, max_iterations;
    
    // Flat data storage for points
    std::vector<double> point_data;            // [point_id * total_values + dim_id]
    std::vector<int> point_cluster_id;         // [point_id]
    std::vector<std::string> point_class_name; // [point_id]
    
    // Flat storage for cluster centroids
    std::vector<double> cluster_centroids;     // [cluster_id * total_values + dim_id]
    
    // Store point indices per cluster (after clustering)
    std::vector<std::vector<int>> points_in_cluster; // [cluster_id][point_index_in_cluster]
    
    bool has_class_names;
    
    // Fast inline accessors for point values
    inline double& point_value(int point_id, int dim_id) {
        return point_data[point_id * total_values + dim_id];
    }
    
    inline const double& point_value(int point_id, int dim_id) const {
        return point_data[point_id * total_values + dim_id];
    }
    
    // Fast inline accessors for cluster centroids
    inline double& centroid_value(int cluster_id, int dim_id) {
        return cluster_centroids[cluster_id * total_values + dim_id];
    }
    
    inline const double& centroid_value(int cluster_id, int dim_id) const {
        return cluster_centroids[cluster_id * total_values + dim_id];
    }
    
    // Optimized distance calculation with SIMD vectorization
    double getDistance(int point_id, int cluster_id) const {
        double sum = 0.0;
        
        // Get base pointers to the point and cluster data for this calculation
        const double* p_data = &point_data[point_id * total_values];
        const double* c_data = &cluster_centroids[cluster_id * total_values];
        
        // Use SIMD for vectorized distance calculation
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < total_values; i++) {
            double diff = c_data[i] - p_data[i];
            sum += diff * diff;
        }
        
        return sum; // No need for sqrt during comparisons for efficiency
    }
    
    // Fast nearest center finder with loop unrolling for small K
    int getIDNearestCenter(int point_id) const {
        double min_dist = std::numeric_limits<double>::max();
        int nearest_cluster = 0;
        
        // Special case: fast path for K <= 8 (common case)
        if (K <= 8) {
            // Manually unrolled loop for small K to help compiler optimize
            double distances[8];
            for (int k = 0; k < K; k++) {
                distances[k] = getDistance(point_id, k);
            }
            
            // Find minimum distance (branchless when possible)
            for (int k = 0; k < K; k++) {
                if (distances[k] < min_dist) {
                    min_dist = distances[k];
                    nearest_cluster = k;
                }
            }
        } else {
            // General case for larger K
            for (int k = 0; k < K; k++) {
                double dist = getDistance(point_id, k);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_cluster = k;
                }
            }
        }
        
        return nearest_cluster;
    }
    
    // Aligned struct to prevent false sharing in parallel accumulation
    struct alignas(CACHE_LINE_SIZE) ClusterAccumulator {
        std::vector<double> sum;
        int count;
        std::vector<int> point_indices;
        char padding[CACHE_LINE_SIZE - sizeof(std::vector<double>) - sizeof(int) - sizeof(std::vector<int>)];
        
        ClusterAccumulator(int dims) : sum(dims, 0.0), count(0) {}
    };
    
public:
    KMeans(int K, int max_iterations, bool has_class_names = false) 
        : K(K), max_iterations(max_iterations), has_class_names(has_class_names), 
          total_points(0), total_values(0) {
        // Pre-allocate cluster storage
        points_in_cluster.resize(K);
    }
    
    void loadPoints(const std::string& filename) {
        std::ifstream infile(filename);
        if (!infile) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }
        
        // Read header line with parameters (space-separated)
        std::string header;
        std::getline(infile, header);
        std::istringstream header_stream(header);
        int expected_points, expected_values, expected_k, expected_max_iter, has_name;
        header_stream >> expected_points >> expected_values >> expected_k >> expected_max_iter >> has_name;
        
        // std::cout << "Dataset header: " << expected_points << " points, " 
        //           << expected_values << " dimensions, " << expected_k << " clusters, "
        //           << expected_max_iter << " max iterations, " 
        //           << (has_name ? "with class names" : "no class names") << std::endl;
        
        // Update our parameters based on header
        total_values = expected_values;
        has_class_names = (has_name == 1);
        
        // Preallocate memory for performance
        point_data.reserve(expected_points * total_values);
        point_cluster_id.resize(expected_points, -1);
        if (has_class_names) {
            point_class_name.resize(expected_points);
        }
        
        std::string line;
        int id = 0;
        
        while (std::getline(infile, line)) {
            if (line.empty()) continue;
            
            std::stringstream ss(line);
            std::string value_str;
            int feature_count = 0;
            
            // Read comma-separated values
            while (feature_count < total_values && std::getline(ss, value_str, ',')) {
                // Trim whitespace
                value_str.erase(0, value_str.find_first_not_of(" \t\r\n"));
                value_str.erase(value_str.find_last_not_of(" \t\r\n") + 1);
                
                if (value_str.empty()) continue;
                
                try {
                    double val = std::stod(value_str);
                    point_data.push_back(val);
                    feature_count++;
                } catch (const std::exception& e) {
                    std::cerr << "Error converting value at line " << id+1 << ", feature " << feature_count << ": " << value_str << std::endl;
                    break;
                }
            }
            
            // Check if we read all expected features
            if (feature_count < total_values) {
                std::cerr << "Error: Expected " << total_values << " features but got only " << feature_count << " at line " << id+1 << std::endl;
                continue;
            }
            
            // Read class name if present
            if (has_class_names && std::getline(ss, value_str)) {
                // Trim whitespace
                value_str.erase(0, value_str.find_first_not_of(" \t\r\n"));
                value_str.erase(value_str.find_last_not_of(" \t\r\n") + 1);
                
                point_class_name[id] = value_str;
            }
            
            id++;
        }
        
        total_points = id;
        
        // std::cout << "Successfully loaded: " << total_points << " points, " 
        //           << total_values << " dimensions" << std::endl;
                  
        if (total_points != expected_points) {
            std::cout << "Warning: Expected " << expected_points << " points but got " 
                      << total_points << std::endl;
            
            // Resize vectors to actual size
            point_cluster_id.resize(total_points);
            if (has_class_names) {
                point_class_name.resize(total_points);
            }
        }
        
        // Initialize cluster centroids storage
        cluster_centroids.resize(K * total_values);
    }
    
    // Generate random points for testing
    // void generateRandomPoints(int num_points, int dimensions, double min_val = 0.0, double max_val = 100.0) {
    //     total_points = num_points;
    //     total_values = dimensions;
        
    //     // Preallocate flat arrays
    //     point_data.resize(total_points * total_values);
    //     point_cluster_id.resize(total_points, -1);
    //     cluster_centroids.resize(K * total_values);
        
    //     std::random_device rd;
    //     std::mt19937 gen(741);
    //     std::uniform_real_distribution<> dis(min_val, max_val);
        
    //     // Generate random points in flat array
    //     #pragma omp parallel for
    //     for (int i = 0; i < total_points; i++) {
    //         for (int j = 0; j < dimensions; j++) {
    //             point_value(i, j) = dis(gen);
    //         }
    //     }
        
    //     std::cout << "Generated: " << total_points << " points, " << total_values << " dimensions" << std::endl;
    // }
    
    void run() {
        if (total_points < K) {
            std::cerr << "Error: Number of clusters cannot be greater than number of points" << std::endl;
            return;
        }
        
        // Initialize clusters with first K points
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < total_values; j++) {
                centroid_value(i, j) = point_value(i, j);
            }
        }
        
        // std::cout << "Starting K-Means clustering with K=" << K << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int iteration = 0;
        bool done = false;
        
        // Allocate thread-local storage for reduction
        int num_threads = omp_get_max_threads();
        
        // Main loop
        while (!done && iteration < max_iterations) {
            done = true;
            
            // Clear previous cluster assignments
            for (int i = 0; i < K; i++) {
                points_in_cluster[i].clear();
            }
            
            // Thread-local accumulators to prevent false sharing
            std::vector<std::vector<ClusterAccumulator>> thread_local_accumulators(num_threads);
            
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                thread_local_accumulators[tid].resize(K, ClusterAccumulator(total_values));
                
                // Fused assignment and accumulation loop with optimized scheduling
                #pragma omp for schedule(static, 64) reduction(&&:done) nowait
                for (int i = 0; i < total_points; i++) {
                    int new_cluster_id = getIDNearestCenter(i);
                    int old_cluster_id = point_cluster_id[i];
                    
                    if (new_cluster_id != old_cluster_id) {
                        point_cluster_id[i] = new_cluster_id;
                        done = false;
                    }
                    
                    // Accumulate for centroid recalculation
                    ClusterAccumulator& acc = thread_local_accumulators[tid][new_cluster_id];
                    acc.count++;
                    acc.point_indices.push_back(i);
                    
                    for (int j = 0; j < total_values; j++) {
                        acc.sum[j] += point_value(i, j);
                    }
                }
            }
            
            // Reduce thread-local accumulators to global
            std::vector<std::vector<double>> total_sums(K, std::vector<double>(total_values, 0.0));
            std::vector<int> total_counts(K, 0);
            
            for (int tid = 0; tid < num_threads; tid++) {
                for (int k = 0; k < K; k++) {
                    const ClusterAccumulator& acc = thread_local_accumulators[tid][k];
                    
                    // Reduction of sums
                    for (int j = 0; j < total_values; j++) {
                        total_sums[k][j] += acc.sum[j];
                    }
                    
                    // Reduction of counts
                    total_counts[k] += acc.count;
                    
                    // Collect point indices
                    points_in_cluster[k].insert(
                        points_in_cluster[k].end(),
                        acc.point_indices.begin(),
                        acc.point_indices.end()
                    );
                }
            }
            
            // Update centroids
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < K; k++) {
                if (total_counts[k] > 0) {
                    for (int j = 0; j < total_values; j++) {
                        centroid_value(k, j) = total_sums[k][j] / total_counts[k];
                    }
                }
            }
            
            std::cout << "Iteration " << iteration << std::endl;
            
            iteration++;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        
        std::cout << "K-Means clustering completed in " << iteration << " iterations" << std::endl;
        std::cout << "Time taken: " << duration << " μs" << std::endl;
        
        // Print cluster sizes
        for (int i = 0; i < K; i++) {
            std::cout << "Cluster " << i << " contains " << points_in_cluster[i].size() << " points" << std::endl;
        }
        
        // If we have class names, analyze clusters
        if (has_class_names) {
            analyzeClustersByClass();
        }
    }
    
    // Analyze clusters by class names
    void analyzeClustersByClass() {
        if (!has_class_names) return;
        
        // Get unique class names
        std::vector<std::string> unique_classes;
        for (const auto& cls : point_class_name) {
            if (!cls.empty() && std::find(unique_classes.begin(), unique_classes.end(), cls) == unique_classes.end()) {
                unique_classes.push_back(cls);
            }
        }
        
        std::cout << "\nCluster Analysis by Class:" << std::endl;
        std::cout << "--------------------------" << std::endl;
        
        // Count class instances in each cluster
        for (int i = 0; i < K; i++) {
            std::cout << "Cluster " << i << ":" << std::endl;
            
            // Map to store count of each class in this cluster
            std::map<std::string, int> class_counts;
            for (const auto& cls : unique_classes) {
                class_counts[cls] = 0;
            }
            
            // Count classes using point indices
            for (int point_idx : points_in_cluster[i]) {
                std::string cls = point_class_name[point_idx];
                class_counts[cls]++;
            }
            
            // Print counts and percentages
            int cluster_size = points_in_cluster[i].size();
            for (const auto& pair : class_counts) {
                double percentage = (cluster_size > 0) ? 
                                   (100.0 * pair.second / cluster_size) : 0.0;
                std::cout << "  " << std::left << std::setw(20) << pair.first 
                          << ": " << std::setw(5) << pair.second << " (" 
                          << std::fixed << std::setprecision(2) << percentage << "%)" 
                          << std::endl;
            }
            std::cout << std::endl;
        }
    }
    
    // Calculate sum of squared errors (SSE) with SIMD
    double calculateSSE() {
        double sse = 0.0;
        
        #pragma omp parallel for reduction(+:sse) schedule(static, 64)
        for (int i = 0; i < total_points; i++) {
            int cluster_id = point_cluster_id[i];
            sse += getDistance(i, cluster_id);
        }
        
        return sse;
    }
    
    void printResults(const std::string& output_file = "") {
        if (!output_file.empty()) {
            std::ofstream outfile(output_file);
            if (!outfile) {
                std::cerr << "Error opening output file: " << output_file << std::endl;
                return;
            }
            
            for (int i = 0; i < total_points; i++) {
                outfile << i << " " << point_cluster_id[i];
                if (has_class_names) {
                    outfile << " " << point_class_name[i];
                }
                outfile << std::endl;
            }
            
            // std::cout << "Results written to " << output_file << std::endl;
        } else {
            double sse = calculateSSE();
            std::cout << "Sum of Squared Errors: " << sse << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    // Parameters from the dataset header: 11247 16 4 100 1
    // total_points, total_values, K, max_iterations, has_name
    int K = 4;
    int max_iterations = 100;
    bool has_class_names = true;
    
    // Environment variables for better thread placement
    // These would typically be set outside the program:
    // export OMP_PROC_BIND=true
    // export OMP_PLACES=cores
    
    // Set number of threads for OpenMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    
    // Set OpenMP dynamic adjustment off for better predictability
    omp_set_dynamic(0);
    
    // Print CPU and thread info
    // std::cout << "Running Final Optimized K-Means with " << num_threads << " threads" << std::endl;
    // std::cout << "Cache line size: " << CACHE_LINE_SIZE << " bytes" << std::endl;
    
    // Initialize K-Means
    KMeans kmeans(K, max_iterations, has_class_names);
    
    // Load the Dry Bean Dataset with a relative path
    std::string dataset_path = argc > 1 ? argv[1] : "./dataset/drybeans.txt";
    kmeans.loadPoints(dataset_path);
    
    // Run the algorithm
    kmeans.run();
    
    // Print results
    kmeans.printResults();
    
    // Optionally save results to a file
    // kmeans.printResults("./results/results-final.txt");
    
    return 0;
} 