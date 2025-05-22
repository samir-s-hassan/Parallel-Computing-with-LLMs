// Highly Optimized Parallel Implementation of the KMeans Algorithm
// Based on: https://github.com/marcoscastro/kmeans
// Parallelized using OpenMP with advanced optimizations

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <string>
#include <omp.h>
#include <random>
#include <sstream>
#include <immintrin.h> // For SIMD intrinsics
#include <cstring>

using namespace std;

// Aligned memory for better SIMD performance
template <typename T>
T* aligned_alloc(size_t size, size_t alignment = 32) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size * sizeof(T)) != 0) {
        throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
}

class Point
{
private:
    int id_point, id_cluster;
    double* values; // Raw pointer for better performance
    int total_values;
    string name;

public:
    Point(int id_point, const vector<double>& input_values, string name = "")
    {
        this->id_point = id_point;
        total_values = input_values.size();
        
        // Allocate aligned memory for SIMD
        values = aligned_alloc<double>(total_values);
        memcpy(values, input_values.data(), total_values * sizeof(double));
        
        this->name = name;
        id_cluster = -1;
    }
    
    ~Point() {
        free(values);
    }
    
    // Move constructor to avoid copying
    Point(Point&& other) noexcept : 
        id_point(other.id_point), 
        id_cluster(other.id_cluster),
        values(other.values),
        total_values(other.total_values),
        name(std::move(other.name)) {
        other.values = nullptr;
    }
    
    // Move assignment operator
    Point& operator=(Point&& other) noexcept {
        if (this != &other) {
            free(values);
            id_point = other.id_point;
            id_cluster = other.id_cluster;
            values = other.values;
            total_values = other.total_values;
            name = std::move(other.name);
            other.values = nullptr;
        }
        return *this;
    }
    
    // Disable copy constructor and assignment
    Point(const Point&) = delete;
    Point& operator=(const Point&) = delete;

    int getID() const { return id_point; }
    void setCluster(int id_cluster) { this->id_cluster = id_cluster; }
    int getCluster() const { return id_cluster; }
    double getValue(int index) const { return values[index]; }
    int getTotalValues() const { return total_values; }
    const double* getValues() const { return values; }
    double* getValues() { return values; }
    string getName() const { return name; }
};

class KMeans
{
private:
    int K; // number of clusters
    int total_values, total_points, max_iterations;
    double** cluster_centers; // Raw pointers for better performance
    std::mt19937 gen; // Random number generator

    // Highly optimized squared distance calculation using AVX2
    inline double calculateSquaredDistanceAVX2(const double* p1, const double* p2) {
        double sum = 0.0;
        int i = 0;
        
        #ifdef __AVX2__
        // Process 4 doubles at a time using AVX2
        for (; i + 3 < total_values; i += 4) {
            __m256d v1 = _mm256_load_pd(p1 + i);
            __m256d v2 = _mm256_load_pd(p2 + i);
            __m256d diff = _mm256_sub_pd(v1, v2);
            __m256d squared = _mm256_mul_pd(diff, diff);
            
            // Horizontal sum
            __m128d low = _mm256_castpd256_pd128(squared);
            __m128d high = _mm256_extractf128_pd(squared, 1);
            __m128d sum128 = _mm_add_pd(low, high);
            sum128 = _mm_hadd_pd(sum128, sum128);
            sum += _mm_cvtsd_f64(sum128);
        }
        #endif
        
        // Handle remaining elements
        for (; i < total_values; i++) {
            double diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        
        return sum;
    }

    // Return ID of nearest center
    inline int getIDNearestCenter(const double* point_values) {
        double min_dist = calculateSquaredDistanceAVX2(point_values, cluster_centers[0]);
        int id_cluster_center = 0;

        for(int i = 1; i < K; i++)
        {
            double dist = calculateSquaredDistanceAVX2(point_values, cluster_centers[i]);
            if(dist < min_dist)
            {
                min_dist = dist;
                id_cluster_center = i;
            }
        }

        return id_cluster_center;
    }

public:
    KMeans(int K, int total_points, int total_values, int max_iterations)
    {
        this->K = K;
        this->total_points = total_points;
        this->total_values = total_values;
        this->max_iterations = max_iterations;
        this->gen = std::mt19937(714); // Set the specified seed for reproducibility
        
        // Allocate cluster centers
        cluster_centers = new double*[K];
        for (int i = 0; i < K; i++) {
            cluster_centers[i] = aligned_alloc<double>(total_values);
        }
    }
    
    ~KMeans() {
        for (int i = 0; i < K; i++) {
            free(cluster_centers[i]);
        }
        delete[] cluster_centers;
    }

    void run(vector<Point>& points)
    {
        auto begin = chrono::high_resolution_clock::now();

        if(K > total_points) return;

        // Initialize cluster centers
        vector<int> prohibited_indexes;
        std::uniform_int_distribution<> distrib(0, total_points - 1);

        // Choose K distinct values for the centers of the clusters
        for(int i = 0; i < K; i++)
        {
            while(true)
            {
                int index_point = distrib(gen);

                if(find(prohibited_indexes.begin(), prohibited_indexes.end(), index_point) == prohibited_indexes.end())
                {
                    prohibited_indexes.push_back(index_point);
                    memcpy(cluster_centers[i], points[index_point].getValues(), total_values * sizeof(double));
                    break;
                }
            }
        }

        auto end_phase1 = chrono::high_resolution_clock::now();

        int iter = 1;
        int* assignments = new int[total_points];
        memset(assignments, -1, total_points * sizeof(int));
        
        // Pre-allocate buffers for parallel reduction
        int num_threads = omp_get_max_threads();
        double*** thread_centers = new double**[num_threads];
        int** thread_counts = new int*[num_threads];
        
        for (int t = 0; t < num_threads; t++) {
            thread_centers[t] = new double*[K];
            thread_counts[t] = new int[K];
            for (int i = 0; i < K; i++) {
                thread_centers[t][i] = aligned_alloc<double>(total_values);
            }
        }

        bool done = false;
        while(!done && iter <= max_iterations)
        {
            done = true;

            // Reset thread-local buffers
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                for (int i = 0; i < K; i++) {
                    memset(thread_centers[thread_id][i], 0, total_values * sizeof(double));
                    thread_counts[thread_id][i] = 0;
                }
            }

            // Associates each point to the nearest center
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                bool local_done = true;
                
                #pragma omp for schedule(static) nowait
                for(int i = 0; i < total_points; i++)
                {
                    int old_cluster = assignments[i];
                    int new_cluster = getIDNearestCenter(points[i].getValues());

                    if(old_cluster != new_cluster)
                    {
                        assignments[i] = new_cluster;
                        local_done = false;
                    }
                    
                    // Accumulate for new centers
                    thread_counts[thread_id][new_cluster]++;
                    const double* point_values = points[i].getValues();
                    double* center_sum = thread_centers[thread_id][new_cluster];
                    
                    // Vectorized accumulation
                    int j = 0;
                    #ifdef __AVX2__
                    for (; j + 3 < total_values; j += 4) {
                        __m256d sum = _mm256_load_pd(center_sum + j);
                        __m256d values = _mm256_load_pd(point_values + j);
                        sum = _mm256_add_pd(sum, values);
                        _mm256_store_pd(center_sum + j, sum);
                    }
                    #endif
                    
                    for (; j < total_values; j++) {
                        center_sum[j] += point_values[j];
                    }
                }
                
                // Collect local_done into global done
                if (!local_done) {
                    #pragma omp atomic write
                    done = false;
                }
            }

            // Compute new centers by combining thread-local sums
            #pragma omp parallel for
            for(int i = 0; i < K; i++) {
                double total_count = 0;
                
                // Initialize accumulator
                memset(cluster_centers[i], 0, total_values * sizeof(double));
                
                // Combine thread-local results
                for (int t = 0; t < num_threads; t++) {
                    total_count += thread_counts[t][i];
                    
                    // Vectorized accumulation
                    int j = 0;
                    #ifdef __AVX2__
                    for (; j + 3 < total_values; j += 4) {
                        __m256d center = _mm256_load_pd(cluster_centers[i] + j);
                        __m256d local = _mm256_load_pd(thread_centers[t][i] + j);
                        center = _mm256_add_pd(center, local);
                        _mm256_store_pd(cluster_centers[i] + j, center);
                    }
                    #endif
                    
                    for (; j < total_values; j++) {
                        cluster_centers[i][j] += thread_centers[t][i][j];
                    }
                }
                
                // Compute average
                if (total_count > 0) {
                    double inv_count = 1.0 / total_count;
                    
                    int j = 0;
                    #ifdef __AVX2__
                    __m256d inv_count_vec = _mm256_set1_pd(inv_count);
                    for (; j + 3 < total_values; j += 4) {
                        __m256d center = _mm256_load_pd(cluster_centers[i] + j);
                        center = _mm256_mul_pd(center, inv_count_vec);
                        _mm256_store_pd(cluster_centers[i] + j, center);
                    }
                    #endif
                    
                    for (; j < total_values; j++) {
                        cluster_centers[i][j] *= inv_count;
                    }
                }
            }

            iter++;
        }
        
        auto end = chrono::high_resolution_clock::now();

        // Output minimal results
        cout << "Break in iteration " << iter - 1 << "\n\n";
        
        // Minimal output to avoid performance overhead
        for(int i = 0; i < K; i++) {
            cout << "Cluster " << i + 1 << " has points" << endl;
            cout << "Cluster values: ";
            for(int j = 0; j < min(5, total_values); j++)
                cout << cluster_centers[i][j] << " ";
            if (total_values > 5)
                cout << "...";
            cout << "\n\n";
        }

        // Calculate total time in microseconds
        auto time_us = chrono::duration_cast<chrono::microseconds>(end - begin).count();
        cout << "Total time: " << time_us << endl;

        // Cleanup
        delete[] assignments;
        for (int t = 0; t < num_threads; t++) {
            for (int i = 0; i < K; i++) {
                free(thread_centers[t][i]);
            }
            delete[] thread_centers[t];
            delete[] thread_counts[t];
        }
        delete[] thread_centers;
        delete[] thread_counts;
    }
};

int main(int argc, char *argv[])
{
    // Disable sync with stdio for faster I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int total_points, total_values, K, max_iterations, has_name;

    // Parse first line with metadata
    cin >> total_points >> total_values >> K >> max_iterations >> has_name;
    cin.ignore(); // Skip newline

    vector<Point> points;
    points.reserve(total_points);
    
    // Buffer for reading lines
    string line;
    line.reserve(1024);
    
    // Parse data points
    for(int i = 0; i < total_points; i++)
    {
        getline(cin, line);
        
        // Replace commas with spaces
        for (char& c : line) {
            if (c == ',') c = ' ';
        }
        
        istringstream line_stream(line);
        vector<double> values;
        values.reserve(total_values);
        
        double value;
        for(int j = 0; j < total_values; j++)
        {
            line_stream >> value;
            values.push_back(value);
        }

        string point_name = "";
        if(has_name) {
            line_stream >> point_name;
        }
        
        points.emplace_back(i, values, point_name);
    }

    KMeans kmeans(K, total_points, total_values, max_iterations);
    kmeans.run(points);

    return 0;
}