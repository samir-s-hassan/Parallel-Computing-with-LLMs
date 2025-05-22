#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>     // Needed for free, strtof (if not using C++17 allocator)
#include <ctime>
#include <algorithm>
#include <chrono>
#include <limits>      // Added for numeric_limits
#include <numeric>
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>
#include <new>         // For bad_alloc, bad_array_new_length, align_val_t

// Use float for potentially better performance
using DataType = float;
// Define alignment for AVX2 (32 Bytes)
constexpr size_t ALIGNMENT = 32;

using namespace std;

// --- Aligned Allocator (Using C++17 std::align_val_t) ---
// *** Requires compilation with -std=c++17 or later ***
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    using align_val_t = std::align_val_t; // C++17 type

    template <class U> struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept {}
    template <class U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        // Use C++17 aligned new
        void* p = ::operator new[](n * sizeof(T), align_val_t{Alignment});
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t n) noexcept {
        ::operator delete[](p, align_val_t{Alignment});
    }

    friend bool operator==(const AlignedAllocator&, const AlignedAllocator&) noexcept { return true; }
    friend bool operator!=(const AlignedAllocator&, const AlignedAllocator&) noexcept { return false; }
};


// --- Cluster Class (Using Aligned Vector for Centroids) ---
// (No changes needed)
class Cluster {
private:
    int id_cluster;
    vector<DataType, AlignedAllocator<DataType, ALIGNMENT>> central_values;
    int total_values;
public:
    Cluster(int id_cluster, int dimensions) :
        id_cluster(id_cluster), central_values(dimensions), total_values(dimensions)
    { std::fill(central_values.begin(), central_values.end(), 0.0f); }
    DataType getCentralValue(int index) const { return central_values[index]; }
    void setCentralValue(int index, DataType value) { central_values[index] = value; }
    int getID() const { return id_cluster; }
    int getTotalValues() const { return total_values; }
    DataType* getCentroidData() { return central_values.data(); }
    const DataType* getCentroidData() const { return central_values.data(); }
};


// --- KMeans Class (Optimized with Aligned SoA and Explicit AVX2 SIMD) ---
// (No changes needed in KMeans class implementation itself)
class KMeans {
private:
    int K;
    int total_values, total_points, max_iterations;
    vector<Cluster> clusters;
    vector<DataType, AlignedAllocator<DataType, ALIGNMENT>> all_point_values;
    vector<int> point_cluster_ids;
    vector<string> point_names;

    inline DataType getSquaredDistSoA_AVX2_Aligned(int point_idx, const Cluster& cluster) const {
        const DataType* point_data_ptr = &all_point_values[point_idx * total_values];
        const DataType* centroid_data_ptr = cluster.getCentroidData();
        __m256 sum_vec = _mm256_setzero_ps();
        int i = 0;
        int limit = total_values - (total_values % 8);
        for (; i < limit; i += 8) {
            __m256 point_vec = _mm256_load_ps(point_data_ptr + i);
            __m256 centroid_vec = _mm256_load_ps(centroid_data_ptr + i);
            __m256 diff = _mm256_sub_ps(centroid_vec, point_vec);
            #ifdef __FMA__
                sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
            #else
                __m256 mul = _mm256_mul_ps(diff, diff);
                sum_vec = _mm256_add_ps(sum_vec, mul);
            #endif
        }
        __m128 sum_vec_128;
        __m128 lo_128 = _mm256_castps256_ps128(sum_vec);
        __m128 hi_128 = _mm256_extractf128_ps(sum_vec, 1);
        sum_vec_128 = _mm_add_ps(lo_128, hi_128);
        sum_vec_128 = _mm_hadd_ps(sum_vec_128, sum_vec_128);
        sum_vec_128 = _mm_hadd_ps(sum_vec_128, sum_vec_128);
        DataType final_sum = _mm_cvtss_f32(sum_vec_128);
        for (; i < total_values; ++i) {
            DataType diff = centroid_data_ptr[i] - point_data_ptr[i];
            final_sum += diff * diff;
        }
        return final_sum;
    }

    int getIDNearestCenterSoA(int point_idx) const {
        DataType min_dist_sq = std::numeric_limits<DataType>::max();
        int id_cluster_center = 0;
        for (int i = 0; i < K; ++i) {
            DataType dist_sq = getSquaredDistSoA_AVX2_Aligned(point_idx, clusters[i]);
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                id_cluster_center = i;
            }
        }
        return id_cluster_center;
    }

public:
    KMeans(int K, int total_points, int total_values, int max_iterations,
           vector<DataType>&& point_values_in, vector<string>&& names_in) :
        K(K), total_points(total_points), total_values(total_values),
        max_iterations(max_iterations),
        all_point_values( (size_t)total_points * total_values ),
        point_cluster_ids(total_points, -1), point_names(std::move(names_in))
    {
         if (K <= 0 || K > total_points) throw std::runtime_error("Error: K must be > 0 and <= total_points.");
         if (total_points == 0) throw std::runtime_error("Error: No points provided.");
         if (point_values_in.size() != all_point_values.size()) throw std::runtime_error("Error: Mismatch in point data size.");
         std::copy(point_values_in.begin(), point_values_in.end(), all_point_values.begin());
    }

    void run() {
        auto begin = chrono::high_resolution_clock::now();
        // Phase 1: Initialization (Same as before)
        vector<int> initial_indices(total_points);
        std::iota(initial_indices.begin(), initial_indices.end(), 0);
        std::random_shuffle(initial_indices.begin(), initial_indices.end());
        for (int i = 0; i < K; ++i) {
            clusters.emplace_back(i, total_values);
            int point_idx = initial_indices[i];
            const DataType* point_data_ptr = &all_point_values[point_idx * total_values];
            DataType* centroid_data_ptr = clusters[i].getCentroidData();
            std::copy(point_data_ptr, point_data_ptr + total_values, centroid_data_ptr);
        }
        // Phase 2: Main Iteration Loop (Same as before)
        int iter = 1;
        bool changed = true;
        vector<vector<DataType, AlignedAllocator<DataType, ALIGNMENT>>> C_sums(K, vector<DataType, AlignedAllocator<DataType, ALIGNMENT>>(total_values));
        vector<int> C_counts(K);
        while (iter <= max_iterations && changed) {
            changed = false;
            #pragma omp parallel for schedule(static) reduction(||:changed)
            for (int i = 0; i < total_points; ++i) {
                int nearest_cluster = getIDNearestCenterSoA(i);
                if (point_cluster_ids[i] != nearest_cluster) { point_cluster_ids[i] = nearest_cluster; changed = true; }
            }
            if (!changed) break;
            std::fill(C_counts.begin(), C_counts.end(), 0);
            for(int k=0; k<K; ++k) std::fill(C_sums[k].begin(), C_sums[k].end(), 0.0f);
            #pragma omp parallel
            {
                vector<vector<DataType, AlignedAllocator<DataType, ALIGNMENT>>> local_C_sums(K, vector<DataType, AlignedAllocator<DataType, ALIGNMENT>>(total_values, 0.0f));
                vector<int> local_C_counts(K, 0);
                #pragma omp for schedule(static) nowait
                for (int i = 0; i < total_points; ++i) {
                    int cluster_id = point_cluster_ids[i];
                    if (cluster_id != -1) {
                        const DataType* point_data_ptr = &all_point_values[i * total_values];
                        DataType* local_sum_ptr = local_C_sums[cluster_id].data();
                        #pragma omp simd
                        for (int j = 0; j < total_values; ++j) { local_sum_ptr[j] += point_data_ptr[j]; }
                        local_C_counts[cluster_id]++;
                    }
                }
                for(int k=0; k<K; ++k) {
                    if (local_C_counts[k] > 0) {
                        #pragma omp atomic update
                        C_counts[k] += local_C_counts[k];
                        #pragma omp critical
                        {
                            DataType* global_sum_ptr = C_sums[k].data();
                            const DataType* local_sum_ptr = local_C_sums[k].data();
                            #pragma omp simd
                            for (int j = 0; j < total_values; ++j) { global_sum_ptr[j] += local_sum_ptr[j]; }
                        }
                    }
                }
            } // End parallel region
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < K; ++i) {
                if (C_counts[i] > 0) {
                    DataType count_inv = 1.0f / C_counts[i];
                    DataType* centroid_data_ptr = clusters[i].getCentroidData();
                    const DataType* sum_data_ptr = C_sums[i].data();
                    #pragma omp simd
                    for (int j = 0; j < total_values; ++j) { centroid_data_ptr[j] = sum_data_ptr[j] * count_inv; }
                }
            }
            if (iter >= max_iterations) break;
            iter++;
        } // End while loop
        auto end = chrono::high_resolution_clock::now();
        cout << "TOTAL EXECUTION TIME = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << " us\n";
    } // End run()
}; // End KMeans class


// --- main Function (FIXED Input Reading Start) ---
int main(int argc, char* argv[]) {
    srand(741);
    ios_base::sync_with_stdio(false); cin.tie(NULL);

    int total_points, total_values, K, max_iterations, has_name;
    if (!(cin >> total_points >> total_values >> K >> max_iterations >> has_name)) {
        cerr << "Error reading input parameters." << endl; return 1;
    }
    if (total_points <= 0 || total_values <= 0 || K <= 0 || max_iterations <= 0) {
        cerr << "Error: Input parameters must be positive." << endl; return 1;
    }

    // Allocate non-aligned vector first for reading input
    vector<DataType> point_values_in( (size_t)total_points * total_values );
    vector<string> names_in(has_name ? total_points : 0);

    string line;
    // *** CORRECTED: Consume the rest of the parameter line correctly ***
    cin.ignore(numeric_limits<streamsize>::max(), '\n');


    for (int i = 0; i < total_points; ++i) {
        // *** This getline now reads the correct data line (0 to total_points-1) ***
        if (!getline(cin, line)) {
             cerr << "Error reading line for point " << i << ". Possible premature EOF or I/O error." << endl;
             cerr << "Expected " << total_points << " data lines, check input file." << endl;
             return 1;
        }

        // (Parsing logic for the line remains the same as the corrected version from previous step)
        size_t current_pos = 0;
        DataType* point_data_start = &point_values_in[ (size_t)i * total_values ];
        char* end_ptr = nullptr;
        for (int j = 0; j < total_values; ++j) {
            while (current_pos < line.length() && isspace(line[current_pos])) current_pos++;
            if (current_pos == line.length()) { cerr << "Error: Insufficient values for point " << i << ". Expected " << total_values << " got " << j << " on line: '" << line << "'" << endl; return 1; }
            errno = 0;
            point_data_start[j] = strtof(line.c_str() + current_pos, &end_ptr);
            if (end_ptr == line.c_str() + current_pos) {
                bool eol = false; char* c = end_ptr; while (*c != '\0' && isspace(*c)) c++; if (*c == '\0') eol = true;
                if (!eol || j < total_values -1) { cerr << "Error converting value for point " << i << ", dim " << j << ". No digits at pos " << (line.c_str() + current_pos - line.c_str()) << " in line: '" << line << "'" << endl; return 1; }
                else if (eol && j == total_values -1){ current_pos = end_ptr - line.c_str(); break; }
            }
            if (errno == ERANGE) { cerr << "Error converting value for point " << i << ", dim " << j << ". Out of range." << endl; return 1; }
            current_pos = end_ptr - line.c_str();
            if (current_pos < line.length() && line[current_pos] == ',') { current_pos++; }
            else if (j < total_values - 1) {
                 size_t next_non_space = current_pos; while(next_non_space < line.length() && isspace(line[next_non_space])) next_non_space++;
                 if (next_non_space < line.length()) { current_pos = next_non_space; }
                 else { cerr << "Error: Insufficient values for point " << i << ". Expected " << total_values << " got " << (j + 1) << " on line: '" << line << "'" << endl; return 1; }
            }
        } // End loop over values (j)
    } // End loop over points (i)

    try {
        KMeans kmeans(K, total_points, total_values, max_iterations, std::move(point_values_in), std::move(names_in));
        kmeans.run();
    } catch (const std::exception& e) {
        cerr << "Error during K-Means execution: " << e.what() << endl; return 1;
    }
    return 0;
}