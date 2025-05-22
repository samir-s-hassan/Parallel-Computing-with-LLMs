#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>
#include <omp.h>

// --------------------------------------------
// Assume this struct exists somewhere above:
struct Point {
    int          id;
    double*      vals;    // pointer into your flat `data[]`
    int          cluster; // current assignment
    std::string  name;
    Point(int id, double* vptr, const std::string& n = "")
      : id(id), vals(vptr), cluster(-1), name(n) {}
};
// --------------------------------------------

int main(int argc, char** argv) {
    // 1) Read header:
    int N, D, K, max_iter, has_name;
    if (!(std::cin >> N >> D >> K >> max_iter >> has_name)) {
        std::cerr << "Error: expected `N D K max_iter has_name` on stdin\n";
        return 1;
    }

    max_iter = 10000;
    // 2) Allocate storage:
    std::vector<double> data(N * D);
    std::vector<Point>  points;
    points.reserve(N);

    // consume the remainder of the first line
    std::string line;
    std::getline(std::cin, line);

    // 3) Read and parse each CSV row:
    for (int i = 0; i < N; ++i) {
        if (!std::getline(std::cin, line)) {
            std::cerr << "Error: missing row " << i << "\n";
            return 1;
        }
        std::istringstream ss(line);
        std::string token;
        for (int d = 0; d < D; ++d) {
            if (!std::getline(ss, token, ',')) {
                std::cerr << "Bad CSV format on row " << i << "\n";
                return 1;
            }
            data[i*D + d] = std::stod(token);
        }
        if (has_name) {
            std::string name;
            ss >> name;  // if there’s a name after the numbers
            points.emplace_back(i, &data[i*D], name);
        } else {
            points.emplace_back(i, &data[i*D]);
        }
    }

    // 4) Initialize centroids by sampling K distinct points:
    std::mt19937_64            rng((unsigned)time(NULL));
    std::uniform_int_distribution<int> uid(0, N-1);
    std::vector<int>           chosen;
    chosen.reserve(K);
    std::vector<double>        centroids(K*D);
    while ((int)chosen.size() < K) {
        int idx = uid(rng);
        if (std::find(chosen.begin(), chosen.end(), idx) == chosen.end()) {
            chosen.push_back(idx);
            std::copy(
              &data[idx*D],
              &data[idx*D + D],
              &centroids[(chosen.size()-1)*D]
            );
        }
    }

    // 5) Prepare thread‑local accumulators:
    int T = omp_get_max_threads();
    std::vector<std::vector<double>> local_sums  (T, std::vector<double>(K*D));
    std::vector<std::vector<int>>    local_counts(T, std::vector<int>(K));
    std::vector<char>                local_moved (T);

    bool converged = false;
    auto t0 = std::chrono::high_resolution_clock::now();

    // 6) OpenMP k‑means loop:
    #pragma omp parallel shared(centroids, converged)
    {
        int tid = omp_get_thread_num();
        auto& sum        = local_sums[tid];
        auto& cnt        = local_counts[tid];
        auto& moved_flag = local_moved[tid];

        for (int iter = 0; iter < max_iter; ++iter) {
            moved_flag = 0;
            std::fill(sum.begin(), sum.end(), 0.0);
            std::fill(cnt.begin(), cnt.end(), 0);

            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                // find nearest centroid
                double best_dist = std::numeric_limits<double>::infinity();
                int    best_k    = -1;
                double* vptr     = points[i].vals;
                for (int k = 0; k < K; ++k) {
                    double s = 0.0;
                    double* cptr = &centroids[k*D];
                    #pragma omp simd reduction(+:s)
                    for (int d = 0; d < D; ++d) {
                        double diff = vptr[d] - cptr[d];
                        s += diff*diff;
                    }
                    if (s < best_dist) {
                        best_dist = s;
                        best_k    = k;
                    }
                }
                // mark if moved
                if (points[i].cluster != best_k) {
                    points[i].cluster = best_k;
                    moved_flag = 1;
                }
                // accumulate for that thread
                for (int d = 0; d < D; ++d)
                    sum[best_k*D + d] += vptr[d];
                cnt[best_k]++;
            }

            // combine & update centroids
            #pragma omp barrier
            #pragma omp single
            {
                std::vector<double> new_centroids(K*D, 0.0);
                std::vector<int>    total_cnt(K, 0);
                // sum over threads
                for (int t = 0; t < T; ++t) {
                    for (int k = 0; k < K; ++k) {
                        total_cnt[k] += local_counts[t][k];
                        for (int d = 0; d < D; ++d)
                            new_centroids[k*D + d] += local_sums[t][k*D + d];
                    }
                }
                // finalize & check convergence
                converged = true;
                for (int k = 0; k < K; ++k) {
                    if (total_cnt[k] > 0) {
                        for (int d = 0; d < D; ++d)
                            new_centroids[k*D + d] /= total_cnt[k];
                    }
                    if (converged) {
                        for (int t = 0; t < T; ++t) {
                            if (local_moved[t]) {
                                converged = false;
                                break;
                            }
                        }
                    }
                }
                centroids.swap(new_centroids);
            }
            #pragma omp barrier
            if (converged) break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Gather cluster memberships for output
    std::vector<std::vector<int>> clusters(K);
    for (int i = 0; i < N; ++i) {
        clusters[points[i].cluster].push_back(i);
    }

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << std::endl;

    return 0;
}

// #include <iostream>
// #include <vector>
// #include <string>
// #include <random>
// #include <cmath>
// #include <algorithm>
// #include <chrono>
// #include <omp.h>
// #include <sstream>


// // Parallel K-Means clustering with OpenMP, per-thread accumulators, and SIMD directives.
// // Compile with: g++ -O3 -fopenmp -march=native KMeans_parallel.cpp -o kmeans

// struct Point {
//     int id;
//     int cluster;
//     double* vals;
//     std::string name;
//     Point(int _id, double* _vals, const std::string& _name = "")
//         : id(_id), cluster(-1), vals(_vals), name(_name) {}
// };

// int main(int argc, char** argv) {
//     int N, D, K, max_iter, has_name;
//     std::cin >> N >> D >> K >> max_iter >> has_name;

//     std::cout << N << " " << D << " " << K << " " << max_iter << " " << has_name << "\n";
//     // Read input data into a flat array (Structure of Arrays pattern)
//     std::vector<double> data(N * D);
//     std::vector<Point> points;
//     points.reserve(N);
//     std::string line;
//     std::getline(std::cin, line);               // consume end of first line
//     for(int i = 0; i < N; ++i) {
//         std::getline(std::cin, line);
//         std::istringstream ss(line);
//         std::string token;
//         for(int d = 0; d < D; ++d) {
//             if(!std::getline(ss, token, ',')) {
//                 std::cerr << "Bad CSV format on line " << i << "\n";
//                 std::exit(1);
//             }
//             data[i*D + d] = std::stod(token);
//         }
//         if(has_name) {
//             // if your names are after a space, e.g. "val,val,val name"
//             // ss >> names[i];
//         }
//     }

//     // Initialize centroids by sampling K distinct points
//     std::vector<double> centroids(K * D);
//     std::mt19937_64 rng(std::random_device{}());
//     std::uniform_int_distribution<int> uid(0, N - 1);
//     for (int k = 0; k < K; ++k) {
//         int idx = uid(rng);
//         std::copy(&data[idx * D], &data[idx * D + D], &centroids[k * D]);
//     }

//     // Thread-local accumulators: sums and counts per centroid
//     int T = omp_get_max_threads();
//     std::vector<std::vector<double>> local_sums(T, std::vector<double>(K * D, 0.0));
//     std::vector<std::vector<int>>    local_counts(T, std::vector<int>(K, 0));
//     std::vector<char>                local_moved(T, 0);

//     bool converged = false;
//     auto t0 = std::chrono::high_resolution_clock::now();

//     #pragma omp parallel shared(points, centroids, converged)
//     {
//         int tid = omp_get_thread_num();
//         auto& sum = local_sums[tid];
//         auto& cnt = local_counts[tid];
//         char& moved_flag = local_moved[tid];

//         for (int iter = 0; iter < max_iter; ++iter) {
//             // Reset per-thread state
//             moved_flag = 0;
//             std::fill(sum.begin(), sum.end(), 0.0);
//             std::fill(cnt.begin(), cnt.end(), 0);

//             // Assignment step: each point independently
//             #pragma omp for schedule(static)
//             for (int i = 0; i < N; ++i) {
//                 double best_dist = INFINITY;
//                 int best_k = 0;
//                 double* vptr = points[i].vals;
//                 // Find nearest centroid
//                 for (int k = 0; k < K; ++k) {
//                     double s = 0.0;
//                     double* cptr = &centroids[k * D];
//                     #pragma omp simd reduction(+:s)
//                     for (int d = 0; d < D; ++d) {
//                         double diff = vptr[d] - cptr[d];
//                         s += diff * diff;
//                     }
//                     if (s < best_dist) {
//                         best_dist = s;
//                         best_k = k;
//                     }
//                 }
//                 // Update cluster assignment
//                 if (points[i].cluster != best_k) {
//                     points[i].cluster = best_k;
//                     moved_flag = 1;
//                 }
//                 // Accumulate for centroid update
//                 for (int d = 0; d < D; ++d) {
//                     sum[best_k * D + d] += vptr[d];
//                 }
//                 cnt[best_k]++;
//             }

//             // Synchronize before reduction
//             #pragma omp barrier
//             #pragma omp single
//             {
//                 // Combine thread-local sums and counts
//                 std::vector<double> new_centroids(K * D, 0.0);
//                 std::vector<int>    total_count(K, 0);
//                 for (int t = 0; t < T; ++t) {
//                     for (int k = 0; k < K; ++k) {
//                         total_count[k] += local_counts[t][k];
//                         for (int d = 0; d < D; ++d) {
//                             new_centroids[k * D + d] += local_sums[t][k * D + d];
//                         }
//                     }
//                 }
//                 // Finalize centroids and check convergence
//                 converged = true;
//                 for (int k = 0; k < K; ++k) {
//                     if (total_count[k] > 0) {
//                         for (int d = 0; d < D; ++d) {
//                             new_centroids[k * D + d] /= total_count[k];
//                         }
//                     }
//                     // Check if any thread moved a point in this cluster
//                     if (converged) {
//                         for (int t = 0; t < T; ++t) {
//                             if (local_moved[t]) {
//                                 converged = false;
//                                 break;
//                             }
//                         }
//                     }
//                 }
//                 centroids.swap(new_centroids);
//             }
//             #pragma omp barrier

//             if (converged)
//                 break;
//         }
//     }

//     auto t1 = std::chrono::high_resolution_clock::now();

//     // Gather cluster memberships for output
//     std::vector<std::vector<int>> clusters(K);
//     for (int i = 0; i < N; ++i) {
//         clusters[points[i].cluster].push_back(i);
//     }

//     // std::cout << "Converged in " << (converged ? "< max_iter" : "max_iter")
//     //           << " iterations\n";
//     // std::cout << "Execution time (us): "
//     //           << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
//     //           << "\n\n";

//     // // Print clusters
//     // for (int k = 0; k < K; ++k) {
//     //     std::cout << "Cluster " << k + 1 << ":\n";
//     //     for (int idx : clusters[k]) {
//     //         std::cout << " Point " << idx + 1 << ": ";
//     //         for (int d = 0; d < D; ++d) {
//     //             std::cout << data[idx * D + d] << " ";
//     //         }
//     //         if (has_name) std::cout << "- " << points[idx].name;
//     //         std::cout << "\n";
//     //     }
//     //     std::cout << " Centroid: ";
//     //     for (int d = 0; d < D; ++d) {
//     //         std::cout << centroids[k * D + d] << " ";
//     //     }
//     //     std::cout << "\n\n";
//     // }

//     return 0;
// }
