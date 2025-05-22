// kmeans_parallel.cpp
// Parallel k-means with OpenMP, SoA layout, SIMD distance, and thread-local reductions

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <sstream>
#include <chrono>

int main(int argc, char* argv[]) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int N, D, K, max_iter, has_name;
    if (!(std::cin >> N >> D >> K >> max_iter >> has_name)) return 0;
    if (N <= 0 || D <= 0 || K <= 0) return 0;

    // Read data
    std::vector<double> data(N * D);
    std::vector<std::string> names;
    if (has_name) names.reserve(N);

    std::string line;
std::getline(std::cin, line);  // eat the header line
for (int i = 0; i < N; ++i) {
    if (!std::getline(std::cin, line)) {
        std::cerr << "Missing data row " << i << "\n";
        return 1;
    }
    std::istringstream ss(line);
    std::string tok;
    for (int d = 0; d < D; ++d) {
        if (!std::getline(ss, tok, ',')) {
            std::cerr << "Bad CSV at row " << i << ", col " << d << "\n";
            return 1;
        }
        data[i*D + d] = std::stod(tok);
    }
    if (has_name) {
        std::string name;
        ss >> name;            // if you have a name trailing the numbers
        names.push_back(name);
    }
}


    auto t0 = std::chrono::high_resolution_clock::now();

    // Initialize centroids by selecting K distinct random points
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int> distr(0, N-1);
    std::vector<int> chosen;
    chosen.reserve(K);
    std::vector<double> centroids(K * D);
    while ((int)chosen.size() < K) {
        int idx = distr(rng);
        if (std::find(chosen.begin(), chosen.end(), idx) == chosen.end()) {
            chosen.push_back(idx);
            for (int d = 0; d < D; ++d)
                centroids[(chosen.size()-1)*D + d] = data[idx*D + d];
        }
    }

    std::vector<int> assignment(N, -1);
    bool converged = false;
    int iter = 0;

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<double>> local_sums(num_threads, std::vector<double>(K * D));
    std::vector<std::vector<int>> local_counts(num_threads, std::vector<int>(K));

    while (!converged && iter < max_iter) {
        ++iter;
        converged = true;

        // Reset local accumulators
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::fill(local_sums[tid].begin(), local_sums[tid].end(), 0.0);
            std::fill(local_counts[tid].begin(), local_counts[tid].end(), 0);
        }

        // Assignment + local accumulation
        bool any_changed = false;
        #pragma omp parallel for schedule(static) reduction(||:any_changed)
        for (int i = 0; i < N; ++i) {
            // Find nearest centroid
            double best_dist = std::numeric_limits<double>::max();
            int best_k = 0;
            for (int k = 0; k < K; ++k) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int d = 0; d < D; ++d) {
                    double diff = data[i*D + d] - centroids[k*D + d];
                    sum += diff * diff;
                }
                if (sum < best_dist) {
                    best_dist = sum;
                    best_k = k;
                }
            }
            if (assignment[i] != best_k) {
                any_changed = true;
                assignment[i] = best_k;
            }

            // Update thread-local sums/counts
            int tid = omp_get_thread_num();
            auto &lsum = local_sums[tid];
            auto &lcnt = local_counts[tid];
            lcnt[best_k]++;
            for (int d = 0; d < D; ++d)
                lsum[best_k*D + d] += data[i*D + d];
        }
        converged = !any_changed;

        // Combine local accumulators into global sums/counts
        std::vector<double> new_centroids(K * D, 0.0);
        std::vector<int> new_counts(K, 0);
        for (int t = 0; t < num_threads; ++t) {
            for (int k = 0; k < K; ++k) {
                new_counts[k] += local_counts[t][k];
                for (int d = 0; d < D; ++d) {
                    new_centroids[k*D + d] += local_sums[t][k*D + d];
                }
            }
        }

        // Recompute centroids
        for (int k = 0; k < K; ++k) {
            if (new_counts[k] > 0) {
                for (int d = 0; d < D; ++d) {
                    centroids[k*D + d] = new_centroids[k*D + d] / new_counts[k];
                }
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << std::endl;

    return 0;
}
