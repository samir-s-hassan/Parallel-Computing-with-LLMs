#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;

typedef vector<double> Vec;

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N, D, K, max_iters, has_name;
    cin >> N >> D >> K >> max_iters >> has_name;

    vector<double> data(N * D);
    string dummy;
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d)
            cin >> data[i*D + d];
        if (has_name) cin >> dummy;  // skip names
    }

    // Random initialization of centroids
    vector<double> centroids(K * D);
    {
        mt19937 gen(0);
        uniform_int_distribution<> dist(0, N-1);
        for (int k = 0; k < K; ++k) {
            int idx = dist(gen);
            for (int d = 0; d < D; ++d)
                centroids[k*D + d] = data[idx*D + d];
        }
    }

    vector<int> assign(N, -1), assign_new(N);
    vector<double> new_centroids(K * D);
    vector<int> counts(K);

    auto t0 = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;

        // 1) Assignment step
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            int best_k = 0;
            double best_dist = 0;
            // compute initial distance
#pragma omp simd reduction(+:best_dist)
            for (int d = 0; d < D; ++d) {
                double diff = centroids[d] - data[i*D + d];
                best_dist += diff * diff;
            }
            // square‐root not needed for comparison
            for (int k = 1; k < K; ++k) {
                double dist = 0;
#pragma omp simd reduction(+:dist)
                for (int d = 0; d < D; ++d) {
                    double diff = centroids[k*D + d] - data[i*D + d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }
            assign_new[i] = best_k;
            if (best_k != assign[i]) changed = true;
        }

        if (!changed) break;
        assign.swap(assign_new);

        // 2) Update step
        fill(new_centroids.begin(), new_centroids.end(), 0.0);
        fill(counts.begin(), counts.end(), 0);

#pragma omp parallel
        {
            // Thread‐local buffers
            vector<double> local_cent(K * D, 0.0);
            vector<int> local_cnt(K, 0);

#pragma omp for nowait schedule(static)
            for (int i = 0; i < N; ++i) {
                int k = assign[i];
                local_cnt[k]++;
#pragma omp simd
                for (int d = 0; d < D; ++d)
                    local_cent[k*D + d] += data[i*D + d];
            }

#pragma omp critical
            {
                for (int k = 0; k < K; ++k) {
                    counts[k] += local_cnt[k];
                    for (int d = 0; d < D; ++d)
                        new_centroids[k*D + d] += local_cent[k*D + d];
                }
            }
        }

        // finalize centroids
        for (int k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                for (int d = 0; d < D; ++d)
                    centroids[k*D + d] = new_centroids[k*D + d] / counts[k];
            }
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    long long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    cout << elapsed << "\n";

    // (Optional) print final assignments or centroids here
    return 0;
}