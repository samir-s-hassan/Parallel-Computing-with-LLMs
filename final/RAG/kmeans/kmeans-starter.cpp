// kmeans_parallel_csv.cpp
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>

using namespace std;

int main(int argc, char *argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Read header: N, D, K, max_iters, has_name
    int N, D, K, max_iters, has_name;
    if (!(cin >> N >> D >> K >> max_iters >> has_name)) return 0;
    if (N <= 0 || D <= 0 || K <= 0) return 0;

    // Prepare flat storage + optional names
    vector<double> data(N * D);
    vector<string> names;
    if (has_name) names.reserve(N);

    string line;
    getline(cin, line); // eat end‑of‑header newline

    // Read each CSV row
    for (int i = 0; i < N; ++i) {
        if (!getline(cin, line)) {
            cerr << "Missing data row " << i << "\n";
            return 1;
        }
        istringstream ss(line);
        string tok;

        // Parse D comma‑separated values
        for (int d = 0; d < D; ++d) {
            if (!getline(ss, tok, ',')) {
                cerr << "Bad CSV at row " << i << ", col " << d << "\n";
                return 1;
            }
            data[i * D + d] = stod(tok);
        }
        // Optional trailing name
        if (has_name) {
            string name;
            ss >> name;
            names.push_back(name);
        }
    }

    // Unflatten into points[row][col]
    vector<vector<double>> points(N, vector<double>(D));
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            points[i][d] = data[i * D + d];
        }
    }
    // (we no longer need 'data' beyond this point)

    // --- K-means setup ---
    vector<int> cluster_id(N, -1);
    vector<vector<double>> centroids(K, vector<double>(D));
    srand(55);
    vector<bool> chosen(N, false);
    for (int c = 0; c < K; ++c) {
        int idx;
        do {
            idx = rand() % N;
        } while (chosen[idx]);
        chosen[idx] = true;
        cluster_id[idx] = c;
        centroids[c] = points[idx];
    }

    // Working arrays for the fused step
    vector<vector<double>> sums(K, vector<double>(D));
    vector<int> counts(K);

    double t_start = omp_get_wtime();
    int iter = 0;

    while (iter < max_iters) {
        // Zero global accumulators
        for (int c = 0; c < K; ++c) {
            counts[c] = 0;
            for (int d = 0; d < D; ++d)
                sums[c][d] = 0.0;
        }

        int changed = 0;

        // Fused assign + sum
        #pragma omp parallel
        {
            vector<vector<double>> local_sums(K, vector<double>(D, 0.0));
            vector<int> local_counts(K, 0);
            int local_changed = 0;

            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                // find nearest centroid
                double best_dist = numeric_limits<double>::max();
                int best_c = -1;
                for (int c = 0; c < K; ++c) {
                    double dist2 = 0.0;
                    for (int d = 0; d < D; ++d) {
                        double diff = points[i][d] - centroids[c][d];
                        dist2 += diff * diff;
                    }
                    if (dist2 < best_dist) {
                        best_dist = dist2;
                        best_c = c;
                    }
                }

                if (cluster_id[i] != best_c) {
                    local_changed++;
                    cluster_id[i] = best_c;
                }

                local_counts[best_c]++;
                for (int d = 0; d < D; ++d) {
                    local_sums[best_c][d] += points[i][d];
                }
            }

            // merge into global
            #pragma omp critical
            {
                for (int c = 0; c < K; ++c) {
                    counts[c] += local_counts[c];
                    for (int d = 0; d < D; ++d) {
                        sums[c][d] += local_sums[c][d];
                    }
                }
            }
            #pragma omp atomic
            changed += local_changed;
        }

        // recompute centroids in parallel
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < K; ++c) {
            if (counts[c] > 0) {
                for (int d = 0; d < D; ++d) {
                    centroids[c][d] = sums[c][d] / counts[c];
                }
            }
        }

        if (changed == 0) break;
        ++iter;
    }
    double t_end = omp_get_wtime();
    // print microseconds to match original
    cout << static_cast<long long>((t_end - t_start) * 1e6) << "\n";
    return 0;
}
