// parallel_kmeans.cpp
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using Clock = chrono::high_resolution_clock;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    //--- 1) Read header line exactly like serial version
    int total_points, total_values, K, max_iterations, has_name;
    if (!(cin >> total_points >> total_values >> K >> max_iterations >> has_name)) {
        cerr << "Error reading header\n";
        return 1;
    }

    //--- NEW: Log number of rows
    cout << "Number of rows: " << total_points << "\n";

    //--- 2) Load data points
    // vector<vector<double>> points(total_points, vector<double>(total_values));
    // string tmp_name;
    // for (int i = 0; i < total_points; i++) {
    //     for (int j = 0; j < total_values; j++) {
    //         cin >> points[i][j];
    //     }
    //     if (has_name) {
    //         cin >> tmp_name;  // skip name
    //     }
    // }
    vector<vector<double>> points(total_points, vector<double>(total_values));
    string line, tmp_name;
    // skip any leftover newline after header
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    for (int i = 0; i < total_points; i++) {
        if (!getline(cin, line)) {
            cerr << "Error reading data row " << i << "\n";
            return 1;
        }
        // replace commas with spaces so stringstream can parse
        for (char &c : line) if (c == ',') c = ' ';
        istringstream ss(line);
        for (int j = 0; j < total_values; j++) {
            ss >> points[i][j];
        }
        if (has_name) {
            ss >> tmp_name;  // skip any trailing name
        }
    }

    //--- 3) Initialize centroids with fixed seed
    mt19937 gen(714);
    uniform_int_distribution<int> dist(0, total_points - 1);
    vector<vector<double>> centroids(K, vector<double>(total_values));
    vector<int> pick;
    pick.reserve(K);
    while ((int)pick.size() < K) {
        int idx = dist(gen);
        if (find(pick.begin(), pick.end(), idx) == pick.end()) {
            pick.push_back(idx);
            centroids[pick.size()-1] = points[idx];
        }
    }

    vector<int> assignments(total_points, -1);
    bool changed = true;
    int iter = 0;

    auto t0 = chrono::high_resolution_clock::now();

    //--- 4) Main Lloyd loop
    while (changed && iter < max_iterations) {
        changed = false;
        iter++;

        // 4a) Assignment step
        #pragma omp parallel for schedule(static) reduction(|:changed)
        for (int i = 0; i < total_points; i++) {
            int best = 0;
            double best_dist = 0;
            for (int d = 0; d < total_values; d++) {
                double diff = points[i][d] - centroids[0][d];
                best_dist += diff * diff;
            }
            for (int c = 1; c < K; c++) {
                double dist_sq = 0;
                for (int d = 0; d < total_values; d++) {
                    double diff = points[i][d] - centroids[c][d];
                    dist_sq += diff * diff;
                }
                if (dist_sq < best_dist) {
                    best_dist = dist_sq;
                    best = c;
                }
            }
            if (assignments[i] != best) {
                changed = true;
                assignments[i] = best;
            }
        }

        if (!changed) break;

        // 4b) Update step with per-thread accumulators
        int nthreads = omp_get_max_threads();
        vector<vector<vector<double>>> local_sums(nthreads,
            vector<vector<double>>(K, vector<double>(total_values, 0.0)));
        vector<vector<int>> local_counts(nthreads, vector<int>(K, 0));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < total_points; i++) {
                int c = assignments[i];
                local_counts[tid][c]++;
                for (int d = 0; d < total_values; d++)
                    local_sums[tid][c][d] += points[i][d];
            }
        }

        // reduce per-thread accumulators
        for (int c = 0; c < K; c++) {
            vector<double> sum_d(total_values, 0.0);
            int count = 0;
            for (int t = 0; t < nthreads; t++) {
                count += local_counts[t][c];
                for (int d = 0; d < total_values; d++)
                    sum_d[d] += local_sums[t][c][d];
            }
            if (count > 0) {
                for (int d = 0; d < total_values; d++)
                    centroids[c][d] = sum_d[d] / count;
            }
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    long long us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();

    //--- 5) Updated Logging
    cout << "Total time: " << us << "\n";
    cout << "Break in iteration: " << iter << "\n\n";

    //--- 6) Print centroids for each cluster
    for (int c = 0; c < K; c++) {
        cout << "Cluster " << (c+1) << ":";
        for (int d = 0; d < total_values; d++) {
            cout << " " << centroids[c][d];
        }
        cout << "\n";
    }

    return 0;
}
