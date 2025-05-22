// kmeans_soa_avx2_tbb.cpp
// C++17, Intel TBB, AVX2 implementation of K-Means with SoA + task-parallel accumulators
//
// Compile with:
//   g++ -O3 -march=native -mavx2 -std=c++17 \
//       -I"$TBBROOT/include" -L"$TBBROOT/lib/intel64/gcc4.8" \
//       src/kmeans_soa_avx2_tbb.cpp -o kmeans_parallel -ltbb

#include <tbb/tbb.h>
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace tbb;

// horizontal sum of an __m256d
inline double hsum256_pd(__m256d v)
{
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    double tmp[2];
    _mm_storeu_pd(tmp, lo);
    return tmp[0] + tmp[1];
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N, D, K, max_iter, has_name;
    if (!(cin >> N >> D >> K >> max_iter >> has_name))
    {
        cerr << "Failed to read header\n";
        return 1;
    }

    // Read points into a flat vector
    vector<double> points(size_t(N) * D);
    string line, name;
    getline(cin, line); // consume end of first line
    for (int i = 0; i < N; ++i)
    {
        getline(cin, line);
        stringstream ss(line);
        for (int d = 0; d < D; ++d)
        {
            string item;
            getline(ss, item, ',');
            points[size_t(i) * D + d] = stod(item);
        }
        if (has_name)
        {
            cin >> name;
            getline(cin, line);
        }
    }

    // Initialize centroids by sampling K distinct points
    vector<double> centroids(size_t(K) * D);
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);
    mt19937_64 rng(714);
    shuffle(indices.begin(), indices.end(), rng);
    for (int k = 0; k < K; ++k)
    {
        int idx = indices[k];
        for (int d = 0; d < D; ++d)
        {
            centroids[size_t(k) * D + d] = points[size_t(idx) * D + d];
        }
    }

    vector<int> assign(N, -1);

    bool moved = true;
    int iter = 0;
    double total_iter_ms = 0.0, max_iter_ms = 0.0;

    auto t_start = chrono::high_resolution_clock::now();

    while (moved && iter < max_iter)
    {
        ++iter;
        auto it_start = chrono::high_resolution_clock::now();

        // <-- here’s the fix: fully qualify std::atomic -->
        std::atomic<bool> moved_flag(false);

        // thread-local accumulators
        combinable<vector<double>> local_sums([&]
                                              { return vector<double>(size_t(K) * D, 0.0); });
        combinable<vector<int>> local_counts([&]
                                             { return vector<int>(K, 0); });

        parallel_for(
            blocked_range<size_t>(0, N),
            [&](auto const &range)
            {
                auto &ls = local_sums.local();
                auto &lc = local_counts.local();
                for (size_t i = range.begin(); i < range.end(); ++i)
                {
                    double best_dist = numeric_limits<double>::infinity();
                    int best_k = 0;
                    // find nearest centroid
                    for (int k = 0; k < K; ++k)
                    {
                        __m256d sumv = _mm256_setzero_pd();
                        int d = 0;
                        for (; d + 3 < D; d += 4)
                        {
                            __m256d x = _mm256_loadu_pd(&points[i * D + d]);
                            __m256d c = _mm256_loadu_pd(&centroids[k * D + d]);
                            __m256d diff = _mm256_sub_pd(x, c);
                            sumv = _mm256_fmadd_pd(diff, diff, sumv);
                        }
                        double dist2 = hsum256_pd(sumv);
                        for (; d < D; ++d)
                        {
                            double diff = points[i * D + d] - centroids[k * D + d];
                            dist2 += diff * diff;
                        }
                        if (dist2 < best_dist)
                        {
                            best_dist = dist2;
                            best_k = k;
                        }
                    }
                    if (assign[i] != best_k)
                    {
                        moved_flag.store(true, memory_order_relaxed);
                        assign[i] = best_k;
                    }
                    // accumulate
                    int base = best_k * D;
                    for (int d = 0; d < D; ++d)
                    {
                        ls[size_t(base + d)] += points[size_t(i * D + d)];
                    }
                    lc[size_t(best_k)] += 1;
                }
            },
            auto_partitioner());

        moved = moved_flag.load(memory_order_relaxed);

        // merge accumulators
        vector<double> global_sums(size_t(K) * D, 0.0);
        local_sums.combine_each([&](vector<double> const &ls)
                                {
            for (size_t i = 0; i < ls.size(); ++i)
                global_sums[i] += ls[i]; });
        vector<int> global_counts(K, 0);
        local_counts.combine_each([&](vector<int> const &lc)
                                  {
            for (int k = 0; k < K; ++k)
                global_counts[k] += lc[k]; });

        // recompute centroids
        for (int k = 0; k < K; ++k)
        {
            int cnt = global_counts[k];
            if (cnt > 0)
            {
                int base = k * D;
                for (int d = 0; d < D; ++d)
                    centroids[base + d] = global_sums[base + d] / cnt;
            }
        }

        // per-iteration timing
        auto it_end = chrono::high_resolution_clock::now();
        double iter_ms = chrono::duration<double, milli>(it_end - it_start).count();
        total_iter_ms += iter_ms;
        if (iter_ms > max_iter_ms)
            max_iter_ms = iter_ms;
    }

    auto t_end = chrono::high_resolution_clock::now();
    double elapsed_ms = chrono::duration<double, milli>(t_end - t_start).count();
    double avg_iter_ms = iter > 0 ? total_iter_ms / iter : 0.0;

    for (int k = 0; k < K; ++k)
    {
        cout << "\nCluster " << (k + 1) << " centroid: ";
        for (int d = 0; d < D; ++d)
            cout << centroids[k * D + d] << (d + 1 < D ? ", " : "\n");
    }

    // output
    cout << "\nConverged in " << iter << " iterations, "
         << elapsed_ms << " ms\n";
    cout << "PARALLEL, AVERAGE TIME PER ITERATION = " << avg_iter_ms << " ms\n";
    cout << "PARALLEL, MAX TIME PER ITERATION     = " << max_iter_ms << " ms\n";

    return 0;
}
