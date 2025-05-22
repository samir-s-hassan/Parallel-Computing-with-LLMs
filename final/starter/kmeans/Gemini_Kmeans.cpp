// Implementation of the KMeans Algorithm with TBB Parallelization and Optimizations
// reference: https://github.com/marcoscastro/kmeans

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <string>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <numeric>
#include <immintrin.h> // For SIMD instructions

using namespace std;
using namespace tbb;

class Point {
private:
    int id_point;
    int id_cluster;
    vector<double> values;
    int total_values;
    string name;

public:
    Point(int id_point, vector<double>& values, string name = "") :
        id_point(id_point), values(values), total_values(values.size()), name(name), id_cluster(-1) {}

    int getID() const {
        return id_point;
    }

    void setCluster(int id_cluster) {
        this->id_cluster = id_cluster;
    }

    int getCluster() const {
        return id_cluster;
    }

    double getValue(int index) const {
        return values[index];
    }

    int getTotalValues() const {
        return total_values;
    }

    void addValue(double value) {
        values.push_back(value);
    }

    string getName() const {
        return name;
    }
};

class Cluster {
private:
    int id_cluster;
    vector<double> central_values;
    vector<int> point_indices; // Store indices of points in this cluster

public:
    Cluster(int id_cluster, const Point& point) : id_cluster(id_cluster), central_values(point.getTotalValues()) {
        for (int i = 0; i < point.getTotalValues(); ++i) {
            central_values[i] = point.getValue(i);
        }
        point_indices.push_back(point.getID());
    }

    void addPointIndex(int point_index) {
        point_indices.push_back(point_index);
    }

    bool removePointIndex(int point_index) {
        auto it = find(point_indices.begin(), point_indices.end(), point_index);
        if (it != point_indices.end()) {
            point_indices.erase(it);
            return true;
        }
        return false;
    }

    double getCentralValue(int index) const {
        return central_values[index];
    }

    void setCentralValue(int index, double value) {
        central_values[index] = value;
    }

    const vector<int>& getPointIndices() const {
        return point_indices;
    }

    int getTotalPoints() const {
        return point_indices.size();
    }

    int getID() const {
        return id_cluster;
    }

    void clearPoints() {
        point_indices.clear();
    }
};

class KMeans {
private:
    int K; // number of clusters
    int total_values;
    int max_iterations;
    vector<Cluster> clusters; // Declare without initial size
    vector<Point>& points;

    // Calculate squared Euclidean distance using SIMD if possible
    double squaredEuclideanDistance(const Point& p1, const vector<double>& center) const {
        double sum = 0.0;
        int n = total_values;

#ifdef __AVX__
        int i = 0;
        __m256d sum_vec = _mm256_setzero_pd();
        for (; i + 4 <= n; i += 4) {
            __m256d p1_vec = _mm256_loadu_pd(&p1.getValue(i));
            __m256d center_vec = _mm256_loadu_pd(&center[i]);
            __m256d diff_vec = _mm256_sub_pd(p1_vec, center_vec);
            sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(diff_vec, diff_vec));
        }
        double temp[4];
        _mm256_storeu_pd(temp, sum_vec);
        sum += temp[0] + temp[1] + temp[2] + temp[3];

        for (; i < n; ++i) {
            double diff = p1.getValue(i) - center[i];
            sum += diff * diff;
        }
#else
        for (int i = 0; i < n; ++i) {
            double diff = p1.getValue(i) - center[i];
            sum += diff * diff;
        }
#endif
        return sum;
    }

    // return ID of nearest center (uses squared Euclidean distance)
    int getIDNearestCenter(const Point& point) const {
        double min_dist_sq = 0.0;
        for (int j = 0; j < total_values; ++j) {
            double diff = point.getValue(j) - clusters[0].getCentralValue(j);
            min_dist_sq += diff * diff;
        }
        int id_cluster_center = 0;

        for (int i = 1; i < K; ++i) {
            double dist_sq = 0.0;
            for (int j = 0; j < total_values; ++j) {
                double diff = point.getValue(j) - clusters[i].getCentralValue(j);
                dist_sq += diff * diff;
            }
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                id_cluster_center = i;
            }
        }
        return id_cluster_center;
    }

public:
    KMeans(int K, int total_values, int max_iterations, vector<Point>& points) :
        K(K), total_values(total_values), max_iterations(max_iterations), points(points) {}

    void initializeClusters() {
        if (K > points.size()) {
            return;
        }

        vector<int> prohibited_indices;
        for (int i = 0; i < K; ++i) {
            while (true) {
                int index_point = rand() % points.size();
                if (find(prohibited_indices.begin(), prohibited_indices.end(), index_point) == prohibited_indices.end()) {
                    prohibited_indices.push_back(index_point);
                    points[index_point].setCluster(i);
                    clusters.emplace_back(i, points[index_point]); // Use emplace_back to construct in place
                    break;
                }
            }
        }
    }

    void run() {
        auto begin = chrono::high_resolution_clock::now();

        initializeClusters();
        auto end_phase1 = chrono::high_resolution_clock::now();

        for (int iter = 0; iter < max_iterations; ++iter) {
            bool done = true;

            // Parallel assignment of points to clusters
            parallel_for(blocked_range<size_t>(0, points.size()),
                [&](const blocked_range<size_t>& r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        int old_cluster = points[i].getCluster();
                        int nearest_center = getIDNearestCenter(points[i]);
                        if (old_cluster != nearest_center) {
                            points[i].setCluster(nearest_center);
                            done = false;
                        }
                    }
                });

            // Clear cluster assignments
            for (auto& cluster : clusters) {
                cluster.clearPoints();
            }

            // Assign points to their new clusters
            for (size_t i = 0; i < points.size(); ++i) {
                if (points[i].getCluster() != -1) {
                    clusters[points[i].getCluster()].addPointIndex(points[i].getID());
                }
            }

            // Parallel recalculation of cluster centers
            parallel_for(blocked_range<size_t>(0, K),
                [&](const blocked_range<size_t>& r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        int total_points_cluster = clusters[i].getTotalPoints();
                        if (total_points_cluster > 0) {
                            vector<double> sum(total_values, 0.0);
                            const auto& point_indices = clusters[i].getPointIndices();

                            for (int point_index : point_indices) {
                                for (int j = 0; j < total_values; ++j) {
                                    sum[j] += points[point_index].getValue(j);
                                }
                            }
                            for (int j = 0; j < total_values; ++j) {
                                clusters[i].setCentralValue(j, sum[j] / total_points_cluster);
                            }
                        }
                    }
                });

            if (done) {
                cout << "Break in iteration " << iter + 1 << "\n\n";
                break;
            }
        }
        auto end = chrono::high_resolution_clock::now();

        // Output cluster information (can be optimized if needed)
        // for (int i = 0; i < K; ++i) {
        //     cout << "Cluster " << clusters[i].getID() + 1 << endl;
        //     const auto& point_indices = clusters[i].getPointIndices();
        //     for (int point_index : point_indices) {
        //         cout << "Point " << points[point_index].getID() + 1 << ": ";
        //         for (int p = 0; p < total_values; ++p) {
        //             cout << points[point_index].getValue(p) << " ";
        //         }
        //         string point_name = points[point_index].getName();
        //         if (!point_name.empty()) {
        //             cout << "- " << point_name;
        //         }
        //         cout << endl;
        //     }
        //     cout << "Cluster values: ";
        //     for (int j = 0; j < total_values; ++j) {
        //         cout << clusters[i].getCentralValue(j) << " ";
        //     }
        //     cout << "\n\n";
        // }

        cout << "TOTAL EXECUTION TIME = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << " microseconds\n";
        cout << "TIME PHASE 1 (Initialization) = " << chrono::duration_cast<chrono::microseconds>(end_phase1 - begin).count() << " microseconds\n";
        cout << "TIME PHASE 2 (Iteration) = " << chrono::duration_cast<chrono::microseconds>(end - end_phase1).count() << " microseconds\n";
    }
};

int main(int argc, char* argv[]) {
    srand(time(NULL));

    int total_points, total_values, K, max_iterations, has_name;

    cin >> total_points >> total_values >> K >> max_iterations >> has_name;

    vector<Point> points;
    string point_name;

    for (int i = 0; i < total_points; ++i) {
        vector<double> values(total_values);
        for (int j = 0; j < total_values; ++j) {
            cin >> values[j];
        }
        if (has_name) {
            cin >> point_name;
            points.emplace_back(i, values, point_name);
        } else {
            points.emplace_back(i, values);
        }
    }

    KMeans kmeans(K, total_values, max_iterations, points);
    kmeans.run();

    return 0;
}