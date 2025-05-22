// chat_kmeans.cpp
// A parallelized k-means clustering program using Intel TBB.
// Compile with: g++ -pthread -std=c++17 -O3 -march=native chat_kmeans.cpp -ltbb -o chat_kmeans

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <atomic>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>

using namespace std;

class Point {
private:
    int id_point;
    int id_cluster;
    vector<double> values;
    int total_values;
    string name;
public:
    Point(int id_point, const vector<double>& values, const string &name = "") {
        this->id_point = id_point;
        total_values = values.size();
        for (int i = 0; i < total_values; i++)
            this->values.push_back(values[i]);
        this->name = name;
        id_cluster = -1;
    }

    int getID() const { return id_point; }
    void setCluster(int id_cluster) { this->id_cluster = id_cluster; }
    int getCluster() const { return id_cluster; }

    // Marked as const
    double getValue(int index) const { return values[index]; }

    // Marked as const
    int getTotalValues() const { return total_values; }

    void addValue(double value) { values.push_back(value); }

    // Marked as const
    string getName() const { return name; }
};

class Cluster {
private:
    int id_cluster;
    vector<double> central_values;
    vector<Point> points;
public:
    Cluster(int id_cluster, const Point &point) {
        this->id_cluster = id_cluster;
        int total_values = point.getTotalValues();
        for (int i = 0; i < total_values; i++) {
            central_values.push_back(point.getValue(i));
        }
        points.push_back(point);
    }
    void addPoint(const Point &point) {
        points.push_back(point);
    }
    bool removePoint(int id_point) {
        int total_points = points.size();
        for (int i = 0; i < total_points; i++) {
            if (points[i].getID() == id_point) {
                points.erase(points.begin() + i);
                return true;
            }
        }
        return false;
    }
    double getCentralValue(int index) {
        return central_values[index];
    }
    void setCentralValue(int index, double value) {
        central_values[index] = value;
    }
    Point getPoint(int index) {
        return points[index];
    }
    int getTotalPoints() {
        return points.size();
    }
    int getID() { return id_cluster; }
    // Utility: clears the points assigned to this cluster.
    void clearPoints() { points.clear(); }
};

class KMeans {
private:
    int K; // Number of clusters.
    int total_values, total_points, max_iterations;
    vector<Cluster> clusters;

    // Calculate the squared Euclidean distance.
    // (No need to take the square root since we only compare distances.)
    int getIDNearestCenter(const Point &point) {
        double min_dist = 1e12;
        int id_cluster_center = 0;
        for (int i = 0; i < K; i++) {
            double sum = 0.0;
            for (int j = 0; j < total_values; j++) {
                double diff = clusters[i].getCentralValue(j) - point.getValue(j);
                sum += diff * diff;
            }
            if (sum < min_dist) {
                min_dist = sum;
                id_cluster_center = i;
            }
        }
        return id_cluster_center;
    }
public:
    KMeans(int K, int total_points, int total_values, int max_iterations) {
        this->K = K;
        this->total_points = total_points;
        this->total_values = total_values;
        this->max_iterations = max_iterations;
    }

    void run(vector<Point> &points) {
        auto begin = chrono::high_resolution_clock::now();

        if (K > total_points)
            return;

        vector<int> prohibited_indexes;

        // Step 1: Choose K distinct initial centers.
        for (int i = 0; i < K; i++) {
            while (true) {
                int index_point = rand() % total_points;
                if (find(prohibited_indexes.begin(), prohibited_indexes.end(), index_point) == prohibited_indexes.end()) {
                    prohibited_indexes.push_back(index_point);
                    points[index_point].setCluster(i);
                    Cluster cluster(i, points[index_point]);
                    clusters.push_back(cluster);
                    break;
                }
            }
        }
        auto end_phase1 = chrono::high_resolution_clock::now();

        int iter = 1;
        while (true) {
            bool done = true;
            vector<int> new_assignment(total_points, -1);

            // Step 1: Assign nearest center in parallel
            tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); i++) {
                    new_assignment[i] = getIDNearestCenter(points[i]);
                }
            });

            // Step 2: Update cluster assignment (parallel + atomic flag for early termination)
            std::atomic<bool> change_detected(false);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); i++) {
                    if (points[i].getCluster() != new_assignment[i]) {
                        change_detected = true;
                        points[i].setCluster(new_assignment[i]);
                    }
                }
            });
            done = !change_detected.load();

            // Step 3: Use thread-local buffers for cluster assignment to avoid races
            tbb::combinable<std::vector<std::vector<Point>>> local_clusters([&]() {
            return std::vector<std::vector<Point>>(K); // One vector per cluster
            });

            tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& local = local_clusters.local();
                for (size_t i = r.begin(); i != r.end(); i++) {
                    int cid = points[i].getCluster();
                    local[cid].push_back(points[i]);
                }
            });

            // Step 4: Clear existing clusters
            for (int i = 0; i < K; i++) {
            clusters[i].clearPoints();
            }

            // Step 5: Merge all local cluster buffers
            local_clusters.combine_each([&](const std::vector<std::vector<Point>>& local) {
            for (int cid = 0; cid < K; cid++) {
                for (const auto& pt : local[cid]) {
                    clusters[cid].addPoint(pt);
                }
            }
            });

            // Parallel: Recalculate cluster centers (each cluster is independent).
            tbb::parallel_for(0, K, [&](int i) {
                for (int j = 0; j < total_values; j++) {
                    int total_points_cluster = clusters[i].getTotalPoints();
                    if (total_points_cluster > 0) {
                        double sum = 0.0;
                        // The inner loop may be auto-vectorized.
                        for (int p = 0; p < total_points_cluster; p++) {
                            sum += clusters[i].getPoint(p).getValue(j);
                        }
                        clusters[i].setCentralValue(j, sum / total_points_cluster);
                    }
                }
            });

            if (done == true || iter >= max_iterations) {
                cout << "Break in iteration " << iter << "\n\n";
                break;
            }
            iter++;
        }
        auto end = chrono::high_resolution_clock::now();

        // // Print clustering result.
        // for (int i = 0; i < K; i++) {
        //     int total_points_cluster = clusters[i].getTotalPoints();
        //     cout << "Cluster " << clusters[i].getID() + 1 << endl;
        //     for (int j = 0; j < total_points_cluster; j++) {
        //         cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
        //         for (int p = 0; p < total_values; p++)
        //             cout << clusters[i].getPoint(j).getValue(p) << " ";
        //         string point_name = clusters[i].getPoint(j).getName();
        //         if (point_name != "")
        //             cout << "- " << point_name;
        //         cout << endl;
        //     }
        //     cout << "Cluster values: ";
        //     for (int j = 0; j < total_values; j++)
        //         cout << clusters[i].getCentralValue(j) << " ";
        //     cout << "\n\n";
        // }

        // Print execution timing information.
        cout << "TOTAL EXECUTION TIME = "
             << chrono::duration_cast<chrono::microseconds>(end - begin).count()
             << "\n";
        cout << "TIME PHASE 1 (initialization) = "
             << chrono::duration_cast<chrono::microseconds>(end_phase1 - begin).count()
             << "\n";
        cout << "TIME PHASE 2 (clustering iterations) = "
             << chrono::duration_cast<chrono::microseconds>(end - end_phase1).count()
             << "\n";
    }
};

int main(int argc, char *argv[]) {
    srand(time(NULL));

    int total_points, total_values, K, max_iterations, has_name;
    cin >> total_points >> total_values >> K >> max_iterations >> has_name;

    vector<Point> points;
    string point_name;

    for (int i = 0; i < total_points; i++) {
        vector<double> values;
        for (int j = 0; j < total_values; j++) {
            double value;
            cin >> value;
            values.push_back(value);
        }
        if (has_name) {
            cin >> point_name;
            Point p(i, values, point_name);
            points.push_back(p);
        }
        else {
            Point p(i, values);
            points.push_back(p);
        }
    }

    // --- Performance tuning hints ---
    // To control the number of threads manually, you can initialize the TBB scheduler,
    // for example using tbb::global_control in newer TBB versions.
    // Also, compile with optimization flags (e.g., -O3 -march=native) to enable SIMD auto-vectorization.

    KMeans kmeans(K, total_points, total_values, max_iterations);
    kmeans.run(points);

    return 0;
}
