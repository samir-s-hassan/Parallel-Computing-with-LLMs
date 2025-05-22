// Implementation of the KMeans Algorithm with parallel optimization
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <numeric> 
#include <random>
#include <atomic>
#include <omp.h>
#include <cfloat>

using namespace std;

class Point
{
private:
    int id_point, id_cluster;
    vector<double> values;
    int total_values;
    string name;

public:
    Point(int id_point, vector<double> &values, string name = "")
    {
        this->id_point = id_point;
        total_values = values.size();

        for (int i = 0; i < total_values; i++)
            this->values.push_back(values[i]);

        this->name = name;
        id_cluster = -1;
    }

    int getID() const
    {
        return id_point;
    }

    void setCluster(int id_cluster)
    {
        this->id_cluster = id_cluster;
    }

    int getCluster() const
    {
        return id_cluster;
    }

    double getValue(int index) const  // Added const here
    {
        return values[index];
    }

    int getTotalValues() const  // Added const here
    {
        return total_values;
    }

    void addValue(double value)
    {
        values.push_back(value);
    }

    string getName() const  // Added const here
    {
        return name;
    }
};

class Cluster
{
private:
    int id_cluster;
    vector<double> central_values;
    vector<Point> points;

public:
    Cluster(int id_cluster, Point point)
    {
        this->id_cluster = id_cluster;

        int total_values = point.getTotalValues();

        for (int i = 0; i < total_values; i++)
            central_values.push_back(point.getValue(i));

        points.push_back(point);
    }

    void addPoint(Point point)
    {
        points.push_back(point);
    }

    bool removePoint(int id_point)
    {
        int total_points = points.size();

        for (int i = 0; i < total_points; i++)
        {
            if (points[i].getID() == id_point)
            {
                points.erase(points.begin() + i);
                return true;
            }
        }
        return false;
    }

    double getCentralValue(int index) const  // Added const here
    {
        return central_values[index];
    }

    void setCentralValue(int index, double value)
    {
        central_values[index] = value;
    }

    Point getPoint(int index) const  // Added const here
    {
        return points[index];
    }

    int getTotalPoints() const  // Added const here
    {
        return points.size();
    }

    int getID() const  // Added const here
    {
        return id_cluster;
    }
};

class KMeans
{
private:
    int K; // number of clusters
    int total_values, total_points, max_iterations;
    vector<Cluster> clusters;

    // Thread-local storage for cluster updates
    struct ClusterUpdate {
        vector<vector<double>> sums;
        vector<int> counts;

        ClusterUpdate(int k, int dims) {
            sums.resize(k, vector<double>(dims, 0.0));
            counts.resize(k, 0);
        }

        void reset(int k, int dims) {
            for (int i = 0; i < k; i++) {
                counts[i] = 0;
                fill(sums[i].begin(), sums[i].end(), 0.0);
            }
        }
    };

    // Optimized method to find nearest center using squared distances
    int getIDNearestCenter(const Point& point)
    {
        double min_dist_squared = DBL_MAX;
        int id_cluster_center = 0;

        // Find cluster with minimum distance
        for (int i = 0; i < K; i++)
        {
            double dist_squared = 0.0;

            for (int j = 0; j < total_values; j++)
            {
                double diff = clusters[i].getCentralValue(j) - point.getValue(j);
                dist_squared += diff * diff;
            }

            if (dist_squared < min_dist_squared)
            {
                min_dist_squared = dist_squared;
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
    }

    void run(vector<Point> &points)
    {
        long long max_time_per_iteration = 0;
        auto begin = chrono::high_resolution_clock::now();

        if (K > total_points)
            return;

        // Initialize clusters with random points (same as serial version)
        vector<int> indices(total_points);
        iota(indices.begin(), indices.end(), 0);
        mt19937_64 rng(714); // same seed for reproducibility
        shuffle(indices.begin(), indices.end(), rng);

        for (int i = 0; i < K; i++)
        {
            int index_point = indices[i];
            points[index_point].setCluster(i);
            Cluster cluster(i, points[index_point]);
            clusters.push_back(cluster);
        }
        auto end_phase1 = chrono::high_resolution_clock::now();

        // Determine optimal number of threads
        int num_threads = omp_get_max_threads();
        cout << "Using " << num_threads << " threads\n";

        int iter = 1;
        vector<ClusterUpdate> thread_local_updates(num_threads, ClusterUpdate(K, total_values));

        // Main iteration loop
        while (true)
        {
            auto iter_start = chrono::high_resolution_clock::now();
            atomic<bool> done{true};

            // Reset thread-local storage
            #pragma omp parallel for
            for (int t = 0; t < num_threads; t++) {
                thread_local_updates[t].reset(K, total_values);
            }

            // Phase 1: Point assignment and local accumulation (parallel)
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                ClusterUpdate& local_update = thread_local_updates[tid];

                #pragma omp for
                for (int i = 0; i < total_points; i++)
                {
                    int id_old_cluster = points[i].getCluster();
                    int id_nearest_center = getIDNearestCenter(points[i]);

                    if (id_old_cluster != id_nearest_center)
                    {
                        points[i].setCluster(id_nearest_center);
                        done.store(false, std::memory_order_relaxed);
                    }

                    // Accumulate data for centroid recalculation
                    int cluster_id = points[i].getCluster();
                    local_update.counts[cluster_id]++;
                    
                    for (int j = 0; j < total_values; j++) {
                        local_update.sums[cluster_id][j] += points[i].getValue(j);
                    }
                }
            }

            // Phase 2: Merge thread-local updates and recalculate centroids
            vector<vector<double>> global_sums(K, vector<double>(total_values, 0.0));
            vector<int> global_counts(K, 0);

            // Reduction step: aggregate thread-local data
            for (int t = 0; t < num_threads; t++) {
                for (int i = 0; i < K; i++) {
                    global_counts[i] += thread_local_updates[t].counts[i];
                    
                    for (int j = 0; j < total_values; j++) {
                        global_sums[i][j] += thread_local_updates[t].sums[i][j];
                    }
                }
            }

            // Update centroids based on aggregated data
            for (int i = 0; i < K; i++) {
                if (global_counts[i] > 0) {
                    for (int j = 0; j < total_values; j++) {
                        clusters[i].setCentralValue(j, global_sums[i][j] / global_counts[i]);
                    }
                }
            }

            auto iter_end = chrono::high_resolution_clock::now();
            long long iter_duration = chrono::duration_cast<chrono::milliseconds>(iter_end - iter_start).count();
            if (iter_duration > max_time_per_iteration)
                max_time_per_iteration = iter_duration;

            if (done.load() || iter >= max_iterations)
            {
                cout << "Break in iteration " << iter << "\n\n";
                break;
            }

            iter++;
        }
        auto end = chrono::high_resolution_clock::now();

        // Update clusters with final point assignments
        // Clear existing points in clusters first
        for (int i = 0; i < K; i++) {
            while (clusters[i].getTotalPoints() > 0) {
                clusters[i].removePoint(clusters[i].getPoint(0).getID());
            }
        }
        
        // Add points to their final clusters
        for (int i = 0; i < total_points; i++) {
            int cluster_id = points[i].getCluster();
            if (cluster_id >= 0) {
                clusters[cluster_id].addPoint(points[i]);
            }
        }

        // Output cluster information
        // for (int i = 0; i < K; i++)
        // {
        //     int total_points_cluster = clusters[i].getTotalPoints();

        //     cout << "Cluster " << clusters[i].getID() + 1 << endl;
        //     cout << "Points in cluster: " << total_points_cluster << endl;
            
        //     cout << "Cluster centroid: ";
        //     for (int j = 0; j < total_values; j++)
        //         cout << clusters[i].getCentralValue(j) << " ";

        //     cout << "\n\n";
        // }

        // Calculate and display performance metrics
        double total_time_ms = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
        double phase1_time_ms = chrono::duration_cast<chrono::milliseconds>(end_phase1 - begin).count();
        double phase2_time_ms = chrono::duration_cast<chrono::milliseconds>(end - end_phase1).count();
        
        cout << "PARALLEL, TOTAL EXECUTION TIME = " << total_time_ms << " ms\n";
        cout << "PARALLEL, PHASE 1 TIME = " << phase1_time_ms << " ms\n";
        cout << "PARALLEL, PHASE 2 TIME = " << phase2_time_ms << " ms\n";
        cout << "Total Iterations run: " << iter << "\n";

        if (iter > 1) {
            double avg_time_per_iteration = phase2_time_ms / iter;
            cout << "PARALLEL, AVERAGE TIME PER ITERATION = " << avg_time_per_iteration << " ms\n";
            cout << "PARALLEL, MAX TIME PER ITERATION = " << max_time_per_iteration << " ms\n";
        }
    }
};

int main(int argc, char *argv[])
{
    // Process command line arguments for controlling OpenMP behavior
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            int threads = atoi(argv[i+1]);
            if (threads > 0) {
                omp_set_num_threads(threads);
                // cout << "Setting thread count to " << threads << endl;
            }
            i++;
        }
    }

    int total_points, total_values, K, max_iterations, has_name;

    cin >> total_points >> total_values >> K >> max_iterations >> has_name;

    vector<Point> points;
    string point_name;

    for (int i = 0; i < total_points; i++)
    {
        vector<double> values;
        string line;
        getline(cin >> ws, line); // read the full line, skipping leading whitespace

        stringstream ss(line);
        string item;

        while (getline(ss, item, ','))
        {
            values.push_back(stod(item));
        }

        if (has_name)
        {
            cin >> point_name;
            points.emplace_back(i, values, point_name);
        }
        else
        {
            points.emplace_back(i, values);
        }
    }
    
    KMeans kmeans(K, total_points, total_values, max_iterations);
    kmeans.run(points);

    return 0;
}