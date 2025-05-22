// Parallel Implementation of the KMeans Algorithm 
// Based on: https://github.com/marcoscastro/kmeans
// Parallelized using OpenMP

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <string>
#include <random>
#include <sstream>
#include <fstream>

using namespace std;

class Point
{
private:
    int id_point, id_cluster;
    vector<double> values;
    int total_values;
    string name;

public:
    Point(int id_point, vector<double>& values, string name = "")
    {
        this->id_point = id_point;
        total_values = values.size();
        this->values = values;
        this->name = name;
        id_cluster = -1;
    }

    int getID() const { return id_point; }
    void setCluster(int id_cluster) { this->id_cluster = id_cluster; }
    int getCluster() const { return id_cluster; }
    double getValue(int index) const { return values[index]; }
    int getTotalValues() const { return total_values; }
    const vector<double>& getValues() const { return values; }
    string getName() const { return name; }
};

class Cluster
{
private:
    int id_cluster;
    vector<double> central_values;
    vector<int> point_ids; // Store only point IDs instead of entire Points

public:
    Cluster(int id_cluster, const Point& point)
    {
        this->id_cluster = id_cluster;
        central_values = point.getValues();
        point_ids.push_back(point.getID());
    }

    void addPoint(int point_id) { point_ids.push_back(point_id); }

    bool removePoint(int id_point)
    {
        auto it = find(point_ids.begin(), point_ids.end(), id_point);
        if (it != point_ids.end()) {
            point_ids.erase(it);
            return true;
        }
        return false;
    }

    double getCentralValue(int index) const { return central_values[index]; }
    void setCentralValue(int index, double value) { central_values[index] = value; }
    const vector<double>& getCentralValues() const { return central_values; }
    int getPointID(int index) const { return point_ids[index]; }
    int getTotalPoints() const { return point_ids.size(); }
    int getID() const { return id_cluster; }
    void clearPoints() { point_ids.clear(); }
};

class KMeans
{
private:
    int K; // number of clusters
    int total_values, total_points, max_iterations;
    vector<Cluster> clusters;
    std::mt19937 gen; // Random number generator

    // Calculate squared Euclidean distance between a point and cluster center
    double calculateSquaredDistance(const vector<double>& point_values, const vector<double>& center)
    {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for(int i = 0; i < total_values; i++)
        {
            double diff = center[i] - point_values[i];
            sum += diff * diff;
        }
        return sum;
    }

    // Return ID of nearest center (uses euclidean distance)
    int getIDNearestCenter(const vector<double>& point_values)
    {
        double min_dist = calculateSquaredDistance(point_values, clusters[0].getCentralValues());
        int id_cluster_center = 0;

        for(int i = 1; i < K; i++)
        {
            double dist = calculateSquaredDistance(point_values, clusters[i].getCentralValues());
            if(dist < min_dist)
            {
                min_dist = dist;
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
        this->gen = std::mt19937(714); // Set the specified seed for reproducibility
    }

    void run(vector<Point>& points)
    {
        auto begin = chrono::high_resolution_clock::now();

        if(K > total_points) {
            cout << "Error: K cannot be greater than total points" << endl;
            return;
        }

        // Initialize cluster centers
        vector<int> prohibited_indexes;
        std::uniform_int_distribution<> distrib(0, total_points - 1);

        // Choose K distinct values for the centers of the clusters
        for(int i = 0; i < K; i++)
        {
            while(true)
            {
                int index_point = distrib(gen);

                if(find(prohibited_indexes.begin(), prohibited_indexes.end(), index_point) == prohibited_indexes.end())
                {
                    prohibited_indexes.push_back(index_point);
                    points[index_point].setCluster(i);
                    Cluster cluster(i, points[index_point]);
                    clusters.push_back(cluster);
                    break;
                }
            }
        }

        auto end_phase1 = chrono::high_resolution_clock::now();
        cout << "Initial centers selected." << endl;

        int iter = 1;
        vector<int> assignments(total_points, -1); // Store cluster assignments

        // Store point values separately for efficient memory access
        vector<vector<double>> point_values(total_points);
        for (int i = 0; i < total_points; i++) {
            point_values[i] = points[i].getValues();
        }

        while(true)
        {
            cout << "Starting iteration " << iter << endl;
            bool done = true;

            // Reset for computing new centers
            vector<vector<double>> new_centers(K, vector<double>(total_values, 0.0));
            vector<int> cluster_counts(K, 0);

            // Associates each point to the nearest center in parallel
            #pragma omp parallel
            {
                vector<vector<double>> local_centers(K, vector<double>(total_values, 0.0));
                vector<int> local_counts(K, 0);

                #pragma omp for reduction(&&:done) schedule(dynamic, 256)
                for(int i = 0; i < total_points; i++)
                {
                    int id_old_cluster = assignments[i];
                    int id_nearest_center = getIDNearestCenter(point_values[i]);

                    if(id_old_cluster != id_nearest_center)
                    {
                        assignments[i] = id_nearest_center;
                        points[i].setCluster(id_nearest_center);
                        done = false;
                    }
                    
                    // Add to local sums
                    local_counts[id_nearest_center]++;
                    for(int j = 0; j < total_values; j++) {
                        local_centers[id_nearest_center][j] += point_values[i][j];
                    }
                }

                // Combine results with critical section
                #pragma omp critical
                {
                    for(int i = 0; i < K; i++) {
                        cluster_counts[i] += local_counts[i];
                        for(int j = 0; j < total_values; j++) {
                            new_centers[i][j] += local_centers[i][j];
                        }
                    }
                }
            }

            // Update cluster centers
            for(int i = 0; i < K; i++) {
                if(cluster_counts[i] > 0) {
                    for(int j = 0; j < total_values; j++) {
                        clusters[i].setCentralValue(j, new_centers[i][j] / cluster_counts[i]);
                    }
                }
            }

            cout << "Iteration " << iter << " completed." << endl;

            // Check termination criteria
            if(done || iter >= max_iterations)
            {
                cout << "Break in iteration " << iter << "\n\n";
                break;
            }

            iter++;
        }
        
        auto end = chrono::high_resolution_clock::now();

        // Rebuild clusters based on final assignments
        for (int i = 0; i < K; i++) {
            clusters[i].clearPoints();
        }
        
        for (int i = 0; i < total_points; i++) {
            int cluster_id = assignments[i];
            if (cluster_id >= 0 && cluster_id < K) {
                clusters[cluster_id].addPoint(i);
            }
        }

        // Shows elements of clusters (limited to prevent overwhelming output)
        for(int i = 0; i < K; i++)
        {
            int total_points_cluster = clusters[i].getTotalPoints();

            cout << "Cluster " << clusters[i].getID() + 1 << " has " << total_points_cluster << " points" << endl;
            
            // Only show first few points for each cluster to avoid excessive output
            int points_to_show = min(5, total_points_cluster);
            
            for(int j = 0; j < points_to_show; j++)
            {
                int point_id = clusters[i].getPointID(j);
                cout << "Point " << point_id + 1 << ": ";
                for(int p = 0; p < min(5, total_values); p++) // Show only first 5 values
                    cout << points[point_id].getValue(p) << " ";
                
                if (total_values > 5)
                    cout << "...";

                string point_name = points[point_id].getName();
                if(point_name != "")
                    cout << "- " << point_name;

                cout << endl;
            }
            
            if(total_points_cluster > points_to_show) {
                cout << "... and " << (total_points_cluster - points_to_show) << " more points" << endl;
            }

            cout << "Cluster values: ";
            for(int j = 0; j < min(5, total_values); j++)
                cout << clusters[i].getCentralValue(j) << " ";
            
            if (total_values > 5)
                cout << "...";
                
            cout << "\n\n";
        }

        // Calculate total time in microseconds (changed from milliseconds)
        auto time_us = chrono::duration_cast<chrono::microseconds>(end - begin).count();
        cout << "Total time: " << time_us << endl;

        // Output additional timing information in microseconds for analysis
        cout << "TIME PHASE 1 = " << std::chrono::duration_cast<std::chrono::microseconds>(end_phase1 - begin).count() << "\n";
        cout << "TIME PHASE 2 = " << std::chrono::duration_cast<std::chrono::microseconds>(end - end_phase1).count() << "\n";
    }
};

int main(int argc, char *argv[])
{
    int total_points, total_values, K, max_iterations, has_name;

    // Parse first line with metadata
    string first_line;
    if (!getline(cin, first_line)) {
        cout << "Error reading input" << endl;
        return 1;
    }
    
    istringstream iss(first_line);
    if (!(iss >> total_points >> total_values >> K >> max_iterations >> has_name)) {
        cout << "Error parsing first line: " << first_line << endl;
        return 1;
    }
    
    cout << "Dataset info: " << total_points << " points, " << total_values << " dimensions, " 
         << K << " clusters, " << max_iterations << " max iterations" << endl;
    
    // Pre-allocate vectors for performance
    vector<Point> points;
    points.reserve(total_points);
    
    // Parse data points
    string line;
    for(int i = 0; i < total_points; i++)
    {
        if (!getline(cin, line)) {
            cout << "Error reading point " << i << endl;
            if (i > 0) {
                // Continue with points we've read so far
                total_points = i;
                break;
            }
            return 1;
        }
        
        // Replace commas with spaces for easier parsing
        for (char& c : line) {
            if (c == ',') c = ' ';
        }
        
        istringstream line_stream(line);
        vector<double> values;
        values.reserve(total_values);
        
        double value;
        for(int j = 0; j < total_values; j++)
        {
            if(!(line_stream >> value)) {
                cout << "Error reading value at point " << i << ", dimension " << j << endl;
                if (j > 0) {
                    // Fill remaining values with 0
                    for (; j < total_values; j++) {
                        values.push_back(0.0);
                    }
                    break;
                }
                return 1;
            }
            values.push_back(value);
        }
        
        if (values.size() != total_values) {
            cout << "Warning: Point " << i << " has " << values.size() << " values, expected " << total_values << endl;
            // Ensure we have the right number of values
            values.resize(total_values, 0.0);
        }

        string point_name = "";
        if(has_name && line_stream >> point_name) {
            Point p(i, values, point_name);
            points.push_back(p);
        }
        else {
            Point p(i, values);
            points.push_back(p);
        }
        
        // Print progress every 1M points
        if (i % 1000000 == 0 && i > 0) {
            cout << "Read " << i << " points..." << endl;
        }
    }
    
    cout << "Read " << points.size() << " points. Starting K-means..." << endl;

    KMeans kmeans(K, points.size(), total_values, max_iterations);
    kmeans.run(points);

    return 0;
}