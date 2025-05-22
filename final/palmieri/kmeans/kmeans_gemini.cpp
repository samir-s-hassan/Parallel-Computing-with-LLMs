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
#include <omp.h> // PARALLEL: Include OpenMP header
#include <stdexcept> // For exceptions like out_of_range
#include <limits>   // For numeric_limits

// Compile with (using the script or manually):
// g++ -std=c++17 -O3 -fopenmp -march=native -mavx2 src/fpk_gemini.cpp -o fpk_gemini

// Run with (script handles this):
// export OMP_NUM_THREADS=N
// cat datasets/your_dataset.txt | ./fpk_gemini

using namespace std;

// --- Point Class (Mostly unchanged, added const correctness) ---
class Point
{
private:
    int id_point, id_cluster;
    vector<double> values;
    int total_values;
    string name;

public:
    // Constructor
    Point(int id_point, const vector<double> &values, string name = "") :
        id_point(id_point),
        id_cluster(-1), // Initialize cluster ID to -1 (unassigned)
        values(values), // Use initializer list for efficiency
        total_values(values.size()),
        name(name)
     {}

    // Default constructor needed for vector resizing etc.
    Point() : id_point(-1), id_cluster(-1), total_values(0) {}

    // Copy constructor (using default is fine here)
    Point(const Point& other) = default;

    // Move constructor (using default is fine here)
    Point(Point&& other) noexcept = default;

    // Copy assignment operator (using default is fine here)
    Point& operator=(const Point& other) = default;

     // Move assignment operator (using default is fine here)
    Point& operator=(Point&& other) noexcept = default;

    // --- Getters and Setters ---
    int getID() const { return id_point; }
    void setCluster(int new_id_cluster) { this->id_cluster = new_id_cluster; }
    int getCluster() const { return id_cluster; }

    // Get a single value by dimension index
    double getValue(int index) const {
        // Basic bounds check for safety
        if(index < 0 || index >= total_values) {
            // Consider throwing an exception for better error handling in production
            // throw std::out_of_range("Point::getValue index out of bounds");
            return 0.0; // Return default value or handle error appropriately
        }
        return values[index];
    }
    // Get the total number of dimensions/values
    int getTotalValues() const { return total_values; }
    // Get a const reference to the underlying values vector (efficient for reading)
    const vector<double>& getValues() const { return values; }
    // Get the point's name (if any)
    string getName() const { return name; }
};


// --- Cluster Class (Mostly unchanged, added const correctness, clearPoints) ---
class Cluster
{
private:
    int id_cluster; // ID (often the index in the clusters vector)
    vector<double> central_values; // Centroid coordinates
    vector<Point> points; // Points currently assigned to this cluster
                          // Note: Storing full Point objects can be memory intensive
                          // for large datasets. Consider storing indices or pointers.
    int total_values; // Number of dimensions (cached for efficiency)

public:
    // Constructor: Initializes a cluster with an ID and its initial centroid point
    Cluster(int id_cluster, const Point& initial_centroid_point) :
        id_cluster(id_cluster),
        central_values(initial_centroid_point.getValues()), // Copy centroid values from the point
        total_values(initial_centroid_point.getTotalValues())
     {
        // The 'points' vector starts empty. It will be populated during
        // the first iteration's update phase (or subsequent ones).
     }

    // --- Methods ---
    // Remove all points from the cluster (used before rebuilding assignments)
    void clearPoints() { points.clear(); }
    // Add a point (by const reference) to this cluster's list
    void addPoint(const Point& point) {
        points.push_back(point); // Adds a copy of the point
    }

    // Get a specific coordinate of the centroid
    double getCentralValue(int index) const {
         if(index < 0 || index >= total_values) {
             // throw std::out_of_range("Cluster::getCentralValue index out of bounds");
             return 0.0;
         }
         return central_values[index];
    }
    // Set a specific coordinate of the centroid
    void setCentralValue(int index, double value) {
         if(index < 0 || index >= total_values) {
             // throw std::out_of_range("Cluster::setCentralValue index out of bounds");
             return;
         }
        central_values[index] = value;
    }

    // Get a const reference to a point within the cluster by its index in the 'points' vector
    const Point& getPoint(int index) const {
        if (index < 0 || index >= points.size()) {
            throw std::out_of_range("Cluster::getPoint index out of bounds");
        }
        return points[index];
    }
    // Get the number of points currently assigned to this cluster
    int getTotalPoints() const { return points.size(); }
    // Get the number of dimensions for this cluster's centroid
    int getTotalValues() const { return total_values; }
    // Get the cluster's ID
    int getID() const { return id_cluster; }
};


// --- KMeans Class (Contains the core algorithm logic) ---
class KMeans
{
private:
    int K; // Target number of clusters
    int total_values; // Number of dimensions per point
    int total_points; // Total number of points in the dataset
    int max_iterations; // Maximum iterations to perform if convergence is not met
    vector<Cluster> clusters; // Vector holding all the cluster objects

    // Calculates the squared Euclidean distance between a point and a cluster's centroid
    // Avoids sqrt for faster comparison. Returns the index of the nearest cluster.
    // Marked const as it doesn't modify the KMeans object state.
    int getIDNearestCenter(const Point& point) const
    {
        double min_dist_sq = numeric_limits<double>::max(); // Initialize with max possible value
        int id_cluster_center = 0; // Default to the first cluster (index 0)

        // Handle case where clusters might not be initialized (shouldn't happen in run)
        if (clusters.empty()) {
             return -1; // Indicate error or uninitialized state
        }

        // Iterate through all existing clusters
        for (int i = 0; i < clusters.size(); ++i) // Use clusters.size() for safety
        {
            double current_dist_sq = 0.0;
            const Cluster& cluster = clusters[i]; // Use const reference for efficiency

            // Calculate squared Euclidean distance using SIMD hint
            // The reduction clause sums the partial sums from different SIMD lanes/threads
            #pragma omp simd reduction(+:current_dist_sq)
            for (int j = 0; j < total_values; ++j)
            {
                 // Difference in the j-th dimension
                 double diff = cluster.getCentralValue(j) - point.getValue(j);
                 // Add the square of the difference to the sum
                 current_dist_sq += diff * diff;
            }

            // If this cluster is closer than the current minimum, update the minimum
            if (current_dist_sq < min_dist_sq)
            {
                min_dist_sq = current_dist_sq;
                id_cluster_center = i; // Store the index of this closer cluster
            }
        }
        return id_cluster_center; // Return the index of the nearest cluster
    }


public:
    // Constructor for KMeans class
    KMeans(int K, int total_points, int total_values, int max_iterations) :
        K(K),
        total_points(total_points),
        total_values(total_values),
        max_iterations(max_iterations)
    {}

    // The main method to run the K-Means algorithm
    void run(vector<Point> &points) // Pass points vector by reference (allows modification)
    {
        // --- Input Validation ---
        if (K <= 0 || K > total_points) {
            cerr << "Error: Invalid K value (" << K << "). Must be > 0 and <= total_points (" << total_points << ")." << endl;
            // Consider throwing an exception or returning an error code
            return;
        }
        if (max_iterations < 0) {
             cerr << "Error: max_iterations cannot be negative." << endl;
             return;
        }

        // --- Timing Initialization ---
        auto begin_overall = chrono::high_resolution_clock::now(); // Start total time measurement

        // --- Initial Cluster Initialization (Sequential is generally fast enough) ---
        // cout << "ℹ️ Initializing " << K << " clusters..." << endl;
        clusters.clear(); // Ensure clusters vector is empty
        clusters.reserve(K); // Pre-allocate memory for K clusters
        vector<int> indices(total_points); // Vector to hold indices 0 to total_points-1
        iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...

        // Use a fixed seed for the random number generator for reproducible initial centroids
        mt19937_64 rng(714);
        shuffle(indices.begin(), indices.end(), rng); // Randomly shuffle the indices

        // Create the initial K clusters using the first K points from the shuffled list
        for (int i = 0; i < K; ++i) {
            // Use emplace_back for potentially better efficiency constructing clusters in place
            clusters.emplace_back(i, points[indices[i]]); // Cluster ID is its index 'i'
        }
        auto end_phase1 = chrono::high_resolution_clock::now(); // End initialization time measurement


        // --- Parallel Iteration Setup ---
        // Vector to store the assigned cluster index for each point in the current iteration
        vector<int> new_cluster_assignments(total_points);
        bool converged = false; // Flag to track if the algorithm has converged
        int iter = 0; // Iteration counter
        double total_iteration_time_ms = 0; // Accumulator for iteration times
        double max_time_per_iteration_ms = 0.0; // Track the longest iteration time

        // cout << "ℹ️ Starting K-Means iterations (max: " << max_iterations << ")..." << endl;
        // Print the number of threads being used by OpenMP
        #pragma omp parallel // Start a parallel region once
        {
            #pragma omp master // Only one thread executes this block
            {
                //  cout << "ℹ️ Using " << omp_get_num_threads() << " OpenMP threads." << endl;
            }
        } // End of single parallel region

        // --- Main K-Means Iteration Loop ---
        while (iter < max_iterations && !converged)
        {
            iter++; // Increment iteration counter
            auto iter_start = chrono::high_resolution_clock::now(); // Start timing this iteration
            converged = true; // Assume convergence for this iteration until a point changes cluster

            // === PARALLEL Assignment Step ===
            // Distribute the loop iterations across OpenMP threads
            // 'shared' variables are accessible by all threads
            // 'converged' needs careful handling (atomic write or reduction) if modified in parallel
            // 'schedule(static)' divides iterations statically among threads (good if work per point is similar)
            #pragma omp parallel for shared(points, clusters, new_cluster_assignments, converged) schedule(static)
            for (int i = 0; i < total_points; ++i) // Loop through all points
            {
                int id_old_cluster = points[i].getCluster(); // Get the point's current cluster index
                int id_nearest_center = getIDNearestCenter(points[i]); // Find the index of the nearest cluster centroid

                // If the point's nearest cluster is different from its current one
                if (id_old_cluster != id_nearest_center) {
                    new_cluster_assignments[i] = id_nearest_center; // Assign the point to the new cluster index
                    // If any point changes cluster, the algorithm hasn't converged yet.
                    // Use atomic write to safely set 'converged' to false from multiple threads.
                     #pragma omp atomic write
                     converged = false;
                } else {
                    // If the point stays in the same cluster, keep its assignment
                    new_cluster_assignments[i] = id_old_cluster;
                }
            } // End of parallel assignment loop

            // --- Check for Convergence ---
            // If 'converged' is still true after the parallel loop, no points changed cluster.
            if (converged) {
                 auto iter_end = chrono::high_resolution_clock::now(); // End timing *before* breaking
                 double iter_duration_ms = chrono::duration_cast<chrono::microseconds>(iter_end - iter_start).count() / 1000.0;
                 total_iteration_time_ms += iter_duration_ms;
                 // Update max time if this last iteration was the longest
                 if (iter_duration_ms > max_time_per_iteration_ms) { max_time_per_iteration_ms = iter_duration_ms; }
                 cout << "✅ Converged in iteration " << iter << "." << endl;
                 break; // Exit the main loop
            }


            // === Sequential Update Phase (Applying assignments) ===
            // This part is sequential to avoid race conditions when modifying the
            // 'points' vector within each 'Cluster' object simultaneously.
            // 1. Clear the points list in all clusters
            for (int i = 0; i < K; ++i) {
                clusters[i].clearPoints();
            }
            // 2. Update point objects and add points (copies) to their newly assigned clusters
            for (int i = 0; i < total_points; ++i) {
                int new_cluster_idx = new_cluster_assignments[i];
                points[i].setCluster(new_cluster_idx); // Update the Point object itself
                 // Add the point to the list of the corresponding cluster
                 if (new_cluster_idx >= 0 && new_cluster_idx < K) { // Safety check
                    clusters[new_cluster_idx].addPoint(points[i]);
                 }
            }


            // === PARALLEL Update Step (Recalculating Centroids) ===
            // This loop can be parallelized because each thread works on a different cluster,
            // modifying distinct 'central_values' vectors.
             #pragma omp parallel for shared(clusters) schedule(static)
             for (int i = 0; i < K; ++i) // Loop over cluster indices 0 to K-1
             {
                 int total_points_cluster = clusters[i].getTotalPoints(); // Points in this specific cluster
                 int cluster_total_values = clusters[i].getTotalValues(); // Dimensions for this cluster

                 // Only recalculate if the cluster has points
                 if (total_points_cluster > 0) {
                     // Vector to store the sum of coordinates for each dimension
                     vector<double> sums(cluster_total_values, 0.0);

                     // Sum the coordinates of all points in this cluster
                     // This inner loop could also be parallelized/vectorized if needed,
                     // but often the outer loop parallelization is sufficient.
                     for (int p = 0; p < total_points_cluster; ++p) {
                          const Point& current_point = clusters[i].getPoint(p); // Get point by const reference
                          // Add each dimension's value to the sums vector
                          for (int j = 0; j < cluster_total_values; ++j) {
                              sums[j] += current_point.getValue(j);
                         }
                     }

                     // Calculate the new centroid by dividing sums by the number of points
                     for (int j = 0; j < cluster_total_values; ++j) {
                         clusters[i].setCentralValue(j, sums[j] / total_points_cluster);
                     }
                 }
                 // Else: Cluster is empty. Centroid remains unchanged.
                 // More advanced implementations might handle empty clusters (e.g., re-initialize).
             } // End of parallel centroid update loop

            // --- Iteration Timing ---
            auto iter_end = chrono::high_resolution_clock::now(); // End timing this iteration
            double iter_duration_ms = chrono::duration_cast<chrono::microseconds>(iter_end - iter_start).count() / 1000.0;
            total_iteration_time_ms += iter_duration_ms; // Accumulate total iteration time
            // Update the maximum iteration time if the current iteration was longer
            if (iter_duration_ms > max_time_per_iteration_ms) {
                 max_time_per_iteration_ms = iter_duration_ms;
            }

            // // Print progress periodically
            //  if (iter % 10 == 0) {
            //      cout << "   Iteration " << iter << " completed (" << iter_duration_ms << " ms)" << endl;
            //  }

        } // End of main K-Means while loop

        // --- Post-Loop Timing and Information ---
        auto end_overall = chrono::high_resolution_clock::now(); // End total time measurement

        // Check if max iterations were reached without convergence
        if(iter >= max_iterations && !converged) {
             cout << "⚠️ Reached max iterations (" << max_iterations << ") without converging." << endl;
        }

        // // --- Output Final Clustering Summary ---
        // cout << "\n--- Final Clustering Summary ---" << endl;
        // for (int i = 0; i < K; ++i) {
        //     cout << "Cluster " << clusters[i].getID() + 1 << ": " << clusters[i].getTotalPoints() << " points." << endl;
        //     // Optional: Print final centroid coordinates
        //      cout << "  Centroid: ";
        //      for (int j = 0; j < total_values; j++)
        //          cout << clusters[i].getCentralValue(j) << " ";
        //      cout << endl;
        // }


        // --- Performance Metrics Output (Matches script parsing format) ---
        cout << "\n--- Performance Metrics ---" << endl;
        long long overall_duration_us = chrono::duration_cast<chrono::microseconds>(end_overall - begin_overall).count();
        long long phase1_duration_us = chrono::duration_cast<chrono::microseconds>(end_phase1 - begin_overall).count();
        long long iterations_duration_us = chrono::duration_cast<chrono::microseconds>(end_overall - end_phase1).count();

        // Determine the actual number of iterations completed
        int actual_iterations = iter; // If converged/finished, 'iter' holds the count

        // Print total times
        cout << "PARALLEL, TOTAL EXECUTION TIME = " << overall_duration_us / 1000.0 << " ms\n";
        cout << "PARALLEL, INIT TIME (Phase 1) = " << phase1_duration_us / 1000.0 << " ms\n";
        cout << "PARALLEL, ITERATION LOOP TIME (Phase 2) = " << iterations_duration_us / 1000.0 << " ms\n";

        // Print per-iteration metrics (average and max)
        if (actual_iterations > 0)
        {
            // Calculate average using the accumulated time and actual iterations
            double avg_time_per_iteration = total_iteration_time_ms / actual_iterations;
            cout << "PARALLEL, AVERAGE TIME PER ITERATION = " << avg_time_per_iteration << " ms\n";
            // Print the maximum iteration time tracked during the loop
            cout << "PARALLEL, MAX TIME PER ITERATION = " << max_time_per_iteration_ms << " ms\n";
        } else {
             // Handle case where no iterations ran (e.g., max_iterations = 0)
             cout << "PARALLEL, No iterations run." << endl;
             cout << "PARALLEL, AVERAGE TIME PER ITERATION = 0 ms\n";
             cout << "PARALLEL, MAX TIME PER ITERATION = 0 ms\n";
        }
        // Print the total number of iterations run
        cout << "Total Iterations run: " << actual_iterations << endl;

    } // End of run() method
};


// --- Main Function (Handles input reading and initiates KMeans) ---
int main(int argc, char *argv[])
{
    // Optional: Faster C++ I/O streams (can sometimes help)
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    // --- Read Header Information ---
    int total_points_hdr, total_values_hdr, K_hdr, max_iterations_hdr, has_name_hdr;
    if (!(cin >> total_points_hdr >> total_values_hdr >> K_hdr >> max_iterations_hdr >> has_name_hdr)) {
        cerr << "Error reading header line from input." << endl; return 1;
    }
    // Basic validation of header values
    if (total_points_hdr <= 0 || total_values_hdr <= 0 || K_hdr <= 0 || max_iterations_hdr < 0) {
         cerr << "Error: Invalid parameters in header (points, values, K must be > 0; max_iterations >= 0)." << endl; return 1;
    }
    if (has_name_hdr != 0 && has_name_hdr != 1) {
         cerr << "Error: has_name flag in header must be 0 or 1." << endl; return 1;
    }
    // cout << "ℹ️ Header: Points=" << total_points_hdr << ", Values/Dim=" << total_values_hdr
    //      << ", K=" << K_hdr << ", MaxIter=" << max_iterations_hdr << ", HasName=" << has_name_hdr << endl;


    // --- Read Point Data ---
    vector<Point> points;
    points.reserve(total_points_hdr); // Pre-allocate memory for efficiency
    string point_name = ""; // Initialize name variable

    // cout << "ℹ️ Reading " << total_points_hdr << " points..." << endl;
    for (int i = 0; i < total_points_hdr; ++i) {
        vector<double> values;
        values.reserve(total_values_hdr); // Pre-allocate for dimensions
        string line;

        // Read the entire line containing coordinate values
        if (!getline(cin >> ws, line)) { // 'ws' consumes leading whitespace
             cerr << "Error reading data line for point index " << i << endl; return 1;
        }

        stringstream ss(line); // Use stringstream to parse the comma-separated values
        string item;
        int value_count = 0;
        // Parse comma-separated values
        while (getline(ss, item, ',')) {
            try {
                values.push_back(stod(item)); // Convert string item to double
                value_count++;
            } catch (const std::invalid_argument& ia) {
                 cerr << "Error: Invalid numeric value '" << item << "' for point index " << i << " on line: " << line << endl; return 1;
            } catch (const std::out_of_range& oor) {
                 cerr << "Error: Numeric value '" << item << "' out of range for point index " << i << " on line: " << line << endl; return 1;
            }
        }

        // Validate that the correct number of values were read for this point
        if(value_count != total_values_hdr) {
            cerr << "Error: Expected " << total_values_hdr << " values, found " << value_count << " for point index " << i << " on line: " << line << endl;
            return 1;
        }

        // Read the point name if the header indicates it exists
        if (has_name_hdr) {
             if (!(cin >> point_name)) { // Read the name (assumed to be on the next line)
                 cerr << "Error reading name for point index " << i << endl; return 1;
             }
             points.emplace_back(i, values, point_name); // Create Point with name
        } else {
            points.emplace_back(i, values); // Create Point without name
        }
    }

    // Final check if the correct total number of points were read
    if (points.size() != total_points_hdr) {
         cerr << "Error: Read " << points.size() << " points, but expected " << total_points_hdr << " based on header." << endl;
         return 1;
    }
    // cout << "✅ Dataset loaded successfully." << endl;


    // --- Run K-Means ---
    // Create KMeans object using header values
    KMeans kmeans(K_hdr, total_points_hdr, total_values_hdr, max_iterations_hdr);
    // Execute the algorithm
    kmeans.run(points); // Pass the vector of points

    return 0; // Indicate successful execution
}
