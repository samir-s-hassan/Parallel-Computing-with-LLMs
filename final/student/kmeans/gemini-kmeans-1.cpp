// Parallel Implementation of the KMeans Algorithm using OpenMP
// Based on: https://github.com/marcoscastro/kmeans
// Optimized for parallel execution and outputs time in microseconds.

#include <iostream>       // For standard I/O (cin, cout, cerr)
#include <vector>         // For std::vector
#include <cmath>          // For std::sqrt
#include <cstdlib>        // For std::stod (string to double)
#include <ctime>          // Not strictly needed after removing srand(time(NULL))
#include <algorithm>      // For std::find (used in original serial, could be used in alt init)
#include <chrono>         // For timing (high_resolution_clock, duration_cast)
#include <limits>         // For std::numeric_limits
#include <random>         // For std::mt19937 and std::uniform_int_distribution
#include <omp.h>          // For OpenMP directives and functions
#include <sstream>        // For std::stringstream (parsing input lines)
#include <stdexcept>      // For exception handling in stod

using namespace std;

// --- Point Class ---
// Represents a single data point with its coordinates and assigned cluster ID.
class Point
{
private:
	int id_point;         // Unique identifier for the point
	int id_cluster;       // ID of the cluster this point belongs to (-1 if unassigned)
	vector<double> values;// Coordinates of the point (dimensions)
	int total_values;     // Number of dimensions

public:
    // Constructor: Initializes a point with its ID and coordinate values.
	Point(int id_point, vector<double>& values)
	{
		this->id_point = id_point;
		this->total_values = values.size();
		this->values = values; // Use vector assignment for efficiency
		this->id_cluster = -1; // Initially unassigned
	}

    // Returns the point's unique ID.
	int getID() const
	{
		return id_point;
	}

    // Assigns the point to a specific cluster.
	void setCluster(int id_cluster)
	{
		this->id_cluster = id_cluster;
	}

    // Returns the ID of the cluster this point currently belongs to.
	int getCluster() const
	{
		return id_cluster;
	}

    // Returns the coordinate value for a specific dimension index.
	double getValue(int index) const
	{
		// Consider adding bounds checking for robustness if necessary:
		// if (index < 0 || index >= total_values) throw std::out_of_range("Invalid index");
		return values[index];
	}

    // Returns the total number of dimensions for this point.
	int getTotalValues() const
	{
		return total_values;
	}

    // Returns a const reference to the internal coordinate vector.
    const vector<double>& getValues() const {
        return values;
    }
};


// --- Cluster Class ---
// Represents a cluster, primarily holding its centroid coordinates.
// The list of points belonging to the cluster is managed externally
// in the parallel version to avoid concurrency issues.
class Cluster
{
private:
	int id_cluster;          // Unique identifier for the cluster
	vector<double> central_values; // Coordinates of the cluster's centroid
	int total_values;      // Number of dimensions (same as points)

public:
    // Constructor: Initializes a cluster with its ID and initial centroid coordinates.
	Cluster(int id_cluster, const vector<double>& initial_centroid_values)
	{
		this->id_cluster = id_cluster;
		this->total_values = initial_centroid_values.size();
		this->central_values = initial_centroid_values; // Copy initial centroid
	}

    // Returns the centroid coordinate value for a specific dimension index.
	double getCentralValue(int index) const
	{
		return central_values[index];
	}

    // Sets the centroid coordinate value for a specific dimension index.
	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

    // Returns a const reference to the centroid coordinate vector.
    const vector<double>& getCentralValues() const {
        return central_values;
    }

    // Returns the total number of dimensions.
	int getTotalValues() const
	{
		return total_values;
	}

    // Returns the cluster's unique ID.
	int getID() const
	{
		return id_cluster;
	}
};


// --- KMeans Class ---
// Orchestrates the K-means clustering algorithm using OpenMP for parallelization.
class KMeans
{
private:
	int K;                 // Target number of clusters
	int total_values;      // Number of dimensions per point
    int total_points;      // Total number of points in the dataset
    int max_iterations;    // Maximum number of iterations allowed
	vector<Cluster> clusters; // Vector holding the K cluster objects

	// Calculates the squared Euclidean distance (faster than Euclidean for comparison)
    // and returns the ID of the nearest cluster centroid for a given point.
	int getIDNearestCenter(const Point& point) const
	{
        // Initialize minimum distance to the maximum possible double value
		double min_dist_sq = numeric_limits<double>::max();
		int id_cluster_center = 0; // Default to the first cluster

		for (int i = 0; i < K; i++) // Iterate through each cluster
		{
			double current_dist_sq = 0.0; // Squared distance to the current cluster
            const vector<double>& central_vals = clusters[i].getCentralValues();

            // Calculate squared Euclidean distance sum using SIMD for potential speedup.
            // Using (a-b)*(a-b) is often faster and more SIMD-friendly than pow(a-b, 2.0).
            #pragma omp simd reduction(+:current_dist_sq) // Request SIMD vectorization with reduction
			for (int j = 0; j < total_values; j++) // Iterate through dimensions
			{
                double diff = central_vals[j] - point.getValue(j);
				current_dist_sq += diff * diff; // Accumulate squared difference
			}
            // No need to sqrt here, comparing squared distances is equivalent and faster.

			if (current_dist_sq < min_dist_sq) // If this cluster is closer
			{
				min_dist_sq = current_dist_sq; // Update minimum squared distance
				id_cluster_center = i;         // Update the ID of the nearest cluster
			}
		}
		return id_cluster_center; // Return the ID of the closest cluster found
	}

public:
    // Constructor: Initializes K-means parameters.
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

    // Executes the parallel K-means algorithm.
	void run(vector<Point>& points) // Pass points by non-const reference (cluster IDs change)
	{
        // Basic validation
		if (K > total_points) {
            cerr << "Error: Number of clusters K (" << K
                 << ") cannot exceed total points (" << total_points << ")." << endl;
            return;
        }
        if (K <= 0) {
             cerr << "Error: Number of clusters K must be positive." << endl;
             return;
        }

        // Start timing the entire execution
        auto total_start_time = chrono::high_resolution_clock::now();

		// --- Phase 1: Initialization (Serial) ---
        // Initialize clusters by selecting K distinct random points as initial centroids.
        // Use the required fixed random seed for reproducibility.
        cout << "Initializing " << K << " clusters using random seed 714..." << endl;
        std::mt19937 gen(714); // Mersenne Twister engine seeded with 714
        std::uniform_int_distribution<> distrib(0, total_points - 1); // Distribution for point indices

		vector<int> used_point_indexes; // Keep track of points already chosen as centroids
		clusters.reserve(K); // Pre-allocate space for cluster objects
		for (int i = 0; i < K; i++)
		{
            int index_point;
            bool found_new = false;
            // Ensure we pick a unique point index
            while(!found_new) {
                index_point = distrib(gen); // Generate a random point index
                bool already_used = false;
                for(int used_idx : used_point_indexes) { // Check if this index was used before
                    if (used_idx == index_point) {
                        already_used = true;
                        break;
                    }
                }
                if (!already_used) { // If not used, mark it and exit the loop
                    used_point_indexes.push_back(index_point);
                    found_new = true;
                }
                // Handle unlikely case where we can't find K unique points (shouldn't happen if K <= total_points)
                if (used_point_indexes.size() > total_points) {
                     cerr << "Error: Could not find enough unique initial points." << endl;
                     return; // Or handle differently
                }
            }
            // Assign the chosen point to this initial cluster
			points[index_point].setCluster(i);
            // Create the cluster object using the chosen point's coordinates as the initial centroid
            clusters.emplace_back(i, points[index_point].getValues()); // Use emplace_back for efficiency
		}
        cout << "Initialization complete." << endl;

		// --- Phase 2: Iteration Loop (Parallel Assignment & Update) ---
		int iter = 1;  // Iteration counter
		bool converged = false; // Flag to check if assignments stabilized
		cout << "Starting iterations (max " << max_iterations << ")..." << endl;

		while (!converged && iter <= max_iterations)
		{
            converged = true; // Assume convergence for this iteration until proven otherwise

            // Global accumulators for centroid updates (sum of coordinates and point counts per cluster)
            // Initialized to zero for each iteration.
            vector<vector<double>> global_sums(K, vector<double>(total_values, 0.0));
            vector<int> global_counts(K, 0);

            // Start the parallel region. Variables declared inside are thread-private by default.
			#pragma omp parallel
			{
                // Thread-local accumulators (each thread calculates its contribution)
                vector<vector<double>> local_sums(K, vector<double>(total_values, 0.0));
                vector<int> local_counts(K, 0);
                bool thread_converged = true; // Thread-local convergence flag

                // Distribute the loop over all points across the available threads.
                // Schedule(static) can be efficient if points have similar computation time.
                // Consider schedule(dynamic) or schedule(guided) if computation is uneven.
                #pragma omp for schedule(static)
				for (int i = 0; i < total_points; i++)
				{
					int id_old_cluster = points[i].getCluster(); // Get current cluster
					int id_nearest_center = getIDNearestCenter(points[i]); // Find nearest cluster

					// If the point needs to change cluster
					if (id_old_cluster != id_nearest_center)
					{
						points[i].setCluster(id_nearest_center); // Assign to the new cluster
                        thread_converged = false; // Mark that a change occurred in this thread's work
					}

                    // Accumulate point's values for its (potentially new) cluster's centroid calculation
                    int current_cluster_id = points[i].getCluster();
                    // Ensure the point is assigned to a valid cluster before accumulating
                    if (current_cluster_id != -1) { // Should always be true after first iteration
                        for(int j=0; j<total_values; ++j) {
                            local_sums[current_cluster_id][j] += points[i].getValue(j);
                        }
                        local_counts[current_cluster_id]++;
                    }
				} // End of parallel for loop

                // Combine thread-local accumulators into the global ones.
                // This critical section ensures only one thread modifies the global sums/counts at a time, preventing race conditions.
                #pragma omp critical
                {
                    for(int k=0; k<K; ++k) {
                        for(int j=0; j<total_values; ++j) {
                            global_sums[k][j] += local_sums[k][j]; // Add thread's sum contribution
                        }
                        global_counts[k] += local_counts[k]; // Add thread's count contribution
                    }
                    // If any thread found a change, the overall iteration has not converged.
                    if (!thread_converged) {
                        converged = false;
                    }
                } // End of critical section

            } // End of parallel region

			// --- Update Centroids (Serial Part) ---
            // After all threads finish, update the centroids based on the globally accumulated sums and counts.
            // This part is typically fast as it only loops K times.
            if (!converged) { // Only update centroids if convergence hasn't happened
                for (int i = 0; i < K; i++) // Iterate through each cluster
                {
                    if (global_counts[i] > 0) // Check if the cluster has any points
                    {
                        for (int j = 0; j < total_values; j++) // Iterate through dimensions
                        {
                            // Calculate the new centroid coordinate (mean)
                            clusters[i].setCentralValue(j, global_sums[i][j] / global_counts[i]);
                        }
                    }
                    // Optional: Handle empty clusters. If a cluster becomes empty (global_counts[i] == 0),
                    // its centroid remains unchanged from the previous iteration. You might want to
                    // re-initialize its centroid randomly or assign a point furthest from its center.
                    // else { cerr << "Warning: Cluster " << i << " became empty in iteration " << iter << endl; }
                }
            }

            // Progress reporting (optional)
            // cout << "Iteration " << iter << " complete. Converged: " << (converged ? "Yes" : "No") << endl;

            // Check for termination conditions
            if (converged) {
                 cout << "Algorithm converged in iteration " << iter << "." << endl;
                 break; // Exit loop if converged
            } else if (iter == max_iterations) {
                 cout << "Reached maximum iterations (" << max_iterations << ") without convergence." << endl;
                 // Proceed to output results even if not fully converged
            }

			iter++; // Increment iteration counter
		} // End of while loop

        // Stop timing
        auto total_end_time = chrono::high_resolution_clock::now();
        // Calculate duration in microseconds
        auto duration_us = chrono::duration_cast<chrono::microseconds>(total_end_time - total_start_time).count();

        // --- Final Output ---
        // Print the total execution time in microseconds as required.
        cout << "Total time: " << duration_us << endl;


        // --- Optional: Print final cluster information (can be slow) ---
        /*
        cout << "\nFinal Cluster Information:\n";
		vector<int> final_counts(K, 0);
        #pragma omp parallel for
        for(int p=0; p<total_points; ++p) {
            int cluster_id = points[p].getCluster();
            if (cluster_id != -1) {
                 #pragma omp atomic update
                 final_counts[cluster_id]++;
            }
        }

		for (int i = 0; i < K; i++)
		{
			cout << "Cluster " << clusters[i].getID() + 1 << " (" << final_counts[i] << " points)" << endl;
			cout << "  Centroid: ";
			for (int j = 0; j < total_values; j++) {
				cout << clusters[i].getCentralValue(j) << " ";
            }
			cout << "\n" << endl;
		}
        */
	} // End of run()
};


// --- Main Function ---
// Reads input data, sets up K-means, runs the algorithm, and handles basic errors.
int main(int argc, char *argv[])
{
	int total_points=0, total_values=0, K=0, max_iterations=0, has_name=0;

    // Read the header line from standard input
	if (!(cin >> total_points >> total_values >> K >> max_iterations >> has_name)) {
        cerr << "Error: Failed to read header line from input." << endl;
        return 1;
    }

    // Validate header values
    if (total_points <= 0 || total_values <= 0 || K <= 0 || max_iterations <= 0) {
        cerr << "Error: Header values (points, values, K, iterations) must be positive." << endl;
        cerr << "Read: points=" << total_points << ", values=" << total_values
             << ", K=" << K << ", iterations=" << max_iterations << endl;
        return 1;
    }

    // Use getline to consume the rest of the header line (including potential newline)
    // and prepare for reading data lines. `ws` consumes leading whitespace.
    string header_remainder;
    getline(cin >> ws, header_remainder);

	vector<Point> points; // Vector to store all data points
    points.reserve(total_points); // Pre-allocate memory for efficiency
	string line;          // String to hold each data line read from input

    cout << "Reading " << total_points << " points (" << total_values << " values each)..." << endl;
	for (int i = 0; i < total_points; ++i)
	{
        // Read one line of data point coordinates
        if (!getline(cin, line)) {
             cerr << "Error: Input ended unexpectedly while reading data line " << i+1
                  << ". Read " << i << " points." << endl;
             total_points = i; // Adjust total_points to the actual number read
             break; // Stop reading
        }

        vector<double> values; // Vector to store coordinates for the current point
        values.reserve(total_values);
        stringstream ss(line); // Use stringstream to parse the comma-separated line
        string value_str;     // String to hold each coordinate value before conversion

        bool read_success = true;
        for (int j = 0; j < total_values; ++j) {
            // Read values separated by comma
            if (!getline(ss, value_str, ',')) {
                 cerr << "Error: Could not read value " << j+1 << " (expected " << total_values
                      << ") on data line " << i+1 << "." << endl;
                 read_success = false;
                 break; // Stop parsing this line
            }
            try {
                 // Convert the string value to double
                 values.push_back(stod(value_str));
            } catch (const std::invalid_argument& ia) {
                cerr << "Error: Invalid double format '" << value_str << "' for value "
                     << j+1 << " on data line " << i+1 << "." << endl;
                read_success = false;
                break;
            } catch (const std::out_of_range& oor) {
                cerr << "Error: Double value out of range '" << value_str << "' for value "
                     << j+1 << " on data line " << i+1 << "." << endl;
                read_success = false;
                break;
            }
        } // End loop reading values for one point

        // If all values were read successfully for this line, create the Point object
        if (read_success && values.size() == total_values) {
            // Note: Assumes has_name is 0 based on problem description.
            // If has_name were 1, code to read the name from the end of the stringstream `ss` would go here.
             points.emplace_back(i, values); // Create Point object directly in the vector
        } else {
             cerr << "Skipping point " << i+1 << " due to read errors." << endl;
             // Adjust total points count if we decide to skip invalid points.
             // For simplicity, we currently proceed but the final count might be lower.
             // It might be better to exit if data format errors occur.
        }
	} // End loop reading all point lines

    // Adjust total_points if some lines were skipped or reading ended early
    if (points.size() != total_points) {
        cout << "Warning: Actual number of points read (" << points.size()
             << ") differs from header value (" << total_points << "). Using actual count." << endl;
        total_points = points.size(); // Use the number of points actually created
    }

    // Final check before running KMeans
    if (total_points < K) {
        cerr << "Error: Not enough valid points read (" << total_points
             << ") for the requested number of clusters K (" << K << ")." << endl;
        return 1;
    }
     if (total_points == 0) {
         cerr << "Error: No valid data points were read from the input." << endl;
         return 1;
     }

    // Create KMeans object and run the algorithm
    cout << "Read " << total_points << " valid points. Starting K-means..." << endl;
	KMeans kmeans(K, total_points, total_values, max_iterations);
	kmeans.run(points); // Execute the parallel K-means algorithm

    cout << "K-means execution finished." << endl;
	return 0; // Indicate successful completion
}