#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <string>
#include <omp.h>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <limits>
#include <immintrin.h> // For explicit SIMD
#include <random>

using namespace std;

// Structure for a data point
struct Point {
    vector<double> values;
    string name;
    int cluster;
    
    Point(int dimensions) : values(dimensions, 0.0), cluster(-1) {}
};

// Cache-friendly Structure of Arrays for better vectorization
class PointDatabase {
private:
    int numPoints;
    int numDimensions;
    
    // Store each dimension contiguously for better vectorization
    vector<vector<double>> dimensions; // dimensions[dim][point]
    vector<int> clusterAssignments;
    vector<string> names;
    
public:
    PointDatabase(int points, int dims) 
        : numPoints(points), numDimensions(dims), clusterAssignments(points, -1) {
        
        // Allocate dimensions as continuous arrays
        dimensions.resize(dims);
        for (int d = 0; d < dims; d++) {
            dimensions[d].resize(points, 0.0);
        }
    }
    
    void setPoint(int pointIdx, const vector<double>& values, const string& name = "") {
        if (pointIdx < 0 || pointIdx >= numPoints) {
            cerr << "Error: Invalid point index " << pointIdx << endl;
            return;
        }
        
        for (int d = 0; d < min(numDimensions, (int)values.size()); d++) {
            dimensions[d][pointIdx] = values[d];
        }
        
        if (!name.empty()) {
            if (names.empty()) names.resize(numPoints);
            names[pointIdx] = name;
        }
    }
    
    void setCluster(int pointIdx, int cluster) {
        if (pointIdx >= 0 && pointIdx < numPoints) {
            clusterAssignments[pointIdx] = cluster;
        }
    }
    
    int getCluster(int pointIdx) const {
        if (pointIdx >= 0 && pointIdx < numPoints) {
            return clusterAssignments[pointIdx];
        }
        return -1;
    }
    
    double getValue(int pointIdx, int dim) const {
        if (pointIdx >= 0 && pointIdx < numPoints && dim >= 0 && dim < numDimensions) {
            return dimensions[dim][pointIdx];
        }
        return 0.0;
    }
    
    const string& getName(int pointIdx) const {
        static const string empty = "";
        if (names.empty() || pointIdx < 0 || pointIdx >= numPoints) {
            return empty;
        }
        return names[pointIdx];
    }
    
    bool hasNames() const {
        return !names.empty();
    }
    
    // Get entire dimension array for vectorized operations
    const vector<double>& getDimension(int dim) const {
        static const vector<double> empty;
        if (dim >= 0 && dim < numDimensions) {
            return dimensions[dim];
        }
        return empty;
    }
    
    // Fill point coordinates into provided array
    void getPointCoords(double* coords, int pointIdx) const {
        if (!coords || pointIdx < 0 || pointIdx >= numPoints) {
            return;
        }
        
        for (int d = 0; d < numDimensions; d++) {
            coords[d] = dimensions[d][pointIdx];
        }
    }
    
    int getNumPoints() const { return numPoints; }
    int getNumDimensions() const { return numDimensions; }
};

// Thread-local storage structure to avoid false sharing
struct alignas(64) ThreadLocalData {
    vector<vector<double>> sums; // [cluster][dimension]
    vector<int> counts;          // [cluster]
    vector<double> pointBuffer;  // For temporary point storage
    vector<double> centroidBuffer; // Pre-allocated buffer for centroid points
    vector<double> distances;     // Distance buffer for each cluster
    
    ThreadLocalData(int k, int dims) : 
        counts(k, 0), 
        pointBuffer(dims, 0.0), 
        centroidBuffer(dims, 0.0),
        distances(k, 0.0) {
        sums.resize(k, vector<double>(dims, 0.0));
    }
    
    void reset() {
        for (auto& cluster_sums : sums) {
            fill(cluster_sums.begin(), cluster_sums.end(), 0.0);
        }
        fill(counts.begin(), counts.end(), 0);
    }
};

// Ultra-optimized K-means implementation
class KMeansHPC {
private:
    int K;                  // Number of clusters
    int maxIterations;      // Maximum iterations
    double convergenceThreshold; // Stop when centroids move less than this amount
    int miniBatchSize;      // Size of mini-batches (0 = full batch)
    int earlyStoppingPatience; // Number of iterations with minimal improvement before stopping
    
    const PointDatabase& points; // Reference to point database
    vector<vector<double>> centroids; // [dimension][cluster]
    vector<double> centroidDistances; // For K-means++ initialization
    
    // Upper bounds for triangle inequality optimization
    vector<double> upperBounds; // Upper bounds on distances to assigned centroid
    vector<vector<double>> centroidDistMatrix; // Distances between centroids
    vector<bool> needsUpdate; // Whether a point needs distance recalculation
    
    // Exponential moving average of convergence rate
    double convergenceRate = 1.0;
    int stagnantIterations = 0;
    double lastShift = numeric_limits<double>::max();
    
    // Safely check SIMD availability at runtime
    bool hasAVX2() const {
        #ifdef __AVX2__
            return true;
        #else
            return false;
        #endif
    }
    
    // Safe distance calculation with AVX2 vectorization when available
    inline double calculateDistance(const double* point, int centroidIdx, int dims) const {
        if (hasAVX2() && dims >= 4) {
            double dist = 0.0;
            int d = 0;
            
            #ifdef __AVX2__
            __m256d sumVec = _mm256_setzero_pd();
            
            // Use aligned loads if possible
            for (; d + 3 < dims; d += 4) {
                // Load 4 elements at a time
                __m256d pVec = _mm256_loadu_pd(&point[d]);
                
                // Load centroid values - this is the safe way
                double c[4] = {
                    centroids[d][centroidIdx],
                    centroids[d+1][centroidIdx],
                    centroids[d+2][centroidIdx],
                    centroids[d+3][centroidIdx]
                };
                __m256d cVec = _mm256_loadu_pd(c);
                
                // Calculate (p-c)²
                __m256d diffVec = _mm256_sub_pd(pVec, cVec);
                sumVec = _mm256_fmadd_pd(diffVec, diffVec, sumVec);
            }
            
            // Extract sum
            alignas(32) double result[4];
            _mm256_store_pd(result, sumVec);
            dist = result[0] + result[1] + result[2] + result[3];
            #endif
            
            // Handle remaining elements
            for (; d < dims; d++) {
                double diff = point[d] - centroids[d][centroidIdx];
                dist += diff * diff;
            }
            
            return dist;
        }
        else {
            // Scalar version as fallback
            double dist = 0.0;
            for (int d = 0; d < dims; d++) {
                double diff = point[d] - centroids[d][centroidIdx];
                dist += diff * diff;
            }
            return dist;
        }
    }
    
    // Find nearest centroid with triangle inequality optimization
    inline int findNearestCentroid(const double* point, double* distances, int pointIdx, int dims) {
        if (!point || !distances) {
            return 0;
        }
        
        int currentCluster = points.getCluster(pointIdx);
        if (currentCluster >= 0 && currentCluster < K) {
            // Check if distance needs recomputation
            if (!needsUpdate[pointIdx]) {
                return currentCluster;
            }
            
            // Compute true distance to current cluster
            double currentDist = calculateDistance(point, currentCluster, dims);
            distances[currentCluster] = currentDist;
            upperBounds[pointIdx] = currentDist;
            
            // Start with current assignment as best
            int bestCluster = currentCluster;
            double minDist = currentDist;
            
            // Check other clusters using triangle inequality
            for (int c = 0; c < K; c++) {
                if (c != currentCluster) {
                    // If twice the distance to current cluster is less than the 
                    // distance between centroids, we can skip this centroid
                    if (2 * minDist < centroidDistMatrix[currentCluster][c]) {
                        continue;
                    }
                    
                    // Otherwise calculate actual distance
                    double dist = calculateDistance(point, c, dims);
                    distances[c] = dist;
                    
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = c;
                    }
                }
            }
            
            // Update upper bound
            upperBounds[pointIdx] = minDist;
            return bestCluster;
        }
        else {
            // First assignment - calculate all distances
            distances[0] = calculateDistance(point, 0, dims);
            int bestCluster = 0;
            double minDist = distances[0];
            
            for (int c = 1; c < K; c++) {
                distances[c] = calculateDistance(point, c, dims);
                if (distances[c] < minDist) {
                    minDist = distances[c];
                    bestCluster = c;
                }
            }
            
            upperBounds[pointIdx] = minDist;
            return bestCluster;
        }
    }
    
    // Safe K-means++ initialization with stochastic seeding
    void initializeKMeansPlusPlus() {
        int numPoints = points.getNumPoints();
        int numDimensions = points.getNumDimensions();
        
        // Create a random seed selection
        const int seedSampleSize = min(10000, numPoints);
        vector<int> seedIndices(numPoints);
        iota(seedIndices.begin(), seedIndices.end(), 0);
        
        // Random shuffle to select initial sample
        random_device rd;
        mt19937 g(714);
        shuffle(seedIndices.begin(), seedIndices.end(), g);
        
        // Choose first centroid from seed samples
        int firstCentroidIdx = seedIndices[0];
        for (int d = 0; d < numDimensions; d++) {
            centroids[d][0] = points.getValue(firstCentroidIdx, d);
        }
        
        // Initialize distance array for K-means++
        vector<double> minDistances(seedSampleSize, numeric_limits<double>::max());
        vector<double> pointBuffer(numDimensions);
        
        // Choose remaining centroids from the seed sample
        for (int k = 1; k < K; k++) {
            // Calculate minimum distances for seed points
            #pragma omp parallel for
            for (int i = 0; i < seedSampleSize; i++) {
                int pointIdx = seedIndices[i];
                vector<double> localPointBuffer(numDimensions);
                
                // Load point coordinates
                points.getPointCoords(localPointBuffer.data(), pointIdx);
                
                // Calculate distance to the new centroid
                double dist = calculateDistance(localPointBuffer.data(), k-1, numDimensions);
                
                // Update minimum distance if smaller
                if (dist < minDistances[i]) {
                    minDistances[i] = dist;
                }
            }
            
            // Choose next centroid with probability proportional to squared distance
            double totalWeight = 0.0;
            for (int i = 0; i < seedSampleSize; i++) {
                totalWeight += minDistances[i];
            }
            
            if (totalWeight <= 0.0) {
                // Fallback to random selection if weights are zero
                int nextSeedIdx = k % seedSampleSize;
                int nextCentroidIdx = seedIndices[nextSeedIdx];
                
                for (int d = 0; d < numDimensions; d++) {
                    centroids[d][k] = points.getValue(nextCentroidIdx, d);
                }
                continue;
            }
            
            double threshold = totalWeight * static_cast<double>(rand()) / RAND_MAX;
            double cumWeight = 0.0;
            int nextCentroidIdx = seedIndices[seedSampleSize - 1];
            
            for (int i = 0; i < seedSampleSize; i++) {
                cumWeight += minDistances[i];
                if (cumWeight >= threshold) {
                    nextCentroidIdx = seedIndices[i];
                    break;
                }
            }
            
            // Set the new centroid
            for (int d = 0; d < numDimensions; d++) {
                centroids[d][k] = points.getValue(nextCentroidIdx, d);
            }
        }
    }
    
    // Calculate distance between centroids
    void updateCentroidDistanceMatrix(int dims) {
        // Update distances between all centroids
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                if (i == j) {
                    centroidDistMatrix[i][j] = 0.0;
                } else {
                    double dist = 0.0;
                    for (int d = 0; d < dims; d++) {
                        double diff = centroids[d][i] - centroids[d][j];
                        dist += diff * diff;
                    }
                    centroidDistMatrix[i][j] = dist;
                }
            }
        }
    }
    
    // Mark points that need to be updated after centroid movement
    void markPointsForUpdate(const vector<vector<double>>& oldCentroids, int dims) {
        int numPoints = points.getNumPoints();
        
        // Calculate how much each centroid moved
        vector<double> centroidShifts(K, 0.0);
        
        for (int c = 0; c < K; c++) {
            double shift = 0.0;
            for (int d = 0; d < dims; d++) {
                double diff = centroids[d][c] - oldCentroids[d][c];
                shift += diff * diff;
            }
            centroidShifts[c] = sqrt(shift);
        }
        
        // Mark points for updates based on centroid shifts and bounds
        #pragma omp parallel for
        for (int i = 0; i < numPoints; i++) {
            int c = points.getCluster(i);
            if (c >= 0 && c < K) {
                double upperBound = upperBounds[i];
                
                // Mark for update if:
                // 1. The current centroid moved, OR
                // 2. Any other centroid moved enough to potentially become closer
                bool needsUpdate = false;
                
                if (centroidShifts[c] > 0) {
                    needsUpdate = true;
                } else {
                    for (int j = 0; j < K; j++) {
                        if (j != c) {
                            // If another centroid moved close enough to this point's
                            // bound, we need to recalculate
                            double distBetweenCentroids = sqrt(centroidDistMatrix[c][j]);
                            if (upperBound > abs(distBetweenCentroids - centroidShifts[j])) {
                                needsUpdate = true;
                                break;
                            }
                        }
                    }
                }
                
                this->needsUpdate[i] = needsUpdate;
            } else {
                // Always update if no valid cluster assigned
                this->needsUpdate[i] = true;
            }
        }
    }
    
public:
    // Constructor takes references to avoid copying data
    KMeansHPC(int K, int maxIterations, const PointDatabase& points, 
              double convergenceThreshold = 1e-6, int miniBatchSize = 0)
        : K(K), maxIterations(maxIterations), points(points), 
          convergenceThreshold(convergenceThreshold), miniBatchSize(miniBatchSize),
          earlyStoppingPatience(30), centroidDistances(K, 0.0) {
        
        // Initialize centroids
        int dims = points.getNumDimensions();
        centroids.resize(dims);
        for (int d = 0; d < dims; d++) {
            centroids[d].resize(K, 0.0);
        }
        
        // Initialize triangle inequality optimization structures
        upperBounds.resize(points.getNumPoints(), numeric_limits<double>::max());
        needsUpdate.resize(points.getNumPoints(), true);
        
        // Initialize centroid distance matrix
        centroidDistMatrix.resize(K);
        for (int i = 0; i < K; i++) {
            centroidDistMatrix[i].resize(K, 0.0);
        }
    }
    
    // Main clustering algorithm
    void run(bool debugOutput = false) {
        auto begin = chrono::high_resolution_clock::now();
        
        int numPoints = points.getNumPoints();
        int numDimensions = points.getNumDimensions();
        
        if (K > numPoints) {
            cerr << "Error: K cannot be greater than the number of points" << endl;
            return;
        }
        
        if (debugOutput) {
            // cout << "Starting K-means clustering with " << numPoints << " points, " 
            //      << numDimensions << " dimensions, K=" << K << endl;
        }
        
        // Enable mini-batch for very large datasets
        if (miniBatchSize <= 0 && numPoints > 1000000) {
            miniBatchSize = 100000; // Default mini-batch size for large datasets
        }
        
        try {
            // Step 1: Fast initialization of centroids
            initializeKMeansPlusPlus();
            
            if (debugOutput) {
                // cout << "Initialization complete" << endl;
            }
            
            auto end_phase1 = chrono::high_resolution_clock::now();
            
            // Create thread-local storage
            int numThreads = omp_get_max_threads();
            if (debugOutput) {
                // cout << "Using " << numThreads << " threads" << endl;
            }
            
            vector<ThreadLocalData> threadData;
            threadData.reserve(numThreads);
            for (int t = 0; t < numThreads; t++) {
                threadData.emplace_back(K, numDimensions);
            }
            
            // For convergence detection
            vector<vector<double>> oldCentroids(numDimensions);
            for (int d = 0; d < numDimensions; d++) {
                oldCentroids[d].resize(K);
            }
            
            // Index vector for mini-batch selection
            vector<int> indices(numPoints);
            iota(indices.begin(), indices.end(), 0);
            
            // Random engine for mini-batch selection
            mt19937 g(chrono::system_clock::now().time_since_epoch().count());
            
            // Main K-means loop
            bool changed = true;
            int iteration = 0;
            double maxCentroidShift = numeric_limits<double>::max();
            
            while (changed && iteration < maxIterations && maxCentroidShift > convergenceThreshold) {
                // Save current centroids for convergence check
                for (int d = 0; d < numDimensions; d++) {
                    for (int c = 0; c < K; c++) {
                        oldCentroids[d][c] = centroids[d][c];
                    }
                }
                
                changed = false;
                iteration++;
                
                // Reset all thread-local storages
                for (auto& data : threadData) {
                    data.reset();
                }
                
                // Determine point set to use for this iteration
                vector<int> batchIndices;
                int batchSize;
                
                if (miniBatchSize > 0 && miniBatchSize < numPoints) {
                    // Use mini-batch: select random subset of points
                    batchSize = miniBatchSize;
                    batchIndices.resize(batchSize);
                    
                    // Shuffle indices and use the first miniBatchSize elements
                    shuffle(indices.begin(), indices.end(), g);
                    copy(indices.begin(), indices.begin() + batchSize, batchIndices.begin());
                } else {
                    // Use full dataset
                    batchSize = numPoints;
                    batchIndices = indices;
                }
                
                // Update centroid distance matrix for triangle inequality
                updateCentroidDistanceMatrix(numDimensions);
                
                // Mark points that need distance recalculation
                markPointsForUpdate(oldCentroids, numDimensions);
                
                // Step 2: Assign points to nearest centroids
                #pragma omp parallel reduction(|:changed)
                {
                    int threadId = omp_get_thread_num();
                    ThreadLocalData& localData = threadData[threadId];
                    
                    #pragma omp for schedule(dynamic, 1024)
                    for (int i = 0; i < batchSize; i++) {
                        int pointIdx = batchIndices[i];
                        
                        // Load point into buffer for optimization
                        points.getPointCoords(localData.pointBuffer.data(), pointIdx);
                        
                        // Find nearest centroid using triangle inequality optimization
                        int oldCluster = points.getCluster(pointIdx);
                        int newCluster = findNearestCentroid(
                            localData.pointBuffer.data(), 
                            localData.distances.data(),
                            pointIdx,
                            numDimensions
                        );
                        
                        // Update cluster assignment if changed
                        if (oldCluster != newCluster) {
                            const_cast<PointDatabase&>(points).setCluster(pointIdx, newCluster);
                            changed = true;
                        }
                        
                        // Update local sums and counts
                        localData.counts[newCluster]++;
                        for (int d = 0; d < numDimensions; d++) {
                            localData.sums[newCluster][d] += localData.pointBuffer[d];
                        }
                    }
                }
                
                // Step 3: Recalculate centroids
                // First reset centroids
                for (int d = 0; d < numDimensions; d++) {
                    fill(centroids[d].begin(), centroids[d].end(), 0.0);
                }
                
                // Combine thread-local sums and counts
                vector<int> clusterCounts(K, 0);
                
                for (int t = 0; t < numThreads; t++) {
                    for (int c = 0; c < K; c++) {
                        clusterCounts[c] += threadData[t].counts[c];
                        for (int d = 0; d < numDimensions; d++) {
                            centroids[d][c] += threadData[t].sums[c][d];
                        }
                    }
                }
                
                // Calculate new centroids
                #pragma omp parallel for collapse(2) schedule(static)
                for (int d = 0; d < numDimensions; d++) {
                    for (int c = 0; c < K; c++) {
                        if (clusterCounts[c] > 0) {
                            centroids[d][c] /= clusterCounts[c];
                        }
                    }
                }
                
                // Handle empty clusters
                for (int c = 0; c < K; c++) {
                    if (clusterCounts[c] == 0) {
                        // Find the largest cluster
                        int largestCluster = 0;
                        int maxSize = clusterCounts[0];
                        
                        for (int i = 1; i < K; i++) {
                            if (clusterCounts[i] > maxSize) {
                                maxSize = clusterCounts[i];
                                largestCluster = i;
                            }
                        }
                        
                        // Split the largest cluster by adding random noise
                        for (int d = 0; d < numDimensions; d++) {
                            // Add small random perturbation
                            double noise = (rand() / (double)RAND_MAX - 0.5) * 0.01;
                            centroids[d][c] = centroids[d][largestCluster] * (1.0 + noise);
                        }
                        
                        changed = true;
                    }
                }
                
                // Check for convergence - maximum centroid shift
                maxCentroidShift = 0.0;
                for (int c = 0; c < K; c++) {
                    double shift = 0.0;
                    for (int d = 0; d < numDimensions; d++) {
                        double diff = centroids[d][c] - oldCentroids[d][c];
                        shift += diff * diff;
                    }
                    shift = sqrt(shift);
                    maxCentroidShift = max(maxCentroidShift, shift);
                }
                
                // Early stopping - check if convergence is slowing down
                double relativeDiff = abs(lastShift - maxCentroidShift) / lastShift;
                lastShift = maxCentroidShift;
                
                // Update convergence rate with exponential moving average
                convergenceRate = 0.9 * convergenceRate + 0.1 * relativeDiff;
                
                if (convergenceRate < 0.001) {
                    stagnantIterations++;
                } else {
                    stagnantIterations = 0;
                }
                
                // Stop if we've had many iterations with minimal improvement
                if (stagnantIterations >= earlyStoppingPatience) {
                    // cout << "Early stopping at iteration " << iteration 
                    //      << " - convergence has stagnated" << endl;
                    break;
                }
                
                if (iteration % 10 == 0 && debugOutput) {
                    // cout << "Iteration " << iteration << ": max centroid shift = " 
                    //      << maxCentroidShift << ", convergence rate = " 
                    //      << convergenceRate << endl;
                }
            }
            
            // Final assignment pass on the full dataset if using mini-batches
            if (miniBatchSize > 0 && miniBatchSize < numPoints) {
                // cout << "Performing final assignment on full dataset..." << endl;
                
                // Reset all thread-local storages
                for (auto& data : threadData) {
                    data.reset();
                }
                
                // Assign all points to their nearest centroid
                #pragma omp parallel
                {
                    int threadId = omp_get_thread_num();
                    ThreadLocalData& localData = threadData[threadId];
                    
                    #pragma omp for schedule(dynamic, 1024)
                    for (int i = 0; i < numPoints; i++) {
                        // Load point into buffer for optimization
                        points.getPointCoords(localData.pointBuffer.data(), i);
                        
                        // Find nearest centroid
                        int newCluster = findNearestCentroid(
                            localData.pointBuffer.data(), 
                            localData.distances.data(),
                            i,
                            numDimensions
                        );
                        
                        // Update cluster assignment
                        const_cast<PointDatabase&>(points).setCluster(i, newCluster);
                        
                        // Update local counts (needed for reporting)
                        localData.counts[newCluster]++;
                    }
                }
                
                // Combine thread-local counts for reporting
                vector<int> clusterCounts(K, 0);
                for (int t = 0; t < numThreads; t++) {
                    for (int c = 0; c < K; c++) {
                        clusterCounts[c] += threadData[t].counts[c];
                    }
                }
            }
            
            auto end = chrono::high_resolution_clock::now();
            
            // Output results
            // cout << "Completed in " << iteration << " iterations" << endl;
            // cout << "Final max centroid shift: " << maxCentroidShift << endl;
            
            // Count points per cluster for output
            vector<int> pointsPerCluster(K, 0);
            for (int i = 0; i < numPoints; i++) {
                int cluster = points.getCluster(i);
                if (cluster >= 0 && cluster < K) {
                    pointsPerCluster[cluster]++;
                }
            }
            
            // Print cluster sizes
            // for (int c = 0; c < K; c++) {
            //     // cout << "Cluster " << c << ": " << pointsPerCluster[c] << " points" << endl;
            // }
            
            cout << "TOTAL EXECUTION TIME = " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << " ms" << endl;
            cout << "TIME PHASE 1 = " << chrono::duration_cast<chrono::milliseconds>(end_phase1-begin).count() << " ms" << endl;
            cout << "TIME PHASE 2 = " << chrono::duration_cast<chrono::milliseconds>(end-end_phase1).count() << " ms" << endl;
        }
        catch (const exception& e) {
            cerr << "Error during K-means clustering: " << e.what() << endl;
        }
        catch (...) {
            cerr << "Unknown error during K-means clustering" << endl;
        }
    }
};

int main(int argc, char *argv[]) {
    try {
        // Set thread count from command line if provided
        if (argc > 1) {
            omp_set_num_threads(atoi(argv[1]));
        }
        
        // Set NUMA memory policy for better performance with large datasets
        #pragma omp parallel
        {
            #pragma omp single
            omp_get_num_threads();
        }
        
        // Use better random seed
        // unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        srand(714);
        
        int numPoints, numDimensions, K, maxIterations, hasNameField;
        
        // Read basic parameters
        if (!(cin >> numPoints >> numDimensions >> K >> maxIterations >> hasNameField)) {
            // cerr << "Error reading dataset parameters" << endl;
            return 1;
        }
        
        if (numPoints <= 0 || numDimensions <= 0 || K <= 0 || maxIterations <= 0) {
            // cerr << "Invalid parameters: points=" << numPoints << ", dimensions=" << numDimensions 
            //     << ", K=" << K << ", maxIter=" << maxIterations << endl;
            return 1;
        }
        
        // cout << "Dataset: " << numPoints << " points, " << numDimensions << " dimensions, K=" << K << endl;
        
        // Set a more reasonable max iterations for extremely large datasets
        if (numPoints > 1000000 && maxIterations > 1000) {
            // cout << "Adjusting max iterations from " << maxIterations << " to 1000 for large dataset" << endl;
            maxIterations = 1000;
        }
        
        auto startLoad = chrono::high_resolution_clock::now();
        
        // Create point database with Structure of Arrays layout
        PointDatabase points(numPoints, numDimensions);
        
        // Pre-allocate buffer for faster parsing
        string line;
        line.reserve(numDimensions * 20); // Reserve space for up to 20 chars per dimension
        vector<double> values(numDimensions);
        
        // Skip any leftover characters from previous line
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        
        // Use a faster method for extremely large datasets
        if (numPoints > 1000000) {
            // cout << "Using optimized loading for large dataset..." << endl;
            
            // Pre-allocate a large buffer for reading
            const size_t BUFFER_SIZE = 1024 * 1024 * 64; // 64MB buffer
            char* buffer = new char[BUFFER_SIZE];
            cin.rdbuf()->pubsetbuf(buffer, BUFFER_SIZE);
            
            // Load data in chunks with minimal string operations
            #pragma omp parallel
            {
                string localLine;
                localLine.reserve(numDimensions * 20);
                vector<double> localValues(numDimensions);
                
                #pragma omp for schedule(dynamic, 10000)
                for (int i = 0; i < numPoints; i++) {
                    string pointName = "";
                    
                    #pragma omp critical(file_io)
                    {
                        getline(cin, localLine);
                    }
                    
                    if (localLine.empty() && i > 0) {
                        continue; // Skip empty lines
                    }
                    
                    // Parse faster with manual parsing
                    size_t pos = 0;
                    size_t nextPos = 0;
                    
                    for (int j = 0; j < numDimensions; j++) {
                        // Find next comma
                        nextPos = localLine.find(',', pos);
                        if (nextPos == string::npos) {
                            if (j < numDimensions - 1) {
                                // Missing columns, fill with zeros
                                for (; j < numDimensions; j++) {
                                    localValues[j] = 0.0;
                                }
                                break;
                            } else {
                                // Last column
                                nextPos = localLine.length();
                            }
                        }
                        
                        // Extract and convert value
                        try {
                            localValues[j] = stod(localLine.substr(pos, nextPos - pos));
                        } catch (...) {
                            localValues[j] = 0.0;
                        }
                        
                        pos = nextPos + 1;
                    }
                    
                    // Set the point data
                    points.setPoint(i, localValues, pointName);
                }
            }
            
            delete[] buffer;
        } else {
            // Standard loading for smaller datasets
            for (int i = 0; i < numPoints; i++) {
                string pointName = "";
                
                if (!getline(cin, line)) {
                    cerr << "Error reading data at point " << i << endl;
                    if (i > 0) {
                        // Continue with what we've got so far
                        numPoints = i;
                        break;
                    } else {
                        return 1;
                    }
                }
                
                stringstream ss(line);
                string token;
                
                // Parse the line safely
                for (int j = 0; j < numDimensions; j++) {
                    if (getline(ss, token, ',')) {
                        try {
                            values[j] = stod(token);
                        } catch (...) {
                            cerr << "Error parsing value '" << token << "' at dimension " << j << " for point " << i << endl;
                            values[j] = 0.0;
                        }
                    } else {
                        cerr << "Missing dimension " << j << " for point " << i << endl;
                        values[j] = 0.0;
                    }
                }
                
                // Get name if specified
                if (hasNameField && getline(ss, pointName)) {
                    pointName.erase(0, pointName.find_first_not_of(", "));
                }
                
                // Set the point data
                points.setPoint(i, values, pointName);
            }
        }
        
        auto endLoad = chrono::high_resolution_clock::now();
        // cout << "Data loading time: " << chrono::duration_cast<chrono::milliseconds>(endLoad-startLoad).count() << " ms" << endl;
        
        // Create and run K-means with early convergence detection
        double convergenceThreshold = 1e-4; // Less strict convergence for faster termination
        int miniBatchSize = (numPoints > 5000000) ? 500000 : 0; // Use mini-batch for very large datasets
        
        KMeansHPC kmeans(K, maxIterations, points, convergenceThreshold, miniBatchSize);
        kmeans.run(true); // Set to true for debug output
        
        return 0;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error occurred" << endl;
        return 1;
    }
}