// parallel_kmeans_opt.cpp
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using Clock = chrono::high_resolution_clock;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int total_points, total_values, K, max_iter, has_name;
    if(!(cin >> total_points >> total_values >> K >> max_iter >> has_name)){
        cerr << "Error reading header\n";
        return 1;
    }
    cout << "Number of rows: " << total_points << "\n";

    // Allocate flat arrays
    size_t P = size_t(total_points) * total_values;
    float *points   = (float*)malloc(P * sizeof(float));
    int   *assign   = (int*)  malloc(total_points * sizeof(int));

    string line, tmp;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    // Read & parse CSV lines
    for(int i=0;i<total_points;i++){
        getline(cin,line);
        for(char &c: line) if(c==',') c=' ';
        istringstream ss(line);
        for(int d=0; d<total_values; d++){
            float v; ss >> v;
            points[size_t(i)*total_values + d] = v;
        }
        if(has_name) ss >> tmp;
        assign[i] = -1;
    }

    // Centroids flat array
    float *centroids = (float*)malloc(size_t(K)*total_values*sizeof(float));
    // init with distinct picks, fixed seed
    mt19937 gen(714);
    uniform_int_distribution<int> dist(0, total_points-1);
    vector<int> pick;
    pick.reserve(K);
    while((int)pick.size()<K){
        int idx = dist(gen);
        if(find(pick.begin(), pick.end(), idx)==pick.end()){
            pick.push_back(idx);
            memcpy(
              centroids + size_t(pick.size()-1)*total_values,
              points  + size_t(idx)*total_values,
              total_values * sizeof(float)
            );
        }
    }

    bool changed = true;
    int iter = 0;
    auto t0 = chrono::high_resolution_clock::now();

    // Main Lloyd loop
    while(changed && iter < max_iter){
        changed = false;
        ++iter;

        // === 1) Assignment ===
        #pragma omp parallel for schedule(dynamic,1024) reduction(|:changed)
        for(int i=0; i<total_points; i++){
            float *pi = points + size_t(i)*total_values;
            int    best = 0;
            float  best_d = 0;
            // centroid 0
            {
                float sum=0;
                float *c0 = centroids + 0*total_values;
                #pragma omp simd reduction(+:sum)
                for(int d=0; d<total_values; d++){
                    float diff = pi[d] - c0[d];
                    sum += diff*diff;
                }
                best_d = sum;
            }
            // other centroids
            for(int c=1; c<K; c++){
                float sum=0;
                float *cc = centroids + size_t(c)*total_values;
                #pragma omp simd reduction(+:sum)
                for(int d=0; d<total_values; d++){
                    float diff = pi[d] - cc[d];
                    sum += diff*diff;
                }
                if(sum < best_d){
                    best_d = sum;
                    best   = c;
                }
            }
            if(assign[i] != best){
                changed = true;
                assign[i] = best;
            }
        }
        if(!changed) break;

        // === 2) Update ===
        int nthreads = omp_get_max_threads();
        // per-thread sums & counts
        float   *local_sums = (float*)calloc(size_t(nthreads)*K*total_values, sizeof(float));
        int     *local_cnts = (int*)  calloc(size_t(nthreads)*K,           sizeof(int));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float *lsum = local_sums + size_t(tid)*K*total_values;
            int   *lcnt = local_cnts + size_t(tid)*K;
            #pragma omp for schedule(static)
            for(int i=0; i<total_points; i++){
                int c = assign[i];
                lcnt[c]++;
                float *pi = points + size_t(i)*total_values;
                float *cs = lsum + size_t(c)*total_values;
                #pragma omp simd
                for(int d=0; d<total_values; d++){
                    cs[d] += pi[d];
                }
            }
        }

        // reduce into centroids
        for(int c=0; c<K; c++){
            float *cptr = centroids + size_t(c)*total_values;
            vector<double> sumd(total_values, 0.0);
            int count=0;
            for(int t=0; t<nthreads; t++){
                int   cnt = local_cnts[size_t(t)*K + c];
                float *ls  = local_sums + size_t(t)*K*total_values + size_t(c)*total_values;
                count += cnt;
                for(int d=0; d<total_values; d++)
                    sumd[d] += ls[d];
            }
            if(count>0){
                for(int d=0; d<total_values; d++)
                    cptr[d] = float(sumd[d] / count);
            }
        }
        free(local_sums);
        free(local_cnts);
    }

    auto t1 = chrono::high_resolution_clock::now();
    long long us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();

    cout << "Total time: " << us << "\n";
    cout << "Break in iteration: " << iter << "\n\n";

    // === 3) Print centroids ===
    for(int c=0; c<K; c++){
        cout << "Cluster " << (c+1) << ":";
        float *cptr = centroids + size_t(c)*total_values;
        for(int d=0; d<total_values; d++)
            cout << " " << cptr[d];
        cout << "\n";
    }

    free(points);
    free(centroids);
    free(assign);
    return 0;
}
