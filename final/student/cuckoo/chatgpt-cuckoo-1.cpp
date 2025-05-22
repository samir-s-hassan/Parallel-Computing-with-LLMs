// concurrent_cuckoo.cpp
#include <bits/stdc++.h>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <chrono>

using namespace std;

//––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Must include this line verbatim, for a fixed seed everywhere:
static std::mt19937 gen(714);

// Striped (fixed-locks) cuckoo hash set from §13.4.3
class StripedCuckooHashSet {
    static constexpr size_t PROBE_SIZE = 4;
    static constexpr size_t THRESHOLD  = PROBE_SIZE/2;

    size_t capacity;
    vector<vector<int>> table0, table1;
    vector<recursive_mutex> locks0, locks1;

    // two independent hash functions
    size_t h0(int x) const {
        return std::hash<int>{}(x) % capacity;
    }
    size_t h1(int x) const {
        // shift to get a second hash
        return (std::hash<int>{}(x) >> 16) % capacity;
    }

    // acquire locks for bucket‐0 and bucket‐1 for x
    void acquire(int x) {
        size_t i0 = h0(x), i1 = h1(x);
        if (i0 <= i1) {
            locks0[i0].lock();
            locks1[i1].lock();
        } else {
            locks1[i1].lock();
            locks0[i0].lock();
        }
    }
    void release(int x) {
        size_t i0 = h0(x), i1 = h1(x);
        if (i0 <= i1) {
            locks1[i1].unlock();
            locks0[i0].unlock();
        } else {
            locks0[i0].unlock();
            locks1[i1].unlock();
        }
    }

    // resize doubles capacity and re-inserts all items
    void resize() {
        // lock all of locks0 to quiesce add/remove/contains
        for (auto &m : locks0) m.lock();

        size_t oldCap = capacity;
        auto old0 = move(table0);
        auto old1 = move(table1);

        capacity *= 2;
        table0.assign(capacity, {});
        table1.assign(capacity, {});
        for (size_t i = 0; i < capacity; i++) {
            table0[i].reserve(PROBE_SIZE);
            table1[i].reserve(PROBE_SIZE);
        }

        // re-insert everything (will recursively call add)
        for (size_t i = 0; i < oldCap; i++) {
            for (int v : old0[i]) add(v);
            for (int v : old1[i]) add(v);
        }

        for (auto &m : locks0) m.unlock();
    }

public:
    StripedCuckooHashSet(size_t initialCap)
      : capacity(initialCap),
        table0(initialCap), table1(initialCap),
        locks0(initialCap), locks1(initialCap)
    {
        for (size_t i = 0; i < capacity; i++) {
            table0[i].reserve(PROBE_SIZE);
            table1[i].reserve(PROBE_SIZE);
        }
    }

    // add x if absent; return true on success
    bool add(int x) {
        acquire(x);
        // check presence
        size_t i0 = h0(x), i1 = h1(x);
        for (int v : table0[i0]) if (v == x) { release(x); return false; }
        for (int v : table1[i1]) if (v == x) { release(x); return false; }

        bool needResize = false;
        // preferentially go into the small probe sets
        if (table0[i0].size() < THRESHOLD) {
            table0[i0].push_back(x);
        }
        else if (table1[i1].size() < THRESHOLD) {
            table1[i1].push_back(x);
        }
        // overflow slots up to PROBE_SIZE
        else if (table0[i0].size() < PROBE_SIZE) {
            table0[i0].push_back(x);
        }
        else if (table1[i1].size() < PROBE_SIZE) {
            table1[i1].push_back(x);
        }
        else {
            needResize = true;
        }
        release(x);

        if (needResize) {
            resize();
            return add(x);
        }
        return true;
    }

    // remove x if present; return true on success
    bool remove(int x) {
        acquire(x);
        size_t i0 = h0(x), i1 = h1(x);
        // bucket 0
        {
            auto &b = table0[i0];
            for (auto it = b.begin(); it != b.end(); ++it) {
                if (*it == x) { b.erase(it); release(x); return true; }
            }
        }
        // bucket 1
        {
            auto &b = table1[i1];
            for (auto it = b.begin(); it != b.end(); ++it) {
                if (*it == x) { b.erase(it); release(x); return true; }
            }
        }
        release(x);
        return false;
    }

    // contains(x)?
    bool contains(int x) {
        acquire(x);
        size_t i0 = h0(x), i1 = h1(x);
        for (int v : table0[i0]) if (v == x) { release(x); return true; }
        for (int v : table1[i1]) if (v == x) { release(x); return true; }
        release(x);
        return false;
    }

    // non-thread-safe count of all elements
    size_t size() const {
        size_t cnt = 0;
        for (size_t i = 0; i < capacity; i++) {
            cnt += table0[i].size() + table1[i].size();
        }
        return cnt;
    }

    size_t getCapacity() const { return capacity; }

    // non-thread-safe bulk initialize with random ints
    void populate(size_t n) {
        uniform_int_distribution<int> dist(0, int(capacity*2));
        for (size_t i = 0; i < n; i++) {
            add(dist(gen));
        }
    }
};

//––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <num_operations> <num_threads>\n";
        return 1;
    }
    const size_t numOps    = stoull(argv[1]);
    const int    numThreads= stoi(argv[2]);

    const size_t initialCap = 10'000'000;
    StripedCuckooHashSet set(initialCap);

    // 1) populate to half capacity
    const size_t initialSize = initialCap / 2;
    set.populate(initialSize);

    // 2) record the starting size
    const size_t startSize = set.size();

    // 3) prepare per-thread seeds
    vector<uint32_t> seeds(numThreads);
    for (int t = 0; t < numThreads; t++) {
        seeds[t] = gen();
    }

    atomic<size_t> adds{0}, removes{0};
    vector<thread> workers;
    workers.reserve(numThreads);

    // 4) benchmark
    auto t0 = chrono::high_resolution_clock::now();
    for (int t = 0; t < numThreads; t++) {
        workers.emplace_back([&,t](){
            mt19937  rng(seeds[t]);
            uniform_int_distribution<int> opDist(1,100);
            uniform_int_distribution<int> valDist(0,int(initialCap*2));

            // split operations as evenly as possible
            size_t base = numOps / numThreads;
            size_t extra= (t==numThreads-1) ? (numOps % numThreads) : 0;
            for (size_t i = 0, M = base+extra; i < M; i++) {
                int op = opDist(rng);
                int v  = valDist(rng);
                if (op <= 80) {
                    set.contains(v);
                }
                else if (op <= 90) {
                    if (set.add(v)) adds++;
                }
                else {
                    if (set.remove(v)) removes++;
                }
            }
        });
    }
    for (auto &th : workers) th.join();
    auto t1 = chrono::high_resolution_clock::now();

    // 5) compute times & sizes
    auto totalUs = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    double avgUs = double(totalUs) / numOps;
    size_t expected = startSize + adds.load() - removes.load();
    size_t finalSize = set.size();
    size_t finalCap  = set.getCapacity();

    // 6) print _exactly_ this block, in this order:
    cout << "Total time: "               << totalUs      << "\n";
    cout << "Average time per operation: " << avgUs        << "\n";
    cout << "Hashset initial size: "      << startSize    << "\n";
    cout << "Hashset initial capacity: "  << initialCap   << "\n";
    cout << "Expected size: "             << expected     << "\n";
    cout << "Final hashset size: "        << finalSize    << "\n";
    cout << "Final hashset capacity: "    << finalCap     << "\n";

    return 0;
}
