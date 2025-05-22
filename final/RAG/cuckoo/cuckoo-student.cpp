// hashset.cpp
// To compile sequential & lock versions:
//   g++ -std=c++17 -O3 hashset.cpp -o hashset
// To compile the transactional version (GCC 4.7+ with TM enabled):
//   g++ -std=gnu++17 -O3 -fgnu-tm -DUSE_TM hashset.cpp -o hashset_tm

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// -----------------------------------------------------------------------------
// Sequential open-addressed hash set with linear probing
// -----------------------------------------------------------------------------
template<typename T>
class SeqHashSet {
  enum State : uint8_t { EMPTY = 0, OCC = 1, DEL = 2 };
  struct Entry { T key; State state = EMPTY; };

public:
  SeqHashSet(size_t capacity)
    : cap_(nextPowerOfTwo(capacity)),
      mask_(cap_ - 1),
      table_(cap_)
  {}

  bool add(const T& key) {
    size_t idx = hash_(key) & mask_;
    while (true) {
      Entry& e = table_[idx];
      if (e.state == EMPTY || e.state == DEL) {
        e.key = key;
        e.state = OCC;
        return true;
      }
      if (e.state == OCC && e.key == key) {
        return false;  // already present
      }
      idx = (idx + 1) & mask_;
    }
  }

  bool remove(const T& key) {
    size_t idx = hash_(key) & mask_;
    while (true) {
      Entry& e = table_[idx];
      if (e.state == EMPTY) {
        return false;  // not found
      }
      if (e.state == OCC && e.key == key) {
        e.state = DEL;
        return true;
      }
      idx = (idx + 1) & mask_;
    }
  }

  bool contains(const T& key) const {
    size_t idx = hash_(key) & mask_;
    while (true) {
      const Entry& e = table_[idx];
      if (e.state == EMPTY) {
        return false;
      }
      if (e.state == OCC && e.key == key) {
        return true;
      }
      idx = (idx + 1) & mask_;
    }
  }

  // Non-thread-safe
  size_t size() const {
    size_t cnt = 0;
    for (const auto& e : table_) {
      if (e.state == OCC) ++cnt;
    }
    return cnt;
  }

  // Non-thread-safe bulk initialization
  void populate(const T start, const T end) {
    for (T v = start; v <= end; ++v) {
      add(v);
    }
  }

private:
  static size_t nextPowerOfTwo(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
  }

  size_t hash_(const T& key) const {
    return std::hash<T>()(key);
  }

  const size_t cap_, mask_;
  std::vector<Entry> table_;
};

// -----------------------------------------------------------------------------
// Striped-lock wrapper around SeqHashSet
// -----------------------------------------------------------------------------
template<typename T>
class LockHashSet {
public:
  LockHashSet(size_t capacity, size_t numLocks = 256)
    : set_(capacity),
      locks_(numLocks)
  {}

  bool add(const T& key) {
    auto& m = stripeFor(key);
    std::lock_guard<std::mutex> lg(m);
    return set_.add(key);
  }

  bool remove(const T& key) {
    auto& m = stripeFor(key);
    std::lock_guard<std::mutex> lg(m);
    return set_.remove(key);
  }

  bool contains(const T& key) const {
    auto& m = stripeFor(key);
    std::lock_guard<std::mutex> lg(m);
    return set_.contains(key);
  }

  // Non-thread-safe
  size_t size() const { return set_.size(); }
  void populate(const T start, const T end) { set_.populate(start, end); }

private:
  std::mutex& stripeFor(const T& key) const {
    size_t idx = std::hash<T>()(key) % locks_.size();
    return locks_[idx];
  }

  SeqHashSet<T> set_;
  mutable std::vector<std::mutex> locks_;
};

// -----------------------------------------------------------------------------
// Transactional-memory wrapper (GCC +fgnu-tm)
// -----------------------------------------------------------------------------
#ifdef USE_TM
template<typename T>
class TMHashSet {
public:
  TMHashSet(size_t capacity)
    : set_(capacity)
  {}

  bool add(const T& key) {
    bool res;
    __transaction_atomic {
      res = set_.add(key);
    }
    return res;
  }

  bool remove(const T& key) {
    bool res;
    __transaction_atomic {
      res = set_.remove(key);
    }
    return res;
  }

  bool contains(const T& key) const {
    bool res;
    __transaction_atomic {
      res = set_.contains(key);
    }
    return res;
  }

  // Non-thread-safe
  size_t size() const { return set_.size(); }
  void populate(const T start, const T end) { set_.populate(start, end); }

private:
  SeqHashSet<T> set_;
};
#endif

// -----------------------------------------------------------------------------
// Test harness
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <version: seq|lock|tm> <numThreads> <opsPerThread>\n";
    return 1;
  }
  std::string version = argv[1];
  int numThreads     = std::stoi(argv[2]);
  int opsPerThread   = std::stoi(argv[3]);
  const size_t tableCapacity     = 1 << 20;      // 1,048,576 slots
  const int    initialInsertions = 500000;        // keys 1..500,000
  const int    maxValue          = 1000000;       // operations on range 1..1,000,000
  const int    totalOps          = numThreads * opsPerThread;

  // Pre-generate operation sequence
  std::vector<std::pair<char,int>> ops(totalOps);
  std::vector<char>                outcomes(totalOps);
  std::vector<bool>                sim(maxValue+1, false);
  sim.reserve(maxValue+1);
  std::mt19937                    rng(0);
  std::uniform_int_distribution<> typeDist(0, 99);
  std::uniform_int_distribution<> keyDist(1, maxValue);

  // Simulate initial populate for expected size
  size_t expectedSize = initialInsertions;
  for (int i = 1; i <= initialInsertions; ++i) {
    sim[i] = true;
  }

  // Build ops[] and simulate on 'sim' to get expectedSize
  for (int i = 0; i < totalOps; ++i) {
    int  td  = typeDist(rng);
    int  key = keyDist(rng);
    char op  = (td < 80 ? 'c' : (td < 90 ? 'a' : 'r'));
    ops[i]    = {op, key};

    // update expectedSize
    if (op == 'a' && !sim[key]) {
      sim[key] = true;
      ++expectedSize;
    } else if (op == 'r' && sim[key]) {
      sim[key] = false;
      --expectedSize;
    }
  }

  // Instantiate the chosen set
  std::string ver = version;
  std::transform(ver.begin(), ver.end(), ver.begin(), ::tolower);
  if (ver == "seq") {
    auto set = std::make_unique<SeqHashSet<int>>(tableCapacity);
    set->populate(1, initialInsertions);

    auto worker = [&](int tid) {
      int start = tid * opsPerThread;
      int end   = start + opsPerThread;
      for (int i = start; i < end; ++i) {
        bool r = false;
        if (ops[i].first == 'c')
          r = set->contains(ops[i].second);
        else if (ops[i].first == 'a')
          r = set->add(ops[i].second);
        else
          r = set->remove(ops[i].second);
        outcomes[i] = r;
      }
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t)
      threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();
  long long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    size_t actual = set->size();

    std::cout << elapsed << "\n";

  } else if (ver == "lock") {
    auto set = std::make_unique<LockHashSet<int>>(tableCapacity);
    set->populate(1, initialInsertions);

    auto worker = [&](int tid) {
      int start = tid * opsPerThread;
      int end   = start + opsPerThread;
      for (int i = start; i < end; ++i) {
        bool r = false;
        if (ops[i].first == 'c')
          r = set->contains(ops[i].second);
        else if (ops[i].first == 'a')
          r = set->add(ops[i].second);
        else
          r = set->remove(ops[i].second);
        outcomes[i] = r;
      }
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t)
      threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    long long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    size_t actual = set->size();

    std::cout << elapsed << "\n";

#ifdef USE_TM
  } else if (ver == "tm") {
    auto set = std::make_unique<TMHashSet<int>>(tableCapacity);
    set->populate(1, initialInsertions);

    auto worker = [&](int tid) {
      int start = tid * opsPerThread;
      int end   = start + opsPerThread;
      for (int i = start; i < end; ++i) {
        bool r = false;
        if (ops[i].first == 'c')
          r = set->contains(ops[i].second);
        else if (ops[i].first == 'a')
          r = set->add(ops[i].second);
        else
          r = set->remove(ops[i].second);
        outcomes[i] = r;
      }
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t)
      threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

        long long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    size_t actual = set->size();

    std::cout << elapsed << "\n";
#else
  } else if (ver == "tm") {
    std::cerr << "Error: transactional version not compiled; rebuild with -DUSE_TM -fgnu-tm\n";
    return 1;
#endif

  } else {
    std::cerr << "Unknown version: " << version << "\n";
    return 1;
  }

  return 0;
}
