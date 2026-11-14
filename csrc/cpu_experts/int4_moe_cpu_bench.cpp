// Standalone benchmark of the host int4 MoE op (no torch): the decode and
// prefill shapes of one MoE layer call, the experts drawn from a set larger
// than the last-level cache, at several thread counts; and a plain streaming
// read with the same threads (the bandwidth ceiling).
//
// Build (any g++ or clang++ with C++20; nothing else):
//   g++ -O3 -std=c++20 -pthread -ffp-contract=off
//       int4_moe_cpu_bench.cpp -o int4_moe_cpu_bench
// Run:
//   ./int4_moe_cpu_bench [--threads 1,2,4,6] [--experts 256] [--reps 60]
//                        [--gap-us 2000] [--shapes 30x1,30x2,30x8,...]
//                        [--cores 0,1,2,...] [--spin-us 100]
//                        [--stream-mib 1024] [--isa 0|1|2] [--arith 0..3]
// --cores pins pool worker i to cores[i]; the streaming read pins its
// thread t to cores[t].
// A shape EXT means E distinct experts per call, each with T tokens (the
// call list is token-major, top-k order inside a token, like the engine's).
// --gap-us sleeps between calls, as the engine does between two MoE layers
// (the card works meanwhile), so pool wake-ups are in the numbers.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "int4_moe_cpu_impl.h"

#ifdef __linux__
  #include <pthread.h>
  #include <sched.h>
#endif

using namespace cpu_moe;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<int> parse_list(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) v.push_back(std::atoi(item.c_str()));
  return v;
}

struct Shape {
  int E, T;
};

std::vector<Shape> parse_shapes(const std::string& s) {
  std::vector<Shape> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const auto x = item.find('x');
    if (x == std::string::npos) continue;
    v.push_back(
        {std::atoi(item.substr(0, x).c_str()),
         std::atoi(item.substr(x + 1).c_str())});
  }
  return v;
}

uint16_t rand_bf16(std::mt19937& g, float scale) {
  std::normal_distribution<float> n(0.f, scale);
  return f32_to_bf16(n(g));
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double pct(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))];
}

// Streaming read of `bytes` split across `threads` threads: GB/s.
double stream_read(
    const uint8_t* buf,
    size_t bytes,
    int threads,
    const std::vector<int>& cores) {
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<double> secs(threads);
  std::vector<uint64_t> sink(threads * 8);
  std::vector<std::thread> ts;
  for (int t = 0; t < threads; ++t) {
    ts.emplace_back([&, t] {
#ifdef __linux__
      if (!cores.empty()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cores[t % cores.size()], &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      }
#endif
      const size_t per = (bytes / threads) & ~size_t(63);
      const uint64_t* p = reinterpret_cast<const uint64_t*>(buf + per * t);
      const size_t n = per / 8;
      ready.fetch_add(1);
      while (!go.load()) {
      }
      const auto t0 = Clock::now();
      uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      for (int rep = 0; rep < 3; ++rep)
        for (size_t i = 0; i + 8 <= n; i += 8) {
          a0 ^= p[i] + p[i + 4];
          a1 ^= p[i + 1] + p[i + 5];
          a2 ^= p[i + 2] + p[i + 6];
          a3 ^= p[i + 3] + p[i + 7];
        }
      secs[t] = std::chrono::duration<double>(Clock::now() - t0).count();
      sink[t * 8] = a0 ^ a1 ^ a2 ^ a3;
    });
  }
  while (ready.load() < threads) {
  }
  go.store(true);
  for (auto& t : ts)
    t.join();
  const double worst = *std::max_element(secs.begin(), secs.end());
  uint64_t s = 0;
  for (auto v : sink)
    s ^= v;
  if (s == 42) std::printf(" ");
  return 3.0 * ((bytes / threads) & ~size_t(63)) * threads / worst / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> threads = {1, 2, 4, 6};
  std::vector<Shape> shapes =
      parse_shapes("30x1,30x2,30x4,30x8,30x16,30x64,30x256");
  std::vector<int> cores;
  int U = 256, reps = 60, gap_us = 2000, H = 2560, I = 640, G = 128;
  int stream_mib = 1024;
  int isa_arg = 0;
  int spin_us = -1;
  int arith = 0;
  for (int a = 1; a + 1 < argc; a += 2) {
    const std::string k = argv[a], v = argv[a + 1];
    if (k == "--threads")
      threads = parse_list(v);
    else if (k == "--experts")
      U = std::atoi(v.c_str());
    else if (k == "--reps")
      reps = std::atoi(v.c_str());
    else if (k == "--gap-us")
      gap_us = std::atoi(v.c_str());
    else if (k == "--shapes")
      shapes = parse_shapes(v);
    else if (k == "--cores")
      cores = parse_list(v);
    else if (k == "--stream-mib")
      stream_mib = std::atoi(v.c_str());
    else if (k == "--arith")
      arith = std::atoi(v.c_str());
    else if (k == "--spin-us")
      spin_us = std::atoi(v.c_str());
    else if (k == "--isa")
      isa_arg = std::atoi(v.c_str());
    else if (k == "--H")
      H = std::atoi(v.c_str());
    else if (k == "--I")
      I = std::atoi(v.c_str());
    else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  Layout L;
  L.hidden = H;
  L.inter = I;
  L.group = G;
  const int64_t w13 = 2LL * I * H / 2, w2 = 1LL * H * I / 2;
  const int64_t s13 = 2LL * I * (H / G) * 2, s2 = 1LL * H * (I / G) * 2;
  L.off_w13 = 0;
  L.off_w2 = w13;
  L.off_s13 = w13 + w2;
  L.off_s2 = w13 + w2 + s13;
  L.unit_bytes = ((w13 + w2 + s13 + s2 + 4095) / 4096) * 4096;
  const int64_t payload = w13 + w2 + s13 + s2;
  std::printf(
      "H %d I %d G %d, unit %lld B (payload %lld B), %d experts = %.0f MiB, "
      "avx2 %d, isa %s, arith %d, gap %d us, reps %d\n",
      H,
      I,
      G,
      (long long)L.unit_bytes,
      (long long)payload,
      U,
      U * L.unit_bytes / 1048576.0,
      (int)cpu_has_avx2(),
      resolve_isa(static_cast<Isa>(isa_arg)) == Isa::kAvx2 ? "avx2" : "scalar",
      arith,
      gap_us,
      reps);
  std::printf("pool cores: %s\n", cores.empty() ? "(none)" : "given");
  std::mt19937 g(1);
  std::vector<uint8_t> units(static_cast<size_t>(U) * L.unit_bytes);
  for (size_t i = 0; i < units.size(); i += 4) {
    const uint32_t r = g();
    std::memcpy(&units[i], &r, 4);
  }
  for (int u = 0; u < U; ++u) {
    uint8_t* base = units.data() + static_cast<int64_t>(u) * L.unit_bytes;
    auto* sa = reinterpret_cast<uint16_t*>(base + L.off_s13);
    for (int64_t i = 0; i < s13 / 2; ++i)
      sa[i] = rand_bf16(g, 0.01f);
    auto* sb = reinterpret_cast<uint16_t*>(base + L.off_s2);
    for (int64_t i = 0; i < s2 / 2; ++i)
      sb[i] = rand_bf16(g, 0.01f);
  }
  // streaming ceiling
  {
    std::vector<uint8_t> big(static_cast<size_t>(stream_mib) << 20);
    for (size_t i = 0; i < big.size(); i += 4096)
      big[i] = static_cast<uint8_t>(i);
    std::printf("stream read (%d MiB, 3 passes):", stream_mib);
    for (int t : threads) {
      double best = 0;
      for (int r = 0; r < 3; ++r)
        best = std::max(best, stream_read(big.data(), big.size(), t, cores));
      std::printf("  %d thr %.1f GB/s", t, best);
    }
    std::printf("\n");
  }
  std::printf(
      "%-8s %4s %10s %10s %12s %10s %10s\n",
      "shape",
      "thr",
      "med us",
      "p90 us",
      "us/exp-tok",
      "wGB/s",
      "MAC/cyc*");
  for (const Shape& sh : shapes) {
    const int E = std::min(sh.E, U), T = sh.T;
    // tokens 0..T-1; each token routes to the same E experts (E calls per
    // token), so each expert has T tokens: the call list is token-major.
    std::vector<uint16_t> x(static_cast<size_t>(T) * H);
    for (auto& v : x)
      v = rand_bf16(g, 1.0f);
    std::vector<uint16_t> out(x.size());
    std::vector<int32_t> cu(static_cast<size_t>(E) * T), ct(cu.size());
    std::vector<float> cw(cu.size());
    std::uniform_real_distribution<float> wd(0.01f, 0.3f);
    for (int tn : threads) {
      Pool pool(tn);
      pool.set_cores(cores);
      if (spin_us >= 0) pool.set_spin_us(spin_us);
      std::vector<double> ts;
      std::vector<int> perm(U);
      for (int i = 0; i < U; ++i)
        perm[i] = i;
      int cursor = U;
      for (int r = 0; r < reps + 5; ++r) {
        // E distinct experts, cycling through all U before any repeats
        std::vector<int> pick;
        for (int e = 0; e < E; ++e) {
          if (cursor >= U) {
            std::shuffle(perm.begin(), perm.end(), g);
            cursor = 0;
          }
          pick.push_back(perm[cursor++]);
        }
        for (int t = 0; t < T; ++t)
          for (int e = 0; e < E; ++e) {
            cu[t * E + e] = pick[e];
            ct[t * E + e] = t;
            cw[t * E + e] = wd(g);
          }
        if (gap_us > 0)
          std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
        const auto t0 = Clock::now();
        run_calls(
            pool,
            static_cast<Isa>(isa_arg),
            L,
            out.data(),
            x.data(),
            T,
            units.data(),
            U,
            cu.data(),
            ct.data(),
            cw.data(),
            static_cast<int64_t>(cu.size()),
            arith);
        const auto t1 = Clock::now();
        if (r >= 5)
          ts.push_back(
              std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
      const double med = median(ts), p90 = pct(ts, 0.9);
      const double macs = static_cast<double>(E) * T * (3.0 * I * H);
      std::printf(
          "%3dx%-4d %4d %10.0f %10.0f %12.1f %10.1f %10.2f\n",
          E,
          T,
          tn,
          med,
          p90,
          med / (E * T),
          E * payload / med / 1e3,
          macs / (med * 1e-6) / (tn * 3.1e9));
      std::fflush(stdout);
    }
  }
  std::printf("* MAC/cyc per thread at a nominal 3.1 GHz\n");
  return 0;
}
