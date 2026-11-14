// Standalone check of the host int4 MoE experts (no torch):
//   1. the scalar and the AVX2 code agree bit for bit, at 1 and at N threads;
//   2. against an emulation of the card's int4 GEMV (16 work-items, each
//      summing 32-element chunks in order, a butterfly sub-group reduction)
//      at the same rounding points: how many bf16 outputs differ, and by how
//      many units in the last place;
//   3. against float64 arithmetic with the same bf16 rounding points: the
//      error of each of the two.
// Build: g++ -O3 -std=c++17 -pthread -ffp-contract=off
// int4_moe_cpu_selftest.cpp Run:   ./a.out [threads] [reps]   (reps 0: no
// timing)

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "int4_moe_cpu_impl.h"

using namespace cpu_moe;

namespace {

struct Problem {
  Layout L;
  int64_t T = 0, U = 0;
  std::vector<uint8_t> units;
  std::vector<uint16_t> x;
  std::vector<int32_t> cu, ct;
  std::vector<float> cw;
};

uint16_t rand_bf16(std::mt19937& g, float scale) {
  std::normal_distribution<float> n(0.f, scale);
  return f32_to_bf16(n(g));
}

Problem make(
    int H,
    int I,
    int G,
    int T,
    int U,
    int topk,
    uint32_t seed,
    bool shared = false) {
  Problem p;
  p.L.hidden = H;
  p.L.inter = I;
  p.L.group = G;
  const int64_t w13 = 2LL * I * H / 2, w2 = 1LL * H * I / 2;
  const int64_t s13 = 2LL * I * (H / G) * 2, s2 = 1LL * H * (I / G) * 2;
  p.L.off_w13 = 0;
  p.L.off_w2 = w13;
  p.L.off_s13 = w13 + w2;
  p.L.off_s2 = w13 + w2 + s13;
  p.L.unit_bytes = ((w13 + w2 + s13 + s2 + 4095) / 4096) * 4096;
  p.T = T;
  p.U = U;
  std::mt19937 g(seed);
  p.units.resize(static_cast<size_t>(U) * p.L.unit_bytes);
  for (auto& b : p.units)
    b = static_cast<uint8_t>(g());
  for (int u = 0; u < U; ++u) {
    uint8_t* base = p.units.data() + static_cast<int64_t>(u) * p.L.unit_bytes;
    auto* sa = reinterpret_cast<uint16_t*>(base + p.L.off_s13);
    for (int64_t i = 0; i < s13 / 2; ++i)
      sa[i] = rand_bf16(g, 0.01f);
    auto* sb = reinterpret_cast<uint16_t*>(base + p.L.off_s2);
    for (int64_t i = 0; i < s2 / 2; ++i)
      sb[i] = rand_bf16(g, 0.01f);
  }
  p.x.resize(static_cast<size_t>(T) * H);
  for (auto& v : p.x)
    v = rand_bf16(g, 1.0f);
  std::uniform_real_distribution<float> wd(0.01f, 0.3f);
  std::vector<int> common;
  for (int t = 0; t < T; ++t) {
    std::vector<int> used;
    for (int k = 0; k < topk; ++k) {
      int u;
      if (shared && t > 0) {
        u = common[k];
      } else {
        do {
          u = static_cast<int>(g() % U);
        } while (std::find(used.begin(), used.end(), u) != used.end());
        if (t == 0) common.push_back(u);
      }
      used.push_back(u);
      p.cu.push_back(u);
      p.ct.push_back(t);
      p.cw.push_back(wd(g));
    }
  }
  return p;
}

// --- the card's GEMV, emulated -------------------------------------------

template <typename Acc>
Acc card_dot(
    const uint8_t* w,
    const uint16_t* s,
    int K,
    int G,
    const std::vector<Acc>& a,
    bool butterfly) {
  // 16 lanes; lane l takes the 32-element chunks starting at l*32, l*32+512..
  Acc lane[16];
  for (int l = 0; l < 16; ++l) {
    Acc acc = 0;
    for (int kb = l * 32; kb < K; kb += 16 * 32) {
      Acc d = 0;
      for (int k = kb; k < kb + 32; ++k) {
        const int q = nibble(w[k / 2], k & 1);
        d = d + static_cast<Acc>(q) * a[k];
      }
      acc = acc + d * static_cast<Acc>(bf16_to_f32(s[kb / G]));
    }
    lane[l] = acc;
  }
  if (butterfly) {
    for (int off = 8; off > 0; off /= 2) {
      Acc next[16];
      for (int l = 0; l < 16; ++l)
        next[l] = lane[l] + lane[l ^ off];
      for (int l = 0; l < 16; ++l)
        lane[l] = next[l];
    }
    return lane[0];
  }
  Acc sum = 0;
  for (int l = 0; l < 16; ++l)
    sum += lane[l];
  return sum;
}

template <typename Acc>
std::vector<uint16_t> reference(const Problem& p, bool butterfly) {
  const Layout& L = p.L;
  const int H = L.hidden, I = L.inter, G = L.group;
  std::vector<float> outacc(static_cast<size_t>(p.T) * H, 0.f);
  std::vector<double> outacc64(static_cast<size_t>(p.T) * H, 0.0);
  for (size_t c = 0; c < p.cu.size(); ++c) {
    const uint8_t* u = p.units.data() + p.cu[c] * L.unit_bytes;
    std::vector<Acc> xa(H);
    for (int k = 0; k < H; ++k)
      xa[k] = bf16_to_f32(p.x[p.ct[c] * H + k]);
    std::vector<float> h(2 * I);
    for (int n = 0; n < 2 * I; ++n)
      h[n] = round_bf16(
          static_cast<float>(card_dot<Acc>(
              u + L.off_w13 + static_cast<int64_t>(n) * (H / 2),
              reinterpret_cast<const uint16_t*>(u + L.off_s13) + n * (H / G),
              H,
              G,
              xa,
              butterfly)));
    std::vector<Acc> act(I);
    for (int i = 0; i < I; ++i) {
      const float s = round_bf16(h[i] / (1.0f + std::exp(-h[i])));
      act[i] = round_bf16(s * h[I + i]);
    }
    for (int n = 0; n < H; ++n) {
      const float y = round_bf16(
          static_cast<float>(card_dot<Acc>(
              u + L.off_w2 + static_cast<int64_t>(n) * (I / 2),
              reinterpret_cast<const uint16_t*>(u + L.off_s2) + n * (I / G),
              I,
              G,
              act,
              butterfly)));
      if (sizeof(Acc) == 8)
        outacc64[p.ct[c] * H + n] += static_cast<double>(y) * p.cw[c];
      else
        outacc[p.ct[c] * H + n] = outacc[p.ct[c] * H + n] + y * p.cw[c];
    }
  }
  std::vector<uint16_t> out(outacc.size());
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = sizeof(Acc) == 8 ? f32_to_bf16(static_cast<float>(outacc64[i]))
                              : f32_to_bf16(outacc[i]);
  return out;
}

struct Diff {
  int64_t n = 0, differ = 0, max_ulp = 0;
  double max_abs = 0, max_ref = 0;
  // The largest difference in units of the bf16 spacing at the output's
  // largest magnitude: a difference at the precision the output carries.
  double scaled() const {
    if (max_ref == 0) return max_abs == 0 ? 0 : 1e9;
    const double spacing = std::ldexp(1.0, std::ilogb(max_ref) - 7);
    return max_abs / spacing;
  }
};

int64_t ulp_distance(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int64_t {
    return (v & 0x8000) ? -static_cast<int64_t>(v & 0x7fff) : (v & 0x7fff);
  };
  return std::llabs(key(a) - key(b));
}

Diff compare(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  Diff d;
  for (size_t i = 0; i < a.size(); ++i) {
    ++d.n;
    if (a[i] != b[i]) {
      ++d.differ;
      d.max_ulp = std::max(d.max_ulp, ulp_distance(a[i], b[i]));
    }
    d.max_abs = std::max(
        d.max_abs, std::fabs(double(bf16_to_f32(a[i])) - bf16_to_f32(b[i])));
    d.max_ref = std::max(d.max_ref, std::fabs(double(bf16_to_f32(b[i]))));
  }
  return d;
}

void print(const char* what, const Diff& d) {
  std::printf(
      "%-44s %8lld of %8lld differ (%.4f %%), max %lld ulp of the element, "
      "max |diff| %.3g = %.2f bf16 steps at max |ref| %.3g\n",
      what,
      (long long)d.differ,
      (long long)d.n,
      100.0 * d.differ / d.n,
      (long long)d.max_ulp,
      d.max_abs,
      d.scaled(),
      d.max_ref);
}

std::vector<uint16_t> run(const Problem& p, Pool& pool, Isa isa) {
  std::vector<uint16_t> out(static_cast<size_t>(p.T) * p.L.hidden);
  run_calls(
      pool,
      isa,
      p.L,
      out.data(),
      p.x.data(),
      p.T,
      p.units.data(),
      p.U,
      p.cu.data(),
      p.ct.data(),
      p.cw.data(),
      static_cast<int64_t>(p.cu.size()));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const int threads = argc > 1 ? std::atoi(argv[1]) : 8;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 50;
  std::printf("avx2 available: %d, threads %d\n", (int)cpu_has_avx2(), threads);
  Pool one(1), many(threads);
  int failures = 0;
  struct Case {
    int H, I, G, T, U, topk;
    uint32_t seed;
  };
  const Case cases[] = {
      {2560, 640, 128, 1, 4, 1, 1},
      {2560, 640, 128, 3, 6, 2, 2},
      {2560, 640, 128, 8, 12, 3, 3},
      {2560, 640, 128, 16, 20, 10, 4},
      {256, 128, 32, 5, 3, 2, 5},
      {256, 128, 64, 7, 2, 2, 6},
      {2560, 640, 128, 13, 2, 2, 7},
      {256, 128, 128, 40, 3, 3, 8},
  };
  for (const Case& cs : cases) {
    Problem p = make(cs.H, cs.I, cs.G, cs.T, cs.U, cs.topk, cs.seed);
    std::printf(
        "-- H %d I %d G %d tokens %d units %d calls %zu\n",
        cs.H,
        cs.I,
        cs.G,
        cs.T,
        cs.U,
        p.cu.size());
    const auto s1 = run(p, one, Isa::kScalar);
    const auto sN = run(p, many, Isa::kScalar);
    if (s1 != sN) {
      std::printf("FAIL scalar 1 thread vs %d threads\n", threads);
      ++failures;
    }
    if (cpu_has_avx2()) {
      const auto a1 = run(p, one, Isa::kAvx2);
      const auto aN = run(p, many, Isa::kAvx2);
      if (a1 != s1 || aN != s1) {
        print("FAIL avx2 vs scalar", compare(a1, s1));
        ++failures;
      } else {
        std::printf(
            "avx2 == scalar, 1 and %d threads: bit-identical\n", threads);
      }
    }
    const auto card = reference<float>(p, true);
    const auto card_seq = reference<float>(p, false);
    const auto truth = reference<double>(p, false);
    print("cpu kernel vs card emulation (butterfly)", compare(s1, card));
    print("card emulation: butterfly vs sequential", compare(card, card_seq));
    print("cpu kernel vs float64", compare(s1, truth));
    print("card emulation vs float64", compare(card, truth));
    // Pass: within one bf16 step at the output's scale of the card
    // emulation, and no farther from float64 than the card emulation is
    // plus one such step.
    const Diff dc = compare(s1, card);
    const Diff d_cpu = compare(s1, truth), d_card = compare(card, truth);
    if (dc.scaled() > 1.0 || d_cpu.scaled() > d_card.scaled() + 1.0) {
      std::printf("FAIL: beyond one bf16 step of the card emulation\n");
      ++failures;
    }
  }
  // Timing at the model's shapes (reps 0: none).
  if (reps > 0)
    std::printf("-- timing (median of %d), H 2560 I 640 G 128\n", reps);
  for (int topk : {1, 10}) {
    if (reps <= 0) break;
    for (int T : {1, 4, 16}) {
      // every token through the same experts
      Problem p = make(2560, 640, 128, T, 64, topk, 100 + T, true);
      std::vector<double> ts;
      for (int r = 0; r < reps + 5; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        run(p, many, Isa::kAuto);
        const auto t1 = std::chrono::steady_clock::now();
        if (r >= 5)
          ts.push_back(
              std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
      std::sort(ts.begin(), ts.end());
      const double med = ts[ts.size() / 2];
      std::printf(
          "experts/token %2d tokens %2d calls %3zu: %8.0f us per call, %6.1f "
          "us "
          "per expert-token\n",
          topk,
          T,
          p.cu.size(),
          med,
          med / p.cu.size());
    }
  }
  std::printf(
      failures ? "SELFTEST FAILED (%d)\n" : "SELFTEST PASSED\n", failures);
  return failures ? 1 : 0;
}
