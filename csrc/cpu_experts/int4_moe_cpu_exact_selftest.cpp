// Standalone check of the op's card-exact arithmetic (arith 7,
// int4_moe_cpu_exact.h), no torch. Every level the host (or SDE's emulated
// CPU) has is checked against an independent reference that sums each
// block in __float128 (113 significant bits: exact for every block of the
// data below, whose products and accumulators span at most 109 bits) and
// rounds once to fp32:
//   1. round32_sum and the 576-bit exact sum (Wide) against __float128 on
//      random pairs, forced midpoints (acc + B rounds in double onto an fp32
//      midpoint, the exact sum on either side), fp32 subnormals and
//      overflow, and cases beyond __float128's reach with known answers;
//   2. rows_exact (both GEMMs' inner function), every level, against the
//      reference on crafted rows: random wide exponent spans (blocks beyond
//      the 30-exponent window take the exact integer sum), forced midpoints,
//      subnormal and overflowing accumulators, non-finite inputs (the
//      canonical NaN / inf rule);
//   3. the op (run_calls, arith 7) at QFN's shapes and smaller ones,
//      every level, 1 and N threads, inputs gathered and read in place,
//      against the reference model of the card (GEMM1, the table's silu,
//      GEMM2, the gather) with 0 differing values, and every level against
//      the scalar code bit for bit;
//   4. the counters: how many blocks took each slow path (they must be
//      exercised).
//
// Build: g++ -O2 -std=c++20 -pthread -ffp-contract=off -I csrc
//        int4_moe_cpu_exact_selftest.cpp       (clang++, icpx likewise)
// Run:   ./a.out [threads]                       exit status 0 when all pass
//        ./a.out work LEVEL ARITH TOKENS EXPERTS [REPS]   REPS calls of QFN's
//                                                shapes, 1 thread (sde -mix)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cpu_experts/int4_moe_cpu_impl.h"

using namespace cpu_moe;

namespace {

int failures = 0, checks = 0;

uint32_t fbits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

// equal bits, or both NaN
bool same(float a, float b) {
  return (std::isnan(a) && std::isnan(b)) || fbits(a) == fbits(b);
}

float ref_trunc_bf16(float f) {  // the reference's own truncation
  uint32_t b = fbits(f) & 0xffff0000u;
  float r;
  std::memcpy(&r, &b, 4);
  return r;
}

int ref_nibble(uint8_t byte, int high) {
  const int n = high ? byte >> 4 : byte & 15;
  return n < 8 ? n : n - 16;
}

// --------------------------------------------------------------------------
// 1. the primitives
// --------------------------------------------------------------------------

float q_round(float acc, double b) {
  return static_cast<float>(static_cast<__float128>(acc) + b);
}

void check_primitives() {
  std::mt19937_64 g(7);
  int64_t n = 0, bad = 0, ties = 0;
  auto one = [&](float acc, double b) {
    ++n;
    const float want = q_round(acc, b);
    const float got = round32_sum(acc, b);
    if (!same(got, want)) {
      if (bad < 5)
        std::printf(
            "  FAIL round32_sum(%a, %a) = %a, want %a\n", acc, b, got, want);
      ++bad;
    }
    // the exact integer sum too
    Wide w;
    w.add(acc);
    w.add(b);
    if (!same(w.to_float(), want)) {
      if (bad < 5)
        std::printf(
            "  FAIL Wide(%a + %a) = %a, want %a\n", acc, b, w.to_float(), want);
      ++bad;
    }
  };
  std::uniform_int_distribution<int> ed(-60, 60), md(0, (1 << 23) - 1);
  std::uniform_int_distribution<int> sh(25, 80);
  for (int i = 0; i < 200000; ++i) {
    // acc an fp32 value; b a sum of bf16 x bf16 products: up to 40 bits
    const float acc =
        std::ldexp(1.0f + md(g) * 0x1p-23f, ed(g)) * (g() & 1 ? -1 : 1);
    // (the lowest bit of b at most 90 below acc's top: __float128 exact)
    const double b = std::ldexp(
        static_cast<double>(g() & ((1ULL << 40) - 1)) * (g() & 1 ? -1 : 1),
        std::ilogb(acc) - 90 + static_cast<int>(g() % 70));
    one(acc, b);
    // a forced midpoint: acc + half an fp32 ulp of acc + a tiny remainder
    // of either sign, far enough below for double to round it away
    const int ea = std::ilogb(acc);
    const double half = std::ldexp(1.0, ea - 24) * (acc < 0 ? -1 : 1);
    const double tiny = std::ldexp(1.0, ea - 24 - sh(g)) * (g() & 1 ? -1 : 1);
    if (static_cast<double>(acc) + (half + tiny) ==
        static_cast<double>(acc) + half)
      ++ties;
    one(acc, half + tiny);
    one(acc, -half + tiny);
  }
  // fp32 subnormals and overflow
  for (int i = 0; i < 20000; ++i) {
    const float acc =
        std::ldexp(static_cast<float>(md(g)), -149) * (g() & 1 ? -1 : 1);
    one(acc,
        std::ldexp(
            static_cast<double>(g() & 0xFFFFFF), -175 - (int)(g() % 20)) *
            (g() & 1 ? -1 : 1));
    one(acc, std::ldexp(1.0, -150) + std::ldexp(1.0, -200));
    one(FLT_MAX,
        std::ldexp(1.0, 103) - std::ldexp(1.0, 50) * (g() & 1 ? 1 : -1));
    one(-FLT_MAX, -std::ldexp(1.0, 103) + std::ldexp(1.0, 60));
  }
  one(FLT_MAX, std::ldexp(1.0, 103));  // exactly FLT_MAX + half an ulp: inf
  std::printf(
      "-- primitives: round32_sum and Wide vs __float128: %lld cases (%lld "
      "midpoints in double), %lld wrong\n",
      (long long)n,
      (long long)ties,
      (long long)bad);
  ++checks;
  if (bad || ties < 1000) ++failures;
  // beyond __float128: known answers
  struct Known {
    float acc;
    std::vector<double> p;
    float want;
    const char* what;
  };
  const Known known[] = {
      {0x1p100f, {0x1p-100, 0x1p-200}, 0x1p100f, "2^100 + tiny"},
      {0x1.000002p100f,
       {0x1p76, 0x1p-200},
       0x1.000004p100f,
       "midpoint + 2^-200 up"},
      {0x1.000002p100f,
       {0x1p76, -0x1p-200},
       0x1.000002p100f,
       "midpoint - 2^-200 down"},
      {0x1p100f, {0x1p76, -0x1p-250}, 0x1p100f, "even midpoint - tiny"},
      {0.0f, {0x1p-266, 0x1p-266}, 0.0f, "2^-265: below fp32"},
      {0.0f, {0x1p-150, 0x1p-266}, 0x1p-149f, "2^-150 + tiny: up to 2^-149"},
      {0.0f, {0x1p-150}, 0.0f, "2^-150 exactly: even, 0"},
      {0.0f, {0x1p250, -0x1p250, 0x1p-250}, 0.0f, "cancel to 2^-250 (0)"},
  };
  int kb = 0;
  for (const Known& k : known) {
    const float got =
        exact_step_slow(k.acc, k.p.data(), static_cast<int>(k.p.size()));
    ++checks;
    if (!same(got, k.want)) {
      std::printf("  FAIL known %s: %a, want %a\n", k.what, got, k.want);
      ++failures;
      ++kb;
    }
  }
  // non-finite rule
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double pi[] = {1.0, inf, 2.0}, pn[] = {inf, -inf}, pq[] = {-1.0, nan};
  ++checks;
  if (exact_step_slow(1.0f, pi, 3) != std::numeric_limits<float>::infinity() ||
      fbits(exact_step_slow(1.0f, pn, 2)) != 0x7fc00000u ||
      fbits(exact_step_slow(1.0f, pq, 2)) != 0x7fc00000u ||
      exact_step_slow(-std::numeric_limits<float>::infinity(), pi, 1) !=
          -std::numeric_limits<float>::infinity()) {
    std::printf("  FAIL the non-finite rule\n");
    ++failures;
    ++kb;
  }
  std::printf(
      "-- known answers beyond __float128 and the non-finite rule: %s\n",
      kb ? "FAIL" : "ok");
}

// --------------------------------------------------------------------------
// 2. rows: the reference and crafted rows
// --------------------------------------------------------------------------

// One output of the card's chain, from the raw row and the inputs.
// Each block: its exact sum (in double when the nonzero inputs span at most
// 30 exponents, which bounds it to 53 bits; else in __float128), then
// acc + sum in __float128 rounded once to fp32.
float ref_dot(
    const uint8_t* w,
    const uint16_t* s,
    int K,
    int G,
    const std::vector<double>& x /* k order */) {
  float acc = 0.0f;
  for (int b = 0; b < K / 16; ++b) {
    int lo = 100000, hi = -100000;
    bool finite = std::isfinite(acc);
    for (int k = 16 * b; k < 16 * b + 16; ++k) {
      finite = finite && std::isfinite(x[k]);
      if (x[k] != 0 && std::isfinite(x[k])) {
        lo = std::min(lo, std::ilogb(x[k]));
        hi = std::max(hi, std::ilogb(x[k]));
      }
    }
    double dsum = 0;
    for (int k = 16 * b; k < 16 * b + 16; ++k) {
      const float scale = bf16_to_f32(s[k / G]);
      const float wq = ref_trunc_bf16(
          static_cast<float>(ref_nibble(w[k / 2], k & 1)) * scale);
      dsum += static_cast<double>(wq) * x[k];
      finite = finite && std::isfinite(static_cast<double>(wq) * x[k]);
    }
    if (finite && hi - lo <= 30) {
      acc = static_cast<float>(
          static_cast<__float128>(acc) + static_cast<__float128>(dsum));
      continue;
    }
    __float128 sum = acc;
    bool nan = std::isnan(acc);
    bool pinf = std::isinf(acc) && acc > 0, ninf = std::isinf(acc) && acc < 0;
    for (int k = 16 * b; k < 16 * b + 16; ++k) {
      const float scale = bf16_to_f32(s[k / G]);
      const float wq = ref_trunc_bf16(
          static_cast<float>(ref_nibble(w[k / 2], k & 1)) * scale);
      const double p = static_cast<double>(wq) * x[k];
      nan = nan || std::isnan(p);
      pinf = pinf || (std::isinf(p) && p > 0);
      ninf = ninf || (std::isinf(p) && p < 0);
      sum += static_cast<__float128>(wq) * static_cast<__float128>(x[k]);
    }
    if (nan || (pinf && ninf))
      acc = std::numeric_limits<float>::quiet_NaN();
    else if (pinf || ninf)
      acc = pinf ? std::numeric_limits<float>::infinity()
                 : -std::numeric_limits<float>::infinity();
    else
      acc = static_cast<float>(sum);
  }
  return acc;
}

struct RowsCase {
  int K, G, N, T;
  std::vector<uint8_t> W;
  std::vector<uint16_t> S;
  std::vector<uint16_t> x;  // [T, K] bf16
};

// Inputs in the op's scratch layout: doubles split even/odd, flags, the
// int16 forms.
struct Scratch {
  std::vector<double> xe, xo, pw;
  std::vector<uint8_t> fl;
  std::vector<int16_t> xi;
};

Scratch scratch_of(const RowsCase& c) {
  Scratch z;
  z.xe.assign(static_cast<size_t>(c.T) * c.K / 2, 0);
  z.xo.assign(z.xe.size(), 0);
  z.fl.assign(static_cast<size_t>(c.T) * c.K / 16, 0);
  z.pw.assign(z.fl.size(), 0);
  z.xi.assign(static_cast<size_t>(c.T) * c.K, 0);
  for (int t = 0; t < c.T; ++t) {
    const uint16_t* r = c.x.data() + static_cast<int64_t>(t) * c.K;
    for (int j = 0; j < c.K / 2; ++j) {
      z.xe[t * c.K / 2 + j] = bf16_to_f32(r[2 * j]);
      z.xo[t * c.K / 2 + j] = bf16_to_f32(r[2 * j + 1]);
    }
    for (int b = 0; b < c.K / 16; ++b) {
      const size_t i = static_cast<size_t>(t) * c.K / 16 + b;
      exact_input_block(
          r + 16 * b, &z.fl[i], &z.xi[t * c.K + 16 * b], &z.pw[i]);
    }
  }
  return z;
}

uint16_t bf16_of(double v) {  // v is a bf16 value
  return f32_to_bf16(static_cast<float>(v));
}

// x of a row: each value sign * 2^e * (1 + m/128), e in [elo, ehi]
uint16_t rand_wide(std::mt19937& g, int elo, int ehi) {
  const int e = elo + static_cast<int>(g() % (ehi - elo + 1));
  return bf16_of(std::ldexp(1.0 + (g() % 128) / 128.0, e) * (g() & 1 ? -1 : 1));
}

RowsCase make_rows(
    int K,
    int G,
    int N,
    int T,
    uint32_t seed,
    int xelo,
    int xehi,
    int selo,
    int sehi) {
  std::mt19937 g(seed);
  RowsCase c{K, G, N, T, {}, {}, {}};
  c.W.resize(static_cast<size_t>(N) * K / 2);
  for (auto& b : c.W)
    b = static_cast<uint8_t>(g());
  c.S.resize(static_cast<size_t>(N) * K / G);
  for (auto& v : c.S)
    v = rand_wide(g, selo, sehi);
  c.x.resize(static_cast<size_t>(T) * K);
  for (auto& v : c.x)
    v = g() % 11 == 0 ? 0 : rand_wide(g, xelo, xehi);
  return c;
}

// rows i and i+1 of the case: acc set by block 0 to a random fp32-ish value
// a (w = 1: q = 1, s = 1), block 1 adds half an fp32 ulp of a plus a tiny
// remainder of either sign 30 exponents lower (inside the window), so
// a + B rounds in double onto a midpoint and the remainder decides.
// gap 60: the remainder is beyond the window (the block is flagged), and a
// double sum of the block would already lose it.
void force_midpoints(RowsCase& c, std::mt19937& g, int gap = 30) {
  const uint16_t one = bf16_of(1.0);
  for (int n = 0; n < c.N; ++n) {
    uint8_t* w = c.W.data() + static_cast<int64_t>(n) * c.K / 2;
    for (int j = 0; j < 16; ++j)
      w[j] = 0;   // blocks 0, 1: q = 0 but where set below
    w[0] = 0x01;  // k = 0: q = 1
    w[8] = 0x11;  // k = 16, 17: q = 1
    for (int gi = 0; gi < 16 / c.G + 1 && gi < c.K / c.G; ++gi)
      c.S[static_cast<int64_t>(n) * (c.K / c.G) + gi] = one;
  }
  for (int t = 0; t < c.T; ++t) {
    uint16_t* r = c.x.data() + static_cast<int64_t>(t) * c.K;
    const int ea = static_cast<int>(g() % 60) - 30;
    const double a =
        std::ldexp(1.0 + (g() % 128) / 128.0, ea) * (g() & 1 ? -1 : 1);
    for (int k = 0; k < 32; ++k)
      r[k] = 0;
    r[0] = bf16_of(a);
    r[16] = bf16_of(std::ldexp(a < 0 ? -1.0 : 1.0, ea - 24));
    r[17] = bf16_of(std::ldexp(g() & 1 ? -1.0 : 1.0, ea - 24 - gap));
  }
}

void check_rows(
    const char* what, const RowsCase& c, const std::vector<int>& levels) {
  const Scratch z = scratch_of(c);
  int64_t flagged = 0, no_int = 0;
  for (uint8_t f : z.fl) {
    flagged += f & 1;
    no_int += (f >> 1) & 1;
  }
  // reference
  std::vector<float> want(static_cast<size_t>(c.T) * c.N);
  for (int t = 0; t < c.T; ++t) {
    std::vector<double> x(c.K);
    for (int k = 0; k < c.K; ++k)
      x[k] = bf16_to_f32(c.x[static_cast<int64_t>(t) * c.K + k]);
    for (int n = 0; n < c.N; ++n)
      want[static_cast<size_t>(t) * c.N + n] = ref_dot(
          c.W.data() + static_cast<int64_t>(n) * c.K / 2,
          c.S.data() + static_cast<int64_t>(n) * (c.K / c.G),
          c.K,
          c.G,
          x);
  }
  std::vector<int> all = {1};
  all.insert(all.end(), levels.begin(), levels.end());
  // every level with the int16 forms and without (double sums only)
  for (int run = 0; run < 2 * static_cast<int>(all.size()); ++run) {
    const int l = all[run / 2];
    const bool ints = run % 2 == 0;
    const ExactInputs in{
        z.xe.data(),
        z.xo.data(),
        z.fl.data(),
        ints ? z.xi.data() : nullptr,
        ints ? z.pw.data() : nullptr};
    std::vector<float> got(want.size(), -1.0f);
    // split the rows the way run_calls' items do (16) plus a ragged tail
    for (int n0 = 0; n0 < c.N; n0 += 16)
      rows_exact(
          l,
          c.W.data(),
          c.S.data(),
          c.K,
          c.G,
          n0,
          std::min(c.N, n0 + 16),
          in,
          c.T,
          got.data(),
          c.N);
    int64_t differ = 0;
    for (size_t i = 0; i < want.size(); ++i)
      if (!same(got[i], want[i])) {
        if (differ < 3)
          std::printf(
              "    %s level %d%s: t %zu n %zu got %a want %a\n",
              what,
              l,
              ints ? "" : " (doubles only)",
              i / c.N,
              i % c.N,
              got[i],
              want[i]);
        ++differ;
      }
    ++checks;
    if (differ) {
      std::printf(
          "  FAIL %-32s level %d (%s)%s: %lld / %zu differ\n",
          what,
          l,
          level_name(l),
          ints ? "" : " doubles only",
          (long long)differ,
          want.size());
      ++failures;
    }
  }
  std::printf(
      "  %-34s %zu values, %zu input blocks: %lld flagged, %lld without "
      "an int16 form: done\n",
      what,
      want.size(),
      z.fl.size(),
      (long long)flagged,
      (long long)no_int);
}

// --------------------------------------------------------------------------
// 3. the op
// --------------------------------------------------------------------------

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
    bool shared,
    uint32_t seed,
    float xscale = 1.0f) {
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
    v = rand_bf16(g, xscale);
  std::uniform_real_distribution<float> wd(0.01f, 0.3f);
  std::vector<int> first;
  for (int t = 0; t < T; ++t) {
    std::vector<int> used;
    for (int k = 0; k < topk; ++k) {
      int u;
      if (shared && t > 0) {
        u = first[k];
      } else {
        do {
          u = static_cast<int>(g() % U);
        } while (std::find(used.begin(), used.end(), u) != used.end());
        if (t == 0) first.push_back(u);
      }
      used.push_back(u);
      p.cu.push_back(u);
      p.ct.push_back(t);
      p.cw.push_back(wd(g));
    }
  }
  return p;
}

// A table standing in for the card's: the host formula, then every 97th
// entry moved one bf16 step (so a wrong index shows).
std::vector<uint16_t> test_table() {
  std::vector<uint16_t> t(65536);
  for (int i = 0; i < 65536; ++i) {
    const float g = bf16_to_f32(static_cast<uint16_t>(i));
    uint16_t v = f32_to_bf16(g / (1.0f + std::exp(-g)));
    if (i % 97 == 0 && (v & 0x7f80) != 0x7f80) v = static_cast<uint16_t>(v + 1);
    t[i] = v;
  }
  return t;
}

std::vector<uint16_t>
run(const Problem& p,
    Pool& pool,
    int level,
    const std::vector<uint16_t>& table,
    bool inplace = false) {
  const int64_t C = static_cast<int64_t>(p.cu.size());
  if (!inplace) {
    std::vector<uint16_t> out(static_cast<size_t>(p.T) * p.L.hidden);
    run_calls(
        pool,
        static_cast<Isa>(level),
        p.L,
        out.data(),
        p.x.data(),
        p.T,
        p.units.data(),
        p.U,
        p.cu.data(),
        p.ct.data(),
        p.cw.data(),
        C,
        kArithCardExact,
        nullptr,
        0,
        table.data());
    return out;
  }
  // one output row per call, inputs read in place; summed back per token
  std::vector<int32_t> iota(C);
  std::vector<float> ones(C, 1.0f);
  for (int64_t c = 0; c < C; ++c)
    iota[c] = static_cast<int32_t>(c);
  std::vector<uint16_t> rows(static_cast<size_t>(C) * p.L.hidden);
  run_calls(
      pool,
      static_cast<Isa>(level),
      p.L,
      rows.data(),
      p.x.data(),
      C,
      p.units.data(),
      p.U,
      p.cu.data(),
      iota.data(),
      ones.data(),
      C,
      kArithCardExact,
      p.ct.data(),
      p.T,
      table.data());
  // the gather as run_calls does it, from the per-call rows
  std::vector<uint16_t> out(static_cast<size_t>(p.T) * p.L.hidden, 0);
  const int H = p.L.hidden;
  int64_t c = 0;
  while (c < C) {
    const int t = p.ct[c];
    std::vector<float> acc(H, 0.f);
    for (; c < C && p.ct[c] == t; ++c)
      for (int j = 0; j < H; ++j)
        acc[j] = acc[j] + bf16_to_f32(rows[c * H + j]) * p.cw[c];
    for (int j = 0; j < H; ++j)
      out[static_cast<int64_t>(t) * H + j] = f32_to_bf16(acc[j]);
  }
  return out;
}

// The card's model of the layer call, from the reference chain.
std::vector<uint16_t>
ref_model(const Problem& p, const std::vector<uint16_t>& table) {
  const int H = p.L.hidden, I = p.L.inter, G = p.L.group;
  std::vector<uint16_t> out(static_cast<size_t>(p.T) * H, 0);
  int64_t c = 0;
  const int64_t C = static_cast<int64_t>(p.cu.size());
  while (c < C) {
    const int t = p.ct[c];
    std::vector<float> acc(H, 0.f);
    for (; c < C && p.ct[c] == t; ++c) {
      const uint8_t* u =
          p.units.data() + static_cast<int64_t>(p.cu[c]) * p.L.unit_bytes;
      const uint8_t* w13 = u + p.L.off_w13;
      const uint8_t* w2 = u + p.L.off_w2;
      const auto* s13 = reinterpret_cast<const uint16_t*>(u + p.L.off_s13);
      const auto* s2 = reinterpret_cast<const uint16_t*>(u + p.L.off_s2);
      std::vector<double> x(H), a(I);
      for (int k = 0; k < H; ++k)
        x[k] = bf16_to_f32(p.x[static_cast<int64_t>(t) * H + k]);
      for (int i = 0; i < I; ++i) {
        const uint16_t gb = f32_to_bf16(ref_dot(
            w13 + static_cast<int64_t>(i) * H / 2,
            s13 + static_cast<int64_t>(i) * (H / G),
            H,
            G,
            x));
        const float up = bf16_to_f32(f32_to_bf16(ref_dot(
            w13 + static_cast<int64_t>(I + i) * H / 2,
            s13 + static_cast<int64_t>(I + i) * (H / G),
            H,
            G,
            x)));
        a[i] = bf16_to_f32(f32_to_bf16(bf16_to_f32(table[gb]) * up));
      }
      for (int j = 0; j < H; ++j) {
        const float y = bf16_to_f32(f32_to_bf16(ref_dot(
            w2 + static_cast<int64_t>(j) * I / 2,
            s2 + static_cast<int64_t>(j) * (I / G),
            I,
            G,
            a)));
        acc[j] = acc[j] + y * p.cw[c];
      }
    }
    for (int j = 0; j < H; ++j)
      out[static_cast<int64_t>(t) * H + j] = f32_to_bf16(acc[j]);
  }
  return out;
}

int64_t ndiff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  int64_t d = 0;
  for (size_t i = 0; i < a.size(); ++i)
    d += a[i] != b[i];
  return d;
}

int work(int argc, char** argv) {
  if (argc < 6) return 2;
  const int level = std::atoi(argv[2]);
  const int arith = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]), E = std::atoi(argv[5]);
  const int reps = argc > 6 ? std::atoi(argv[6]) : 1;
  Problem p = make(2560, 640, 128, T, E, E, true, 7);
  const auto table = test_table();
  Pool one(1);
  std::vector<uint16_t> out(static_cast<size_t>(p.T) * p.L.hidden);
  for (int r = 0; r < reps; ++r)
    run_calls(
        one,
        static_cast<Isa>(level),
        p.L,
        out.data(),
        p.x.data(),
        p.T,
        p.units.data(),
        p.U,
        p.cu.data(),
        p.ct.data(),
        p.cw.data(),
        static_cast<int64_t>(p.cu.size()),
        arith,
        nullptr,
        0,
        table.data());
  uint32_t h = 2166136261u;
  for (uint16_t v : out)
    h = (h ^ v) * 16777619u;
  const ExactCounters n = exact_counters();
  std::printf(
      "work level %d arith %d tokens %d experts %d: fnv %08x (slow blocks "
      "%lld, "
      "fixed lanes %lld)\n",
      level,
      arith,
      T,
      E,
      h,
      (long long)n.wide,
      (long long)n.fixed);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "work") == 0) return work(argc, argv);
  const int threads = argc > 1 ? std::atoi(argv[1]) : 4;
  const int best = static_cast<int>(resolve_isa(Isa::kAuto));
  std::printf("auto resolves to level %d (%s)\n", best, level_name(best));
  std::vector<int> levels;
  for (int l = 2; l <= kIsaMax; ++l)
    if (level_supported(l)) levels.push_back(l);
  std::printf("levels present:");
  for (int l : levels)
    std::printf(" %d", l);
  std::printf("\n");
  const ExactCounters before = exact_counters();

  check_primitives();

  std::printf("-- rows_exact, every level vs the __float128 reference\n");
  {
    // QFN's K and group, normal inputs (all inside the window)
    check_rows(
        "normal inputs, K 2560",
        make_rows(2560, 128, 20, 3, 1, -8, 2, -9, -5),
        levels);
    check_rows(
        "normal inputs, K 640, 1 token",
        make_rows(640, 128, 36, 1, 2, -8, 2, -9, -5),
        levels);
    // spans 40..70: most blocks beyond the window (exact integer sums)
    check_rows(
        "wide spans (x 2^-60..2^10)",
        make_rows(640, 128, 20, 5, 3, -60, 10, -10, 5),
        levels);
    check_rows(
        "wide spans, group 32",
        make_rows(256, 32, 12, 7, 4, -60, 10, -10, 5),
        levels);
    // spans up to 6: int16 forms (vpmaddwd) for most blocks; then the same
    // inputs with scales below the int16 tables' range (double sums)
    check_rows(
        "narrow spans (int16 forms)",
        make_rows(640, 128, 20, 5, 11, -3, 3, -9, 5),
        levels);
    check_rows(
        "narrow spans, group 32, 4 tokens",
        make_rows(256, 32, 12, 4, 12, -3, 3, -9, 5),
        levels);
    check_rows(
        "narrow spans, tiny scales",
        make_rows(256, 128, 8, 3, 13, -3, 3, -125, -120),
        levels);
    // spans up to 30: the double sums are exact, larger cancellation
    check_rows(
        "spans up to 30", make_rows(512, 64, 16, 4, 5, -25, 5, -10, 5), levels);
    // forced midpoints in double: the exact sum decides
    {
      RowsCase c = make_rows(512, 128, 20, 8, 6, -8, 2, -9, -5);
      std::mt19937 g(66);
      force_midpoints(c, g);
      check_rows("forced midpoints (acc + B in double)", c, levels);
      RowsCase w = make_rows(512, 128, 20, 8, 16, -8, 2, -9, -5);
      force_midpoints(w, g, 60);
      check_rows("forced midpoints, remainder 60 below", w, levels);
    }
    // fp32 subnormal accumulators: tiny inputs and scales
    check_rows(
        "subnormal accumulators",
        make_rows(256, 128, 8, 3, 8, -133, -120, -30, -20),
        levels);
    // overflow to inf, then inf + finite
    check_rows(
        "overflowing accumulators",
        make_rows(256, 128, 8, 3, 9, 90, 120, 20, 30),
        levels);
    // non-finite inputs: inf and NaN in x
    {
      RowsCase c = make_rows(256, 128, 8, 4, 10, -8, 2, -9, -5);
      c.x[5] = 0x7f80;          // +inf
      c.x[256 + 40] = 0xff80;   // -inf
      c.x[256 + 41] = 0x7f80;   // +inf in the same block
      c.x[512 + 100] = 0x7fc1;  // NaN
      check_rows("non-finite inputs", c, levels);
    }
  }

  std::printf(
      "-- the op (arith 7), every level vs the reference model and scalar\n");
  const auto table = test_table();
  Pool one(1), many(threads);
  struct Case {
    int H, I, G, T, U, topk;
    bool shared;
    uint32_t seed;
    float xscale;
    const char* what;
  };
  const Case cases[] = {
      {2560, 640, 128, 1, 10, 10, false, 1, 1.0f, "QFN decode, top-10"},
      {2560, 640, 128, 3, 12, 10, false, 2, 1.0f, "QFN 3 tokens"},
      {2560, 640, 128, 9, 4, 2, true, 3, 1.0f, "QFN 9 rows per expert"},
      {256, 128, 32, 5, 3, 2, false, 4, 1.0f, "group 32"},
      {512, 256, 64, 7, 2, 2, false, 5, 1.0f, "group 64"},
      {512, 256, 128, 40, 3, 2, true, 6, 1.0f, "40 rows per expert"},
      {512, 256, 128, 6, 3, 2, false, 7, 0x1p-100f, "tiny inputs"},
  };
  for (const Case& cs : cases) {
    Problem p = make(
        cs.H, cs.I, cs.G, cs.T, cs.U, cs.topk, cs.shared, cs.seed, cs.xscale);
    const auto want = ref_model(p, table);
    const auto scalar = run(p, one, 1, table);
    const int64_t dref = ndiff(scalar, want);
    ++checks;
    if (dref) {
      std::printf(
          "  FAIL %-26s scalar vs model: %lld / %zu differ\n",
          cs.what,
          (long long)dref,
          want.size());
      ++failures;
    }
    if (ndiff(run(p, many, 1, table), scalar) ||
        ndiff(run(p, many, 1, table, true), scalar)) {
      std::printf(
          "  FAIL %-26s scalar: threads or in place change the result\n",
          cs.what);
      ++failures;
    }
    for (int l : levels) {
      const int64_t a = ndiff(run(p, one, l, table), scalar);
      const int64_t b = ndiff(run(p, many, l, table), scalar);
      const int64_t c = ndiff(run(p, many, l, table, true), scalar);
      ++checks;
      if (a || b || c) {
        std::printf(
            "  FAIL %-26s level %d (%s): %lld / %lld / %lld differ from scalar "
            "(1 thread, %d threads, in place)\n",
            cs.what,
            l,
            level_name(l),
            (long long)a,
            (long long)b,
            (long long)c,
            threads);
        ++failures;
      }
    }
    std::printf(
        "  %-26s %zu values, model = scalar = every level: %s\n",
        cs.what,
        want.size(),
        dref ? "NO" : "yes");
  }
  // the refusal without a table
  {
    Problem p = make(256, 128, 32, 1, 2, 1, false, 9);
    std::vector<uint16_t> out(p.L.hidden);
    bool refused = false;
    try {
      run_calls(
          one,
          Isa::kAuto,
          p.L,
          out.data(),
          p.x.data(),
          1,
          p.units.data(),
          p.U,
          p.cu.data(),
          p.ct.data(),
          p.cw.data(),
          1,
          kArithCardExact);
    } catch (const std::runtime_error&) {
      refused = true;
    }
    ++checks;
    if (!refused) {
      std::printf("  FAIL arith 7 without a table was not refused\n");
      ++failures;
    }
  }
  const ExactCounters after = exact_counters();
  const int64_t wide = after.wide - before.wide,
                fixed = after.fixed - before.fixed;
  std::printf(
      "-- slow paths taken: %lld blocks summed in the exact integer, %lld "
      "lanes rounded by round32_sum (midpoints, subnormal range)\n",
      (long long)wide,
      (long long)fixed);
  ++checks;
  if (wide == 0 || (fixed == 0 && !levels.empty())) {
    std::printf("  FAIL: a slow path was never taken\n");
    ++failures;
  }
  std::printf(
      "%s: %d checks, %d failures, levels checked: 1",
      failures ? "EXACT SELFTEST FAILED" : "EXACT SELFTEST PASSED",
      checks,
      failures);
  for (int l : levels)
    std::printf(" %d", l);
  std::printf("\n");
  return failures ? 1 : 0;
}
