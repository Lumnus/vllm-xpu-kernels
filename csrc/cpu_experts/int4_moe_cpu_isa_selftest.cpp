// Standalone check of every dispatch level of the host MoE ops against
// their scalar code (no torch; c10's header-only types for the W4A8 op's
// activation quantization, -I <torch>/include). On a host without a level,
// run it under Intel SDE with that CPU (sde64 -spr -- ./a.out): the levels
// the (emulated) CPU has are checked, the others are listed as absent.
//
//   1. which level each request resolves to, and the names;
//   2. W4A16, arith 0..3, every level, 1 and N threads: bit-identical to the
//      scalar code (level 8's card-class AMX-BF16 prefill: the differences
//      are counted and bounded, not required to be zero);
//   3. the AVX-512 BF16 guard: subnormal and tiny inputs take the AVX-512 F
//      code (the call still bit-identical, the fallback counted);
//   4. W4A8, four schemes, every level, 1 and N threads: bit-identical;
//   5. W4A16 with the inputs read in place (call_x_row, one output row per
//      call): every level equals its own gathered-input run, and every
//      level but 8 equals the scalar code's.
//
// Build: g++ -O2 -std=c++20 -pthread -ffp-contract=off -I csrc
//        -I <torch>/include int4_moe_cpu_isa_selftest.cpp
// Run:   ./a.out [threads]          checks, exit status 0 when all pass
//        ./a.out work LEVEL ARITH|a8 TOKENS EXPERTS [REPS]   REPS calls,
//                                   1 thread (instruction counts: sde -mix)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include "cpu_experts/int4_moe_cpu_a8_impl.h"

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

// tokens T, each through topk of U experts (all tokens through the same
// experts when shared: many rows per expert)
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

std::vector<uint16_t>
run16(const Problem& p, Pool& pool, int level, int arith) {
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
      static_cast<int64_t>(p.cu.size()),
      arith);
  return out;
}

// One output row per call (call_token = 0..C-1), the inputs either read in
// place from x by call_x_row or gathered into a copy first.
std::vector<uint16_t>
run16_rows(const Problem& p, Pool& pool, int level, int arith, bool inplace) {
  const int64_t C = static_cast<int64_t>(p.cu.size());
  const int H = p.L.hidden;
  std::vector<int32_t> iota(C);
  for (int64_t c = 0; c < C; ++c)
    iota[c] = static_cast<int32_t>(c);
  std::vector<uint16_t> xg;
  if (!inplace) {
    xg.resize(static_cast<size_t>(C) * H);
    for (int64_t c = 0; c < C; ++c)
      std::memcpy(
          xg.data() + c * H,
          p.x.data() + static_cast<int64_t>(p.ct[c]) * H,
          H * sizeof(uint16_t));
  }
  std::vector<uint16_t> out(static_cast<size_t>(C) * H);
  run_calls(
      pool,
      static_cast<Isa>(level),
      p.L,
      out.data(),
      inplace ? p.x.data() : xg.data(),
      C,
      p.units.data(),
      p.U,
      p.cu.data(),
      iota.data(),
      p.cw.data(),
      C,
      arith,
      inplace ? p.ct.data() : nullptr,
      inplace ? p.T : 0);
  return out;
}

std::vector<uint16_t>
run8(const Problem& p, Pool& pool, int level, const a8::Scheme& sc) {
  std::vector<uint16_t> out(static_cast<size_t>(p.T) * p.L.hidden);
  a8::run_calls(
      pool,
      static_cast<Isa>(level),
      p.L,
      sc,
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

// differing values, and the largest difference in bf16 steps at the
// output's scale (the largest |value| of the reference)
struct Diff {
  int64_t differ = 0;
  double steps = 0;
};

Diff compare(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  Diff d;
  double scale = 0, worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    scale = std::max(scale, std::fabs(double(bf16_to_f32(b[i]))));
    if (a[i] != b[i]) {
      ++d.differ;
      worst = std::max(
          worst, std::fabs(double(bf16_to_f32(a[i])) - bf16_to_f32(b[i])));
    }
  }
  // one bf16 step at the output's scale: 2^(floor(log2 scale) - 7)
  const double step = scale > 0 ? std::ldexp(1.0, std::ilogb(scale) - 7) : 1;
  d.steps = worst / step;
  return d;
}

int work(int argc, char** argv) {
  if (argc < 6) return 2;
  const int level = std::atoi(argv[2]);
  const bool a8m = std::strcmp(argv[3], "a8") == 0;
  const int arith = a8m ? 0 : std::atoi(argv[3]);
  const int T = std::atoi(argv[4]), E = std::atoi(argv[5]);
  const int reps = argc > 6 ? std::atoi(argv[6]) : 1;
  // QFN's shapes, every token through the same E experts
  Problem p = make(2560, 640, 128, T, E, E, true, 7);
  Pool one(1);
  std::vector<uint16_t> out;
  for (int r = 0; r < reps; ++r)
    out = a8m ? run8(p, one, level, a8::Scheme{}) : run16(p, one, level, arith);
  uint32_t h = 2166136261u;
  for (uint16_t v : out)
    h = (h ^ v) * 16777619u;
  std::printf(
      "work level %d (%s) %s tokens %d experts %d: fnv %08x\n",
      level,
      level_name(level),
      a8m ? "a8" : ("arith " + std::to_string(arith)).c_str(),
      T,
      E,
      h);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "work") == 0) return work(argc, argv);
  const int threads = argc > 1 ? std::atoi(argv[1]) : 4;
  const CpuFeatures& f = cpu_features();
  std::printf(
      "cpu: avx2 %d fma %d avxvnni %d avx512 f/bw/dq/vl %d%d%d%d vnni %d "
      "bf16 %d vbmi %d amx tile/int8/bf16 %d%d%d os avx/avx512/amx %d%d%d "
      "amx granted %d\n",
      f.avx2,
      f.fma,
      f.avxvnni,
      f.avx512f,
      f.avx512bw,
      f.avx512dq,
      f.avx512vl,
      f.avx512vnni,
      f.avx512bf16,
      f.avx512vbmi,
      f.amx_tile,
      f.amx_int8,
      f.amx_bf16,
      f.os_avx,
      f.os_avx512,
      f.os_amx,
      f.amx_granted);
  const int best = static_cast<int>(resolve_isa(Isa::kAuto));
  std::printf("auto resolves to level %d (%s)\n", best, level_name(best));
  std::vector<int> levels;
  for (int l = 2; l <= kIsaMax; ++l) {
    const bool ok = level_supported(l);
    std::printf("level %d %-42s %s\n", l, level_name(l), ok ? "yes" : "absent");
    if (ok) levels.push_back(l);
  }
#if INT4_MOE_CPU_X86
  if (level_supported(6))
    std::printf(
        "vdpbf16ps probe (two rounded fused multiply-adds, high word "
        "first): %s\n",
        bf16_body_ok() ? "yes" : "NO: the BF16 code is off");
#endif
  Pool one(1), many(threads);
  int failures = 0, checks = 0;
  struct Case {
    int H, I, G, T, U, topk;
    bool shared;
    uint32_t seed;
    float xscale;
    const char* what;
  };
  const Case cases[] = {
      {2560, 640, 128, 1, 8, 8, false, 1, 1.0f, "decode, 8 experts"},
      {2560, 640, 128, 3, 6, 2, false, 2, 1.0f, "3 tokens"},
      {256, 128, 32, 5, 3, 2, false, 3, 1.0f, "group 32"},
      {256, 128, 64, 7, 2, 2, false, 4, 1.0f, "group 64"},
      {512, 256, 128, 40, 3, 2, true, 5, 1.0f, "40 rows per expert"},
      {256, 128, 128, 20, 2, 1, true, 6, 1.0f, "20 rows per expert"},
      {512, 256, 128, 18, 2, 2, true, 7, 0x1p-100f, "tiny inputs (BF16 guard)"},
  };
  std::printf("-- W4A16 (arith 0..3), every level vs scalar\n");
  for (const Case& cs : cases) {
    Problem p = make(
        cs.H, cs.I, cs.G, cs.T, cs.U, cs.topk, cs.shared, cs.seed, cs.xscale);
    for (int arith = 0; arith <= 3; ++arith) {
      const auto ref = run16(p, one, 1, arith);
      for (int l : levels) {
        const auto a = run16(p, one, l, arith);
        const auto b = run16(p, many, l, arith);
        const Diff da = compare(a, ref), db = compare(b, ref);
        ++checks;
        // level 8 at 16+ tokens per expert runs the card class for the
        // grouped GEMM's arithmetic: counted, bounded (2 bf16 steps)
        const bool cls = l == 8 && arith != 0 && cs.shared && cs.T >= 16;
        if (cls) {
          std::printf(
              "  %-26s arith %d level 8: card class, %lld / %lld values "
              "differ from scalar, largest %.2f bf16 steps\n",
              cs.what,
              arith,
              (long long)std::max(da.differ, db.differ),
              (long long)ref.size(),
              std::max(da.steps, db.steps));
          if (da.steps > 2.0 || db.steps > 2.0) ++failures;
        } else if (da.differ || db.differ) {
          std::printf(
              "  FAIL %-26s arith %d level %d (%s): %lld / %lld differ (1 "
              "thread), %lld (%d threads)\n",
              cs.what,
              arith,
              l,
              level_name(l),
              (long long)da.differ,
              (long long)ref.size(),
              (long long)db.differ,
              threads);
          ++failures;
        }
      }
    }
    std::printf("  %-26s done\n", cs.what);
  }
#if INT4_MOE_CPU_X86
  // subnormal inputs: the BF16 code must step aside (DAZ would change them)
  if (level_supported(6)) {
    Problem p = make(512, 256, 128, 4, 3, 2, false, 11);
    for (size_t i = 0; i < p.x.size(); i += 7)
      p.x[i] = static_cast<uint16_t>((p.x[i] & 0x8000) | 0x0011);  // subnormal
    const int64_t before = bf16_fallbacks().load();
    for (int arith = 1; arith <= 3; arith += 2) {
      const auto ref = run16(p, one, 1, arith);
      for (int l : levels) {
        if (l < 6) continue;
        ++checks;
        const Diff d = compare(run16(p, many, l, arith), ref);
        if (d.differ) {
          std::printf("  FAIL subnormal inputs, level %d arith %d\n", l, arith);
          ++failures;
        }
      }
    }
    const int64_t fell = bf16_fallbacks().load() - before;
    std::printf(
        "  subnormal inputs: bit-identical; %lld rows() calls took the "
        "AVX-512 F code\n",
        (long long)fell);
    if (fell == 0) {
      std::printf("  FAIL: the BF16 guard did not fire on subnormal inputs\n");
      ++failures;
    }
  }
#endif
  std::printf("-- W4A16, inputs read in place (call_x_row), every level\n");
  {
    const Case rc[] = {cases[0], cases[1], cases[4]};
    for (const Case& cs : rc) {
      Problem p = make(
          cs.H, cs.I, cs.G, cs.T, cs.U, cs.topk, cs.shared, cs.seed, cs.xscale);
      for (int arith = 0; arith <= 3; ++arith) {
        const auto ref = run16_rows(p, one, 1, arith, false);
        if (compare(run16_rows(p, one, 1, arith, true), ref).differ) {
          std::printf(
              "  FAIL %-26s arith %d scalar: in place != gathered\n",
              cs.what,
              arith);
          ++failures;
        }
        for (int l : levels) {
          const auto g = run16_rows(p, many, l, arith, false);
          const Diff dp = compare(run16_rows(p, many, l, arith, true), g);
          const Diff da = compare(run16_rows(p, one, l, arith, true), g);
          const Diff ds = compare(g, ref);
          ++checks;
          if (dp.differ || da.differ || (l != 8 && ds.differ)) {
            std::printf(
                "  FAIL %-26s arith %d level %d (%s): in place vs gathered "
                "%lld / %lld (%d threads), %lld (1 thread); gathered vs "
                "scalar %lld\n",
                cs.what,
                arith,
                l,
                level_name(l),
                (long long)dp.differ,
                (long long)g.size(),
                threads,
                (long long)da.differ,
                (long long)ds.differ);
            ++failures;
          }
        }
      }
      std::printf("  %-26s done\n", cs.what);
    }
  }
  std::printf("-- W4A8, every level vs scalar\n");
  struct Case8 {
    int H, I, G, T, U, topk;
    bool shared;
    uint32_t seed;
    std::vector<int64_t> scheme;
    const char* what;
  };
  const Case8 cases8[] = {
      {2560, 640, 128, 1, 8, 8, false, 21, {}, "decode, card scheme"},
      {2560, 640, 128, 3, 6, 2, false, 22, {}, "3 tokens"},
      {512, 256, 128, 40, 3, 2, true, 23, {}, "40 rows per expert"},
      {512, 256, 128, 17, 2, 2, true, 24, {1, 0, 1, 0, 1, 0}, "amax127, sep"},
      {512, 256, 128, 33, 2, 1, true, 25, {0, 64, 0, 64, 0, 0}, "act g64"},
      {512, 256, 64, 16, 2, 2, true, 26, {0, 0, 1, 128, 0, 1}, "glu fp32"},
      {256, 128, 128, 5, 3, 2, false, 27, {0, 256, 0, 0, 1, 0}, "act g256"},
  };
  for (const Case8& cs : cases8) {
    Problem p = make(cs.H, cs.I, cs.G, cs.T, cs.U, cs.topk, cs.shared, cs.seed);
    const a8::Scheme sc = a8::scheme_from(cs.scheme.data(), cs.scheme.size());
    const auto ref = run8(p, one, 1, sc);
    for (int l : levels) {
      const Diff da = compare(run8(p, one, l, sc), ref);
      const Diff db = compare(run8(p, many, l, sc), ref);
      ++checks;
      if (da.differ || db.differ) {
        std::printf(
            "  FAIL %-22s level %d (%s): %lld / %lld differ (1 thread), "
            "%lld (%d threads)\n",
            cs.what,
            l,
            level_name(l),
            (long long)da.differ,
            (long long)ref.size(),
            (long long)db.differ,
            threads);
        ++failures;
      }
    }
    std::printf("  %-22s done\n", cs.what);
  }
  std::printf(
      "%s: %d checks, %d failures, levels checked:",
      failures ? "ISA SELFTEST FAILED" : "ISA SELFTEST PASSED",
      checks,
      failures);
  for (int l : levels)
    std::printf(" %d", l);
  std::printf("\n");
  return failures ? 1 : 0;
}
