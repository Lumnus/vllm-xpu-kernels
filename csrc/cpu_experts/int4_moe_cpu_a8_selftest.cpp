// Standalone self-test of the W4A8 host experts (int4_moe_cpu_a8_impl.h)
// against an emulation of the card's W4A8 arithmetic written element by
// element from the device sources:
//   activation quant   csrc/quantization/int8/int8_quant_math.h (the same
//                      header the device kernel compiles)
//   GEMM               csrc/xpu/grouped_gemm/xe_2_w4a8/gemm_w4a8_xe2.hpp:
//                      int32 per 128-K group, F += float(s32) * w_scale as a
//                      fused multiply-add (the compiled kernel's `mad`), then
//                      F * a_scale, then bf16
//   fused gemm1 (off in serving) the same file's xe_gemm_w4a8_glu epilogue
//   silu_and_mul       csrc/activation.cpp silu_kernel, act_and_mul_kernel
//   gather             csrc/moe/moe_gather.cpp (`mad` in top-k order)
// Needs only c10's header-only types (-I <torch>/include). No device.
//
// Usage: int4_moe_cpu_a8_selftest [threads] [isa: 0 auto, 1 scalar, 2 avx2]
// Exit status 0 when every exact case is bit-identical.

#include <cstdio>
#include <cstdlib>
#include <random>

#include "cpu_experts/int4_moe_cpu_a8_impl.h"

using namespace cpu_moe;
namespace a8 = cpu_moe::a8;
namespace iq = vllm::int8q;

namespace {

struct Model {
  int H, I, G;
  Layout L;
  std::vector<uint8_t> units;
  int64_t S;
};

Model make_model(int H, int I, int G, int S, std::mt19937& rng, int wfill) {
  Model m{H, I, G, {}, {}, S};
  const int64_t w13 = int64_t(2) * I * H / 2, w2 = int64_t(H) * I / 2;
  const int64_t s13 = int64_t(2) * I * (H / G) * 2,
                s2 = int64_t(H) * (I / G) * 2;
  m.L = Layout{H, I, G, 0, 0, w13, w13 + w2, w13 + w2 + s13};
  m.L.unit_bytes = ((w13 + w2 + s13 + s2 + 4095) / 4096) * 4096;
  m.units.assign(size_t(S) * m.L.unit_bytes, 0);
  std::uniform_int_distribution<int> byte(0, 255);
  std::normal_distribution<float> nd(0.0f, 0.01f);
  for (int s = 0; s < S; ++s) {
    uint8_t* u = m.units.data() + size_t(s) * m.L.unit_bytes;
    for (int64_t j = 0; j < w13 + w2; ++j)
      u[j] = wfill >= 0 ? uint8_t(wfill) : uint8_t(byte(rng));
    uint16_t* sc = reinterpret_cast<uint16_t*>(u + m.L.off_s13);
    for (int64_t j = 0; j < (s13 + s2) / 2; ++j)
      sc[j] = f32_to_bf16(wfill >= 0 ? 0.01f : nd(rng));
  }
  return m;
}

// --- the card, element by element ----------------------------------------

struct Card {
  bool fused_acc = true;   // the kernel's mad
  double exp_scale = 1.0;  // models a hardware exp off by a relative amount
  bool glu = false;        // fused gemm1 epilogue (AG 128 for gemm2)
};

int w4(const uint8_t* W, int K, int n, int k) {
  return nibble(W[int64_t(n) * (K / 2) + k / 2], k & 1);
}

// D[n] for one row of int8 a (scale per row as0, or per AG group asg).
void card_gemm(
    const Card& c,
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int N,
    int G,
    const int8_t* q,
    float as0,
    const float* asg,
    int AG,
    bool rnd,
    float* out) {
  const int FL = (AG && AG < G) ? AG : G;
  for (int n = 0; n < N; ++n) {
    float F = 0.0f;
    for (int k0 = 0; k0 < K; k0 += FL) {
      int32_t s32 = 0;
      for (int k = k0; k < k0 + FL; ++k)
        s32 += int32_t(q[k]) * w4(W, K, n, k);
      float m = bf16_to_f32(S[int64_t(n) * (K / G) + k0 / G]);
      if (AG) m = m * asg[k0 / AG];
      F = c.fused_acc ? std::fma(float(s32), m, F) : F + float(s32) * m;
    }
    if (!AG) F = F * as0;
    out[n] = rnd ? round_bf16(F) : F;
  }
}

float card_exp(const Card& c, float x) {
  return float(double(a8::exp_f32(x)) * c.exp_scale);
}

void card_quant_bf16(const float* v, int K, int8_t* q, float* s) {
  float amax = 0.0f;
  for (int k = 0; k < K; ++k)
    amax = iq::max_f(amax, iq::abs_f(v[k]));
  *s = iq::token_scale_from_absmax<c10::BFloat16>(amax);
  for (int k = 0; k < K; ++k)
    q[k] = iq::quantize_one<c10::BFloat16>(v[k], *s);
}

// One expert applied to one token, the card's way: y [H].
void card_expert(
    const Card& c, const Model& m, int unit, const uint16_t* xt, float* y) {
  const int H = m.H, I = m.I, G = m.G;
  const uint8_t* u = m.units.data() + size_t(unit) * m.L.unit_bytes;
  const uint8_t* w13 = u + m.L.off_w13;
  const uint8_t* w2 = u + m.L.off_w2;
  const uint16_t* s13 = reinterpret_cast<const uint16_t*>(u + m.L.off_s13);
  const uint16_t* s2 = reinterpret_cast<const uint16_t*>(u + m.L.off_s2);
  std::vector<float> xv(H), h(2 * I), act(I);
  for (int k = 0; k < H; ++k)
    xv[k] = bf16_to_f32(xt[k]);
  std::vector<int8_t> q1(H), q2(I);
  float s1;
  card_quant_bf16(xv.data(), H, q1.data(), &s1);
  if (!c.glu) {
    card_gemm(
        c, w13, s13, H, 2 * I, G, q1.data(), s1, nullptr, 0, true, h.data());
    for (int i = 0; i < I; ++i) {
      const float g = h[i];
      const float sl = round_bf16(g / (1.0f + card_exp(c, -g)));
      act[i] = round_bf16(sl * h[I + i]);
    }
    float s2a;
    card_quant_bf16(act.data(), I, q2.data(), &s2a);
    card_gemm(c, w2, s2, I, H, G, q2.data(), s2a, nullptr, 0, true, y);
  } else {
    // xe_gemm_w4a8_glu: g = Fg * as, u = Fu * as, v = g / (1 + exp(-g)) * u
    // in fp32; per (row, 128 columns): scale = mx * (1/127), q = rint(v *
    // 127/mx) in +-127; gemm2 with AG = 128.
    std::vector<float> F(2 * I);
    card_gemm(
        c, w13, s13, H, 2 * I, G, q1.data(), 1.0f, nullptr, 0, false, F.data());
    for (int i = 0; i < I; ++i) {
      const float g = F[i] * s1, uu = F[I + i] * s1;
      act[i] = g / (1.0f + card_exp(c, -g)) * uu;
    }
    std::vector<float> qs(I / 128);
    for (int a = 0; a < I / 128; ++a) {
      float mx = 0.0f;
      for (int i = a * 128; i < a * 128 + 128; ++i)
        mx = std::fmax(mx, std::fabs(act[i]));
      qs[a] = mx * (1.0f / 127.0f);
      const float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
      for (int i = a * 128; i < a * 128 + 128; ++i) {
        float qq = std::nearbyint(act[i] * inv);
        q2[i] = int8_t(std::fmin(127.0f, std::fmax(-127.0f, qq)));
      }
    }
    card_gemm(c, w2, s2, I, H, G, q2.data(), 0.0f, qs.data(), 128, true, y);
  }
}

void card_calls(
    const Card& c,
    const Model& m,
    const std::vector<uint16_t>& x,
    const std::vector<int32_t>& cu,
    const std::vector<int32_t>& ct,
    const std::vector<float>& cw,
    int64_t T,
    std::vector<uint16_t>& out) {
  const int H = m.H;
  out.assign(size_t(T) * H, 0);
  std::vector<float> acc(H), y(H);
  size_t i = 0;
  while (i < cu.size()) {
    const int t = ct[i];
    std::fill(acc.begin(), acc.end(), 0.0f);
    for (; i < cu.size() && ct[i] == t; ++i) {
      card_expert(c, m, cu[i], x.data() + size_t(t) * H, y.data());
      for (int j = 0; j < H; ++j)
        acc[j] = std::fma(y[j], cw[i], acc[j]);
    }
    for (int j = 0; j < H; ++j)
      out[size_t(t) * H + j] = f32_to_bf16(acc[j]);
  }
}

struct Diff {
  int64_t n = 0, differ = 0;
  double max_steps = 0.0;
};

Diff compare(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  Diff d;
  float ref = 0.0f;
  for (uint16_t v : b)
    ref = std::fmax(ref, std::fabs(bf16_to_f32(v)));
  const double spacing = ref > 0 ? std::ldexp(1.0, std::ilogb(ref) - 7) : 1.0;
  for (size_t i = 0; i < a.size(); ++i) {
    ++d.n;
    if (a[i] != b[i]) {
      ++d.differ;
      d.max_steps = std::fmax(
          d.max_steps,
          std::fabs(bf16_to_f32(a[i]) - bf16_to_f32(b[i])) / spacing);
    }
  }
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  const int threads = argc > 1 ? std::atoi(argv[1]) : 4;
  const Isa want = argc > 2 ? static_cast<Isa>(std::atoi(argv[2])) : Isa::kAuto;
  const Isa isa = resolve_isa(want);
  std::printf(
      "isa %s, %d threads\n", isa == Isa::kAvx2 ? "avx2" : "scalar", threads);
  Pool pool(threads);
  int failures = 0;
  struct Case {
    const char* name;
    int H, I, tokens, experts, topk, wfill;
    float xscale;
    bool glu;
  };
  const Case cases[] = {
      {"1 token x 1 expert", 2560, 640, 1, 1, 1, -1, 1.0f, false},
      {"3 tokens x 2 experts", 2560, 640, 3, 2, 2, -1, 1.0f, false},
      {"8 tokens x 3 experts", 2560, 640, 8, 3, 3, -1, 1.0f, false},
      {"16 tokens x 10 experts", 2560, 640, 16, 10, 10, -1, 1.0f, false},
      {"7 tokens x 5 experts, small shapes",
       256,
       128,
       7,
       5,
       4,
       -1,
       3.0f,
       false},
      {"extremes: all weights +7", 512, 256, 4, 1, 1, 0x77, 50.0f, false},
      {"extremes: all weights -8", 512, 256, 4, 1, 1, 0x88, 50.0f, false},
      {"fused gemm1 epilogue scheme", 2560, 640, 8, 3, 3, -1, 1.0f, true},
  };
  for (const Case& cs : cases) {
    std::mt19937 rng(20261114u + cs.H + cs.tokens);
    Model m = make_model(cs.H, cs.I, 128, cs.experts, rng, cs.wfill);
    std::normal_distribution<float> nd(0.0f, cs.xscale);
    std::vector<uint16_t> x(size_t(cs.tokens) * cs.H);
    for (auto& v : x)
      v = f32_to_bf16(nd(rng));
    if (cs.wfill >= 0)  // a row of one value: the scale rounds below amax/127
      for (int t = 0; t < cs.tokens; ++t)
        for (int k = 0; k < cs.H; ++k)
          x[size_t(t) * cs.H + k] = f32_to_bf16((t & 1 ? -1.0f : 1.0f) * 37.3f);
    std::vector<int32_t> cu, ct;
    std::vector<float> cw;
    std::uniform_real_distribution<float> ud(0.05f, 1.0f);
    for (int t = 0; t < cs.tokens; ++t)
      for (int k = 0; k < cs.topk; ++k) {
        cu.push_back((t + k * 3) % cs.experts);
        ct.push_back(t);
        cw.push_back(ud(rng));
      }
    a8::Scheme sc;
    if (cs.glu) {
      sc.a2_rule = a8::Rule::kAmax127;
      sc.a2_group = 128;
      sc.glu_fp32 = true;
    }
    std::vector<uint16_t> got(x.size()), sca(x.size()), want_out;
    a8::run_calls(
        pool,
        isa,
        m.L,
        sc,
        got.data(),
        x.data(),
        cs.tokens,
        m.units.data(),
        cs.experts,
        cu.data(),
        ct.data(),
        cw.data(),
        int64_t(cu.size()));
    a8::run_calls(
        pool,
        Isa::kScalar,
        m.L,
        sc,
        sca.data(),
        x.data(),
        cs.tokens,
        m.units.data(),
        cs.experts,
        cu.data(),
        ct.data(),
        cw.data(),
        int64_t(cu.size()));
    Card card;
    card.glu = cs.glu;
    card_calls(card, m, x, cu, ct, cw, cs.tokens, want_out);
    const Diff exact = compare(got, want_out);
    const Diff isa_d = compare(got, sca);
    Card sep = card;
    sep.fused_acc = false;
    std::vector<uint16_t> sep_out, hi_out, lo_out;
    card_calls(sep, m, x, cu, ct, cw, cs.tokens, sep_out);
    Card hi = card, lo = card;
    hi.exp_scale = 1.0 + std::ldexp(1.0, -22);
    lo.exp_scale = 1.0 - std::ldexp(1.0, -12);
    card_calls(hi, m, x, cu, ct, cw, cs.tokens, hi_out);
    card_calls(lo, m, x, cu, ct, cw, cs.tokens, lo_out);
    const Diff d_sep = compare(got, sep_out), d_hi = compare(got, hi_out),
               d_lo = compare(got, lo_out);
    const bool ok = exact.differ == 0 && isa_d.differ == 0;
    failures += !ok;
    std::printf(
        "%-38s %s  vs card emulation: %lld of %lld differ; vs scalar: %lld | "
        "if the card's accumulation were mul+add: %lld differ (%.2f steps); "
        "if its exp were off by +2^-22 / -2^-12: %lld / %lld differ (%.2f / "
        "%.2f "
        "steps)\n",
        cs.name,
        ok ? "OK  " : "FAIL",
        (long long)exact.differ,
        (long long)exact.n,
        (long long)isa_d.differ,
        (long long)d_sep.differ,
        d_sep.max_steps,
        (long long)d_hi.differ,
        (long long)d_lo.differ,
        d_hi.max_steps,
        d_lo.max_steps);
  }
  // Refusals.
  {
    std::mt19937 rng(1);
    Model m = make_model(256, 128, 128, 1, rng, -1);
    std::vector<uint16_t> x(256), o(256);
    int32_t cu = 0, ct = 0;
    float cw = 1.0f;
    a8::Scheme bad;
    bad.a1_group = 96;
    try {
      a8::run_calls(
          pool,
          isa,
          m.L,
          bad,
          o.data(),
          x.data(),
          1,
          m.units.data(),
          1,
          &cu,
          &ct,
          &cw,
          1);
      std::printf("refusal of a 96-wide activation group: MISSING\n");
      ++failures;
    } catch (const std::runtime_error& e) {
      std::printf("refusal: %s\n", e.what());
    }
  }
  std::printf("%s\n", failures ? "SELFTEST FAILED" : "selftest passed");
  return failures ? 1 : 0;
}
