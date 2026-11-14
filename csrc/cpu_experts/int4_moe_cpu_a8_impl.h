// Int4 MoE experts with int8 activations (W4A8) computed on the host CPU, in
// the layout the XPU kernels read (the same expert unit as the W4A16 host op
// of int4_moe_cpu_impl.h, whose layout reading, thread pool and bf16 helpers
// this file reuses). No device dependency: the activation quantization is the
// header the device kernel itself compiles (quantization/int8/
// int8_quant_math.h, c10 types only), so the host and the device run the
// same source for it.
//
// What a call computes (the default scheme is the B70 card's W4A8 path,
// vllm_xpu_kernels/moe_w4a8.py with fused_act off, and
// csrc/xpu/grouped_gemm/xe_2_w4a8/gemm_w4a8_xe2.hpp with AG = 0):
//
//   q1, s1 = quant(x[token])          one scale per row (or per a1_group
//                                     elements), int8 symmetric
//   h[n]   = bf16(s1 * F1[n])         F = sum over K groups g, in order, of
//                                     float(int32 dot of the group) * w_scale
//                                     [n, g], accumulated with fused
//                                     multiply-adds (the card's `mad`)
//   a      = bf16(bf16(silu(gate)) * up)
//   q2, s2 = quant(a)
//   y[n]   = bf16(s2 * F2[n])
//   out[t] = bf16(sum over the token's calls, in call order, of
//                 fma(float(y), weight, acc))         (moe_gather's `mad`)
//
// Integer dot products are exact and their order does not matter, and the
// fp32 accumulation over groups is a fixed chain per output, so every step
// except silu is reproducible bit for bit. silu uses exp: the card evaluates
// it with its hardware approximations (math.exp, math.inv); here it is
// fp32(exp(double)) and an IEEE division, a well-defined function a test
// can reproduce, which differs from the card's in the last fp32 bits.
//
// The scheme is a parameter so the same op can follow other W4A8 kernels:
// per-row or per-group activation scales, the rounding rule of the scale
// (bf16-rounded with a floor, as the card's dynamic_per_token_scaled_int8
// _quant; or amax/127 in fp32 with +-127, as the fused gemm1 epilogue and
// the dense HC kernel), fused or separate multiply-add over the groups, and
// an fp32 gemm1 epilogue (silu*up without the bf16 round trips).
//
// The instruction set is a dispatch behind one inner function per GEMM
// phase (quant_row, gemm_rows, act_row, gather). The integer dot of a K
// chunk is the only ISA-specific part, at gemm_rows: scalar, AVX2
// (vpmaddubsw / vpmaddwd), and in int4_moe_cpu_a8_isa.h AVX-VNNI and
// AVX-512 VNNI (vpdpbusd on the same unsigned weights w + 8 with the same
// correction 8 * sum(q)) and AMX-INT8 (tdpbusd tiles of the same operands,
// for 16 or more rows per expert). Not built: Arm NEON / SVE sdot (s8 x s8
// on the nibbles sign-extended, no correction). All give the same int32 per
// chunk, so the outputs stay bit-identical across ISAs.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu_experts/int4_moe_cpu_impl.h"
#include "quantization/int8/int8_quant_math.h"

#if defined(__clang__)
  #pragma clang fp contract(off)
  #pragma clang fp reassociate(off)
#elif defined(__GNUC__)
  #pragma GCC optimize("fp-contract=off")
#endif

namespace cpu_moe {
namespace a8 {

// How a row of activations becomes int8 and its scale.
enum class Rule : int {
  // dynamic_per_token_scaled_int8_quant (csrc/quantization/int8/
  // int8_quant_math.h): s = round_to_input_dtype(max(amax / 127, 1e-5)),
  // q = clamp(round_half_even(x / s), -128, 127) with an exact tie test.
  kInt8q = 0,
  // amax / 127 in fp32 (no rounding of the scale, no floor),
  // q = clamp(rint(x * (127 / amax)), -127, 127); amax 0 gives q 0, s 0.
  kAmax127 = 1,
};

// How a group's int32 dot joins the fp32 accumulator.
enum class Acc : int {
  kFused = 0,     // F = fma(float(d), m, F)   (the card's mad)
  kSeparate = 1,  // F = F + round(float(d) * m)
};

struct Scheme {
  Rule a1_rule = Rule::kInt8q;
  int a1_group = 0;  // 0: one scale per row; else elements per scale
  Rule a2_rule = Rule::kInt8q;
  int a2_group = 0;
  Acc acc = Acc::kFused;
  bool glu_fp32 = false;  // gemm1 epilogue in fp32 (no bf16 h, no bf16 a)
};

// scheme as the op takes it: [a1_rule, a1_group, a2_rule, a2_group, acc,
// glu_fp32]; an empty list is the default (the B70 card's path).
inline Scheme scheme_from(const int64_t* v, size_t n) {
  Scheme s;
  if (n == 0) return s;
  if (n != 6)
    throw std::runtime_error(
        "int4_moe_cpu_a8: scheme is [a1_rule, a1_group, a2_rule, a2_group, "
        "acc, glu_fp32] or empty");
  for (size_t i : {size_t(0), size_t(2)})
    if (v[i] < 0 || v[i] > 1)
      throw std::runtime_error("int4_moe_cpu_a8: a rule is 0 or 1");
  if (v[1] < 0 || v[3] < 0)
    throw std::runtime_error("int4_moe_cpu_a8: a group is 0 or positive");
  if (v[4] < 0 || v[4] > 1 || v[5] < 0 || v[5] > 1)
    throw std::runtime_error("int4_moe_cpu_a8: acc and glu_fp32 are 0 or 1");
  s.a1_rule = static_cast<Rule>(v[0]);
  s.a1_group = static_cast<int>(v[1]);
  s.a2_rule = static_cast<Rule>(v[2]);
  s.a2_group = static_cast<int>(v[3]);
  s.acc = static_cast<Acc>(v[4]);
  s.glu_fp32 = v[5] != 0;
  return s;
}

constexpr int kStep = 64;  // elements per 32 weight bytes (one AVX2 load)
constexpr int kMaxFlush = 512;

// The K chunk an int32 partial covers before it is folded into fp32: the
// weight group, or the activation group when that is finer (as the card's
// FLUSH in gemm_w4a8_xe2.hpp).
inline int flush_size(int K, int G, int ag, const char* which) {
  if (ag != 0) {
    if (K % ag || !(ag % G == 0 || G % ag == 0))
      throw std::runtime_error(
          std::string("int4_moe_cpu_a8: ") + which +
          " activation group must divide K and divide or be a multiple of "
          "the weight group");
  }
  const int fl = (ag != 0 && ag < G) ? ag : G;
  if (fl % kStep || fl > kMaxFlush || K % fl)
    throw std::runtime_error(
        std::string("int4_moe_cpu_a8: ") + which +
        " K chunks must be multiples of 64 elements, at most 512");
  return fl;
}

// --------------------------------------------------------------------------
// Activations of one row in GEMM form
// --------------------------------------------------------------------------

struct QRows {
  int K = 0, fl = 0, ag = 0, nas = 1, nfl = 0;
  std::vector<int8_t> qe, qo;  // per row: even / odd elements, K/2 each
  std::vector<float> as;       // per row: nas scales
  std::vector<int32_t> cs;     // per row: sum of q over each flush chunk
  // AMX only (int4_moe_cpu_a8_isa.h): the activations transposed into
  // tdpbusd's B layout, [K/4][cp] dwords, cp = rows + 16; empty otherwise.
  std::vector<int32_t> qt;
  int64_t cp = 0;
  void shape(int64_t rows, int k, int fl_, int ag_) {
    K = k;
    fl = fl_;
    ag = ag_;
    nas = ag ? K / ag : 1;
    nfl = K / fl;
    qe.resize(static_cast<size_t>(rows) * (K / 2));
    qo.resize(static_cast<size_t>(rows) * (K / 2));
    as.resize(static_cast<size_t>(rows) * nas);
    cs.resize(static_cast<size_t>(rows) * nfl);
    qt.clear();
    cp = 0;
  }
  void shape_transposed(int64_t rows) {
    cp = rows + 16;
    qt.resize(static_cast<size_t>(K / 4) * cp);
  }
};

inline int8_t quant_amax127(float x, float inv) {
  float q = vllm::int8q::rint_half_even(x * inv);
  q = q < -127.0f ? -127.0f : q;
  q = q > 127.0f ? 127.0f : q;
  return static_cast<int8_t>(q);
}

// Phase: one row v [K] (bf16 values when bf16_input, else fp32) into int8
// with its scales, split even/odd, with the chunk sums.
inline void
quant_row(const float* v, bool bf16_input, Rule rule, QRows& Q, int64_t r) {
  const int K = Q.K, len = Q.ag ? Q.ag : K;
  int8_t* qe = Q.qe.data() + r * (K / 2);
  int8_t* qo = Q.qo.data() + r * (K / 2);
  float* as = Q.as.data() + r * Q.nas;
  for (int a = 0; a < Q.nas; ++a) {
    const int k0 = a * len;
    float amax = 0.0f;
    for (int k = k0; k < k0 + len; ++k)
      amax = vllm::int8q::max_f(amax, vllm::int8q::abs_f(v[k]));
    float s;
    if (rule == Rule::kInt8q) {
      s = bf16_input ? vllm::int8q::token_scale_from_absmax<c10::BFloat16>(amax)
                     : vllm::int8q::token_scale_from_absmax<float>(amax);
      for (int k = k0; k < k0 + len; ++k) {
        const int8_t q = bf16_input
                             ? vllm::int8q::quantize_one<c10::BFloat16>(v[k], s)
                             : vllm::int8q::quantize_one<float>(v[k], s);
        (k & 1 ? qo : qe)[k / 2] = q;
      }
    } else {
      s = amax * (1.0f / 127.0f);
      const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
      for (int k = k0; k < k0 + len; ++k)
        (k & 1 ? qo : qe)[k / 2] = quant_amax127(v[k], inv);
    }
    as[a] = s;
  }
  int32_t* cs = Q.cs.data() + r * Q.nfl;
  for (int f = 0; f < Q.nfl; ++f) {
    int32_t sum = 0;
    for (int j = f * Q.fl / 2; j < (f + 1) * Q.fl / 2; ++j)
      sum += static_cast<int32_t>(qe[j]) + static_cast<int32_t>(qo[j]);
    cs[f] = sum;
  }
}

// --------------------------------------------------------------------------
// Phase: 8 output rows [n0, n0 + 8) of an int4 x int8 product for nt rows
// of activations. W [N, K/2] bytes, S [N, K/G] bf16. out[t * ldo + n] gets
// the fp32 result (bf16-rounded when round_out).
// --------------------------------------------------------------------------

struct GemmArgs {
  const uint8_t* W;
  const uint16_t* S;
  int K, G;
  Acc acc;
  bool round_out;
};

constexpr int kRows = 8;  // output rows per block (one AVX2 vector of rows)
constexpr int kTBa = 4;   // activation rows per pass over the block

inline float acc_step(Acc acc, float d, float m, float F) {
  return acc == Acc::kFused ? std::fma(d, m, F) : F + d * m;
}

inline void gemm_rows_scalar(
    const GemmArgs& a,
    int n0,
    const QRows& Q,
    int64_t r0,
    int nt,
    float* out,
    int ldo) {
  const int kb = a.K / 2, ng = a.K / a.G;
  for (int t = 0; t < nt; ++t) {
    const int64_t r = r0 + t;
    const int8_t* qe = Q.qe.data() + r * kb;
    const int8_t* qo = Q.qo.data() + r * kb;
    const float* as = Q.as.data() + r * Q.nas;
    for (int n = n0; n < n0 + kRows; ++n) {
      const uint8_t* w = a.W + static_cast<int64_t>(n) * kb;
      float F = 0.0f;
      for (int f = 0; f < Q.nfl; ++f) {
        int32_t d = 0;
        for (int j = f * Q.fl / 2; j < (f + 1) * Q.fl / 2; ++j)
          d += nibble(w[j], 0) * qe[j] + nibble(w[j], 1) * qo[j];
        const int k0 = f * Q.fl;
        float m = bf16_to_f32(a.S[static_cast<int64_t>(n) * ng + k0 / a.G]);
        if (Q.ag) m = m * as[k0 / Q.ag];
        F = acc_step(a.acc, static_cast<float>(d), m, F);
      }
      if (!Q.ag) F = F * as[0];
      out[static_cast<int64_t>(t) * ldo + n] = a.round_out ? round_bf16(F) : F;
    }
  }
}

#if INT4_MOE_CPU_X86
// The 8 lanes' sums of 8 int32 vectors, as one vector (integer: any order).
__attribute__((target("avx2"))) inline __m256i sum8_epi32(const __m256i* v) {
  const __m256i s01 = _mm256_hadd_epi32(v[0], v[1]);
  const __m256i s23 = _mm256_hadd_epi32(v[2], v[3]);
  const __m256i s45 = _mm256_hadd_epi32(v[4], v[5]);
  const __m256i s67 = _mm256_hadd_epi32(v[6], v[7]);
  const __m256i s03 = _mm256_hadd_epi32(s01, s23);
  const __m256i s47 = _mm256_hadd_epi32(s45, s67);
  const __m256i lo = _mm256_permute2x128_si256(s03, s47, 0x20);
  const __m256i hi = _mm256_permute2x128_si256(s03, s47, 0x31);
  return _mm256_add_epi32(lo, hi);
}

// The int32 dot of a chunk: weights as unsigned u = w + 8 (two's complement
// nibble XOR 8), activations signed, vpmaddubsw then vpmaddwd; the caller
// subtracts 8 * sum(q). No saturation: |u * q| <= 15 * 128 = 1920, a
// vpmaddubsw pair is at most 3840, the add of the even and the odd pair at
// most 7680, all inside int16, and each step is widened to int32 before the
// next one is added. (A VNNI path would replace the three instructions by
// one vpdpbusd per operand pair.)
__attribute__((target("avx2,fma"))) inline void gemm_rows_avx2(
    const GemmArgs& a,
    int n0,
    const QRows& Q,
    int64_t r0,
    int nt,
    float* out,
    int ldo) {
  const int kb = a.K / 2, ng = a.K / a.G;
  const int steps = Q.fl / kStep;
  const __m256i m0f = _mm256_set1_epi8(0x0F);
  const __m256i x88 = _mm256_set1_epi8(static_cast<char>(0x88));
  const __m256i ones = _mm256_set1_epi16(1);
  alignas(32) uint8_t ub[kRows][kMaxFlush / kStep][2][32];
  for (int t0 = 0; t0 < nt; t0 += kTBa) {
    const int tb = nt - t0 < kTBa ? nt - t0 : kTBa;
    __m256 F[kTBa];
    for (int t = 0; t < kTBa; ++t)
      F[t] = _mm256_setzero_ps();
    for (int f = 0; f < Q.nfl; ++f) {
      const int j0 = f * Q.fl / 2;
      for (int r = 0; r < kRows; ++r) {
        const uint8_t* w = a.W + static_cast<int64_t>(n0 + r) * kb + j0;
        for (int s = 0; s < steps; ++s) {
          const __m256i raw = _mm256_xor_si256(
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32 * s)),
              x88);
          _mm256_store_si256(
              reinterpret_cast<__m256i*>(ub[r][s][0]),
              _mm256_and_si256(raw, m0f));
          _mm256_store_si256(
              reinterpret_cast<__m256i*>(ub[r][s][1]),
              _mm256_and_si256(_mm256_srli_epi16(raw, 4), m0f));
        }
      }
      const int k0 = f * Q.fl;
      const int g = k0 / a.G;
      alignas(32) float wl[kRows];
      for (int r = 0; r < kRows; ++r)
        wl[r] = bf16_to_f32(a.S[static_cast<int64_t>(n0 + r) * ng + g]);
      const __m256 wsc = _mm256_load_ps(wl);
      for (int t = 0; t < tb; ++t) {
        const int64_t row = r0 + t0 + t;
        const int8_t* e = Q.qe.data() + row * kb + j0;
        const int8_t* o = Q.qo.data() + row * kb + j0;
        __m256i v[kRows];
        for (int r = 0; r < kRows; ++r) {
          __m256i acc32 = _mm256_setzero_si256();
          for (int s = 0; s < steps; ++s) {
            const __m256i ev = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(e + 32 * s));
            const __m256i ov = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(o + 32 * s));
            const __m256i p = _mm256_add_epi16(
                _mm256_maddubs_epi16(
                    _mm256_load_si256(
                        reinterpret_cast<const __m256i*>(ub[r][s][0])),
                    ev),
                _mm256_maddubs_epi16(
                    _mm256_load_si256(
                        reinterpret_cast<const __m256i*>(ub[r][s][1])),
                    ov));
            acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(p, ones));
          }
          v[r] = acc32;
        }
        const __m256i d = _mm256_sub_epi32(
            sum8_epi32(v), _mm256_set1_epi32(8 * Q.cs[row * Q.nfl + f]));
        const __m256 df = _mm256_cvtepi32_ps(d);
        __m256 m = wsc;
        if (Q.ag)
          m = _mm256_mul_ps(wsc, _mm256_set1_ps(Q.as[row * Q.nas + k0 / Q.ag]));
        F[t] = a.acc == Acc::kFused ? _mm256_fmadd_ps(df, m, F[t])
                                    : _mm256_add_ps(F[t], _mm256_mul_ps(df, m));
      }
    }
    for (int t = 0; t < tb; ++t) {
      const int64_t row = r0 + t0 + t;
      __m256 y = F[t];
      if (!Q.ag) y = _mm256_mul_ps(y, _mm256_set1_ps(Q.as[row * Q.nas]));
      alignas(32) float l[kRows];
      _mm256_store_ps(l, y);
      float* dst = out + static_cast<int64_t>(t0 + t) * ldo + n0;
      for (int r = 0; r < kRows; ++r)
        dst[r] = a.round_out ? round_bf16(l[r]) : l[r];
    }
  }
}
#endif

}  // namespace a8
}  // namespace cpu_moe

// AVX-VNNI, AVX-512 VNNI and AMX-INT8 code for the integer dot.
#include "int4_moe_cpu_a8_isa.h"

namespace cpu_moe {
namespace a8 {

inline void gemm_rows(
    Isa isa,
    const GemmArgs& a,
    int n0,
    const QRows& Q,
    int64_t r0,
    int nt,
    float* out,
    int ldo) {
#if INT4_MOE_CPU_X86
  switch (isa) {
    case Isa::kAvx2:
    case Isa::kAvx512:  // AVX-512 without VNNI: the AVX2 code
      gemm_rows_avx2(a, n0, Q, r0, nt, out, ldo);
      return;
    case Isa::kAvxVnni:
      gemm_rows_avxvnni(a, n0, Q, r0, nt, out, ldo);
      return;
    case Isa::kAmx:
    case Isa::kAmxBf16Class:
      if (!Q.qt.empty() && nt >= kAmxMinTokens) {
        gemm_rows_amx(a, n0, Q, r0, nt, out, ldo);
        return;
      }
      [[fallthrough]];
    case Isa::kAvx512Vnni:
    case Isa::kAvx512Bf16:
      gemm_rows_avx512vnni(a, n0, Q, r0, nt, out, ldo);
      return;
    default:
      break;
  }
#endif
  gemm_rows_scalar(a, n0, Q, r0, nt, out, ldo);
}

// --------------------------------------------------------------------------
// Phase: the activation of one row h [2I] (gate, then up) into v [I].
// --------------------------------------------------------------------------

// fp32(exp(double)): a well-defined exp the host and a test both reproduce.
inline float exp_f32(float x) {
  return static_cast<float>(std::exp(static_cast<double>(x)));
}

inline void act_row(const float* h, int I, bool glu_fp32, float* v) {
  for (int i = 0; i < I; ++i) {
    const float g = h[i], u = h[I + i];
    const float s = g / (1.0f + exp_f32(-g));
    v[i] = glu_fp32 ? s * u : round_bf16(round_bf16(s) * u);
  }
}

// --------------------------------------------------------------------------
// The call (see int4_moe_cpu_impl.h run_calls for the call list contract).
// --------------------------------------------------------------------------

inline void run_calls(
    Pool& pool,
    Isa isa,
    const Layout& L,
    const Scheme& sc,
    uint16_t* out,
    const uint16_t* x,
    int64_t T,
    const uint8_t* units,
    int64_t num_units,
    const int32_t* call_unit,
    const int32_t* call_token,
    const float* call_weight,
    int64_t C) {
  check_layout(L);
  isa = resolve_isa(isa);
  const int H = L.hidden, I = L.inter, G = L.group;
  if (H % kStep || I % kStep)
    throw std::runtime_error(
        "int4_moe_cpu_a8: H and I must be multiples of 64");
  const int fl1 = flush_size(H, G, sc.a1_group, "gemm1");
  const int fl2 = flush_size(I, G, sc.a2_group, "gemm2");
  std::memset(out, 0, static_cast<size_t>(T) * H * sizeof(uint16_t));
  if (C == 0) return;
  for (int64_t c = 0; c < C; ++c) {
    if (call_unit[c] < 0 || call_unit[c] >= num_units)
      throw std::runtime_error(
          "int4_moe_cpu_a8: call " + std::to_string(c) + " names unit " +
          std::to_string(call_unit[c]) + " of " + std::to_string(num_units));
    if (call_token[c] < 0 || call_token[c] >= T)
      throw std::runtime_error(
          "int4_moe_cpu_a8: call " + std::to_string(c) + " names token " +
          std::to_string(call_token[c]) + " of " + std::to_string(T));
    if (c > 0 && call_token[c] < call_token[c - 1])
      throw std::runtime_error(
          "int4_moe_cpu_a8: calls must be in token order (call " +
          std::to_string(c) + ")");
  }
  // Calls grouped by unit, call order kept inside a group.
  std::vector<int32_t> units_used;
  std::vector<std::vector<int64_t>> calls_of;
  {
    std::vector<int32_t> slot(static_cast<size_t>(num_units), -1);
    for (int64_t c = 0; c < C; ++c) {
      int32_t& s = slot[call_unit[c]];
      if (s < 0) {
        s = static_cast<int32_t>(units_used.size());
        units_used.push_back(call_unit[c]);
        calls_of.emplace_back();
      }
      calls_of[s].push_back(c);
    }
  }
  const int64_t E = static_cast<int64_t>(units_used.size());
  std::vector<int64_t> base(E + 1, 0), row_of(C);
  for (int64_t e = 0; e < E; ++e) {
    base[e + 1] = base[e] + static_cast<int64_t>(calls_of[e].size());
    for (size_t i = 0; i < calls_of[e].size(); ++i)
      row_of[calls_of[e][i]] = base[e] + static_cast<int64_t>(i);
  }
  // Scratch per calling thread (workers reach it through these references).
  struct Scratch {
    QRows q1, q2;
    std::vector<float> h, y;
  };
  static thread_local Scratch scratch;
  QRows& q1 = scratch.q1;
  QRows& q2 = scratch.q2;
  auto& h = scratch.h;
  auto& y = scratch.y;
  q1.shape(C, H, fl1, sc.a1_group);
  q2.shape(C, I, fl2, sc.a2_group);
  // AMX: the activations also in tdpbusd's layout, when an expert has
  // enough rows for a tile
  int64_t most = 0;
  for (int64_t e = 0; e < E; ++e)
    most = std::max(most, base[e + 1] - base[e]);
  const bool amx =
      (isa == Isa::kAmx || isa == Isa::kAmxBf16Class) && most >= kAmxMinTokens;
  if (amx) {
    q1.shape_transposed(C);
    q2.shape_transposed(C);
  }
  h.resize(static_cast<size_t>(C) * 2 * I);
  y.resize(static_cast<size_t>(C) * H);
  // Phase 1: each call row's input, quantized (rows by unit group).
  pool.run(C, [&](int64_t c) {
    const uint16_t* xt = x + static_cast<int64_t>(call_token[c]) * H;
    std::vector<float> v(static_cast<size_t>(H));
    for (int k = 0; k < H; ++k)
      v[k] = bf16_to_f32(xt[k]);
    quant_row(v.data(), true, sc.a1_rule, q1, row_of[c]);
    if (amx) transpose_row(q1, row_of[c]);
  });
  // Phase 2: GEMM1 -> h [rows, 2I] (bf16 values, or fp32 for glu_fp32).
  const int64_t blocks1 = 2 * I / kRows;
  pool.run(E * blocks1, [&](int64_t item) {
    const int64_t e = item / blocks1;
    const int n0 = static_cast<int>((item % blocks1) * kRows);
    const uint8_t* u =
        units + static_cast<int64_t>(units_used[e]) * L.unit_bytes;
    const GemmArgs a{
        u + L.off_w13,
        reinterpret_cast<const uint16_t*>(u + L.off_s13),
        H,
        G,
        sc.acc,
        !sc.glu_fp32};
    gemm_rows(
        isa,
        a,
        n0,
        q1,
        base[e],
        static_cast<int>(base[e + 1] - base[e]),
        h.data() + base[e] * 2 * I,
        2 * I);
  });
  // Phase 3: activation and its quantization, per call row.
  pool.run(C, [&](int64_t r) {
    std::vector<float> v(static_cast<size_t>(I));
    act_row(h.data() + r * 2 * I, I, sc.glu_fp32, v.data());
    quant_row(v.data(), !sc.glu_fp32, sc.a2_rule, q2, r);
    if (amx) transpose_row(q2, r);
  });
  // Phase 4: GEMM2 -> y [rows, H], bf16 values.
  const int64_t blocks2 = H / kRows;
  pool.run(E * blocks2, [&](int64_t item) {
    const int64_t e = item / blocks2;
    const int n0 = static_cast<int>((item % blocks2) * kRows);
    const uint8_t* u =
        units + static_cast<int64_t>(units_used[e]) * L.unit_bytes;
    const GemmArgs a{
        u + L.off_w2,
        reinterpret_cast<const uint16_t*>(u + L.off_s2),
        I,
        G,
        sc.acc,
        true};
    gemm_rows(
        isa,
        a,
        n0,
        q2,
        base[e],
        static_cast<int>(base[e + 1] - base[e]),
        y.data() + base[e] * H,
        H);
  });
  // Phase 5: gather, per token in call order, fused multiply-add as the
  // card's moe_gather; columns split across threads.
  constexpr int kCols = 256;
  const int64_t col_blocks = (H + kCols - 1) / kCols;
  pool.run(col_blocks, [&](int64_t b) {
    const int j0 = static_cast<int>(b * kCols);
    const int j1 = j0 + kCols < H ? j0 + kCols : H;
    std::vector<float> acc(static_cast<size_t>(j1 - j0));
    int64_t c = 0;
    while (c < C) {
      const int32_t t = call_token[c];
      std::fill(acc.begin(), acc.end(), 0.f);
      for (; c < C && call_token[c] == t; ++c) {
        const float* yr = y.data() + row_of[c] * H;
        const float w = call_weight[c];
        for (int j = j0; j < j1; ++j)
          acc[j - j0] = std::fma(yr[j], w, acc[j - j0]);
      }
      uint16_t* ot = out + static_cast<int64_t>(t) * H;
      for (int j = j0; j < j1; ++j)
        ot[j] = f32_to_bf16(acc[j - j0]);
    }
  });
}

}  // namespace a8
}  // namespace cpu_moe
