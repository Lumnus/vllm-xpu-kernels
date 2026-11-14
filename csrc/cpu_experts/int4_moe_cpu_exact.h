// The card's W4A16 grouped-GEMM arithmetic, exactly (the op's arith 7,
// "card-exact"). Included by int4_moe_cpu_impl.h.
//
// What the card computes for one output value of either GEMM, measured on a
// B70 by a probe of its per-pair GEMM rows and reproduced from the inputs
// with 0 differing values (153,600 of them):
//   1. each weight is w = bf16_rtz(float(q) * float(s)): apply_scale is a
//      bf16 times fp32 multiply into a bf16 destination that truncates
//      (rounds toward zero). q * s has at most 12 significant bits, so the
//      fp32 product is exact and truncation keeps its top 8;
//   2. the products w * x (bf16 x bf16: at most 16 significant bits, exact)
//      are summed in blocks of 16 consecutive k, aligned at k = 0, in
//      ascending block order, and each block's exact sum is added to the
//      fp32 accumulator (starting at +0) with one rounding to nearest even:
//      acc = fl32(acc + sum of the block);
//   3. the output is bf16(acc), to nearest even (the callers' rounding).
// The activation between the GEMMs is bf16(silu_card[bits(gate)] * up),
// with the card's own bf16 silu of every bf16 gate as a 65,536-entry table
// (run_calls' silu_table), the product of two bf16 values exact in fp32.
//
// How a block is made exact here:
//   1. the products are formed in double (exact: 16 significant bits, and
//      the exponent range of two bf16 values fits double's);
//   2. their sum is exact in double whenever the 16 inputs of the block
//      (x, nonzero) have exponents spanning at most kExactSpan = 30.
//      Proof: with e_s the exponent of the group's scale, a nonzero weight
//      has s <= |w| <= 8 |s| (truncation of |q| * |s| >= |s| to bf16 keeps
//      at least |s|, which is a bf16), so every product is a multiple of
//      2^(e_s - 7 + e_xmin - 7) (8 significant bits each; a subnormal bf16
//      only raises the grain) and below 2^(e_s + 4 + e_xmax + 1); a sum of
//      any 16 or fewer of them is below 2^(e_s + e_xmax + 9). That needs
//      e_xmax - e_xmin + 23 <= 53 bits: span <= 30, in any order of
//      addition (so vector lanes may add in any order). The scale cancels:
//      the bound depends on the inputs only, flagged once per input block
//      (exact_block_flag) for all rows and experts that read it;
//   3. acc + B (B the block's exact sum) is rounded to fp32 once. In double
//      it may round (acc and B far apart); converting that double to fp32
//      is then a second rounding, which differs from one rounding only when
//      the double lands exactly on an fp32 midpoint while the sum was not
//      exact. Fast path: round s = acc + B to fp32 unless s is such a
//      midpoint (bits 0..28 of its mantissa = 1 << 28) with a nonzero
//      TwoSum error, lies in fp32's subnormal range, or is not finite;
//      those take round32_sum (the exact error decides the tie) or
//      exact_step_slow;
//   4. a block whose inputs span more than 30 exponents (or with a non-
//      finite value anywhere) takes exact_step_slow: the exact sum in a
//      576-bit fixed-point integer (Wide), rounded once. So the result is
//      the exact spec for every finite input, not only within the window.
// Non-finite values: a NaN anywhere, or +inf and -inf together, give the
// canonical NaN; otherwise an inf gives that inf (the card's behaviour on
// non-finite inputs is not measured; this makes the result independent of
// the order of addition, which the vector code needs).
//
// The AVX2 + FMA body forms a block's exact sum one of two ways: when the
// block's inputs span at most 7 exponents (and the row's scale is in a
// plain range) both factors are small integers times powers of two, and
// vpmaddwd sums the 16 int16 products exactly in int32 (|W| < 2^11,
// |X| < 2^15: below 2^30); otherwise in double with fused multiply-adds
// (exact by the window above). Four output rows are reduced and rounded
// together, one per double lane.
//
// Every instruction set computes the same exact function, so all agree bit
// for bit by construction. The AVX2 + FMA body serves every level from 2
// up, the AVX-512 levels and level 8 included: for this arithmetic they
// dispatch to it (rows_exact).

#pragma once

#include <atomic>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>

namespace cpu_moe {

constexpr int kArithCardExact = 7;  // run_calls' arith for this arithmetic
constexpr int kExactBlock = 16;     // products per exact block sum
constexpr int kExactSpan = 30;      // input exponent span of an exact sum

// How often the slow paths ran in this process (tests read them): blocks
// summed in the exact integer (Wide), and vector lanes rounded by
// round32_sum (a midpoint in double, fp32's subnormal range).
struct ExactCounters {
  int64_t wide = 0, fixed = 0;
};
inline std::atomic<int64_t>& exact_wide_count() {
  static std::atomic<int64_t> n{0};
  return n;
}
inline std::atomic<int64_t>& exact_fixed_count() {
  static std::atomic<int64_t> n{0};
  return n;
}
inline ExactCounters exact_counters() {
  return {exact_wide_count().load(), exact_fixed_count().load()};
}

// Truncation of an fp32 value to bf16 (toward zero), kept as a float.
inline float bf16_rtz(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  bits &= 0xffff0000u;
  std::memcpy(&f, &bits, 4);
  return f;
}

// 1 when the double sum of a block reading these 16 bf16 inputs may be
// inexact (nonzero exponents span more than kExactSpan) or an input is not
// finite; 0 otherwise.
inline uint8_t exact_block_flag(const uint16_t* v, int stride = 1) {
  int lo = INT_MAX, hi = INT_MIN;
  for (int k = 0; k < kExactBlock; ++k) {
    const uint16_t b = v[k * stride];
    const int e8 = (b >> 7) & 0xFF, m = b & 0x7F;
    if (e8 == 0xFF) return 1;
    int e;
    if (e8 != 0) {
      e = e8 - 127;
    } else if (m != 0) {
      e = -133 + (31 - __builtin_clz(static_cast<unsigned>(m)));
    } else {
      continue;  // zero
    }
    lo = std::min(lo, e);
    hi = std::max(hi, e);
  }
  return hi != INT_MIN && hi - lo > kExactSpan ? 1 : 0;
}

// --------------------------------------------------------------------------
// Exact sums: a two's-complement fixed-point integer of 576 bits, bit 0
// worth 2^-272. Every value it takes is a bf16 x bf16 product (lowest bit at
// least 2^-266, magnitude below 2^256) or an fp32 accumulator; 17 of them
// sum below 2^262, so neither end can be lost.
// --------------------------------------------------------------------------

struct Wide {
  static constexpr int kLimbs = 9;
  static constexpr int kBase = -272;
  uint64_t w[kLimbs] = {};

  // v finite, with its lowest set bit at 2^kBase or above.
  void add(double v) {
    if (v == 0.0) return;
    int e;
    const double f = std::frexp(std::fabs(v), &e);  // [0.5, 1) * 2^e
    uint64_t m = static_cast<uint64_t>(std::ldexp(f, 53));
    int sh = e - 53 - kBase;
    const int tz = __builtin_ctzll(m);
    m >>= tz;
    sh += tz;
    uint64_t part[kLimbs] = {};
    const int li = sh / 64, bi = sh % 64;
    part[li] = m << bi;
    if (bi != 0 && li + 1 < kLimbs) part[li + 1] = m >> (64 - bi);
    if (v > 0) {
      unsigned __int128 c = 0;
      for (int i = 0; i < kLimbs; ++i) {
        c += static_cast<unsigned __int128>(w[i]) + part[i];
        w[i] = static_cast<uint64_t>(c);
        c >>= 64;
      }
    } else {
      uint64_t borrow = 0;
      for (int i = 0; i < kLimbs; ++i) {
        const uint64_t a = w[i], b = part[i];
        const uint64_t d = a - b - borrow;
        borrow = (a < b) || (a - b < borrow) ? 1 : 0;
        w[i] = d;
      }
    }
  }

  // The value rounded once to fp32, to nearest even (overflow: inf).
  float to_float() const {
    uint64_t a[kLimbs];
    std::memcpy(a, w, sizeof(a));
    const bool neg = (a[kLimbs - 1] >> 63) != 0;
    if (neg) {  // magnitude: two's complement negation
      uint64_t c = 1;
      for (int i = 0; i < kLimbs; ++i) {
        a[i] = ~a[i] + c;
        c = (c != 0 && a[i] == 0) ? 1 : 0;
      }
    }
    int top = -1;
    for (int i = kLimbs - 1; i >= 0 && top < 0; --i)
      if (a[i] != 0) top = i * 64 + 63 - __builtin_clzll(a[i]);
    if (top < 0) return 0.0f;
    auto bit = [&](int i) -> uint64_t {
      return i < 0 ? 0 : (a[i / 64] >> (i % 64)) & 1u;
    };
    // the lowest kept bit: 24 significant bits, or 2^-149 below fp32's
    // normal range
    const int q = top + kBase >= -126 ? top - 23 : -149 - kBase;
    uint64_t mant = 0;
    for (int i = top; i >= q; --i)
      mant = (mant << 1) | bit(i);
    const uint64_t guard = bit(q - 1);
    bool sticky = false;
    for (int i = 0; i < q - 1 && !sticky; ++i)
      sticky = bit(i) != 0;
    if (guard && (sticky || (mant & 1))) ++mant;
    const float r = std::ldexp(static_cast<float>(mant), q + kBase);
    return neg ? -r : r;
  }
};

// fl32(a + b) with one rounding, a and b finite doubles (a an fp32 value, b
// a block's exact sum, so a + b cannot overflow double).
inline float round32_sum(double a, double b) {
  const double s = a + b;
  const double bb = s - a;
  const double err = (a - (s - bb)) + (b - bb);  // TwoSum: s + err = a + b
  const float f = static_cast<float>(s);
  if (err == 0.0) return f;
  if (std::isinf(f)) {
    // s at or beyond FLT_MAX + half an ulp; exactly there, err decides
    const double lim = 0x1.ffffffp127;
    if (std::fabs(s) == lim && (s > 0) != (err > 0))
      return s > 0 ? FLT_MAX : -FLT_MAX;
    return f;
  }
  if (static_cast<double>(f) == s) return f;
  const float g = std::nextafter(
      f,
      s > static_cast<double>(f) ? std::numeric_limits<float>::infinity()
                                 : -std::numeric_limits<float>::infinity());
  // s exactly halfway between f and its neighbour g (both differences are
  // exact: Sterbenz): the exact sum lies on err's side of s
  if (s - static_cast<double>(f) == static_cast<double>(g) - s)
    return err > 0 ? std::max(f, g) : std::min(f, g);
  return f;
}

// One block when the double sum may be inexact or a value is not finite:
// fl32(acc + p[0] + ... + p[n-1]) with one rounding.
inline float exact_step_slow(float acc, const double* p, int n) {
  bool nan = std::isnan(acc);
  bool pinf = acc == std::numeric_limits<float>::infinity();
  bool ninf = acc == -std::numeric_limits<float>::infinity();
  for (int k = 0; k < n; ++k) {
    nan = nan || std::isnan(p[k]);
    pinf = pinf || p[k] == std::numeric_limits<double>::infinity();
    ninf = ninf || p[k] == -std::numeric_limits<double>::infinity();
  }
  if (nan || (pinf && ninf)) return std::numeric_limits<float>::quiet_NaN();
  if (pinf) return std::numeric_limits<float>::infinity();
  if (ninf) return -std::numeric_limits<float>::infinity();
  exact_wide_count().fetch_add(1, std::memory_order_relaxed);
  Wide w;
  w.add(acc);
  for (int k = 0; k < n; ++k)
    w.add(p[k]);
  return w.to_float();
}

// The 16 products of block b of one row (weights bf16_rtz(q * scale)) with
// a token's inputs in double, split even/odd as the op's scratch: product
// 2l is byte 8b + l's low nibble times xe[8b + l], 2l + 1 its high nibble
// times xo[8b + l].
inline void exact_products(
    const uint8_t* wrow,
    float scale,
    const double* xe,
    const double* xo,
    int b,
    double* p) {
  for (int l = 0; l < 8; ++l) {
    const int j = 8 * b + l;
    const float wl = bf16_rtz(static_cast<float>(nibble(wrow[j], 0)) * scale);
    const float wh = bf16_rtz(static_cast<float>(nibble(wrow[j], 1)) * scale);
    p[2 * l] = static_cast<double>(wl) * xe[j];
    p[2 * l + 1] = static_cast<double>(wh) * xo[j];
  }
}

// One block of the chain: acc and the block's products (flag: bit 0 of the
// inputs' block flag, the double sum may be inexact).
inline float exact_step(float acc, const double* p, uint8_t flag) {
  bool finite = std::isfinite(acc);
  for (int k = 0; k < kExactBlock && finite; ++k)
    finite = std::isfinite(p[k]);
  if (flag || !finite) return exact_step_slow(acc, p, kExactBlock);
  double B = 0.0;  // exact: any order
  for (int k = 0; k < kExactBlock; ++k)
    B += p[k];
  return round32_sum(acc, B);
}

// --------------------------------------------------------------------------
// The inputs of one GEMM in the forms the bodies read, per token:
//   xe, xo  [K/2] double: the even and the odd inputs (as rows()' scratch);
//   flag    [K/16]: bit 0, the block's double sum may be inexact or an input
//           is not finite (exact_block_flag: the scalar steps); bit 1, the
//           block has no exact int16 form (below; set with bit 0 too);
//   xi      [K] int16, pw [K/16] double (or null: no int16 forms): block b's
//           inputs are xi[16b + i] * pw[b] exactly, pw a power of two.
// A block's int16 form exists when its nonzero inputs span at most 7
// exponents: with L = e_max - 14 every input (8 significant bits, at least
// 2^(e_max - 7)) is an integer multiple of 2^L below 2^15 * 2^L.
// --------------------------------------------------------------------------

// 2^k exactly, without a library call (k within double's normal range).
inline double pow2d(int k) {
  const uint64_t bits = static_cast<uint64_t>(k + 1023) << 52;
  double d;
  std::memcpy(&d, &bits, 8);
  return d;
}

// Row groups of 4 interleaved at one and at two tokens.
#ifndef INT4_MOE_CPU_EXACT_Q1
  #define INT4_MOE_CPU_EXACT_Q1 2
#endif
#ifndef INT4_MOE_CPU_EXACT_Q2
  #define INT4_MOE_CPU_EXACT_Q2 2
#endif
constexpr int kExactQ1 = INT4_MOE_CPU_EXACT_Q1;
constexpr int kExactQ2 = INT4_MOE_CPU_EXACT_Q2;

struct ExactInputs {
  const double* xe;
  const double* xo;
  const uint8_t* flag;
  const int16_t* xi;
  const double* pw;
};

constexpr int kExactIntSpan = 7;

// One block of 16 bf16 inputs (k order) to its flag and int16 form.
inline void
exact_input_block(const uint16_t* v, uint8_t* flag, int16_t* xi, double* pw) {
  int lo = INT_MAX, hi = INT_MIN;
  bool finite = true;
  for (int k = 0; k < kExactBlock; ++k) {
    const uint16_t b = v[k];
    const int e8 = (b >> 7) & 0xFF, m = b & 0x7F;
    if (e8 == 0xFF) {
      finite = false;
      continue;
    }
    int e;
    if (e8 != 0) {
      e = e8 - 127;
    } else if (m != 0) {
      e = -133 + (31 - __builtin_clz(static_cast<unsigned>(m)));
    } else {
      continue;
    }
    lo = std::min(lo, e);
    hi = std::max(hi, e);
  }
  const bool slow = !finite || (hi != INT_MIN && hi - lo > kExactSpan);
  const bool no_int = slow || (hi != INT_MIN && hi - lo > kExactIntSpan);
  *flag = static_cast<uint8_t>((slow ? 1 : 0) | (no_int ? 2 : 0));
  if (xi == nullptr) return;
  if (no_int || hi == INT_MIN) {
    for (int k = 0; k < kExactBlock; ++k)
      xi[k] = 0;
    *pw = 1.0;
    return;
  }
  const int L = hi - 14;
  *pw = pow2d(L);
  const double up = pow2d(-L);  // exact: the product is an integer < 2^15
  for (int k = 0; k < kExactBlock; ++k)
    xi[k] = static_cast<int16_t>(static_cast<double>(bf16_to_f32(v[k])) * up);
}

// Output rows [n0, n1) for nt tokens (in.* per token: K/2, K/16 and K
// apart); out[t * ldo + n] = the fp32 accumulator.
inline void rows_exact_scalar(
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const ExactInputs& in,
    int nt,
    float* out,
    int ldo) {
  const int kb = K / 2, ng = K / G, nb = K / kExactBlock;
  double p[kExactBlock];
  for (int n = n0; n < n1; ++n) {
    const uint8_t* w = W + static_cast<int64_t>(n) * kb;
    const uint16_t* s = S + static_cast<int64_t>(n) * ng;
    for (int t = 0; t < nt; ++t) {
      const double* e = in.xe + static_cast<int64_t>(t) * kb;
      const double* o = in.xo + static_cast<int64_t>(t) * kb;
      const uint8_t* fl = in.flag + static_cast<int64_t>(t) * nb;
      float acc = 0.0f;
      for (int b = 0; b < nb; ++b) {
        exact_products(w, bf16_to_f32(s[(b * kExactBlock) / G]), e, o, b, p);
        acc = exact_step(acc, p, fl[b] & 1);
      }
      out[static_cast<int64_t>(t) * ldo + n] = acc;
    }
  }
}

#if INT4_MOE_CPU_X86

// What a lane needs to redo its block in the scalar steps.
struct ExactSlow {
  const uint8_t* const* w;  // the 4 rows
  const uint16_t* const* s;
  int G;
  const double* xe;  // the token's inputs
  const double* xo;
};

__attribute__((target("avx2,fma"), noinline)) inline void exact_fix4(
    __m256d& next,
    const double* acc,
    const double* B,
    const double* s,
    int mask,
    bool flagged,
    const ExactSlow& c,
    int b) {
  alignas(32) double r[4];
  _mm256_store_pd(r, next);
  double p[kExactBlock];
  for (int l = 0; l < 4; ++l) {
    if (!flagged && !((mask >> l) & 1)) continue;
    if (flagged || !std::isfinite(s[l])) {
      exact_products(
          c.w[l],
          bf16_to_f32(c.s[l][(b * kExactBlock) / c.G]),
          c.xe,
          c.xo,
          b,
          p);
      r[l] = exact_step_slow(static_cast<float>(acc[l]), p, kExactBlock);
    } else {
      exact_fixed_count().fetch_add(1, std::memory_order_relaxed);
      r[l] = round32_sum(acc[l], B[l]);
    }
  }
  next = _mm256_load_pd(r);
}

// Four rows' exact block sums B (one row per lane) added to their
// accumulators acc (fp32 values held as doubles), one rounding each. Lanes
// the fast path cannot round (see the top of this file) and every lane of a
// flagged block go through the scalar steps.
__attribute__((target("avx2,fma"), always_inline)) inline void
exact_step4(__m256d& acc, __m256d B, bool flagged, const ExactSlow& c, int b) {
  const __m256d s = _mm256_add_pd(acc, B);
  // a midpoint in double is a tie only when s is not exact (true ties are
  // common, and to nearest even is then right as it is). s is exact iff
  // s - acc == B and s - B == acc: of the two, the one subtracting the
  // larger operand is exact (Fast2Sum's first step), so an inexact s fails
  // it; NaN fails both and is caught as not finite below.
  const __m256d exact = _mm256_and_pd(
      _mm256_cmp_pd(_mm256_sub_pd(s, acc), B, _CMP_EQ_OQ),
      _mm256_cmp_pd(_mm256_sub_pd(s, B), acc, _CMP_EQ_OQ));
  const __m256i bits = _mm256_castpd_si256(s);
  const __m256i mid = _mm256_andnot_si256(
      _mm256_castpd_si256(exact),
      _mm256_cmpeq_epi64(
          _mm256_and_si256(bits, _mm256_set1_epi64x(0x1FFFFFFF)),
          _mm256_set1_epi64x(0x10000000)));
  const __m256d mag = _mm256_and_pd(
      s, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
  const __m256d small = _mm256_and_pd(
      _mm256_cmp_pd(mag, _mm256_set1_pd(0x1p-125), _CMP_LT_OQ),
      _mm256_cmp_pd(mag, _mm256_setzero_pd(), _CMP_NEQ_OQ));
  const __m256d bad = _mm256_cmp_pd(
      mag,
      _mm256_set1_pd(std::numeric_limits<double>::infinity()),
      _CMP_NLT_UQ);
  const int mask = _mm256_movemask_pd(
      _mm256_or_pd(_mm256_or_pd(_mm256_castsi256_pd(mid), small), bad));
  __m256d next = _mm256_cvtps_pd(_mm256_cvtpd_ps(s));
  if (flagged || mask) {
    alignas(32) double av[4], Bv[4], sv[4];
    _mm256_store_pd(av, acc);
    _mm256_store_pd(Bv, B);
    _mm256_store_pd(sv, s);
    exact_fix4(next, av, Bv, sv, mask, flagged, c, b);
  }
  acc = next;
}

// The four block sums of d[0..3] (4 partial sums each), one per lane.
__attribute__((target("avx2,fma"), always_inline)) inline __m256d
hsum4(const __m256d* d) {
  const __m256d t0 = _mm256_hadd_pd(d[0], d[1]);
  const __m256d t1 = _mm256_hadd_pd(d[2], d[3]);
  return _mm256_add_pd(
      _mm256_permute2f128_pd(t0, t1, 0x20),
      _mm256_permute2f128_pd(t0, t1, 0x31));
}

// The weights of a group, bf16_rtz(q * s) by raw nibble, as CardTable.
__attribute__((target("avx2,fma"))) inline CardTable
card_table_rtz(__m256 scale) {
  return card_table(scale, true);
}

// The same weights as int16 multiples of 2^(e_s - 7) (q * s has at most 12
// significant bits and |w| < 2^(e_s + 4): at most 11 bits as an integer),
// in CardTable's byte tables, for vpmaddwd. Only for a normal scale with
// -119 <= e_s <= 124 (int_ok): the scaling stays in float's range and
// 8 * s does not overflow.
struct IntTable {
  CardTable t;
  double f;  // 2^(e_s - 7)
  bool ok;
};

__attribute__((target("avx2,fma"))) inline IntTable int_table(uint16_t sb) {
  IntTable r;
  const int e8 = (sb >> 7) & 0xFF;
  const bool zero = (sb & 0x7FFF) == 0;
  const int es = e8 - 127;
  r.ok = zero || (e8 != 0 && es >= -119 && es <= 124);
  r.f = zero || !r.ok ? 1.0 : pow2d(es - 7);
  if (!r.ok) {
    r.t = {_mm256_setzero_si256(), _mm256_setzero_si256()};
    return r;
  }
  const __m256 scale = _mm256_set1_ps(bf16_to_f32(sb));
  // 2^(7 - e_s) as a float (7 - e_s in [-117, 126]: normal)
  const __m256 up = _mm256_castsi256_ps(
      _mm256_set1_epi32(zero ? 127 << 23 : (7 - es + 127) << 23));
  const __m256 top =
      _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0xffff0000u)));
  const __m256 qlo =
      _mm256_mul_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), scale);
  const __m256 qhi =
      _mm256_mul_ps(_mm256_setr_ps(-8, -7, -6, -5, -4, -3, -2, -1), scale);
  const __m256i lo =
      _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_and_ps(qlo, top), up));
  const __m256i hi =
      _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_and_ps(qhi, top), up));
  // 16 words in u order, then split into low and high bytes per lane
  __m256i w = _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0xD8);
  const __m256i split = _mm256_setr_epi8(
      0,
      2,
      4,
      6,
      8,
      10,
      12,
      14,
      1,
      3,
      5,
      7,
      9,
      11,
      13,
      15,
      0,
      2,
      4,
      6,
      8,
      10,
      12,
      14,
      1,
      3,
      5,
      7,
      9,
      11,
      13,
      15);
  w = _mm256_shuffle_epi8(w, split);
  const __m128i a = _mm256_castsi256_si128(w);
  const __m128i b = _mm256_extracti128_si256(w, 1);
  r.t = {
      _mm256_broadcastsi128_si256(_mm_unpacklo_epi64(a, b)),
      _mm256_broadcastsi128_si256(_mm_unpackhi_epi64(a, b))};
  return r;
}

// The int16 weights of two blocks (16 bytes) in k order: byte l gives
// k = 2l (low nibble) and 2l + 1 (high nibble); a holds the first 8 bytes'
// 16 weights, b the next 8 bytes'. The nibble indices of both blocks sit in
// one register (a's in the low lane), so two byte lookups (low and high
// bytes of the int16 table) and two unpacks serve 32 weights.
__attribute__((target("avx2,fma"), always_inline)) inline void
int_weights2(const CardTable& t, const uint8_t* p, __m256i& a, __m256i& b) {
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(raw, m4);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
  const __m256i idx = _mm256_inserti128_si256(
      _mm256_castsi128_si256(_mm_unpacklo_epi8(lo, hi)),
      _mm_unpackhi_epi8(lo, hi),
      1);
  const __m256i lb = _mm256_shuffle_epi8(t.tl, idx);
  const __m256i hb = _mm256_shuffle_epi8(t.th, idx);
  const __m256i w0 = _mm256_unpacklo_epi8(lb, hb);  // k 0..7 of each block
  const __m256i w1 = _mm256_unpackhi_epi8(lb, hb);  // k 8..15
  a = _mm256_permute2x128_si256(w0, w1, 0x20);
  b = _mm256_permute2x128_si256(w0, w1, 0x31);
}

// The four rows' int32 block sums of m[0..3] (8 pair sums each), one per
// lane, as doubles.
__attribute__((target("avx2,fma"), always_inline)) inline __m256d
isum4(const __m256i* m) {
  const __m256i h = _mm256_hadd_epi32(
      _mm256_hadd_epi32(m[0], m[1]), _mm256_hadd_epi32(m[2], m[3]));
  return _mm256_cvtepi32_pd(
      _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)));
}

// The double block sums of four rows for one token, block b (the chunk's
// lane vectors lo/hi of each row for that block's 8 bytes).
__attribute__((target("avx2,fma"), always_inline)) inline __m256d
dsum4(const __m256d (*wd)[4], const double* e, const double* o) {
  __m256d d[4];
  for (int r = 0; r < 4; ++r) {
    __m256d x = _mm256_mul_pd(wd[r][0], _mm256_loadu_pd(e));
    x = _mm256_fmadd_pd(wd[r][1], _mm256_loadu_pd(e + 4), x);
    x = _mm256_fmadd_pd(wd[r][2], _mm256_loadu_pd(o), x);
    d[r] = _mm256_fmadd_pd(wd[r][3], _mm256_loadu_pd(o + 4), x);
  }
  return hsum4(d);
}

  // Q groups of 4 rows for NT tokens (Q > 1 at one or two tokens: the
  // accumulators' chains, an add and two conversions per block, interleave).
  // Per block and token, the exact block sum comes from vpmaddwd over the
  // int16 forms (both exist: the token's block flag bit 1 clear, the group's
  // four scales int_ok) or from double fused multiply-adds; either is the
  // exact sum, so the choice never shows.
  #if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
  #endif
template <int NT, int Q>
__attribute__((target("avx2,fma"))) inline void rows_exact_avx2_nt(
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const ExactInputs& in,
    float* out,
    int ldo) {
  constexpr int R = 4 * Q;
  const int64_t kb = K / 2;
  const int ng = K / G, nb = K / kExactBlock;
  const int blocks_per_group = G / kExactBlock;
  const bool have_int = in.xi != nullptr;
  int n = n0;
  for (; n + R <= n1; n += R) {
    const uint8_t* w[R];
    const uint16_t* s[R];
    for (int r = 0; r < R; ++r) {
      w[r] = W + static_cast<int64_t>(n + r) * kb;
      s[r] = S + static_cast<int64_t>(n + r) * ng;
    }
    __m256d acc[Q][NT];
    for (int q = 0; q < Q; ++q)
      for (int t = 0; t < NT; ++t)
        acc[q][t] = _mm256_setzero_pd();
    for (int g = 0; g < ng; ++g) {
      IntTable it[R];
      bool qint[Q];
      __m256d f[Q];
      for (int q = 0; q < Q; ++q) {
        qint[q] = have_int;
        for (int r = 4 * q; r < 4 * q + 4; ++r) {
          if (have_int) {
            it[r] = int_table(s[r][g]);
            qint[q] = qint[q] && it[r].ok;
          }
          if (kPrefetchRows > 0)
            _mm_prefetch(
                reinterpret_cast<const char*>(
                    w[r] + R * kb + g * (blocks_per_group * 8)),
                _MM_HINT_T0);
        }
        f[q] = qint[q] ? _mm256_setr_pd(
                             it[4 * q].f,
                             it[4 * q + 1].f,
                             it[4 * q + 2].f,
                             it[4 * q + 3].f)
                       : _mm256_setzero_pd();
      }
      CardTable ft[R];
      bool ft_ready = false;
      __m256i wi[2][R];
      for (int bg = 0; bg < blocks_per_group; ++bg) {
        const int b = g * blocks_per_group + bg;
        const int j0 = 8 * b;
        // whether some token or row group takes the double sums here
        bool any_double = false;
        for (int q = 0; q < Q; ++q)
          any_double = any_double || !qint[q];
        for (int t = 0; t < NT && !any_double; ++t)
          any_double = (in.flag[t * nb + b] & 2) != 0;
        // int16 weights: both blocks of a 16-byte chunk at its first block
        if ((bg & 1) == 0)
          for (int q = 0; q < Q; ++q)
            if (qint[q])
              for (int r = 4 * q; r < 4 * q + 4; ++r)
                int_weights2(it[r].t, w[r] + j0, wi[0][r], wi[1][r]);
        __m256d wd[R][4];  // set where any_double; read only then
        if (any_double) {
          if (!ft_ready) {
            for (int r = 0; r < R; ++r)
              ft[r] = card_table_rtz(_mm256_set1_ps(bf16_to_f32(s[r][g])));
            ft_ready = true;
          }
          // the chunk holding the block: its first or second 8 bytes
          const int c0 = j0 & ~15, half = (j0 >> 3) & 1;
          for (int r = 0; r < R; ++r) {
            __m256 lo0, hi0, lo1, hi1;
            card_chunk(ft[r], w[r] + c0, lo0, hi0, lo1, hi1);
            const __m256 lo = half ? lo1 : lo0, hi = half ? hi1 : hi0;
            wd[r][0] = _mm256_cvtps_pd(_mm256_castps256_ps128(lo));
            wd[r][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(lo, 1));
            wd[r][2] = _mm256_cvtps_pd(_mm256_castps256_ps128(hi));
            wd[r][3] = _mm256_cvtps_pd(_mm256_extractf128_ps(hi, 1));
          }
        }
        for (int t = 0; t < NT; ++t) {
          const uint8_t fl = in.flag[t * nb + b];
          const __m256i x = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(
                  in.xi ? in.xi + static_cast<int64_t>(t) * K + 16 * b
                        : reinterpret_cast<const int16_t*>(in.flag)));
          for (int q = 0; q < Q; ++q) {
            __m256d B;
            if (qint[q] && !(fl & 2)) {
              __m256i m[4];
              for (int r = 0; r < 4; ++r)
                m[r] = _mm256_madd_epi16(wi[bg & 1][4 * q + r], x);
              B = _mm256_mul_pd(
                  _mm256_mul_pd(isum4(m), f[q]),
                  _mm256_set1_pd(in.pw[t * nb + b]));
            } else {
              B = dsum4(wd + 4 * q, in.xe + t * kb + j0, in.xo + t * kb + j0);
            }
            const ExactSlow c{
                w + 4 * q, s + 4 * q, G, in.xe + t * kb, in.xo + t * kb};
            exact_step4(acc[q][t], B, (fl & 1) != 0, c, b);
          }
        }
      }
    }
    for (int q = 0; q < Q; ++q)
      for (int t = 0; t < NT; ++t) {
        alignas(32) double a[4];
        _mm256_store_pd(a, acc[q][t]);
        for (int r = 0; r < 4; ++r)
          out[static_cast<int64_t>(t) * ldo + n + 4 * q + r] =
              static_cast<float>(a[r]);
      }
  }
  if (n < n1) {
    if constexpr (Q > 1)  // the rest in groups of 4, then row by row
      rows_exact_avx2_nt<NT, 1>(W, S, K, G, n, n1, in, out, ldo);
    else
      rows_exact_scalar(W, S, K, G, n, n1, in, NT, out, ldo);
  }
}
  #if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic pop
  #endif

__attribute__((target("avx2,fma"))) inline void rows_exact_avx2(
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const ExactInputs& in,
    int nt,
    float* out,
    int ldo) {
  const int64_t kb = K / 2, nb = K / kExactBlock;
  for (int t0 = 0; t0 < nt; t0 += 4) {
    const int tb = nt - t0 < 4 ? nt - t0 : 4;
    const ExactInputs part{
        in.xe + t0 * kb,
        in.xo + t0 * kb,
        in.flag + t0 * nb,
        in.xi ? in.xi + static_cast<int64_t>(t0) * K : nullptr,
        in.pw ? in.pw + t0 * nb : nullptr};
    float* y = out + static_cast<int64_t>(t0) * ldo;
    switch (tb) {
      case 1:
        rows_exact_avx2_nt<1, kExactQ1>(W, S, K, G, n0, n1, part, y, ldo);
        break;
      case 2:
        rows_exact_avx2_nt<2, kExactQ2>(W, S, K, G, n0, n1, part, y, ldo);
        break;
      case 3:
        rows_exact_avx2_nt<3, 1>(W, S, K, G, n0, n1, part, y, ldo);
        break;
      default:
        rows_exact_avx2_nt<4, 1>(W, S, K, G, n0, n1, part, y, ldo);
        break;
    }
  }
}

#endif  // INT4_MOE_CPU_X86

// The dispatch of this arithmetic: level 1 the scalar code, every level
// from 2 up (AVX2, AVX-VNNI, the AVX-512 levels, AMX and level 8) the AVX2 +
// FMA body. All compute the same exact function.
inline void rows_exact(
    int level,
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const ExactInputs& in,
    int nt,
    float* out,
    int ldo) {
#if INT4_MOE_CPU_X86
  if (level >= 2) {
    rows_exact_avx2(W, S, K, G, n0, n1, in, nt, out, ldo);
    return;
  }
#endif
  rows_exact_scalar(W, S, K, G, n0, n1, in, nt, out, ldo);
}

}  // namespace cpu_moe
