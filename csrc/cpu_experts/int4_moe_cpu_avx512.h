// The W4A16 inner function rows() on AVX-512 (F, BW, DQ, VL) and with
// AVX-512 BF16, included by int4_moe_cpu_impl.h after its scalar and AVX2
// code. Each function carries its own target attribute, so one build serves
// every host and the dispatch picks by CPUID (int4_moe_cpu_isa.h).
//
// The contract is the scalar code's lane order (int4_moe_cpu_impl.h, the
// comment above rows_scalar): per output row and token, 8 fp32 lanes, lane l
// of chunk c taking bytes j0 + l and j0 + 8 + l in a fixed sequence of
// fused multiply-adds, then lane_sum. A 512-bit register is two such 8-lane
// groups: lanes 0..7 are row n, lanes 8..15 row n + 1 (two rows, not 16
// lanes of one row: the 8-lane order is the contract). Every lane then takes
// exactly the scalar code's operations in its order, so the outputs equal
// the scalar code's bit for bit in every arith mode.
//
// AVX-512 BF16 (vdpbf16ps) is used for the grouped GEMM's arithmetic
// (kBf16Weights) only, and only where it equals the contract:
//   1. It computes, per fp32 lane, acc = fma(a[2i+1], b[2i+1], acc), then
//      acc = fma(a[2i], b[2i], acc), each rounded to nearest even (Intel
//      SDE 9.53's model; probed at first use on the host, below). The
//      weights bf16(q * s) and the inputs (bf16 values) are bf16, so each
//      product is exact in fp32, and with the low nibble's pair in the odd
//      word (taken first) the two steps are the scalar code's two fused
//      multiply-adds of a byte.
//   2. It treats subnormal inputs as zero and flushes subnormal results.
//      The call runs it only when neither can occur: every nonzero product
//      is a multiple of 2^(e_w + e_x - 14) (8-bit significands), so no
//      partial sum is subnormal if e_s,min + e_x,min >= -112 (e_w >= e_s for
//      |q| >= 1); inputs and scales finite and normal, the sums far from
//      overflow, inputs with no bits below bf16. Otherwise the call takes
//      the AVX-512 F code (the same result).
// The int4 GEMV's arithmetic (kChunkScaled) sums 4 products per lane before
// a scale is applied: vdpbf16ps would need bf16 nibble values (exact) but
// rounds after each pair, the same as the scalar code, while the chunk sum
// starts with a plain product, not an accumulate: it is the AVX-512 F code.

#pragma once

#if INT4_MOE_CPU_X86

  #define INT4_MOE_CPU_T512 "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma"
  #define INT4_MOE_CPU_T512BF16 \
    "avx512f,avx512bw,avx512dq,avx512vl,avx512bf16,avx2,fma"

namespace cpu_moe {

// 8 bytes of row a and 8 bytes of row b, zero-extended to 16 dwords (lanes
// 0..7 from a, 8..15 from b).
__attribute__((target(INT4_MOE_CPU_T512), always_inline)) inline __m512i
load2x8_u32(const uint8_t* a, const uint8_t* b) {
  __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a));
  v = _mm_castpd_si128(
      _mm_loadh_pd(_mm_castsi128_pd(v), reinterpret_cast<const double*>(b)));
  return _mm512_cvtepu8_epi32(v);
}

// The same exact float trick as nib_to_ps (int4_moe_cpu_impl.h).
__attribute__((target(INT4_MOE_CPU_T512), always_inline)) inline __m512
nib_to_ps512(__m512i u) {
  const __m512i magic = _mm512_set1_epi32(0x4B000008);
  const __m512 bias = _mm512_set1_ps(8388616.0f);
  return _mm512_sub_ps(_mm512_castsi512_ps(_mm512_xor_si512(u, magic)), bias);
}

// 8 fp32 activations broadcast to both 8-lane halves.
__attribute__((target(INT4_MOE_CPU_T512), always_inline)) inline __m512
bcast8(const float* p) {
  return _mm512_castpd_ps(_mm512_broadcast_f64x4(
      _mm256_loadu_pd(reinterpret_cast<const double*>(p))));
}

__attribute__((target(INT4_MOE_CPU_T512), always_inline)) inline __m512
round_bf16_ps512(__m512 v) {
  const __m512i b = _mm512_castps_si512(v);
  const __m512i lsb =
      _mm512_and_si512(_mm512_srli_epi32(b, 16), _mm512_set1_epi32(1));
  const __m512i r =
      _mm512_add_epi32(b, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7fff)));
  return _mm512_castsi512_ps(
      _mm512_and_si512(r, _mm512_set1_epi32(static_cast<int>(0xffff0000u))));
}

// The grouped GEMM's weights of one group for a row pair: 32 bf16 words,
// bf16(q * s_a) for the raw nibble u = 0..15 of row a (words 0..15), then of
// row b (words 16..31); q = u for u < 8, u - 16 otherwise (as card_table).
__attribute__((target(INT4_MOE_CPU_T512), always_inline)) inline __m512i
card_table512(float sa, float sb) {
  const __m512 q =
      _mm512_setr_ps(0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1);
  const __m512i wa = _mm512_srli_epi32(
      _mm512_castps_si512(
          round_bf16_ps512(_mm512_mul_ps(q, _mm512_set1_ps(sa)))),
      16);
  const __m512i wb = _mm512_srli_epi32(
      _mm512_castps_si512(
          round_bf16_ps512(_mm512_mul_ps(q, _mm512_set1_ps(sb)))),
      16);
  return _mm512_inserti64x4(
      _mm512_castsi256_si512(_mm512_cvtepi32_epi16(wa)),
      _mm512_cvtepi32_epi16(wb),
      1);
}

// Per dword lane l of the result (lanes 0..7 row a byte p[l], 8..15 row b):
// low word bf16(q(hi nibble) * s), high word bf16(q(lo nibble) * s). One
// vpermw against the pair's table.
__attribute__((target(INT4_MOE_CPU_T512), always_inline)) inline __m512i
card_pairs512(const __m512i& tab, const uint8_t* a, const uint8_t* b) {
  const __m512i v = load2x8_u32(a, b);
  const __m512i idx = _mm512_or_si512(
      _mm512_or_si512(
          _mm512_srli_epi32(v, 4),
          _mm512_slli_epi32(_mm512_and_si512(v, _mm512_set1_epi32(0x0F)), 16)),
      _mm512_setr_epi32(
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0x00100010,
          0x00100010,
          0x00100010,
          0x00100010,
          0x00100010,
          0x00100010,
          0x00100010,
          0x00100010));
  return _mm512_permutexvar_epi16(idx, tab);
}

// One 32-element chunk for RP row pairs and NT tokens (AVX-512 F).
// w[2p], w[2p + 1]: the rows of pair p. acc[p][t]: lanes 0..7 row 2p,
// 8..15 row 2p + 1.
template <int NT, int RP, Arith A>
__attribute__((target(INT4_MOE_CPU_T512), always_inline)) inline void chunk512(
    const uint8_t* const* w,
    const __m512* scale,
    const __m512i* tab,
    int j0,
    int64_t kb,
    const float* xe,
    const float* xo,
    __m512 (*acc)[NT]) {
  for (int p = 0; p < RP; ++p) {
    __m512 lo0, hi0, lo1, hi1;
    if constexpr (A == Arith::kBf16Weights) {
      const __m512i hl = _mm512_set1_epi32(static_cast<int>(0xFFFF0000u));
      const __m512i p0 =
          card_pairs512(tab[p], w[2 * p] + j0, w[2 * p + 1] + j0);
      const __m512i p1 =
          card_pairs512(tab[p], w[2 * p] + j0 + 8, w[2 * p + 1] + j0 + 8);
      lo0 = _mm512_castsi512_ps(_mm512_and_si512(p0, hl));
      hi0 = _mm512_castsi512_ps(_mm512_slli_epi32(p0, 16));
      lo1 = _mm512_castsi512_ps(_mm512_and_si512(p1, hl));
      hi1 = _mm512_castsi512_ps(_mm512_slli_epi32(p1, 16));
    } else {
      const __m512i m = _mm512_set1_epi32(0x0F);
      const __m512i v0 = load2x8_u32(w[2 * p] + j0, w[2 * p + 1] + j0);
      const __m512i v1 = load2x8_u32(w[2 * p] + j0 + 8, w[2 * p + 1] + j0 + 8);
      lo0 = nib_to_ps512(_mm512_and_si512(v0, m));
      hi0 = nib_to_ps512(_mm512_srli_epi32(v0, 4));
      lo1 = nib_to_ps512(_mm512_and_si512(v1, m));
      hi1 = nib_to_ps512(_mm512_srli_epi32(v1, 4));
    }
    for (int t = 0; t < NT; ++t) {
      const float* e = xe + t * kb + j0;
      const float* o = xo + t * kb + j0;
      if constexpr (A == Arith::kBf16Weights) {
        __m512 a = _mm512_fmadd_ps(lo0, bcast8(e), acc[p][t]);
        a = _mm512_fmadd_ps(hi0, bcast8(o), a);
        a = _mm512_fmadd_ps(lo1, bcast8(e + 8), a);
        acc[p][t] = _mm512_fmadd_ps(hi1, bcast8(o + 8), a);
      } else {
        __m512 d = _mm512_mul_ps(lo0, bcast8(e));
        d = _mm512_fmadd_ps(hi0, bcast8(o), d);
        d = _mm512_fmadd_ps(lo1, bcast8(e + 8), d);
        d = _mm512_fmadd_ps(hi1, bcast8(o + 8), d);
        acc[p][t] = _mm512_fmadd_ps(d, scale[p], acc[p][t]);
      }
    }
  }
}

// The same chunk with vdpbf16ps (A = kBf16Weights only). xp: per token the
// input pairs, dword j = bf16 x[2j] in the high word, x[2j + 1] in the low.
template <int NT, int RP>
__attribute__((target(INT4_MOE_CPU_T512BF16), always_inline)) inline void
chunk512_bf16(
    const uint8_t* const* w,
    const __m512i* tab,
    int j0,
    int64_t kb,
    const uint32_t* xp,
    __m512 (*acc)[NT]) {
  for (int p = 0; p < RP; ++p) {
    const __m512i p0 = card_pairs512(tab[p], w[2 * p] + j0, w[2 * p + 1] + j0);
    const __m512i p1 =
        card_pairs512(tab[p], w[2 * p] + j0 + 8, w[2 * p + 1] + j0 + 8);
    for (int t = 0; t < NT; ++t) {
      const float* x = reinterpret_cast<const float*>(xp + t * kb + j0);
      __m512 a = _mm512_dpbf16_ps(
          acc[p][t], (__m512bh)p0, (__m512bh)_mm512_castps_si512(bcast8(x)));
      acc[p][t] = _mm512_dpbf16_ps(
          a, (__m512bh)p1, (__m512bh)_mm512_castps_si512(bcast8(x + 8)));
    }
  }
}

// Groups [g0, g1) of the row pairs of [n0, n1) (n1 - n0 even), RP pairs side
// by side, NT tokens; accumulators in acc_io[(pair index) * NT + t].
template <int NT, int RP, Arith A>
__attribute__((target(INT4_MOE_CPU_T512))) inline void panel512(
    const uint8_t* W,
    const uint16_t* S,
    int64_t kb,
    int ng,
    int chunks_per_group,
    int n0,
    int n1,
    int g0,
    int g1,
    const float* xe,
    const float* xo,
    __m512* acc_io) {
  int n = n0;
  for (; n + 2 * RP <= n1; n += 2 * RP) {
    const uint8_t* w[2 * RP];
    const uint16_t* s[2 * RP];
    __m512 acc[RP][NT];
    for (int r = 0; r < 2 * RP; ++r) {
      w[r] = W + static_cast<int64_t>(n + r) * kb;
      s[r] = S + static_cast<int64_t>(n + r) * ng;
    }
    for (int p = 0; p < RP; ++p)
      for (int t = 0; t < NT; ++t)
        acc[p][t] = acc_io[((n - n0) / 2 + p) * NT + t];
    for (int g = g0; g < g1; ++g) {
      __m512 scale[RP];
      __m512i tab[RP];
      for (int p = 0; p < RP; ++p) {
        const float sa = bf16_to_f32(s[2 * p][g]);
        const float sb = bf16_to_f32(s[2 * p + 1][g]);
        if constexpr (A == Arith::kBf16Weights)
          tab[p] = card_table512(sa, sb);
        else
          scale[p] =
              _mm512_insertf32x8(_mm512_set1_ps(sa), _mm256_set1_ps(sb), 1);
        if (kPrefetchRows > 0)
          for (int r = 2 * p; r < 2 * p + 2; ++r)
            _mm_prefetch(
                reinterpret_cast<const char*>(
                    w[r] + kPrefetchRows * kb + g * (chunks_per_group * 16)),
                _MM_HINT_T0);
      }
      for (int cc = 0; cc < chunks_per_group; ++cc)
        chunk512<NT, RP, A>(
            w,
            scale,
            tab,
            (g * chunks_per_group + cc) * (kChunk / 2),
            kb,
            xe,
            xo,
            acc);
    }
    for (int p = 0; p < RP; ++p)
      for (int t = 0; t < NT; ++t)
        acc_io[((n - n0) / 2 + p) * NT + t] = acc[p][t];
  }
  if (n < n1)
    panel512<NT, 1, A>(
        W,
        S,
        kb,
        ng,
        chunks_per_group,
        n,
        n1,
        g0,
        g1,
        xe,
        xo,
        acc_io + ((n - n0) / 2) * NT);
}

template <int NT, int RP, Arith A>
__attribute__((target(INT4_MOE_CPU_T512))) inline void rows512_nt(
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const float* xe,
    const float* xo,
    float* out,
    int ldo) {
  const int64_t kb = K / 2;
  const int ng = K / G;
  const int chunks_per_group = G / kChunk;
  const int per_panel = std::max(1, kPanelBytes / (NT * G * 4));
  alignas(64) __m512 acc[(kRowBlock / 2) * NT];
  for (int b0 = n0; b0 < n1; b0 += kRowBlock) {
    const int b1 = std::min(n1, b0 + kRowBlock);  // b1 - b0 even
    for (int i = 0; i < ((b1 - b0) / 2) * NT; ++i)
      acc[i] = _mm512_setzero_ps();
    for (int g0 = 0; g0 < ng; g0 += per_panel)
      panel512<NT, RP, A>(
          W,
          S,
          kb,
          ng,
          chunks_per_group,
          b0,
          b1,
          g0,
          std::min(ng, g0 + per_panel),
          xe,
          xo,
          acc);
    for (int n = b0; n < b1; n += 2)
      for (int t = 0; t < NT; ++t) {
        alignas(64) float l[16];
        _mm512_store_ps(l, acc[((n - b0) / 2) * NT + t]);
        out[static_cast<int64_t>(t) * ldo + n] = lane_sum(l);
        out[static_cast<int64_t>(t) * ldo + n + 1] = lane_sum(l + 8);
      }
  }
}

  #ifndef INT4_MOE_CPU_RP1
    #define INT4_MOE_CPU_RP1 2
  #endif
constexpr int kRP1 = INT4_MOE_CPU_RP1;  // row pairs side by side at 1 token

template <Arith A>
__attribute__((target(INT4_MOE_CPU_T512))) inline void rows512_a(
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const float* xe,
    const float* xo,
    int nt,
    float* out,
    int ldo) {
  const int64_t kb = K / 2;
  for (int t0 = 0; t0 < nt; t0 += kTBAvx2) {
    const int tb = nt - t0 < kTBAvx2 ? nt - t0 : kTBAvx2;
    const float* e = xe + t0 * kb;
    const float* o = xo + t0 * kb;
    float* y = out + static_cast<int64_t>(t0) * ldo;
    switch (tb) {
      case 1:
        rows512_nt<1, kRP1, A>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      case 2:
        rows512_nt<2, 1, A>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      case 3:
        rows512_nt<3, 1, A>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      case 4:
        rows512_nt<4, 1, A>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      case 5:
        rows512_nt<5, 1, A>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      case 6:
        rows512_nt<6, 1, A>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      case 7:
        rows512_nt<7, 1, A>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      default:
        rows512_nt<8, 1, A>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
    }
  }
}

// rows512_nt / panel512 for the vdpbf16ps chunk (rows [n0, n1), n1 - n0
// even): the same blocking, the accumulators in the same lanes.
template <int NT, int RP>
__attribute__((target(INT4_MOE_CPU_T512BF16))) inline void panel512_bf16(
    const uint8_t* W,
    const uint16_t* S,
    int64_t kb,
    int ng,
    int chunks_per_group,
    int n0,
    int n1,
    int g0,
    int g1,
    const uint32_t* xp,
    __m512* acc_io) {
  int n = n0;
  for (; n + 2 * RP <= n1; n += 2 * RP) {
    const uint8_t* w[2 * RP];
    const uint16_t* s[2 * RP];
    __m512 acc[RP][NT];
    for (int r = 0; r < 2 * RP; ++r) {
      w[r] = W + static_cast<int64_t>(n + r) * kb;
      s[r] = S + static_cast<int64_t>(n + r) * ng;
    }
    for (int p = 0; p < RP; ++p)
      for (int t = 0; t < NT; ++t)
        acc[p][t] = acc_io[((n - n0) / 2 + p) * NT + t];
    for (int g = g0; g < g1; ++g) {
      __m512i tab[RP];
      for (int p = 0; p < RP; ++p) {
        tab[p] = card_table512(
            bf16_to_f32(s[2 * p][g]), bf16_to_f32(s[2 * p + 1][g]));
        if (kPrefetchRows > 0)
          for (int r = 2 * p; r < 2 * p + 2; ++r)
            _mm_prefetch(
                reinterpret_cast<const char*>(
                    w[r] + kPrefetchRows * kb + g * (chunks_per_group * 16)),
                _MM_HINT_T0);
      }
      for (int cc = 0; cc < chunks_per_group; ++cc)
        chunk512_bf16<NT, RP>(
            w, tab, (g * chunks_per_group + cc) * (kChunk / 2), kb, xp, acc);
    }
    for (int p = 0; p < RP; ++p)
      for (int t = 0; t < NT; ++t)
        acc_io[((n - n0) / 2 + p) * NT + t] = acc[p][t];
  }
  if (n < n1)
    panel512_bf16<NT, 1>(
        W,
        S,
        kb,
        ng,
        chunks_per_group,
        n,
        n1,
        g0,
        g1,
        xp,
        acc_io + ((n - n0) / 2) * NT);
}

template <int NT, int RP>
__attribute__((target(INT4_MOE_CPU_T512BF16))) inline void rows512_nt_bf16(
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const uint32_t* xp,
    float* out,
    int ldo) {
  const int64_t kb = K / 2;
  const int ng = K / G;
  const int chunks_per_group = G / kChunk;
  // the pairs are one dword per two elements: half the bytes of xe + xo
  const int per_panel = std::max(1, kPanelBytes / (NT * G * 2));
  alignas(64) __m512 acc[(kRowBlock / 2) * NT];
  for (int b0 = n0; b0 < n1; b0 += kRowBlock) {
    const int b1 = std::min(n1, b0 + kRowBlock);
    for (int i = 0; i < ((b1 - b0) / 2) * NT; ++i)
      acc[i] = _mm512_setzero_ps();
    for (int g0 = 0; g0 < ng; g0 += per_panel)
      panel512_bf16<NT, RP>(
          W,
          S,
          kb,
          ng,
          chunks_per_group,
          b0,
          b1,
          g0,
          std::min(ng, g0 + per_panel),
          xp,
          acc);
    for (int n = b0; n < b1; n += 2)
      for (int t = 0; t < NT; ++t) {
        alignas(64) float l[16];
        _mm512_store_ps(l, acc[((n - b0) / 2) * NT + t]);
        out[static_cast<int64_t>(t) * ldo + n] = lane_sum(l);
        out[static_cast<int64_t>(t) * ldo + n + 1] = lane_sum(l + 8);
      }
  }
}

__attribute__((target(INT4_MOE_CPU_T512BF16))) inline void rows512_bf16(
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const uint32_t* xp,
    int nt,
    float* out,
    int ldo) {
  const int64_t kb = K / 2;
  for (int t0 = 0; t0 < nt; t0 += kTBAvx2) {
    const int tb = nt - t0 < kTBAvx2 ? nt - t0 : kTBAvx2;
    const uint32_t* p = xp + t0 * kb;
    float* y = out + static_cast<int64_t>(t0) * ldo;
    switch (tb) {
      case 1:
        rows512_nt_bf16<1, kRP1>(W, S, K, G, n0, n1, p, y, ldo);
        break;
      case 2:
        rows512_nt_bf16<2, 1>(W, S, K, G, n0, n1, p, y, ldo);
        break;
      case 3:
        rows512_nt_bf16<3, 1>(W, S, K, G, n0, n1, p, y, ldo);
        break;
      case 4:
        rows512_nt_bf16<4, 1>(W, S, K, G, n0, n1, p, y, ldo);
        break;
      case 5:
        rows512_nt_bf16<5, 1>(W, S, K, G, n0, n1, p, y, ldo);
        break;
      case 6:
        rows512_nt_bf16<6, 1>(W, S, K, G, n0, n1, p, y, ldo);
        break;
      case 7:
        rows512_nt_bf16<7, 1>(W, S, K, G, n0, n1, p, y, ldo);
        break;
      default:
        rows512_nt_bf16<8, 1>(W, S, K, G, n0, n1, p, y, ldo);
        break;
    }
  }
}

// AVX-512 F: an odd last row goes to the AVX2 code (the same lane order).
__attribute__((target(INT4_MOE_CPU_T512))) inline void rows_avx512(
    Arith arith,
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const float* xe,
    const float* xo,
    int nt,
    float* out,
    int ldo) {
  const int m = n0 + ((n1 - n0) & ~1);
  if (m > n0) {
    if (arith == Arith::kBf16Weights)
      rows512_a<Arith::kBf16Weights>(W, S, K, G, n0, m, xe, xo, nt, out, ldo);
    else
      rows512_a<Arith::kChunkScaled>(W, S, K, G, n0, m, xe, xo, nt, out, ldo);
  }
  if (m < n1) rows_avx2(arith, W, S, K, G, m, n1, xe, xo, nt, out, ldo);
}

// --------------------------------------------------------------------------
// AVX-512 BF16: the probe of the instruction's semantics and the guard.
// --------------------------------------------------------------------------

namespace detail {

inline uint16_t bf16_bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return static_cast<uint16_t>(b >> 16);
}

__attribute__((target(INT4_MOE_CPU_T512BF16))) inline float
dpbf16_one(float acc, float hi_a, float hi_b, float lo_a, float lo_b) {
  // word 2i + 1 (high) = hi_*, word 2i (low) = lo_*
  const uint32_t a = bf16_bits(lo_a) | (uint32_t(bf16_bits(hi_a)) << 16);
  const uint32_t b = bf16_bits(lo_b) | (uint32_t(bf16_bits(hi_b)) << 16);
  const __m512 r = _mm512_dpbf16_ps(
      _mm512_set1_ps(acc),
      (__m512bh)_mm512_set1_epi32(static_cast<int>(a)),
      (__m512bh)_mm512_set1_epi32(static_cast<int>(b)));
  return _mm512_cvtss_f32(r);
}

// vdpbf16ps behaves as two fused multiply-adds, the high word's first, each
// rounded: checked on values where the order and an intermediate rounding
// change the result.
inline bool probe_dpbf16() {
  // order: (1 + (-1) * 1) + 2^-15 * 2^-15 = 2^-30; the other order gives 0
  if (dpbf16_one(1.0f, -1.0f, 1.0f, 0x1p-15f, 0x1p-15f) != 0x1p-30f)
    return false;
  // a rounding after each step: 1 + 2^-24 + 2^-24 is 1 (twice a tie to
  // even), with one rounding 1 + 2^-23
  if (dpbf16_one(1.0f, 0x1p-12f, 0x1p-12f, 0x1p-12f, 0x1p-12f) != 1.0f)
    return false;
  // nearest even on a tie: (1 + 2^-23) + 2^-24 -> 1 + 2^-22
  if (dpbf16_one(1.0f + 0x1p-23f, 0x1p-12f, 0x1p-12f, 0.0f, 0.0f) !=
      1.0f + 0x1p-22f)
    return false;
  return true;
}

}  // namespace detail

// Whether the vdpbf16ps code may be used on this host (probed once).
inline bool bf16_body_ok() {
  static const bool ok = level_supported(6) && detail::probe_dpbf16();
  return ok;
}

// Bumped by run_calls whenever it is about to rewrite the inputs rows()
// reads; the per-thread caches below are valid for one value. A caller of
// rows() that rewrites an input in place between two calls bumps it too.
inline std::atomic<uint64_t>& rows_input_epoch() {
  static std::atomic<uint64_t> e{0};
  return e;
}

// One input (nt tokens of K elements, as xe/xo) in a derived form, built
// once per input and layer call on each thread that reads it: the GEMM1
// items of an expert all read the same input, the GEMM2 items every
// expert's.
template <typename Entry>
struct InputCache {
  uint64_t epoch = ~uint64_t(0);
  std::vector<Entry> entries;
  Entry* find(const float* xe, int nt, int K, bool& fresh) {
    const uint64_t now = rows_input_epoch().load(std::memory_order_acquire);
    if (now != epoch) {
      entries.clear();
      epoch = now;
    }
    for (Entry& e : entries)
      if (e.xe == xe && e.nt == nt && e.K == K) {
        fresh = false;
        return &e;
      }
    if (entries.size() >= 256) entries.clear();
    entries.emplace_back();
    Entry& e = entries.back();
    e.xe = xe;
    e.nt = nt;
    e.K = K;
    fresh = true;
    return &e;
  }
};

// The input pairs of nt tokens for vdpbf16ps (dword j: bf16 x[2j] high, x[2j+1]
// low) and the input's part of the guard (top of this file).
struct PairInput {
  const float* xe = nullptr;
  int nt = 0, K = 0;
  std::vector<uint32_t> xp;
  bool bf16_values = false;       // no bits below bf16
  int ex_min = 0xFF, ex_max = 0;  // biased exponents of the nonzero values
};

__attribute__((target(INT4_MOE_CPU_T512BF16))) inline void
build_pairs(PairInput& p, const float* xe, const float* xo) {
  const int64_t n = static_cast<int64_t>(p.nt) * (p.K / 2);
  p.xp.resize(static_cast<size_t>(n));
  const __m512i expm = _mm512_set1_epi32(0xFF);
  const __m512i absm = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512i lowm = _mm512_set1_epi32(0xFFFF);
  __m512i emin = _mm512_set1_epi32(0xFF), emax = _mm512_setzero_si512();
  __m512i low = _mm512_setzero_si512();
  for (int64_t i = 0; i < n; i += 16) {  // K / 2 is a multiple of 16
    const __m512i e = _mm512_loadu_si512(xe + i);
    const __m512i o = _mm512_loadu_si512(xo + i);
    _mm512_storeu_si512(
        p.xp.data() + i, _mm512_or_si512(e, _mm512_srli_epi32(o, 16)));
    low = _mm512_or_si512(low, _mm512_and_si512(_mm512_or_si512(e, o), lowm));
    const __mmask16 nze = _mm512_test_epi32_mask(e, absm);
    const __mmask16 nzo = _mm512_test_epi32_mask(o, absm);
    const __m512i exe = _mm512_and_si512(_mm512_srli_epi32(e, 23), expm);
    const __m512i exo = _mm512_and_si512(_mm512_srli_epi32(o, 23), expm);
    emin = _mm512_mask_min_epu32(emin, nze, emin, exe);
    emin = _mm512_mask_min_epu32(emin, nzo, emin, exo);
    emax = _mm512_mask_max_epu32(emax, nze, emax, exe);
    emax = _mm512_mask_max_epu32(emax, nzo, emax, exo);
  }
  p.bf16_values = _mm512_reduce_or_epi32(low) == 0;
  p.ex_min = static_cast<int>(_mm512_reduce_min_epu32(emin));
  p.ex_max = static_cast<int>(_mm512_reduce_max_epu32(emax));
}

// The guard for rows [n0, n1) of S with this input.
inline bool bf16_guard(
    const PairInput& p, const uint16_t* S, int K, int G, int n0, int n1) {
  if (!p.bf16_values) return false;
  const int ng = K / G;
  int es_min = 0xFF, es_max = 0;
  for (int n = n0; n < n1; ++n) {
    const uint16_t* s = S + static_cast<int64_t>(n) * ng;
    for (int g = 0; g < ng; ++g) {
      if ((s[g] & 0x7FFF) == 0) continue;
      const int ex = (s[g] >> 7) & 0xFF;
      es_min = ex < es_min ? ex : es_min;
      es_max = ex > es_max ? ex : es_max;
    }
  }
  // biased exponents (of nonzero values; 0xFF / 0 when there are none):
  // normal and finite inputs and scales (a subnormal has exponent 0,
  // Inf/NaN 255), then no subnormal partial sum and far from overflow
  if (p.ex_min < 1 || es_min < 1 || p.ex_max > 254 || es_max > 254)
    return false;
  if (p.ex_max == 0 || es_max == 0) return true;  // exact zeros only
  return p.ex_min + es_min >= 2 * 127 - 112 &&
         p.ex_max + es_max <= 2 * 127 + 100;
}

inline std::atomic<int64_t>& bf16_fallbacks() {
  static std::atomic<int64_t> n{0};
  return n;
}

// AVX-512 BF16 code for the grouped GEMM's arithmetic where the guard holds;
// everything else is rows_avx512.
__attribute__((target(INT4_MOE_CPU_T512BF16))) inline void rows_avx512bf16(
    Arith arith,
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const float* xe,
    const float* xo,
    int nt,
    float* out,
    int ldo) {
  const int m = n0 + ((n1 - n0) & ~1);
  if (arith != Arith::kBf16Weights || m == n0 || !bf16_body_ok()) {
    rows_avx512(arith, W, S, K, G, n0, n1, xe, xo, nt, out, ldo);
    return;
  }
  static thread_local InputCache<PairInput> cache;
  bool fresh = false;
  PairInput* p = cache.find(xe, nt, K, fresh);
  if (fresh) build_pairs(*p, xe, xo);
  if (!bf16_guard(*p, S, K, G, n0, m)) {
    bf16_fallbacks().fetch_add(1, std::memory_order_relaxed);
    rows_avx512(arith, W, S, K, G, n0, n1, xe, xo, nt, out, ldo);
    return;
  }
  rows512_bf16(W, S, K, G, n0, m, p->xp.data(), nt, out, ldo);
  if (m < n1) rows_avx2(arith, W, S, K, G, m, n1, xe, xo, nt, out, ldo);
}

}  // namespace cpu_moe

#endif  // INT4_MOE_CPU_X86
