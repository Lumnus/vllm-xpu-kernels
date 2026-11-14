// The W4A8 op's integer dot (int4_moe_cpu_a8_impl.h gemm_rows) on AVX-VNNI,
// AVX-512 VNNI and AMX-INT8, included by int4_moe_cpu_a8_impl.h before its
// dispatch. Each function carries its own target attribute; the dispatch
// picks by CPUID (int4_moe_cpu_isa.h).
//
// Exactness: the only ISA-specific step is the int32 dot of a flush chunk,
// sum over the chunk of u * q with u = w + 8 (0..15, unsigned) and q the
// int8 activation, minus 8 * sum(q) (the chunk sums of QRows::cs). Integer
// sums are exact in any order, so every path gives the same int32 per
// (row, token, chunk); the fp32 steps after it (scale, fused or separate
// multiply-add over the chunks in order, the token scale, bf16) are the
// AVX2 code's per output element. So the outputs equal the scalar code's
// bit for bit.
//
// No overflow: vpdpbusd (and tdpbusd) multiply u8 by s8 into 16-bit
// products without saturation and add groups of four into an int32 lane,
// wrapping only beyond 2^31. A lane of a flush chunk (at most kMaxFlush =
// 512 elements) collects at most 512 products of |u * q| <= 15 * 128 =
// 1920: |sum| <= 983,040; the correction 8 * |sum(q)| <= 8 * 512 * 128 =
// 524,288; the corrected dot |sum(w * q)| <= 512 * 8 * 128 = 524,288. All
// far inside int32.

#pragma once

#if INT4_MOE_CPU_X86

  #define INT4_MOE_CPU_TVNNI256 "avx2,fma,avxvnni"
  #define INT4_MOE_CPU_TVNNI512 \
    "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma"
  #define INT4_MOE_CPU_TAMX8 \
    "amx-tile,amx-int8,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma"

namespace cpu_moe {
namespace a8 {

// The fp32 steps of 8 rows for one activation row, after the int32 dots of
// a flush chunk (d: the 8 rows' corrected dots): the AVX2 code's.
__attribute__((target("avx2,fma"), always_inline)) inline __m256 fp32_step(
    const GemmArgs& a,
    const QRows& Q,
    int64_t row,
    int k0,
    __m256i d,
    __m256 wsc,
    __m256 F) {
  const __m256 df = _mm256_cvtepi32_ps(d);
  __m256 m = wsc;
  if (Q.ag)
    m = _mm256_mul_ps(wsc, _mm256_set1_ps(Q.as[row * Q.nas + k0 / Q.ag]));
  return a.acc == Acc::kFused ? _mm256_fmadd_ps(df, m, F)
                              : _mm256_add_ps(F, _mm256_mul_ps(df, m));
}

__attribute__((target("avx2,fma"), always_inline)) inline void store_rows(
    const GemmArgs& a, const QRows& Q, int64_t row, __m256 F, float* dst) {
  __m256 y = F;
  if (!Q.ag) y = _mm256_mul_ps(y, _mm256_set1_ps(Q.as[row * Q.nas]));
  alignas(32) float l[kRows];
  _mm256_store_ps(l, y);
  for (int r = 0; r < kRows; ++r)
    dst[r] = a.round_out ? round_bf16(l[r]) : l[r];
}

// AVX-VNNI (VEX): the AVX2 code with vpdpbusd in place of vpmaddubsw,
// vpmaddwd and the add.
__attribute__((target(INT4_MOE_CPU_TVNNI256))) inline void gemm_rows_avxvnni(
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
            acc32 = _mm256_dpbusd_avx_epi32(
                acc32,
                _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(ub[r][s][0])),
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(e + 32 * s)));
            acc32 = _mm256_dpbusd_avx_epi32(
                acc32,
                _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(ub[r][s][1])),
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(o + 32 * s)));
          }
          v[r] = acc32;
        }
        const __m256i d = _mm256_sub_epi32(
            sum8_epi32(v), _mm256_set1_epi32(8 * Q.cs[row * Q.nfl + f]));
        F[t] = fp32_step(a, Q, row, k0, d, wsc, F[t]);
      }
    }
    for (int t = 0; t < tb; ++t)
      store_rows(
          a,
          Q,
          r0 + t0 + t,
          F[t],
          out + static_cast<int64_t>(t0 + t) * ldo + n0);
  }
}

// 64 bytes, or 32 with the upper half zero.
__attribute__((target(INT4_MOE_CPU_TVNNI512), always_inline)) inline __m512i
load64_or_32(const void* p, bool full) {
  return full ? _mm512_loadu_si512(p)
              : _mm512_zextsi256_si512(
                    _mm256_loadu_si256(static_cast<const __m256i*>(p)));
}

// AVX-512 VNNI: per row and 128-element block of a flush chunk two
// vpdpbusd on 64 bytes: the low nibbles of 64 weight bytes with the 64 even
// activations, the high nibbles with the 64 odd ones (a 64-element rest of a
// chunk: the lower halves, the upper zero). The weights are unpacked at 512
// bits once per row and chunk and kept for the activation rows. NB: blocks
// per chunk, the rest included (1..4).
template <int NB>
__attribute__((target(INT4_MOE_CPU_TVNNI512))) inline void gemm_rows_vnni512_nb(
    const GemmArgs& a,
    int n0,
    const QRows& Q,
    int64_t r0,
    int nt,
    float* out,
    int ldo) {
  const int kb = a.K / 2, ng = a.K / a.G;
  const int blocks = Q.fl / 128;  // full blocks; NB - blocks is the rest (0, 1)
  const __m512i m0f = _mm512_set1_epi8(0x0F);
  const __m512i x88 = _mm512_set1_epi8(static_cast<char>(0x88));
  alignas(64) uint8_t ub[kRows][NB][2][64];
  for (int t0 = 0; t0 < nt; t0 += kTBa) {
    const int tb = nt - t0 < kTBa ? nt - t0 : kTBa;
    __m256 F[kTBa];
    for (int t = 0; t < kTBa; ++t)
      F[t] = _mm256_setzero_ps();
    for (int f = 0; f < Q.nfl; ++f) {
      const int j0 = f * Q.fl / 2;
      for (int r = 0; r < kRows; ++r) {
        const uint8_t* w = a.W + static_cast<int64_t>(n0 + r) * kb + j0;
        for (int b = 0; b < NB; ++b) {
          const __m512i raw =
              _mm512_xor_si512(load64_or_32(w + 64 * b, b < blocks), x88);
          __m512i lo = _mm512_and_si512(raw, m0f);
          __m512i hi = _mm512_and_si512(_mm512_srli_epi16(raw, 4), m0f);
          if (b == blocks) {  // the rest: keep the lower halves only
            lo = _mm512_zextsi256_si512(_mm512_castsi512_si256(lo));
            hi = _mm512_zextsi256_si512(_mm512_castsi512_si256(hi));
          }
          _mm512_store_si512(ub[r][b][0], lo);
          _mm512_store_si512(ub[r][b][1], hi);
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
        __m512i ev[NB], ov[NB];
        for (int b = 0; b < NB; ++b) {
          ev[b] = load64_or_32(e + 64 * b, b < blocks);
          ov[b] = load64_or_32(o + 64 * b, b < blocks);
        }
        __m256i v[kRows];
        for (int r = 0; r < kRows; ++r) {
          // two chains (low and high nibbles), added at the end
          __m512i acc0 = _mm512_dpbusd_epi32(
              _mm512_setzero_si512(), _mm512_load_si512(ub[r][0][0]), ev[0]);
          __m512i acc1 = _mm512_dpbusd_epi32(
              _mm512_setzero_si512(), _mm512_load_si512(ub[r][0][1]), ov[0]);
          for (int b = 1; b < NB; ++b) {
            acc0 = _mm512_dpbusd_epi32(
                acc0, _mm512_load_si512(ub[r][b][0]), ev[b]);
            acc1 = _mm512_dpbusd_epi32(
                acc1, _mm512_load_si512(ub[r][b][1]), ov[b]);
          }
          const __m512i acc = _mm512_add_epi32(acc0, acc1);
          v[r] = _mm256_add_epi32(
              _mm512_castsi512_si256(acc), _mm512_extracti64x4_epi64(acc, 1));
        }
        const __m256i d = _mm256_sub_epi32(
            sum8_epi32(v), _mm256_set1_epi32(8 * Q.cs[row * Q.nfl + f]));
        F[t] = fp32_step(a, Q, row, k0, d, wsc, F[t]);
      }
    }
    for (int t = 0; t < tb; ++t)
      store_rows(
          a,
          Q,
          r0 + t0 + t,
          F[t],
          out + static_cast<int64_t>(t0 + t) * ldo + n0);
  }
}

__attribute__((target(INT4_MOE_CPU_TVNNI512))) inline void gemm_rows_avx512vnni(
    const GemmArgs& a,
    int n0,
    const QRows& Q,
    int64_t r0,
    int nt,
    float* out,
    int ldo) {
  switch ((Q.fl + 127) / 128) {  // flush chunks are 64..512 elements
    case 1:
      gemm_rows_vnni512_nb<1>(a, n0, Q, r0, nt, out, ldo);
      break;
    case 2:
      gemm_rows_vnni512_nb<2>(a, n0, Q, r0, nt, out, ldo);
      break;
    case 3:
      gemm_rows_vnni512_nb<3>(a, n0, Q, r0, nt, out, ldo);
      break;
    default:
      gemm_rows_vnni512_nb<4>(a, n0, Q, r0, nt, out, ldo);
      break;
  }
}

// --------------------------------------------------------------------------
// AMX-INT8 (tdpbusd): 8 weight rows x 16 activation rows per tile product.
//   A (tile 1): the 8 rows' u = w + 8 of a 64-element step, 64 bytes each:
//     bytes 0..31 the low nibbles of the step's 32 weight bytes, 32..63 the
//     high nibbles.
//   B (tile 2): the activations of 16 rows in the "4 bytes per column"
//     layout tdpbusd takes, k in the same order (QRows::qt: row q of a step
//     holds per activation row the 4 bytes qe[4q..4q+3] for q < 8,
//     qo[4(q-8)..] for q >= 8), read in place with the row stride.
//   C (tile 0): int32 [8 rows][16 activation rows], zeroed per flush chunk.
// Rows of other experts that fill a tile's 16 columns are computed and
// dropped (the transposed activations are padded to 16 rows past the end).
// --------------------------------------------------------------------------

// Row r of the activations into QRows::qt (the B layout above).
inline void transpose_row(QRows& Q, int64_t r) {
  const int kb = Q.K / 2;
  const int8_t* qe = Q.qe.data() + r * kb;
  const int8_t* qo = Q.qo.data() + r * kb;
  for (int ks = 0; ks < kb / 32; ++ks)
    for (int q = 0; q < 16; ++q) {
      const int8_t* src =
          q < 8 ? qe + 32 * ks + 4 * q : qo + 32 * ks + 4 * (q - 8);
      int32_t v;
      std::memcpy(&v, src, 4);
      Q.qt[(static_cast<int64_t>(ks) * 16 + q) * Q.cp + r] = v;
    }
}

__attribute__((target(INT4_MOE_CPU_TAMX8))) inline void gemm_rows_amx(
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
  TileConfig cfg;
  cfg.rows[0] = kRows;  // C
  cfg.colsb[0] = 64;
  cfg.rows[1] = kRows;  // A
  cfg.colsb[1] = 64;
  cfg.rows[2] = 16;  // B
  cfg.colsb[2] = 64;
  _tile_loadconfig(&cfg);
  alignas(64) uint8_t ua[kMaxFlush / kStep][kRows][64];
  alignas(64) int32_t cr[kRows][16];
  for (int t0 = 0; t0 < nt; t0 += 16) {
    const int tb = nt - t0 < 16 ? nt - t0 : 16;
    __m512 F[kRows];
    for (int r = 0; r < kRows; ++r)
      F[r] = _mm512_setzero_ps();
    for (int f = 0; f < Q.nfl; ++f) {
      const int j0 = f * Q.fl / 2;
      for (int r = 0; r < kRows; ++r) {
        const uint8_t* w = a.W + static_cast<int64_t>(n0 + r) * kb + j0;
        for (int s = 0; s < steps; ++s) {
          const __m256i raw = _mm256_xor_si256(
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32 * s)),
              x88);
          _mm256_store_si256(
              reinterpret_cast<__m256i*>(ua[s][r]), _mm256_and_si256(raw, m0f));
          _mm256_store_si256(
              reinterpret_cast<__m256i*>(ua[s][r] + 32),
              _mm256_and_si256(_mm256_srli_epi16(raw, 4), m0f));
        }
      }
      _tile_zero(0);
      for (int s = 0; s < steps; ++s) {
        const int ks = (j0 + 32 * s) / 32;
        _tile_loadd(1, ua[s], 64);
        _tile_loadd(
            2,
            Q.qt.data() + static_cast<int64_t>(ks) * 16 * Q.cp + r0 + t0,
            Q.cp * 4);
        _tile_dpbusd(0, 1, 2);
      }
      _tile_stored(0, cr, 64);
      // per activation row: the correction, its scale (per group of the
      // activations), as the AVX2 code's fp32_step, 16 rows at a time
      const int k0 = f * Q.fl;
      const int g = k0 / a.G;
      alignas(64) int32_t csv[16];
      alignas(64) float asv[16];
      for (int t = 0; t < 16; ++t) {
        const int64_t row = r0 + t0 + (t < tb ? t : 0);
        csv[t] = 8 * Q.cs[row * Q.nfl + f];
        asv[t] = Q.ag ? Q.as[row * Q.nas + k0 / Q.ag] : 0.0f;
      }
      const __m512i cv = _mm512_load_si512(csv);
      for (int r = 0; r < kRows; ++r) {
        const __m512 df =
            _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_load_si512(cr[r]), cv));
        __m512 m = _mm512_set1_ps(
            bf16_to_f32(a.S[static_cast<int64_t>(n0 + r) * ng + g]));
        if (Q.ag) m = _mm512_mul_ps(m, _mm512_load_ps(asv));
        F[r] = a.acc == Acc::kFused ? _mm512_fmadd_ps(df, m, F[r])
                                    : _mm512_add_ps(F[r], _mm512_mul_ps(df, m));
      }
    }
    alignas(64) float asv[16];
    for (int t = 0; t < 16; ++t)
      asv[t] = Q.ag ? 0.0f : Q.as[(r0 + t0 + (t < tb ? t : 0)) * Q.nas];
    for (int r = 0; r < kRows; ++r) {
      __m512 y = F[r];
      if (!Q.ag) y = _mm512_mul_ps(y, _mm512_load_ps(asv));
      alignas(64) float l[16];
      _mm512_store_ps(l, y);
      for (int t = 0; t < tb; ++t)
        out[static_cast<int64_t>(t0 + t) * ldo + n0 + r] =
            a.round_out ? round_bf16(l[t]) : l[t];
    }
  }
  _tile_release();
}

}  // namespace a8
}  // namespace cpu_moe

#endif  // INT4_MOE_CPU_X86
