// AMX for the W4A16 op: the card-class AMX-BF16 prefill of dispatch level 8
// (asked for explicitly, never automatic), included by int4_moe_cpu_impl.h
// after the AVX-512 code. The W4A8 op's AMX-INT8 code is in
// int4_moe_cpu_a8_isa.h.
//
// Why it is not bit-identical, and why there is no exact AMX-BF16 code:
// TDPBF16PS adds the two products of a bf16 pair first, rounds that sum to
// fp32, then adds it to the accumulator (Intel SDE 9.53, measured:
// acc 1, products 2^-24 and 2^-24 gives 1 + 2^-23, where two fused
// multiply-adds give 1). The scalar contract adds the low nibble's product
// and then the high nibble's, each rounded, into one of 8 lanes per row. An
// exact tile code would have to put every product alone in its pair and in
// its lane's own column (1 useful product in 16), slower than the AVX-512
// code. So this code computes the grouped GEMM's arithmetic class instead:
// each weight bf16(q * s) exactly as the card's apply_scale, bf16 inputs,
// fp32 accumulation, in the tile's order (one accumulator per output, pairs
// summed first): the card's own grouped GEMM (DPAS) also has its own order,
// so the result is as close to the card as the exact code is, not equal to
// the scalar code. Used for the grouped GEMM's arithmetic (kBf16Weights) at
// 16 or more tokens per expert; everything else takes level 7's code.

#pragma once

#if INT4_MOE_CPU_X86

  #define INT4_MOE_CPU_TAMX \
    "amx-tile,amx-bf16,amx-int8,avx512f,avx512bw,avx512dq,avx512vl,avx2,fma"

namespace cpu_moe {

constexpr int kAmxMinTokens = 16;  // tokens per expert from which AMX runs

struct alignas(64) TileConfig {
  uint8_t palette = 1, start_row = 0;
  uint8_t reserved[14] = {};
  uint16_t colsb[16] = {};
  uint8_t rows[16] = {};
};

// Row pairs of bf16(q * s) for 16 rows of one 32-element chunk in the tile
// layout TDPBF16PS takes for its second operand: row p (byte j0 + p of the
// chunk), column n (output row n0 + n): the dword (low word: the weight of
// the low nibble, high word: of the high nibble).
__attribute__((target(INT4_MOE_CPU_TAMX))) inline void amx_b_tile(
    const uint8_t* W,
    const uint16_t* S,
    int64_t kb,
    int ng,
    int G,
    int n0,
    int c,
    uint32_t* tile) {
  alignas(64) uint32_t col[16][16];  // [n][p]
  const int j0 = c * (kChunk / 2);
  const int g = (c * kChunk) / G;
  for (int n = 0; n < 16; n += 2) {
    const uint8_t* wa = W + static_cast<int64_t>(n0 + n) * kb + j0;
    const uint8_t* wb = wa + kb;
    const __m512i tab = card_table512(
        bf16_to_f32(S[static_cast<int64_t>(n0 + n) * ng + g]),
        bf16_to_f32(S[static_cast<int64_t>(n0 + n + 1) * ng + g]));
    for (int r = 0; r < 2; ++r) {
      const __m512i v = _mm512_cvtepu8_epi32(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(r ? wb : wa)));
      const __m512i off = _mm512_set1_epi32(r ? 0x00100010 : 0);
      const __m512i idx = _mm512_or_si512(
          _mm512_or_si512(
              _mm512_and_si512(v, _mm512_set1_epi32(0x0F)),
              _mm512_slli_epi32(_mm512_srli_epi32(v, 4), 16)),
          off);
      _mm512_store_si512(col[n + r], _mm512_permutexvar_epi16(idx, tab));
    }
  }
  for (int p = 0; p < 16; ++p)
    for (int n = 0; n < 16; ++n)
      tile[p * 16 + n] = col[n][p];
}

// One accumulator tile per 16 tokens, up to 4 at a time (tiles 0..3), the
// activations in tile 4, the weights in tile 5.
template <int NC>
__attribute__((target(INT4_MOE_CPU_TAMX))) inline void amx_class_block(
    const uint32_t* bt,  // [chunks][16 * 16] weight tiles
    int chunks,
    const uint16_t* xa,  // [tokens][K] bf16, natural order
    int64_t K,
    float* c) {  // [NC][16][16]
  _tile_zero(0);
  if (NC > 1) _tile_zero(1);
  if (NC > 2) _tile_zero(2);
  if (NC > 3) _tile_zero(3);
  for (int ch = 0; ch < chunks; ++ch) {
    _tile_loadd(5, bt + static_cast<int64_t>(ch) * 256, 64);
    const uint16_t* a = xa + static_cast<int64_t>(ch) * kChunk;
    _tile_loadd(4, a, K * 2);
    _tile_dpbf16ps(0, 4, 5);
    if (NC > 1) {
      _tile_loadd(4, a + 16 * K, K * 2);
      _tile_dpbf16ps(1, 4, 5);
    }
    if (NC > 2) {
      _tile_loadd(4, a + 32 * K, K * 2);
      _tile_dpbf16ps(2, 4, 5);
    }
    if (NC > 3) {
      _tile_loadd(4, a + 48 * K, K * 2);
      _tile_dpbf16ps(3, 4, 5);
    }
  }
  _tile_stored(0, c, 64);
  if (NC > 1) _tile_stored(1, c + 256, 64);
  if (NC > 2) _tile_stored(2, c + 512, 64);
  if (NC > 3) _tile_stored(3, c + 768, 64);
}

__attribute__((target(INT4_MOE_CPU_TAMX))) inline void rows_amx_class(
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
  const int m = n0 + ((n1 - n0) & ~15);
  if (arith != Arith::kBf16Weights || nt < kAmxMinTokens || m == n0) {
    rows_avx512bf16(arith, W, S, K, G, n0, n1, xe, xo, nt, out, ldo);
    return;
  }
  const int64_t kb = K / 2;
  const int ng = K / G;
  const int chunks = K / kChunk;
  const int tpad = (nt + 15) & ~15;
  // the inputs in bf16, natural order (the even element in the low word),
  // padded with zero tokens to a multiple of 16; once per input and call
  struct Natural {
    const float* xe = nullptr;
    int nt = 0, K = 0;
    std::vector<uint16_t> xa;
  };
  static thread_local InputCache<Natural> cache;
  static thread_local std::vector<uint32_t> bt;
  bool fresh = false;
  Natural* in = cache.find(xe, nt, K, fresh);
  if (fresh) {
    in->xa.assign(static_cast<size_t>(tpad) * K, 0);
    for (int t = 0; t < nt; ++t)
      for (int64_t j = 0; j < kb; j += 16) {
        const __m512i e = _mm512_loadu_si512(xe + t * kb + j);
        const __m512i o = _mm512_loadu_si512(xo + t * kb + j);
        const __m512i d = _mm512_or_si512(
            _mm512_srli_epi32(e, 16),
            _mm512_and_si512(
                o, _mm512_set1_epi32(static_cast<int>(0xFFFF0000u))));
        _mm512_storeu_si512(in->xa.data() + t * K + 2 * j, d);
      }
  }
  const std::vector<uint16_t>& xa = in->xa;
  bt.resize(static_cast<size_t>(chunks) * 256);
  TileConfig cfg;
  for (int i = 0; i < 4; ++i) {
    cfg.rows[i] = 16;
    cfg.colsb[i] = 64;
  }
  cfg.rows[4] = 16;
  cfg.colsb[4] = 64;
  cfg.rows[5] = 16;
  cfg.colsb[5] = 64;
  _tile_loadconfig(&cfg);
  alignas(64) float c[4 * 256];
  for (int r0 = n0; r0 < m; r0 += 16) {
    for (int ch = 0; ch < chunks; ++ch)
      amx_b_tile(W, S, kb, ng, G, r0, ch, bt.data() + ch * 256);
    for (int t0 = 0; t0 < nt; t0 += 64) {
      const int nc = std::min(4, (tpad - t0) / 16);
      const uint16_t* a = xa.data() + static_cast<int64_t>(t0) * K;
      switch (nc) {
        case 1:
          amx_class_block<1>(bt.data(), chunks, a, K, c);
          break;
        case 2:
          amx_class_block<2>(bt.data(), chunks, a, K, c);
          break;
        case 3:
          amx_class_block<3>(bt.data(), chunks, a, K, c);
          break;
        default:
          amx_class_block<4>(bt.data(), chunks, a, K, c);
          break;
      }
      for (int t = t0; t < std::min(nt, t0 + 64); ++t)
        for (int n = 0; n < 16; ++n)
          out[static_cast<int64_t>(t) * ldo + r0 + n] = c[(t - t0) * 16 + n];
    }
  }
  _tile_release();
  if (m < n1) rows_avx512bf16(arith, W, S, K, G, m, n1, xe, xo, nt, out, ldo);
}

}  // namespace cpu_moe

#endif  // INT4_MOE_CPU_X86
