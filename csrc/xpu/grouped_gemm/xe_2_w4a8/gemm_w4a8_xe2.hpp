/***************************************************************************************************
 * W4A8 grouped GEMM mainloop (Xe2 / BMG).
 *
 * D[m, n] = a_scale[m] * sum_g w_scale[n, g] * sum_{k in g} A8[m, k] * W4[n, k]
 *
 *   A8  int8  (M, K) row-major: activations quantized per token (dynamic,
 *              symmetric), one fp32 scale per row.
 *   W4  int4  (N, K) K-contiguous: the int4 bytes XpuFusedMoe holds after
 *              implement_zp (signed two's-complement nibbles, element 2p in
 *              the low nibble of byte p), exactly what the W4A16 path reads.
 *   w_scale    [N, K/GroupSize] (bf16 / fp16) per expert, same as W4A16.
 *
 * The MMA atom is the Xe2 mixed-precision DPAS s8 x s4 -> s32
 * (XE_DPAS_TT<8, int32_t, int8_t, int4_t>, `dpas.s4.s8`): the int4 weights
 * go into the systolic array as nibbles, so there is no unpack at all, only
 * the copy-layout -> VNNI register shuffle that sycl-tla's reorder does for
 * int4 B in its own s8 x s4 GEMM. The int32 partial of each K group is
 * converted to fp32 and scaled by w_scale[n, g] into an fp32 accumulator; the
 * per-row activation scale is applied once at the end, then bf16 out.
 *
 * The C fragment of the s32 atom has the same layout as the f32 one of the
 * bf16 atom with the same WG tile and SG layout (cell IP-D proved this with a
 * static_assert; the launcher re-asserts it), so the fp32 accumulator and the
 * D store reuse the stock W4A16 epilogue machinery via a twin bf16 TiledMMA.
 **************************************************************************************************/
#pragma once

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoEW4A8 {

using namespace cute;

// AG = 0: one activation scale per row (AScales[M]), applied at the end.
// AG > 0: one activation scale per (row, AG-wide K group) (AScales[M, K/AG],
//         as written by the fused gemm1 epilogue below), applied with the
//         weight scale at every AG boundary of the int32 partial.
// PF: prefetch distance in k-tiles (IP-L used 4; stock W4A16 uses 6).
template <
    int GroupSize,
    int AG,
    int PF,
    class GmemTiledCopyD,
    class ATensor,
    class BTensor,
    class DTensor,
    class TiledMMA,
    class TiledMMAF,
    typename ElementS,
    typename ElementBI>
CUTE_DEVICE void xe_gemm_w4a8(
    ATensor const& A,  // (M,K) int8
    BTensor const& B,  // (N,K) int4
    const ElementS* Scales,   // [N, K/GroupSize] of this expert
    const float* AScales,     // [M] (AG == 0) or [M, K/AG] of this expert
    const ElementBI* Bias,    // [N] of this expert or nullptr
    DTensor& C,               // (M,N) bf16/fp16
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma,      // s8 x s4 -> s32
    TiledMMAF const& mma_f,   // twin 16-bit atom, same tile: f32 C layout
    float* slm_as = nullptr) {  // AG > 0: SLM, tile_m * (K / AG) floats
  static constexpr int sg_local_range = 16;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = get_block_2d_copy_D<GmemTiledCopyD>(mma_f, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_mma_f = mma_f.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_c = copy_c.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto tCrC = thr_mma.partition_sg_fragment_C(gC);    // int32 group partial
  auto tCrF = thr_mma_f.partition_sg_fragment_C(gC);  // fp32 running sum
  auto tCcC = thr_mma_f.partition_C(gC);              // (m, n) of each value
  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);
  auto tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);
  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);
  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;

  static constexpr auto ATOM_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr int tile_m = get<0>(wg_tile);
  static constexpr int tile_n = get<1>(wg_tile);
  static constexpr int tile_k = get<2>(wg_tile);
  static constexpr int SG_M = tile_m / ATOM_M;
  static constexpr int SG_N = tile_n / ATOM_N;
  static constexpr int N_COLS = SG_N / sg_local_range;  // columns per lane
  static_assert(SG_N % sg_local_range == 0, "SG_N must be a multiple of 16");
  static_assert(GroupSize % tile_k == 0, "a K tile must not straddle groups");
  static constexpr int k_tiles_per_group = GroupSize / tile_k;
  static constexpr int prefetch_dist = PF;
  // The int32 partial is folded into fp32 every FLUSH K values: at each
  // weight group, and at each activation group when AG is finer.
  static constexpr int FLUSH = (AG > 0 && AG < GroupSize) ? AG : GroupSize;
  static_assert(AG == 0 || (AG % tile_k == 0 && GroupSize % AG == 0) ||
                    (AG % GroupSize == 0),
                "activation groups must align with K tiles and weight groups");
  static_assert(FLUSH % tile_k == 0, "a K tile must not straddle a flush");
  static constexpr int k_tiles_per_flush = FLUSH / tile_k;

  const int gemm_m = static_cast<int>(get<0>(A.shape()));
  const int gemm_n = static_cast<int>(get<0>(B.shape()));
  const int group_num = static_cast<int>(get<1>(A.shape())) / GroupSize;
  const int k_tile_count = ceil_div(get<1>(A.shape()), tile_k);

  const int sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  const int sg_local_id = cutlass::get_sub_group_local_id();
  const int n_lane0 = wg_n * tile_n + sg_local_n_coord * SG_N + sg_local_id;

  // Value i of the C fragment is column n_lane0 + (i / SG_M) * 16 (the
  // layout the stock bias epilogue relies on; s32 == f32 layout).
  float wsc[N_COLS];
  const int ag_num = AG > 0 ? static_cast<int>(get<1>(A.shape())) / AG : 1;

  clear(tCrC);
  clear(tCrF);

  if constexpr (AG > 0) {
    // The tile's activation scales [tile_m, K/AG] go to SLM once, so the
    // flushes in the K loop read them at SLM latency (no global load in the
    // dependency chain, nothing extra live in GRF).
    const int lsz = item.get_local_range(2);
    for (int idx = local_id; idx < tile_m * ag_num; idx += lsz) {
      const int ml = idx / ag_num;
      const int m = wg_m * tile_m + ml;
      slm_as[idx] = m < gemm_m ? AScales[static_cast<int64_t>(m) * ag_num + idx % ag_num] : 0.0f;
    }
    sycl::group_barrier(item.get_group());
  }

  int k_tile_prefetch = 0;
  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile % k_tiles_per_group == 0) {
      // Issue this group's weight-scale loads now; they are consumed only
      // at the end of the group, k_tiles_per_group MMA steps later.
      const int g = k_tile / k_tiles_per_group;
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < N_COLS; ++c) {
        const int n = n_lane0 + c * sg_local_range;
        wsc[c] = n < gemm_n
                     ? static_cast<float>(
                           Scales[static_cast<int64_t>(n) * group_num + g])
                     : 0.0f;
      }
    }

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);

    if ((k_tile + 1) % k_tiles_per_flush == 0 || k_tile + 1 == k_tile_count) {
      if constexpr (AG > 0) {
        // Value i = c * SG_M + sm sits on the row of value sm (the column
        // block varies with c only): one activation scale per sm, loaded
        // here so nothing extra stays live across the MMA.
        const int ag = (k_tile * tile_k) / AG;
        CUTLASS_PRAGMA_UNROLL
        for (int sm = 0; sm < SG_M; ++sm) {
          const int ml = get<0>(tCcC(sm)) - wg_m * tile_m;
          const float as = slm_as[ml * ag_num + ag];
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < N_COLS; ++c) {
            const int i = c * SG_M + sm;
            tCrF(i) += static_cast<float>(tCrC(i)) * (wsc[c] * as);
          }
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < N_COLS; ++c) {
          CUTLASS_PRAGMA_UNROLL
          for (int sm = 0; sm < SG_M; ++sm) {
            const int i = c * SG_M + sm;
            tCrF(i) += static_cast<float>(tCrC(i)) * wsc[c];
          }
        }
      }
      clear(tCrC);
    }
  }

  // Per-token activation scale (one per row; AG > 0 applied it per group
  // already), then the optional bias.
  if constexpr (AG == 0) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrF.size(); ++i) {
      const int m = get<0>(tCcC(i));
      tCrF(i) *= (m < gemm_m) ? AScales[m] : 0.0f;
    }
  }
  if (Bias != nullptr) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < N_COLS; ++c) {
      const int n = n_lane0 + c * sg_local_range;
      const float b = n < gemm_n ? static_cast<float>(Bias[n]) : 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrF(c * SG_M + sm) += b;
      }
    }
  }

  reorder(tCrF, tCrC_out);
  copy(copy_c, tCrC_out, tCgC);
}


/***************************************************************************************************
 * gemm1 with the SiLU-and-mul and the gemm2 activation quantization in the epilogue.
 *
 * w13 is [2*inter, K] per expert, gate rows [0, inter) then up rows [inter, 2*inter) (the
 * layout silu_and_mul reads: out = silu(x[:, :d]) * x[:, d:]). The work-group tile is in
 * activation columns: for act columns [n0, n0 + tile_n) it runs the s8 x s4 mainloop twice
 * per k-tile, over the gate rows n0.. and the up rows inter + n0.., into two accumulators,
 * so gate and up of one column sit in the same lane. The epilogue computes
 *   v = silu(as[m] * Fg + bias_g) * (as[m] * Fu + bias_u)          (fp32, no bf16 round trip)
 * then quantizes v to int8 with one scale per (row, AG columns): the row max over each AG-wide
 * column group is reduced across lanes (sub-group) and across the SGs of the group (SLM).
 * Q [M, inter] int8 and QS [M, inter / AG] fp32 are exactly gemm2's A and its AG scales.
 **************************************************************************************************/
template <
    int GroupSize,
    int AG,
    int PF,
    class ATensor,
    class BTensor,
    class TiledMMA,
    class TiledMMAF,
    typename ElementS,
    typename ElementBI>
CUTE_DEVICE void xe_gemm_w4a8_glu(
    ATensor const& A,          // (M, K) int8
    BTensor const& Bg,         // (inter, K) int4, gate rows
    BTensor const& Bu,         // (inter, K) int4, up rows
    const ElementS* ScalesG,   // [inter, K/GroupSize]
    const ElementS* ScalesU,   // [inter, K/GroupSize]
    const float* AScales,      // [M]
    const ElementBI* BiasG,    // [inter] or nullptr
    const ElementBI* BiasU,    // [inter] or nullptr
    int8_t* Q,                 // [M, inter]
    float* QS,                 // [M, inter / AG]
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma,
    TiledMMAF const& mma_f,
    float* red) {              // SLM, ATOM_N * tile_m floats
  static constexpr int sg_local_range = 16;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto sg = item.get_sub_group();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  const int gemm_m = static_cast<int>(get<0>(A.shape()));
  const int inter = static_cast<int>(get<0>(Bg.shape()));

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(Bg.shape());
  Tensor cC = make_identity_tensor(make_shape(gemm_m, inter));

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_bg = make_block_2d_copy_B(mma, Bg);
  auto copy_bu = make_block_2d_copy_B(mma, Bu);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_mma_f = mma_f.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_bg = copy_bg.get_slice(local_id);
  auto thr_copy_bu = copy_bu.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrBg = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tCrBu = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrBg = thr_copy_bg.partition_sg_fragment_D(gB(_, _, 0));
  auto tBrBu = thr_copy_bu.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgBg = thr_copy_bg.partition_S(gB);
  Tensor tBgBu = thr_copy_bu.partition_S(gB);

  auto tCrCg = thr_mma.partition_sg_fragment_C(gC);
  auto tCrCu = thr_mma.partition_sg_fragment_C(gC);
  auto tCrFg = thr_mma_f.partition_sg_fragment_C(gC);
  auto tCrFu = thr_mma_f.partition_sg_fragment_C(gC);
  auto tCcC = thr_mma_f.partition_C(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_bg = make_block_2d_prefetch(copy_bg);
  auto prefetch_bu = make_block_2d_prefetch(copy_bu);
  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_Bg = prefetch_bg.get_slice(local_id);
  auto thr_prefetch_Bu = prefetch_bu.get_slice(local_id);
  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgBg = thr_prefetch_Bg.partition_S(gB);
  auto pBgBu = thr_prefetch_Bu.partition_S(gB);

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;

  static constexpr auto ATOM_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr int tile_m = get<0>(wg_tile);
  static constexpr int tile_n = get<1>(wg_tile);
  static constexpr int tile_k = get<2>(wg_tile);
  static constexpr int SG_M = tile_m / ATOM_M;
  static constexpr int SG_N = tile_n / ATOM_N;
  static constexpr int N_COLS = SG_N / sg_local_range;
  static_assert(SG_N % sg_local_range == 0, "SG_N must be a multiple of 16");
  static_assert(GroupSize % tile_k == 0, "a K tile must not straddle groups");
  static_assert(AG % SG_N == 0 && tile_n % AG == 0,
                "an activation group must be whole SGs inside one WG tile");
  static constexpr int SPG = AG / SG_N;  // SGs per activation group
  static constexpr int k_tiles_per_group = GroupSize / tile_k;
  static constexpr int prefetch_dist = PF;

  const int group_num = static_cast<int>(get<1>(A.shape())) / GroupSize;
  const int k_tile_count = ceil_div(get<1>(A.shape()), tile_k);

  const int sg_id = cutlass::get_sub_group_id();
  const int sg_n = sg_id % ATOM_N;
  const int sg_local_id = cutlass::get_sub_group_local_id();
  const int n_lane0 = wg_n * tile_n + sg_n * SG_N + sg_local_id;

  float wscg[N_COLS];
  float wscu[N_COLS];

  clear(tCrCg);
  clear(tCrCu);
  clear(tCrFg);
  clear(tCrFu);

  int k_tile_prefetch = 0;
  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_bg, pBgBg(_, _, _, k_tile_prefetch));
    prefetch(prefetch_bu, pBgBu(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_bg, tBgBg(_, _, _, k_tile), tBrBg);
    copy(copy_bu, tBgBu(_, _, _, k_tile), tBrBu);

    if (k_tile % k_tiles_per_group == 0) {
      const int g = k_tile / k_tiles_per_group;
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < N_COLS; ++c) {
        const int n = n_lane0 + c * sg_local_range;
        const bool in = n < inter;
        const int64_t o = static_cast<int64_t>(n) * group_num + g;
        wscg[c] = in ? static_cast<float>(ScalesG[o]) : 0.0f;
        wscu[c] = in ? static_cast<float>(ScalesU[o]) : 0.0f;
      }
    }

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_bg, pBgBg(_, _, _, k_tile_prefetch));
      prefetch(prefetch_bu, pBgBu(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrBg, tCrBg);
    reorder(tBrBu, tCrBu);

    cute::gemm(mma, tCrA, tCrBg, tCrCg);
    cute::gemm(mma, tCrA, tCrBu, tCrCu);

    barrier_wait(barrier_scope);

    if ((k_tile + 1) % k_tiles_per_group == 0 || k_tile + 1 == k_tile_count) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < N_COLS; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int sm = 0; sm < SG_M; ++sm) {
          const int i = c * SG_M + sm;
          tCrFg(i) += static_cast<float>(tCrCg(i)) * wscg[c];
          tCrFu(i) += static_cast<float>(tCrCu(i)) * wscu[c];
        }
      }
      clear(tCrCg);
      clear(tCrCu);
    }
  }

  // silu(gate) * up in fp32, into tCrFg; zero outside [gemm_m, inter).
  CUTLASS_PRAGMA_UNROLL
  for (int c = 0; c < N_COLS; ++c) {
    const int n = n_lane0 + c * sg_local_range;
    const bool nin = n < inter;
    const float bg = (BiasG != nullptr && nin) ? static_cast<float>(BiasG[n]) : 0.0f;
    const float bu = (BiasU != nullptr && nin) ? static_cast<float>(BiasU[n]) : 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int sm = 0; sm < SG_M; ++sm) {
      const int i = c * SG_M + sm;
      const int m = get<0>(tCcC(i));
      const bool in = nin && m < gemm_m;
      const float as = in ? AScales[m] : 0.0f;
      const float g = tCrFg(i) * as + bg;
      const float u = tCrFu(i) * as + bu;
      const float v = g / (1.0f + sycl::native::exp(-g)) * u;
      tCrFg(i) = in ? v : 0.0f;
    }
  }

  // Row max over each AG-wide column group: lanes, then the group's SGs.
  sycl::group_barrier(item.get_group());  // red[] free (previous tile done)
  CUTLASS_PRAGMA_UNROLL
  for (int sm = 0; sm < SG_M; ++sm) {
    float v = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < N_COLS; ++c) {
      v = sycl::fmax(v, sycl::fabs(tCrFg(c * SG_M + sm)));
    }
    v = sycl::reduce_over_group(sg, v, sycl::maximum<float>());
    const int m_local = get<0>(tCcC(sm)) - wg_m * tile_m;
    if (sg_local_id == 0) red[sg_n * tile_m + m_local] = v;
  }
  sycl::group_barrier(item.get_group());

  const int g_first = (sg_n / SPG) * SPG;
  const int ag_num = inter / AG;
  const int ag = (wg_n * tile_n + g_first * SG_N) / AG;
  CUTLASS_PRAGMA_UNROLL
  for (int sm = 0; sm < SG_M; ++sm) {
    const int m = get<0>(tCcC(sm));
    const int m_local = m - wg_m * tile_m;
    float mx = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < SPG; ++s) {
      mx = sycl::fmax(mx, red[(g_first + s) * tile_m + m_local]);
    }
    const float scale = mx * (1.0f / 127.0f);
    const float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
    if (m < gemm_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < N_COLS; ++c) {
        const int n = n_lane0 + c * sg_local_range;
        if (n < inter) {
          float q = sycl::rint(tCrFg(c * SG_M + sm) * inv);
          q = sycl::fmin(127.0f, sycl::fmax(-127.0f, q));
          Q[static_cast<int64_t>(m) * inter + n] = static_cast<int8_t>(q);
        }
      }
      if (sg_n == g_first && sg_local_id == 0) {
        QS[static_cast<int64_t>(m) * ag_num + ag] = scale;
      }
    }
  }
}

}  // namespace MoEW4A8
