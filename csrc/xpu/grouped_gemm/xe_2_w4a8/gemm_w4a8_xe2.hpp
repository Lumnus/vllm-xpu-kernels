/***************************************************************************************************
 * B70 W4A8 grouped GEMM mainloop (Xe2 / BMG).
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
 * bf16 atom with the same WG tile and SG layout (checked by a
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

template <
    int GroupSize,
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
    const float* AScales,     // [M] of this expert
    const ElementBI* Bias,    // [N] of this expert or nullptr
    DTensor& C,               // (M,N) bf16/fp16
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma,      // s8 x s4 -> s32
    TiledMMAF const& mma_f) { // twin 16-bit atom, same tile: f32 C layout
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
  static constexpr int prefetch_dist = 4;

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

  clear(tCrC);
  clear(tCrF);

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

    if ((k_tile + 1) % k_tiles_per_group == 0 || k_tile + 1 == k_tile_count) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < N_COLS; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int sm = 0; sm < SG_M; ++sm) {
          const int i = c * SG_M + sm;
          tCrF(i) += static_cast<float>(tCrC(i)) * wsc[c];
        }
      }
      clear(tCrC);
    }
  }

  // Per-token activation scale (one per row), then the optional bias.
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < tCrF.size(); ++i) {
    const int m = get<0>(tCcC(i));
    tCrF(i) *= (m < gemm_m) ? AScales[m] : 0.0f;
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

}  // namespace MoEW4A8
