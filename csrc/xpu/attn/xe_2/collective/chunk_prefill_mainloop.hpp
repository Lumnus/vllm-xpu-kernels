/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"

#include "block_2d_align.hpp"

namespace cutlass::fmha {
// Arch-tagged inline namespace: gives these definitions a mangled name
// distinct from the other Xe architecture's identically named copies,
// while leaving name lookup (cutlass::fmha::...) unchanged.
inline namespace vllm_xpu_xe2 {

template <int Stages>
class XeDefault {};  // Default FMHA mainloop, P in registers.

}  // namespace vllm_xpu_xe2

};  // namespace cutlass::fmha

namespace cutlass::fmha::collective {
// Arch-tagged inline namespace: gives these definitions a mangled name
// distinct from the other Xe architecture's identically named copies,
// while leaving name lookup (cutlass::fmha::...) unchanged.
inline namespace vllm_xpu_xe2 {

static inline void sbarrier_wait() { asm volatile("sbarrier.wait\n"); }

static inline void sbarrier_signal() { asm volatile("sbarrier.signal\n"); }

static inline void gfence() { asm volatile("lsc_fence.ugm.none.group\n"); }

static inline void barrier() {
  asm volatile("lsc_fence.ugm.none.group\n");
  asm volatile("barrier\n");
}

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    bool CausalMask_,
    bool LocalMask_,
    bool PagedKV_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,  // Optional TiledCopy for loading K
    class TiledCopyV_ = void,  // Optional TiledCopy for loading V
    // INT8 KV only (see Int8Dot below): s8 x s8 -> s32 MMA for Q*K and the
    // 16-bit MMA over the same (q, k, d) tile that loads the query for its
    // quantization.
    class TiledMMAQK8_ = void,
    class TiledMMAQ16_ = void,
    // INT8 P*V (with the int8 Q*K only): s8 x s8 -> s32 MMA over the P*V tile.
    class TiledMMAPV8_ = void>
struct FMHAFwdMainloop {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Could not find a mainloop specialization.");
};

// Exactly rounded fp32 division a / b (round to nearest even) from FMAs, so the
// result does not depend on the TU's floating-point model (the attention TUs
// build with the default fast model, where '/' may be approximate). The
// reciprocal comes from native::recip (opaque to algebraic folding) and two
// Newton steps; the quotient gets one FMA residual correction (Markstein).
// Finite, normal operands only (the callers pass |b| >= 1e-10).
static inline float int8_attn_div_rn(float a, float b) {
  float r = sycl::native::recip(b);
  r = sycl::fma(sycl::fma(-b, r, 1.0f), r, r);
  r = sycl::fma(sycl::fma(-b, r, 1.0f), r, r);
  float q = a * r;
  float e = sycl::fma(-b, q, a);
  return sycl::fma(e, r, q);
}

// The shared INT8 query rule of the attention kernels (paged decode and chunk
// prefill here, prefill in the engine's Triton kernel): per (token, query head)
//   scale = max(absmax / 127, 1e-10), y = x * (1 / scale) (both divisions
//   exactly rounded), code = sign(y) * min(floor(|y| + 1/2), 127),
// with the half-up rounding computed exactly: floor(|y| + 1/2) =
// (trunc(2 |y|) + 1) >> 1 (2 |y| is exact), so no fp addition can round.
static inline float int8_attn_query_scale(float absmax) {
  return sycl::fmax(int8_attn_div_rn(absmax, 127.0f), 1.0e-10f);
}
static inline float int8_attn_query_code(float x, float inv_scale) {
  const float y = x * inv_scale;
  const float a2 = sycl::fabs(y) * 2.0f;
  int c = a2 < 255.0f ? (static_cast<int>(a2) + 1) >> 1 : 127;
  c = c > 127 ? 127 : c;
  return y < 0.0f ? static_cast<float>(-c) : static_cast<float>(c);
}
// INT8 P*V: P * v_scale (>= 0) per query row of a KV tile, scale = max(row max
// / 127, 1e-30) exactly rounded, codes by int8_attn_query_code with the exactly
// rounded inverse (the engine's Triton reference uses the same rule).
static inline float int8_attn_p_scale(float rowmax) {
  return sycl::fmax(int8_attn_div_rn(rowmax, 127.0f), 1.0e-30f);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool CausalMask_,
    bool LocalMask_,
    bool PagedKV_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_,
    class TiledMMAQK8_,
    class TiledMMAQ16_,
    class TiledMMAPV8_>
struct FMHAFwdMainloop<
    XeDefault<Stages>,
    CausalMask_,
    LocalMask_,
    PagedKV_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_,
    TiledMMAQK8_,
    TiledMMAQ16_,
    TiledMMAPV8_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using ElementQ = typename TensorQ::engine_type::value_type;
  using ElementK = typename TensorK::engine_type::value_type;

  using TensorQ2D =
      decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D =
      decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D =
      decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ = conditional_t<
      is_void_v<TiledCopyQ_>,
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})),
      TiledCopyQ_>;
  using TiledCopyK = conditional_t<
      is_void_v<TiledCopyK_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})),
      TiledCopyK_>;
  using TiledCopyV = conditional_t<
      is_void_v<TiledCopyV_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})),
      TiledCopyV_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;  // (atom val,q',v')
  using FragA =
      expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool Fp8KV =
      is_any_of_v<ElementK, float_e5m2_t, float_e4m3_t>;
  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool LocalMask = LocalMask_;
  static constexpr bool PagedKV = PagedKV_;

  // INT8 per-token-head KV (prefill over the cache in place): int8 K/V codes in
  // the block-local split layout plus one fp32 scale per (KV token, KV head),
  // read from [num_blocks, block_size, num_heads_kv] scale caches (scale_k /
  // scale_v point to them); one 2D surface per (block, head), so the block
  // stride is passed in bytes. Paged, causal or full, no window. The arithmetic
  // is the paged decode kernel's (DecodeFwdMainloop): S = (Q*K) * k_scale per
  // column before the mask, fp32 online softmax, P *= v_scale per column after
  // the row sums, P*V in the query's 16-bit type over the widened V codes.
  static constexpr bool Int8KV = is_same_v<ElementK, int8_t>;
  // INT8 Q*K (int8 KV only): the query is quantized per (token, query head)
  // in the kernel with the shared rule (int8_attn_query_scale / _code), Q*K
  // runs as s8 x s8 -> s32 DPAS on the cached K codes and
  // S = s32 * q_scale * k_scale in fp32. Without it, Q*K runs on the 16-bit
  // query over the widened K codes, exactly as the paged decode kernel does.
  static constexpr bool Int8Dot = !is_void_v<TiledMMAQK8_>;
  static_assert(!Int8Dot || Int8KV, "the int8 Q*K dot needs an int8 KV cache");
  static_assert(
      !Int8KV || (PagedKV && !LocalMask),
      "INT8 KV prefill: paged KV cache, no local window");
  static_assert(
      Int8Dot == !is_void_v<TiledMMAQ16_>,
      "the int8 Q*K dot needs both its s8 and its 16-bit query MMA");
  using TiledMMAQK8 = conditional_t<Int8Dot, TiledMMAQK8_, TiledMMAQK_>;
  using TiledMMAQ16 = conditional_t<Int8Dot, TiledMMAQ16_, TiledMMAQK_>;
  // INT8 P*V (with Int8Dot): after the softmax and P *= v_scale, P is
  // quantized per query row of the KV tile (int8_attn_p_scale), P*V runs as
  // s8 x s8 -> s32 DPAS on the cached V codes as loaded (no widening), and
  // O = O * rescale + s32 * p_scale in fp32.
  static constexpr bool Int8PV = !is_void_v<TiledMMAPV8_>;
  static_assert(!Int8PV || Int8Dot, "the int8 P*V needs the int8 Q*K");
  using TiledMMAPV8 = conditional_t<Int8PV, TiledMMAPV8_, TiledMMAPV_>;
  using TileShapeQK8 = decltype(TiledMMAQK8{}.tile_mnk());

  // User-facing arguments
  struct ArgumentsCommon {
    ElementS const scale;
    void* const scale_k;
    void* const scale_v;

    // Paged KV Cache
    const int* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    // Local Mask
    int local_left, local_right;
    // Physical stride between paged blocks in seq-position units.
    int page_stride_elements;
  };
  // INT8 KV: the same fields plus the scale-cache strides (elements) and the
  // block stride of the split layout (bytes). A separate type, so the other
  // instantiations keep their parameter block unchanged.
  struct ArgumentsInt8KV {
    ElementS const scale;
    void* const scale_k;
    void* const scale_v;
    const int* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    int local_left, local_right;
    int page_stride_elements;
    int64_t scale_stride_block = 0;
    int64_t scale_stride_slot = 0;
    int64_t scale_stride_head = 0;
    int64_t page_stride_bytes = 0;
  };
  using Arguments = conditional_t<Int8KV, ArgumentsInt8KV, ArgumentsCommon>;

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data. Int8Dot: each sub-group's s8 Q operand (head_dim / 64 chunks of
  // the 64-wide s8 A fragment), written once per work tile and read back in
  // every KV tile, so the s8 query is not live in registers across the KV
  // loop next to the O accumulator. Stored element-major across the
  // sub-group's lanes ([element][lane] bytes: the fragment's register image),
  // so a sub-group block read of 16 bytes per lane lands in the fragment
  // registers as they are, without per-byte moves. The epilogue of this
  // kernel uses no SLM (ReduceK == 1).
  static constexpr int kInt8DChunks = 4;  // head_dim <= 256, 64-wide chunks
  using FragQ8 = decltype(TiledMMAQK8{}.get_slice(0).partition_sg_fragment_A(
      make_identity_tensor(select<0, 2>(TileShapeQK8{}))));
  static constexpr int kQ8LaneBytes =
      Int8Dot ? int(decltype(size(FragQ8{}.tensor()))::value) : 16;
  static_assert(kQ8LaneBytes % 16 == 0);
  static constexpr int kQ8Blocks = kQ8LaneBytes / 16;  // uchar16 block reads
  struct SharedStorageInt8Dot {
    alignas(64)
        uint8_t q8[SGPerWG::value][kInt8DChunks][kQ8LaneBytes * intel::sg_size];
  };
  using Q8Block = uint8_t __attribute__((ext_vector_type(16)));
  struct SharedStorageNone {};
  using SharedStorage =
      conditional_t<Int8Dot, SharedStorageInt8Dot, SharedStorageNone>;

  Params params;
  SharedStorage* smem_ = nullptr;

  //
  // Methods
  //

  FMHAFwdMainloop(Params const& params_, SharedStorage& smem)
      : params(params_), smem_(&smem) {}

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    Params p{
        val,
        args.scale_k,
        args.scale_v,
        args.ptr_page_table,
        args.page_size,
        args.max_pages_per_seq,
        args.total_seqlen_kv,
        args.local_left,
        args.local_right,
        args.page_stride_elements};
    if constexpr (Int8KV) {
      p.scale_stride_block = args.scale_stride_block;
      p.scale_stride_slot = args.scale_stride_slot;
      p.scale_stride_head = args.scale_stride_head;
      p.page_stride_bytes = args.page_stride_bytes;
    }
    return p;
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_DEVICE int get_paged_idx(int K, int idx_b) {
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int b_offset = idx_b * params.max_pages_per_seq;
    int page_local_idx = K * get<1>(TileShapeQK{}) / params.page_size;

    // Clamp page_local_idx to the valid range [0, max_pages_per_seq - 1]
    if (page_local_idx >= params.max_pages_per_seq) {
      page_local_idx = params.max_pages_per_seq - 1;
    }

    int block_idx = params.ptr_page_table[b_offset + page_local_idx];
    int page_stride_tiles = params.page_stride_elements / get<1>(TileShapeQK{});
    return block_idx * page_stride_tiles + K % tiles_per_page;
  }

  template <bool has_large_surface, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorK2D const& K_2D,  // (k,d)
      TensorV2D const& V_2D,  // (d,k)
      FragA& tArA,            // Output accumulator (q,v)
      FragARow& tA_max,       // Softmax row-wise max accumulator
      FragARow& tA_sum,       // Softmax row-wise sum accumulator
      QVCoord blk_qv,         // WG tile indices: (Q,V)
      int const& idx_b,       // WG tile indices: (B)
      int blk_k0,             // K block range: [K0,K1)
      int blk_k1,
      int blk_k1_causal,
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int blk_local_l_safe,
      int blk_local_r_safe,
      int head_kv = 0) {  // KV head (INT8 KV: indexes the scale caches)
    using namespace sycl::ext::oneapi::this_work_item;

    if constexpr (Int8KV) {
      int8_kv_mainloop(
          Q_2D,
          K_2D,
          V_2D,
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          idx_b,
          blk_k0,
          blk_k1,
          blk_k1_causal,
          thr_id,
          seq_len,
          full_tile_offset,
          head_kv);
      return;
    }

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    constexpr int tile_k = get<1>(TileShapeQK{});

    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* 64B-align the per-head block-2D surfaces (see block_2d_align.hpp) */
    int x_shift_q, x_shift_k, x_shift_v;
    auto Q_2D_al = align_block_2d_base<1>(Q_2D, x_shift_q);
    auto K_2D_al = align_block_2d_base<1>(K_2D, x_shift_k);
    auto V_2D_al = align_block_2d_base<0>(V_2D, x_shift_v);

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ =
        make_shifted_identity_tensor<1>(Q_2D.shape(), x_shift_q);  // (q,d)
    Tensor cK =
        make_shifted_identity_tensor<1>(K_2D.shape(), x_shift_k);  // (k,d)
    Tensor cV =
        make_shifted_identity_tensor<0>(V_2D.shape(), x_shift_v);  // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));   // (q,k)

    auto K_2D_0 = K_2D_al;
    auto V_2D_0 = V_2D_al;

    // --- Xe 2D-block surface-height split
    // ------------------------------------- The Xe LSC 2D block-load descriptor
    // stores the surface height in a 24-bit field, so a (rank-2) K/V surface
    // can address at most 2^24 rows. A paged KV cache may span many more rows
    // (total_seqlen_kv = num_blocks * page_stride_elements), which overflows
    // that field and makes the hardware wrap addresses. To keep the original
    // rank-2 load/prefetch logic we split the K and V surfaces into
    // page-aligned sub-surfaces, each at most 2^24 rows tall, and route every
    // physical tile to the surface that physically contains it.

    // The split state and the two `make_*_surface` lambdas are declared here at
    // function scope (rather than inside the `if constexpr` below) so that the
    // per-tile routing inside the K loop can reuse them. They are only ever
    // exercised when `has_large_surface` is true; otherwise they stay unused.
    [[maybe_unused]] int tiles_per_surface = 0x3fffffff;
    [[maybe_unused]] int rows_per_surface = 0;
    [[maybe_unused]] int k_rows_total = get<0>(K_2D.shape());
    [[maybe_unused]] int v_cols_total = get<1>(V_2D.shape());
    [[maybe_unused]] int64_t k_surf_shift = 0;
    [[maybe_unused]] int64_t v_surf_shift = 0;
    [[maybe_unused]] int k_pre_surf = -1;

    // Lazily materialize an arbitrary overflow sub-surface `surf` (>= 1). The
    // base is shifted by `surf` full surfaces and the height/width is the real
    // remaining extent of that surface, so the 2D block load never reads past
    // the cache. These are only ever built inside the K loop's cold branch, so
    // supporting any number of sub-surfaces adds at most one live descriptor
    // and the common path keeps its original register/descriptor footprint.
    [[maybe_unused]] auto make_k_surface = [&](int surf) {
      int64_t rem = int64_t(k_rows_total) - int64_t(surf) * rows_per_surface;
      int rows = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (rows < 1) {
        rows = 1;
      }
      return make_tensor(
          K_2D_al.data() + int64_t(surf) * k_surf_shift,
          make_layout(
              make_shape(rows, get<1>(K_2D_al.shape())), K_2D_al.stride()));
    };
    [[maybe_unused]] auto make_v_surface = [&](int surf) {
      int64_t rem = int64_t(v_cols_total) - int64_t(surf) * rows_per_surface;
      int cols = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (cols < 1) {
        cols = 1;
      }
      return make_tensor(
          V_2D_al.data() + int64_t(surf) * v_surf_shift,
          make_layout(
              make_shape(get<0>(V_2D_al.shape()), cols), V_2D_al.stride()));
    };

    if constexpr (has_large_surface) {
      constexpr int kXeMaxSurfaceRows = 1 << 24;
      if constexpr (PagedKV) {
        int page_stride_tiles = params.page_stride_elements / tile_k;
        int blocks_per_surface =
            kXeMaxSurfaceRows / params.page_stride_elements;
        if (blocks_per_surface < 1) {
          blocks_per_surface = 1;
        }
        tiles_per_surface = blocks_per_surface * page_stride_tiles;
        rows_per_surface = tiles_per_surface * tile_k;
      }

      // Build the (rank-2) K/V surface 0. Surface 0 keeps the original base
      // pointer and is the only surface addressed on the hot path; overflow
      // surfaces are materialized lazily (see make_k_surface/make_v_surface).
      int k_rows0, v_cols0;
      if constexpr (PagedKV) {
        k_rows0 =
            k_rows_total < rows_per_surface ? k_rows_total : rows_per_surface;
        k_surf_shift = int64_t(rows_per_surface) * get<0>(K_2D.stride());
        v_cols0 =
            v_cols_total < rows_per_surface ? v_cols_total : rows_per_surface;
        v_surf_shift = int64_t(rows_per_surface) * get<1>(V_2D.stride());
      } else {
        k_rows0 = k_rows_total;
        v_cols0 = v_cols_total;
      }

      K_2D_0 = make_tensor(
          K_2D_al.data(),
          make_layout(
              make_shape(k_rows0, get<1>(K_2D_al.shape())), K_2D_al.stride()));
      V_2D_0 = make_tensor(
          V_2D_al.data(),
          make_layout(
              make_shape(get<0>(V_2D_al.shape()), v_cols0), V_2D_al.stride()));

      cK = make_shifted_identity_tensor<1>(
          make_shape(
              PagedKV ? rows_per_surface : k_rows_total, get<1>(K_2D.shape())),
          x_shift_k);  // (k,d)
      cV = make_shifted_identity_tensor<0>(
          make_shape(
              get<0>(V_2D.shape()), PagedKV ? rows_per_surface : v_cols_total),
          x_shift_v);  // (v,k)
    }

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D)
    Tensor gK = local_tile(
        cK,
        TileShapeQK{},
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,d,K,D)
    Tensor gV =
        local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
    Tensor gV_split = local_tile(
        gV,
        TileShapePV{},
        make_coord(_, _, 0),
        Step<X, _1, _1>{});  // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D_al};
    TiledCopyK copy_k{K_2D_0};
    TiledCopyV copy_v{V_2D_0};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));
    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);
    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_0);
    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    /* Masking scalars */
    int lane_id = thr_id % intel::sg_size;
    constexpr int sg_tile_q = get<0>(TileShapeQK{}) / SGPerWG::value;
    int row_base = get<0>(blk_qv) * get<0>(TileShapeQK{}) +
                   (thr_id / intel::sg_size) * sg_tile_q;

    // PagedKV
    [[maybe_unused]] int page_idx = blk_k0;
    [[maybe_unused]] int next_page_idx = blk_k0;
    if constexpr (PagedKV) {
      next_page_idx = get_paged_idx(blk_k0, idx_b);
    }

    /* Q prefetch */
    CUTLASS_PRAGMA_UNROLL
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    bool check_remainder_k = (seq_len % tile_k != 0);

    // FP8 KV Scale (per-tensor): fold scale_k into the softmax scale and
    // defer scale_v to a single post-loop output rescale. Mathematically
    // identical to per-element K/V rescaling (S = Q·K is linear in K and
    // O = sum P_i·V_i is linear in V), but removes both rescales from the
    // inner loop. The remaining fp8 -> ElementQ conversion is performed by
    // reorder() using the subgroup-vectorized cast primitive.
    ElementS effective_scale = params.scale;
    float scale_v = 1.f;
    if constexpr (Fp8KV) {
      const float scale_k = *static_cast<const float*>(params.scale_k);
      scale_v = *static_cast<const float*>(params.scale_v);
      effective_scale = params.scale * ElementS(scale_k);
    }

    [[maybe_unused]] auto route_current = [&](int cur) -> int {
      int k_surf = PagedKV ? cur / tiles_per_surface : 0;
      int local = cur - k_surf * tiles_per_surface;
      if (k_pre_surf != k_surf) {
        k_pre_surf = k_surf;

        auto K_2D_n = make_k_surface(k_surf);
        auto V_2D_n = make_v_surface(k_surf);

        copy_k = TiledCopyK{K_2D_n};
        copy_v = TiledCopyV{V_2D_n};

        auto thr_copy_k_n = copy_k.get_slice(thr_id);
        auto thr_copy_v_n = copy_v.get_slice(thr_id);

        prefetch_k = make_block_2d_prefetch(copy_k);
        prefetch_v =
            make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_n);

        Tensor cK_n = make_shifted_identity_tensor<1>(
            make_shape(get<0>(K_2D_n.shape()), get<1>(K_2D.shape())),
            x_shift_k);  // (k,d)
        Tensor cV_n = make_shifted_identity_tensor<0>(
            make_shape(get<0>(V_2D.shape()), get<1>(V_2D_n.shape())),
            x_shift_v);  // (v,k)

        Tensor gK_n = local_tile(
            cK_n,
            TileShapeQK{},
            make_coord(_, _, _),
            Step<X, _1, _1>{});  // (k,d,K,D)
        Tensor gV_n = local_tile(
            cV_n, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
        Tensor gV_split_n = local_tile(
            gV_n,
            TileShapePV{},
            make_coord(_, _, 0),
            Step<X, _1, _1>{});  // (v,k,VV,K)

        tKgK = thr_copy_k_n.partition_S(gK_n);        // (atom_val,k',d',K,D)
        tVgV = thr_copy_v_n.partition_S(gV_split_n);  // (atom_val,v',k',VV,K)

        pKgK = prefetch_k.get_slice(thr_id).partition_S(gK_n);
        pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_n);
      }
      return local;
    };

    if constexpr (!has_large_surface) {
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_, _, _, next_page_idx, D));
      }
    }

    /* Main loop, blocked in k. */
    for (int K = blk_k0; K < blk_k1; K++) {
      barrier_arrive(ScopeWorkgroup);

      bool need_causal = false;
      if constexpr (CausalMask) {
        need_causal = K >= blk_k1_causal;
      }

      int load_idx;
      if constexpr (has_large_surface) {
        int cur = PagedKV ? get_paged_idx(K, idx_b) : K;
        load_idx = route_current(cur);
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k, pKgK(_, _, _, load_idx, D));
        }
      } else {
        page_idx = next_page_idx;
        next_page_idx = K + 1;
        // next paged_idx
        if constexpr (PagedKV) {
          next_page_idx = get_paged_idx(next_page_idx, idx_b);
        }
        load_idx = PagedKV ? page_idx : K;
      }

      auto tKgK_cache = tKgK(_, _, _, load_idx, _);
      auto tVgV_cache = tVgV(_, _, _, _, load_idx);

      /* V prefetch for GEMM 2 */
      prefetch(prefetch_v, pVgV(_, _, _, load_idx));

      /* GEMM 1: S = Q * K^T */
      clear(tSrS); /* TODO: fuse w/ initial gemm call */
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k, tKgK_cache(_, _, _, D), tKrK);
        reorder(tQrQ, tSrQ);
        // reorder() performs the (vectorized) fp8 -> ElementQ cast; the
        // per-tensor scale_k has been folded into effective_scale above.
        reorder(tKrK, tSrK);
        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* Causal masking and k remainder masking */
      if constexpr (CausalMask) {
        if (need_causal) {
          constexpr int kTileK = tile_k;
          constexpr int n_reps = kTileK / intel::sg_size;
          constexpr int elems_per_n = tSrS.size() / n_reps;
          int k_base = K * kTileK;
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < n_reps; n++) {
            int col = k_base + n * intel::sg_size + lane_id;
            int causal_bound = col - full_tile_offset - row_base;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < elems_per_n; j++) {
              if (j < causal_bound) {
                tSrS(n * elems_per_n + j) = ElementS(-INFINITY);
              }
            }
          }
        }
      }
      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        int k = K * tile_k + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) =
              (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }
      /* Local masking */
      if constexpr (LocalMask) {
        const bool need_left = (K < blk_local_l_safe);
        const bool need_right = (K > blk_local_r_safe);
        if (need_left || need_right) {
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(
              cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
            bool left_mask = col_idx < row_idx - params.local_left;
            bool right_mask = col_idx > row_idx + params.local_right;
            if (left_mask || right_mask) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }
      }

      /* Apply softmax */
      auto rescale =
          softmax(effective_scale, K == blk_k0, tSrS, tA_max, tA_sum);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV_cache(_, _, _, VV), tVrV);
        // reorder() performs the (vectorized) fp8 -> ElementQ cast; the
        // per-tensor scale_v is applied once after the K loop below.
        reorder(tVrV, tArV);
        if (K != blk_k0) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArA.size() / VTiles; i++)
            tArA(_, _, _, VV)(i) *= broadcast<0>(rescale, tArA, i);
        }
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      if constexpr (!has_large_surface) {
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k, pKgK(_, _, _, next_page_idx, D));
        }
      }

      barrier_wait(ScopeWorkgroup);
    }

    // Apply the per-tensor V scale once after the K loop.
    if constexpr (Fp8KV) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tArA.size(); ++i) {
        tArA(i) *= ElementA(scale_v);
      }
    }
  }

  // INT8 per-token-head KV prefill: the chunk-prefill walk (Q tile per
  // work-group, one sub-group per SG_Q rows, KV tiles of tile_k slots, causal
  // mask on absolute positions, workgroup split barrier per tile) over the int8
  // cache in place. Per KV tile: one page-table read, one K and one V surface
  // for that block (the split layout's block stride is in bytes), the tile's
  // K and V scales (0 past seq_len, so stale bytes never multiply), Q*K as
  // s8 x s8 -> s32 (Int8Dot) or on the 16-bit query over widened K codes, the
  // K scale per column, masks, fp32 online softmax, the V scale per column
  // after the row sums, P*V over widened V codes. No dequantized scratch.
  template <typename QVCoord>
  CUTLASS_DEVICE void int8_kv_mainloop(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorK2D const& K_2D,  // (k,d), int8, head offset applied
      TensorV2D const& V_2D,  // (d,k), int8, head offset applied
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      QVCoord blk_qv,
      int const& idx_b,
      int blk_k0,
      int blk_k1,
      int blk_k1_causal,
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int head_kv) {
    using namespace sycl::ext::oneapi::this_work_item;

    constexpr int tile_k = get<1>(TileShapeQK{});
    // Query rows per sub-group (the chunk-prefill policies split Q only).
    constexpr int kSGRows = decltype(get<0>(
        shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{}))))::value;
    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    const int lane = get_sub_group().get_local_id()[0];
    const int sg = thr_id / intel::sg_size;
    const int tiles_per_page = params.page_size / tile_k;
    const int row_base = get<0>(blk_qv) * get<0>(TileShapeQK{}) + sg * kSGRows;

    /* Q: the 16-bit query, 64B-aligned surface as the base path */
    int x_shift_q;
    auto Q_2D_al = align_block_2d_base<1>(Q_2D, x_shift_q);
    Tensor cQ =
        make_shifted_identity_tensor<1>(Q_2D.shape(), x_shift_q);  // (q,d)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));   // (q,k)
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D)

    /* K and V of one cache block: the coordinates do not depend on the block */
    auto layout_kb = make_layout(
        make_shape(params.page_size, get<1>(K_2D.shape())), K_2D.stride());
    auto layout_vb = make_layout(
        make_shape(get<0>(V_2D.shape()), params.page_size), V_2D.stride());
    auto k_block = [&](int pblk) {
      return make_tensor(
          K_2D.data() + int64_t(pblk) * params.page_stride_bytes, layout_kb);
    };
    auto v_block = [&](int pblk) {
      return make_tensor(
          V_2D.data() + int64_t(pblk) * params.page_stride_bytes, layout_vb);
    };
    Tensor cKb = make_identity_tensor(shape(layout_kb));  // (k,d)
    Tensor cVb = make_identity_tensor(shape(layout_vb));  // (v,k)
    Tensor gKb = local_tile(
        cKb,
        TileShapeQK{},
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,d,T,D)
    Tensor gVb = local_tile(
        cVb, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,T)
    Tensor gVb_split = local_tile(
        gVb,
        TileShapePV{},
        make_coord(_, _, 0),
        Step<X, _1, _1>{});  // (v,k,VV,T)

    TiledCopyQ copy_q{Q_2D_al};
    TiledCopyV copy_v{v_block(0)};
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    auto tQgQ = thr_copy_q.partition_S(gQ);         // (atom_val,q',d',D)
    auto tVgV = thr_copy_v.partition_S(gVb_split);  // (atom_val,v',k',VV,T)
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);
    auto tVrV = thr_copy_v.partition_sg_fragment_D(gVb_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gVb_split(_, _, 0, 0));

    /* Q prefetch */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    CUTLASS_PRAGMA_UNROLL
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }

    auto phys_block = [&](int k_tile) {
      int page_local = k_tile * tile_k / params.page_size;
      page_local = page_local < params.max_pages_per_seq
                       ? page_local
                       : params.max_pages_per_seq - 1;
      return params
          .ptr_page_table[idx_b * params.max_pages_per_seq + page_local];
    };
    // fp32 scales of this tile's columns (this lane's: lane, lane + 16, ...);
    // 0 at and past seq_len, where the cache may hold stale bytes. A tile lies
    // inside one block: its slot is t * tile_k + column (t = tile in block).
    auto tile_scales = [&](void* cache, int k_tile, int t, int pblk) {
      FragSCol sc;
      const float* base = static_cast<const float*>(cache) +
                          int64_t(pblk) * params.scale_stride_block +
                          int64_t(head_kv) * params.scale_stride_head;
      int k = k_tile * tile_k + lane;
      int slot = t * tile_k + lane;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < sc.size();
           i++, k += intel::sg_size, slot += intel::sg_size) {
        sc(i) = (k < seq_len)
                    ? ElementS(base[int64_t(slot) * params.scale_stride_slot])
                    : ElementS(0);
      }
      return sc;
    };

    // ---- INT8 Q*K: s8 operands, query quantized here once per work tile ----
    using TiledCopyK8 =
        decltype(make_block_2d_copy_B(TiledMMAQK8{}, TensorK2D{}));
    using TiledCopyQ16 =
        decltype(make_block_2d_copy_A(TiledMMAQ16{}, TensorQ2D{}));
    TiledMMAQK8 mma_qk8{};
    auto thr_mma_qk8 = mma_qk8.get_slice(thr_id);
    auto thr_mma_q16 = TiledMMAQ16{}.get_slice(thr_id);
    Tensor gQ8 = local_tile(
        cQ, TileShapeQK8{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D8)
    Tensor gKb8 = local_tile(
        cKb,
        TileShapeQK8{},
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,d,T,D8)
    TiledCopyK8 copy_k8{k_block(0)};
    auto thr_copy_k8 = copy_k8.get_slice(thr_id);
    auto tKgK8 = thr_copy_k8.partition_S(gKb8);  // (atom_val,k',d',T,D8)
    auto tKrK8 = thr_copy_k8.partition_sg_fragment_D(gKb8(_, _, 0, 0));
    auto tSrK8 = thr_mma_qk8.partition_sg_fragment_B(gKb8(_, _, 0, 0));
    auto tSrQ8 = thr_mma_qk8.partition_sg_fragment_A(gQ8(_, _, 0));
    float row_scale[Int8Dot ? kSGRows : 1];
    row_scale[0] = 1.0f;

    if constexpr (Int8Dot) {
      static_assert(decltype(size(tSrQ8.tensor()))::value == kQ8LaneBytes);
      TiledCopyQ16 copy_q16{Q_2D_al};
      auto thr_copy_q16 = copy_q16.get_slice(thr_id);
      auto tQgQ16 = thr_copy_q16.partition_S(gQ8);  // (atom_val,q',d',D8)
      auto tQrQ16 = thr_copy_q16.partition_sg_fragment_D(gQ8(_, _, 0));
      auto tSrQ16 = thr_mma_q16.partition_sg_fragment_A(gQ8(_, _, 0));
      const int d_chunks = size<3>(tQgQ16);
      // (lane, value) -> (q, d) of the 16-bit A fragment, sub-group local; the
      // row of a value does not depend on the lane in the DPAS A layout, so
      // it folds to a constant per value once the loops are unrolled.
      auto qtv = tSrQ16.tv_layout();
      constexpr int kQVals = decltype(size(tSrQ16.tensor()))::value;

      // Pass 1: per-row absmax over the whole head, then over the sub-group.
      // A sub-group owns its kSGRows query rows whole: no SLM exchange.
      float row_absmax[kSGRows];
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < kSGRows; r++) {
        row_absmax[r] = 0.0f;
      }
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < kInt8DChunks; D++) {
        if (D < d_chunks) {
          copy(copy_q16, tQgQ16(_, _, _, D), tQrQ16);
          reorder(tQrQ16, tSrQ16);
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kQVals; v++) {
            const int r = int(get<0>(qtv(lane, v))) % kSGRows;
            const float a = sycl::fabs(static_cast<float>(tSrQ16.data()[v]));
            CUTLASS_PRAGMA_UNROLL
            for (int rr = 0; rr < kSGRows; rr++) {
              row_absmax[rr] =
                  (r == rr) ? sycl::fmax(row_absmax[rr], a) : row_absmax[rr];
            }
          }
        }
      }
      float row_inv[kSGRows];
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < kSGRows; r++) {
        const float m = sycl::reduce_over_group(
            get_sub_group(), row_absmax[r], sycl::maximum<float>());
        row_scale[r] = int8_attn_query_scale(m);
        row_inv[r] = int8_attn_div_rn(1.0f, row_scale[r]);
      }

      // Pass 2: the codes, as exact integers in the 16-bit A fragment,
      // converted and relaid into the s8 A operand, stored to this
      // work-item's SLM words.
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < kInt8DChunks; D++) {
        if (D < d_chunks) {
          copy(copy_q16, tQgQ16(_, _, _, D), tQrQ16);
          reorder(tQrQ16, tSrQ16);
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kQVals; v++) {
            const int r = int(get<0>(qtv(lane, v))) % kSGRows;
            float inv = row_inv[0];
            CUTLASS_PRAGMA_UNROLL
            for (int rr = 1; rr < kSGRows; rr++) {
              inv = (r == rr) ? row_inv[rr] : inv;
            }
            tSrQ16.data()[v] = static_cast<ElementQ>(int8_attn_query_code(
                static_cast<float>(tSrQ16.data()[v]), inv));
          }
          reorder(tSrQ16, tSrQ8);
          uint8_t* slot = smem_->q8[sg][D];
          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < kQ8LaneBytes; e++) {
            slot[e * intel::sg_size + lane] =
                static_cast<uint8_t>(tSrQ8.data()[e]);
          }
        }
      }
      // The KV loop reads these bytes back with sub-group block reads.
      sycl::group_barrier(get_sub_group());
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    const bool check_remainder_k = (seq_len % tile_k != 0);
    const ElementS effective_scale = params.scale;

    // s32 view of the f32 S registers (Int8Dot): same layout, same width,
    // converted in place.
    using FragS8 = decltype(thr_mma_qk8.partition_sg_fragment_C(cP));
    static_assert(
        !Int8Dot ||
            is_same_v<decltype(FragS8{}.layout()), decltype(tSrS.layout())>,
        "s32 and f32 S fragments must share one layout");
    static_assert(
        !Int8Dot || is_same_v<
                        decltype(FragS8{}.tv_layout()),
                        decltype(tSrS.tv_layout())>,
        "s32 and f32 S fragments must share one TV layout");
    auto tSrS8 = make_subgroup_tensor(
        make_tensor(recast_ptr<int32_t>(tSrS.data()), tSrS.layout()),
        tSrS.tv_layout());
    auto stv = tSrS.tv_layout();  // (lane, value) -> (q, k), sub-group local
    constexpr int kSVals = decltype(size(tSrS.tensor()))::value;

    // ---- INT8 P*V: s8 P (A operand), V codes as the s8 B operand ----
    using TiledCopyV8 =
        decltype(make_block_2d_copy_B(TiledMMAPV8{}, TensorV2D{}));
    TiledMMAPV8 mma_pv8{};
    auto thr_mma_pv8 = mma_pv8.get_slice(thr_id);
    TiledCopyV8 copy_v8{v_block(0)};
    auto thr_copy_v8 = copy_v8.get_slice(thr_id);
    auto tVgV8 = thr_copy_v8.partition_S(gVb_split);  // (atom_val,v',k',VV,T)
    auto tVrV8 = thr_copy_v8.partition_sg_fragment_D(gVb_split(_, _, 0, 0));
    auto tArV8 = thr_mma_pv8.partition_sg_fragment_B(gVb_split(_, _, 0, 0));
    auto tArP8 = thr_mma_pv8.partition_sg_fragment_A(cP);
    auto tArO8 = thr_mma_pv8.partition_sg_fragment_C(
        make_identity_tensor(select<0, 1>(TileShapePV{})));
    static_assert(
        !Int8PV || is_same_v<
                       decltype(tArO8.layout()),
                       decltype(SingleFragA{}.layout())>,
        "s32 P*V and f32 O fragments must share one layout");

    // ---- 16-bit Q*K over widened K codes (without Int8Dot) ----
    using TiledCopyKw = TiledCopyK;
    TiledCopyKw copy_kw{k_block(0)};
    auto thr_copy_kw = copy_kw.get_slice(thr_id);
    auto tKgKw = thr_copy_kw.partition_S(gKb);  // (atom_val,k',d',T,D)
    auto tKrKw = thr_copy_kw.partition_sg_fragment_D(gKb(_, _, 0, 0));
    auto tSrKw = thr_mma_qk.partition_sg_fragment_B(gKb(_, _, 0, 0));

    auto prefetch_k_tile = [&](int k_tile) {
      if (k_tile >= blk_k1) return;
      const int pblk = phys_block(k_tile);
      if constexpr (Int8Dot) {
        TiledCopyK8 copy_blk{k_block(pblk)};
        auto pf = make_block_2d_prefetch(copy_blk);
        auto pKg = pf.get_slice(thr_id).partition_S(gKb8);
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(pKg); D++) {
          prefetch(pf, pKg(_, _, _, k_tile % tiles_per_page, D));
        }
      } else {
        TiledCopyKw copy_blk{k_block(pblk)};
        auto pf = make_block_2d_prefetch(copy_blk);
        auto pKg = pf.get_slice(thr_id).partition_S(gKb);
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(pKg); D++) {
          prefetch(pf, pKg(_, _, _, k_tile % tiles_per_page, D));
        }
      }
    };

    prefetch_k_tile(blk_k0);

    /* Main loop, blocked in k. */
    for (int K = blk_k0; K < blk_k1; K++) {
      barrier_arrive(ScopeWorkgroup);

      bool need_causal = false;
      if constexpr (CausalMask) {
        need_causal = K >= blk_k1_causal;
      }

      const int pblk = phys_block(K);
      const int t = K % tiles_per_page;
      FragSCol ksc = tile_scales(params.scale_k, K, t, pblk);
      FragSCol vsc = tile_scales(params.scale_v, K, t, pblk);
      auto V_blk = v_block(pblk);
      TiledCopyV copy_v_blk{V_blk};

      /* V prefetch for GEMM 2 */
      auto pf_v = make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_blk);
      auto pVgV = pf_v.get_slice(thr_id).partition_S(gVb);
      prefetch(pf_v, pVgV(_, _, _, t));

      /* GEMM 1: S = Q * K^T */
      if constexpr (Int8Dot) {
        TiledCopyK8 copy_k8_blk{k_block(pblk)};
        clear(tSrS8);
        const int d8_chunks = size<4>(tKgK8);
        for (int D = 0; D < d8_chunks; D++) {
          copy(copy_k8_blk, tKgK8(_, _, _, t, D), tKrK8);
          // Sub-group block reads: lane l gets bytes [i * 16 + l] of each
          // 256-byte block, i.e. its fragment elements in register order.
#ifdef __SYCL_DEVICE_ONLY__
          auto slot = sycl::address_space_cast<
              sycl::access::address_space::local_space,
              sycl::access::decorated::yes>(smem_->q8[sg][D]);
#endif
          CUTLASS_PRAGMA_UNROLL
          for (int b = 0; b < kQ8Blocks; b++) {
#ifdef __SYCL_DEVICE_ONLY__
            Q8Block blk = __spirv_SubgroupBlockReadINTEL<Q8Block>(
                slot.get_decorated() + b * 16 * intel::sg_size);
#else
            Q8Block blk{};  // host pass: never executed
#endif
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 16; i++) {
              tSrQ8.data()[b * 16 + i] = static_cast<int8_t>(blk[i]);
            }
          }
          reorder(tKrK8, tSrK8);
          cute::gemm(mma_qk8, tSrQ8, tSrK8, tSrS8);
        }
        // S = s32 * q_scale[row] (k_scale below), exact int32 dot.
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kSVals; v++) {
          const int r = int(get<0>(stv(lane, v))) % kSGRows;
          float qs = row_scale[0];
          CUTLASS_PRAGMA_UNROLL
          for (int rr = 1; rr < kSGRows; rr++) {
            qs = (r == rr) ? row_scale[rr] : qs;
          }
          tSrS.data()[v] = static_cast<ElementS>(tSrS8.data()[v]) * qs;
        }
      } else {
        TiledCopyKw copy_k_blk{k_block(pblk)};
        clear(tSrS);
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(tKgKw); D++) {
          copy(copy_q, tQgQ(_, _, _, D), tQrQ);
          copy(copy_k_blk, tKgKw(_, _, _, t, D), tKrKw);
          reorder(tQrQ, tSrQ);
          reorder(tKrKw, tSrKw);  // int8 -> 16-bit (exact)
          cute::gemm(mma_qk, tSrQ, tSrKw, tSrS);
        }
      }
      prefetch_k_tile(K + 1);

      // S(:, j) *= k_scale[j]: per-token K dequantization (before masking).
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tSrS.size(); i++) {
        tSrS(i) *= broadcast<1>(ksc, tSrS, i);
      }

      /* Causal masking (absolute positions), as the base walk */
      if constexpr (CausalMask) {
        if (need_causal) {
          constexpr int n_reps = tile_k / intel::sg_size;
          constexpr int elems_per_n = tSrS.size() / n_reps;
          int k_base = K * tile_k;
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < n_reps; n++) {
            int col = k_base + n * intel::sg_size + lane;
            int causal_bound = col - full_tile_offset - row_base;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < elems_per_n; j++) {
              if (j < causal_bound) {
                tSrS(n * elems_per_n + j) = ElementS(-INFINITY);
              }
            }
          }
        }
      }
      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        int k = K * tile_k + lane;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) =
              (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }

      /* Softmax; then P(:, j) *= v_scale[j] after the row sums */
      auto rescale =
          softmax(effective_scale, K == blk_k0, tSrS, tA_max, tA_sum);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tSrS.size(); i++) {
        tSrS(i) *= broadcast<1>(vsc, tSrS, i);
      }

      if constexpr (Int8PV) {
        /* P to int8 per query row of this tile; GEMM 2 in s8 x s8 -> s32 */
        auto pmax = reduce<1>(tSrS, sycl::maximum{});
        decltype(pmax) pscale, pinv;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < pmax.size(); i++) {
          pscale(i) = int8_attn_p_scale(pmax(i));
          pinv(i) = int8_attn_div_rn(1.0f, pscale(i));
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = int8_attn_query_code(tSrS(i), broadcast<0>(pinv, tSrS, i));
        }
        reorder(tSrS, tArP8);  // exact integers -> s8 A operand
        TiledCopyV8 copy_v8_blk{V_blk};
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          copy(copy_v8_blk, tVgV8(_, _, _, VV, t), tVrV8);
          reorder(tVrV8, tArV8);  // relayout only, codes as loaded
          clear(tArO8);
          cute::gemm(mma_pv8, tArP8, tArV8, tArO8);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArA.size() / VTiles; i++) {
            ElementA o = tArA(_, _, _, VV)(i);
            if (K != blk_k0) {
              o *= broadcast<0>(rescale, tArA, i);
            }
            tArA(_, _, _, VV)(i) = o + static_cast<ElementA>(tArO8(i)) *
                                           broadcast<0>(pscale, tArA, i);
          }
        }
      } else {
        reorder(tSrS, tArP);

        /* GEMM 2: A += P * V over widened V codes, split in v dimension */
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          copy(copy_v_blk, tVgV(_, _, _, VV, t), tVrV);
          reorder(tVrV, tArV);  // int8 -> 16-bit (exact)
          if (K != blk_k0) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tArA.size() / VTiles; i++)
              tArA(_, _, _, VV)(i) *= broadcast<0>(rescale, tArA, i);
          }
          cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
        }
      }

      barrier_wait(ScopeWorkgroup);
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  FragSRow softmax(
      ElementS scale,      // Effective softmax scale (fp8 K scale folded in)
      bool first_block,    // First softmax block?
      FragS& tS,           // Softmax src/dst block
      FragSRow& tS_max,    // Softmax row-wise max accumulator
      FragSRow& tS_sum) {  // Softmax row-wise sum accumulator

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++) {
      tS(i) = sycl::native::exp2(scale * tS(i) - broadcast<0>(tS_max, tS, i));
    }

    /* Rescale existing S sums */
    if (!first_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) *= rescale(i);
      }
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_sum.size(); i++) {
      tS_sum(i) += tS_bsum(i);
    }
    return rescale;
  }
};

template <
    class DispatchPolicy_,
    bool PagedKV_,
    bool CausalMask_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,  // Optional TiledCopy for loading K
    class TiledCopyV_ = void,  // Optional TiledCopy for loading V
    bool LocalMask_ = false,
    class TiledMMAQK8_ = void>  // INT8 KV only: s8 x s8 -> s32 MMA for Q*K
struct DecodeFwdMainloop {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Could not find a mainloop specialization.");
};

// int8_attn_div_rn / int8_attn_query_scale / int8_attn_query_code: the shared
// INT8 query rule, defined above FMHAFwdMainloop.

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool PagedKV_,
    bool CausalMask_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_,
    bool LocalMask_,
    class TiledMMAQK8_>
struct DecodeFwdMainloop<
    XeDefault<Stages>,
    PagedKV_,
    CausalMask_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_,
    LocalMask_,
    TiledMMAQK8_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using ElementQ = typename TensorQ::engine_type::value_type;
  using ElementK = typename TensorK::engine_type::value_type;

  using TensorQ2D =
      decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D =
      decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D =
      decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ = conditional_t<
      is_void_v<TiledCopyQ_>,
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})),
      TiledCopyQ_>;
  using TiledCopyK = conditional_t<
      is_void_v<TiledCopyK_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})),
      TiledCopyK_>;
  using TiledCopyV = conditional_t<
      is_void_v<TiledCopyV_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})),
      TiledCopyV_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;  // (atom val,q',v')
  using FragA =
      expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  // static_assert(is_same_v<decltype(FragSRow{}.shape()), float>, "dtype
  // mismatched");
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool Fp8KV =
      is_any_of_v<ElementK, float_e5m2_t, float_e4m3_t>;
  // INT8 per-token-head KV: int8 K/V tiles plus one fp32 scale per (KV token,
  // KV head), read from [num_blocks, block_size, num_heads_kv] scale caches
  // (scale_k / scale_v point to them). Paged KV only.
  static constexpr bool Int8KV = is_same_v<ElementK, int8_t>;
  static constexpr bool LocalMask = LocalMask_;
  // INT8 Q*K (int8 KV only): Q is quantized per (token, query head) in the
  // kernel, Q*K runs as s8 x s8 -> s32 DPAS on the cached K codes, and
  // S = s32 * q_scale * k_scale in fp32. TiledMMAQK keeps typing the fp32 S
  // fragment and the Q load; the s32 fragment of TiledMMAQK8 shares its
  // layout (asserted below). Softmax and P*V are unchanged.
  static constexpr bool Int8Dot = !is_void_v<TiledMMAQK8_>;
  using TiledMMAQK8 = conditional_t<Int8Dot, TiledMMAQK8_, TiledMMAQK_>;
  static_assert(!Int8Dot || Int8KV, "the int8 Q*K dot needs an int8 KV cache");

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    void* const scale_k;
    void* const scale_v;
    // Paged KV Cache
    int const* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    // Local Mask
    int window_size_left;
    int window_size_right;
    // Physical stride between paged blocks in seq-position units
    int page_stride_elements;
    // INT8 KV only (see FMHAFwdMainloop::Arguments).
    int64_t scale_stride_block = 0;
    int64_t scale_stride_slot = 0;
    int64_t scale_stride_head = 0;
    int64_t page_stride_bytes = 0;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data. Int8Dot: each work-item's s8 Q operand bytes (four 64-wide head
  // chunks), written once per work tile and read back by the same work-item
  // in every KV tile, so the 32 GRF of s8 Q are not live across the KV loop
  // next to the 128-GRF O accumulator. The kernel's SharedStorage is a union
  // with the epilogue's (larger) reduction buffer; mainloop and epilogue are
  // separated by a barrier there.
  static constexpr int kInt8DotDChunks = 4;  // head_dim <= 256
  using FragQ8 = decltype(TiledMMAQK8{}.get_slice(0).partition_sg_fragment_A(
      make_identity_tensor(select<0, 2>(TileShapeQK{}))));
  static constexpr int kQ8LaneBytes =
      Int8Dot ? int(decltype(size(FragQ8{}.tensor()))::value) : 1;
  struct SharedStorageInt8Dot {
    alignas(64) int8_t
        q8[SGPerWG::value][kInt8DotDChunks][intel::sg_size][kQ8LaneBytes];
  };
  struct SharedStorageNone {};
  using SharedStorage =
      conditional_t<Int8Dot, SharedStorageInt8Dot, SharedStorageNone>;

  Params params;
  SharedStorage* smem_ = nullptr;

  //
  // Methods
  //

  DecodeFwdMainloop(Params const& params_, SharedStorage& smem)
      : params(params_), smem_(&smem) {}

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.scale_k,
        args.scale_v,
        args.ptr_page_table,
        args.page_size,
        args.max_pages_per_seq,
        args.total_seqlen_kv,
        args.window_size_left,
        args.window_size_right,
        args.page_stride_elements,
        args.scale_stride_block,
        args.scale_stride_slot,
        args.scale_stride_head,
        args.page_stride_bytes};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  template <bool has_large_surface, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorK2D const& K_2D,  // (k,d)
      TensorV2D const& V_2D,  // (d,k)
      FragA& tArA,            // Output accumulator (q,v)
      FragARow& tA_max,       // Softmax row-wise max accumulator
      FragARow& tA_sum,       // Softmax row-wise sum accumulator
      QVCoord blk_qv,         // WG tile indices: (Q,V)
      int const& idx_b,       // WG tile indices: (B)
      int blk_k0,             // K block range: [K0,K1)
      int blk_k1,
      int total_blk,  // Total # of K blocks
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int discard_seq_coord,
      int head_kv = 0) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* 64B-align the per-head block-2D surfaces (see block_2d_align.hpp) */
    int x_shift_q, x_shift_k, x_shift_v;
    auto Q_2D_al = align_block_2d_base<1>(Q_2D, x_shift_q);
    auto K_2D_al = align_block_2d_base<1>(K_2D, x_shift_k);
    auto V_2D_al = align_block_2d_base<0>(V_2D, x_shift_v);

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ =
        make_shifted_identity_tensor<1>(Q_2D.shape(), x_shift_q);  // (q,d)
    Tensor cK =
        make_shifted_identity_tensor<1>(K_2D.shape(), x_shift_k);  // (k,d)
    Tensor cV =
        make_shifted_identity_tensor<0>(V_2D.shape(), x_shift_v);  // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));   // (q,k)

    auto K_2D_0 = K_2D_al;
    auto V_2D_0 = V_2D_al;

    // --- Xe 2D-block surface-height split
    // ------------------------------------- The Xe LSC 2D block-load descriptor
    // stores the surface height in a 24-bit field, so a (rank-2) K/V surface
    // can address at most 2^24 rows. A paged KV cache may span many more rows
    // (total_seqlen_kv = num_blocks * page_stride_elements), which overflows
    // that field and makes the hardware wrap addresses. To keep the original
    // rank-2 load/prefetch logic we split the K and V surfaces into
    // page-aligned sub-surfaces, each at most 2^24 rows tall, and route every
    // physical tile to the surface that physically contains it.

    // The split state and the two `make_*_surface` lambdas are declared here at
    // function scope (rather than inside the `if constexpr` below) so that the
    // per-tile routing inside the K loop can reuse them. They are only ever
    // exercised when `has_large_surface` is true; otherwise they stay unused.
    constexpr int tile_k = get<1>(TileShapeQK{});
    [[maybe_unused]] int tiles_per_surface = 0x3fffffff;
    [[maybe_unused]] int rows_per_surface = 0;
    [[maybe_unused]] int k_rows_total = get<0>(K_2D.shape());
    [[maybe_unused]] int v_cols_total = get<1>(V_2D.shape());
    [[maybe_unused]] int64_t k_surf_shift = 0;
    [[maybe_unused]] int64_t v_surf_shift = 0;
    [[maybe_unused]] int k_pre_surf = -1;

    // Lazily materialize an arbitrary overflow sub-surface `surf` (>= 1). The
    // base is shifted by `surf` full surfaces and the height/width is the real
    // remaining extent of that surface, so the 2D block load never reads past
    // the cache. These are only ever built inside the K loop's cold branch, so
    // supporting any number of sub-surfaces adds at most one live descriptor
    // and the common path keeps its original register/descriptor footprint.
    [[maybe_unused]] auto make_k_surface = [&](int surf) {
      int64_t rem = int64_t(k_rows_total) - int64_t(surf) * rows_per_surface;
      int rows = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (rows < 1) {
        rows = 1;
      }
      return make_tensor(
          K_2D_al.data() + int64_t(surf) * k_surf_shift,
          make_layout(
              make_shape(rows, get<1>(K_2D_al.shape())), K_2D_al.stride()));
    };
    [[maybe_unused]] auto make_v_surface = [&](int surf) {
      int64_t rem = int64_t(v_cols_total) - int64_t(surf) * rows_per_surface;
      int cols = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (cols < 1) {
        cols = 1;
      }
      return make_tensor(
          V_2D_al.data() + int64_t(surf) * v_surf_shift,
          make_layout(
              make_shape(get<0>(V_2D_al.shape()), cols), V_2D_al.stride()));
    };

    if constexpr (has_large_surface) {
      constexpr int kXeMaxSurfaceRows = 1 << 24;
      if constexpr (PagedKV) {
        int page_stride_tiles = params.page_stride_elements / tile_k;
        int blocks_per_surface =
            kXeMaxSurfaceRows / params.page_stride_elements;
        if (blocks_per_surface < 1) {
          blocks_per_surface = 1;
        }
        tiles_per_surface = blocks_per_surface * page_stride_tiles;
        rows_per_surface = tiles_per_surface * tile_k;
      }

      // Build the (rank-2) K/V surface 0. Surface 0 keeps the original base
      // pointer and is the only surface addressed on the hot path; overflow
      // surfaces are materialized lazily (see make_k_surface/make_v_surface).
      int k_rows0, v_cols0;
      if constexpr (PagedKV) {
        k_rows0 =
            k_rows_total < rows_per_surface ? k_rows_total : rows_per_surface;
        k_surf_shift = int64_t(rows_per_surface) * get<0>(K_2D.stride());
        v_cols0 =
            v_cols_total < rows_per_surface ? v_cols_total : rows_per_surface;
        v_surf_shift = int64_t(rows_per_surface) * get<1>(V_2D.stride());
      } else {
        k_rows0 = k_rows_total;
        v_cols0 = v_cols_total;
      }

      K_2D_0 = make_tensor(
          K_2D_al.data(),
          make_layout(
              make_shape(k_rows0, get<1>(K_2D_al.shape())), K_2D_al.stride()));
      V_2D_0 = make_tensor(
          V_2D_al.data(),
          make_layout(
              make_shape(get<0>(V_2D_al.shape()), v_cols0), V_2D_al.stride()));

      cK = make_shifted_identity_tensor<1>(
          make_shape(
              PagedKV ? rows_per_surface : k_rows_total, get<1>(K_2D.shape())),
          x_shift_k);  // (k,d)
      cV = make_shifted_identity_tensor<0>(
          make_shape(
              get<0>(V_2D.shape()), PagedKV ? rows_per_surface : v_cols_total),
          x_shift_v);  // (v,k)
    }

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D)
    Tensor gK = local_tile(
        cK,
        TileShapeQK{},
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,d,K,D)
    Tensor gV =
        local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
    Tensor gV_split = local_tile(
        gV,
        TileShapePV{},
        make_coord(_, _, 0),
        Step<X, _1, _1>{});  // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D_al};
    TiledCopyK copy_k{K_2D_0};
    TiledCopyV copy_v{V_2D_0};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_0);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    // ------
    // Kernel
    // ------

    // PagedKV
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int page_stride_tiles = params.page_stride_elements / get<1>(TileShapeQK{});
    int tile_idx = blk_k0;
    int b_offset = idx_b * params.max_pages_per_seq;
    if constexpr (PagedKV) {
      int page_local_idx = tile_idx * get<1>(TileShapeQK{}) / params.page_size;
      int block_idx = params.ptr_page_table[b_offset + page_local_idx];
      tile_idx = block_idx * page_stride_tiles + tile_idx % tiles_per_page;
    }

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }

    [[maybe_unused]] auto update_surface_info = [&](int& next_page_idx) {
      // Route the current physical tile to its sub-surface (see split above).
      int k_surf = PagedKV ? next_page_idx / tiles_per_surface : 0;
      next_page_idx = next_page_idx - k_surf * tiles_per_surface;

      if (k_pre_surf != k_surf) {
        k_pre_surf = k_surf;

        auto K_2D_n = make_k_surface(k_surf);
        auto V_2D_n = make_v_surface(k_surf);

        copy_k = TiledCopyK{K_2D_n};
        copy_v = TiledCopyV{V_2D_n};

        auto thr_copy_k_n = copy_k.get_slice(thr_id);
        auto thr_copy_v_n = copy_v.get_slice(thr_id);

        prefetch_k = make_block_2d_prefetch(copy_k);
        prefetch_v =
            make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_n);

        Tensor cK_n = make_shifted_identity_tensor<1>(
            make_shape(get<0>(K_2D_n.shape()), get<1>(K_2D.shape())),
            x_shift_k);  // (k,d)
        Tensor cV_n = make_shifted_identity_tensor<0>(
            make_shape(get<0>(V_2D.shape()), get<1>(V_2D_n.shape())),
            x_shift_v);  // (v,k)

        Tensor gK_n = local_tile(
            cK_n,
            TileShapeQK{},
            make_coord(_, _, _),
            Step<X, _1, _1>{});  // (k,d,K,D)
        Tensor gV_n = local_tile(
            cV_n, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
        Tensor gV_split_n = local_tile(
            gV_n,
            TileShapePV{},
            make_coord(_, _, 0),
            Step<X, _1, _1>{});  // (v,k,VV,K)

        tKgK = thr_copy_k_n.partition_S(gK_n);        // (atom_val,k',d',K,D)
        tVgV = thr_copy_v_n.partition_S(gV_split_n);  // (atom_val,v',k',VV,K)

        pKgK = prefetch_k.get_slice(thr_id).partition_S(gK_n);
        pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_n);
      }
    };

    if constexpr (has_large_surface) {
      update_surface_info(tile_idx);
    }

    for (int D = 0; D < size<4>(pKgK); D++) {
      if constexpr (!Int8KV) {
        prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
      }
    }

    // INT8 Q*K: the query is quantized here, once per work tile, before the
    // O accumulator comes alive, with the shared rule (int8_attn_query_scale
    // / int8_attn_query_code): one fp32 scale per packed row = (token, query
    // head). The s8 A operand bytes go to this work-item's SLM slot
    // (SharedStorageInt8Dot) and are read back one 64-wide chunk at a time in
    // the KV loop. Unused (and empty after optimization) without Int8Dot.
    constexpr int kQRows = get<0>(TileShapeQK{});
    constexpr int kMaxDChunks = kInt8DotDChunks;
    auto thr_mma_qk8 = TiledMMAQK8{}.get_slice(thr_id);
    float row_scale[Int8Dot ? kQRows : 1];
    if constexpr (Int8Dot) {
      const int d_chunks = size<3>(tQgQ);
      const int lane = get_sub_group().get_local_id()[0];
      // (lane, value) -> (q, d) of the 16-bit A fragment; the row of a value
      // does not depend on the lane in the DPAS A layout, so it folds to a
      // constant per value once the loops are unrolled.
      auto qtv = tSrQ.tv_layout();
      constexpr int kQVals = decltype(size(tSrQ.tensor()))::value;

      // Pass 1: per-row absmax over the whole head (all D chunks), then over
      // the sub-group. Every sub-group holds the full Q tile, so each
      // computes the same scales; no SLM exchange.
      float row_absmax[kQRows];
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < kQRows; r++) {
        row_absmax[r] = 0.0f;
      }
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < kMaxDChunks; D++) {
        if (D < d_chunks) {
          copy(copy_q, tQgQ(_, _, _, D), tQrQ);
          reorder(tQrQ, tSrQ);
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kQVals; v++) {
            const int r = get<0>(qtv(lane, v));
            const float a = sycl::fabs(static_cast<float>(tSrQ.data()[v]));
            CUTLASS_PRAGMA_UNROLL
            for (int rr = 0; rr < kQRows; rr++) {
              row_absmax[rr] =
                  (r == rr) ? sycl::fmax(row_absmax[rr], a) : row_absmax[rr];
            }
          }
        }
      }
      float row_inv[kQRows];
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < kQRows; r++) {
        const float m = sycl::reduce_over_group(
            get_sub_group(), row_absmax[r], sycl::maximum<float>());
        row_scale[r] = int8_attn_query_scale(m);
        row_inv[r] = int8_attn_div_rn(1.0f, row_scale[r]);
      }

      // Pass 2: the codes, as exact integers in the 16-bit A fragment, then
      // converted and relaid into the s8 A operand.
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < kMaxDChunks; D++) {
        if (D < d_chunks) {
          copy(copy_q, tQgQ(_, _, _, D), tQrQ);
          reorder(tQrQ, tSrQ);
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kQVals; v++) {
            const int r = get<0>(qtv(lane, v));
            float inv = row_inv[0];
            CUTLASS_PRAGMA_UNROLL
            for (int rr = 1; rr < kQRows; rr++) {
              inv = (r == rr) ? row_inv[rr] : inv;
            }
            tSrQ.data()[v] = static_cast<ElementQ>(
                int8_attn_query_code(static_cast<float>(tSrQ.data()[v]), inv));
          }
          auto q8 = thr_mma_qk8.partition_sg_fragment_A(gQ(_, _, 0));
          reorder(tSrQ, q8);
          int8_t* slot = smem_->q8[thr_id / intel::sg_size][D][lane];
          CUTLASS_PRAGMA_UNROLL
          for (int b = 0; b < kQ8LaneBytes; b++) {
            slot[b] = q8.data()[b];
          }
        }
      }
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    // FP8 KV Scale (per-tensor): fold scale_k into the softmax scale and
    // defer scale_v to a single post-loop output rescale. Mathematically
    // identical to per-element K/V rescaling (S = Q·K is linear in K and
    // O = sum P_i·V_i is linear in V), but removes both rescales from the
    // inner loop. The remaining fp8 -> ElementQ conversion is performed by
    // reorder() using the subgroup-vectorized cast primitive.
    ElementS effective_scale = params.scale;
    float scale_v = 1.f;
    if constexpr (Fp8KV) {
      const float scale_k = *static_cast<const float*>(params.scale_k);
      scale_v = *static_cast<const float*>(params.scale_v);
      effective_scale = params.scale * ElementS(scale_k);
    }

    // INT8 KV helpers (only instantiated for int8 K/V; see Arguments).
    auto int8_phys_block = [&](int k_tile) {
      int page_local = k_tile * get<1>(TileShapeQK{}) / params.page_size;
      page_local = page_local < params.max_pages_per_seq
                       ? page_local
                       : params.max_pages_per_seq - 1;
      return params
          .ptr_page_table[idx_b * params.max_pages_per_seq + page_local];
    };
    auto int8_tile_in_block = [&](int k_tile) {
      return k_tile % (params.page_size / get<1>(TileShapeQK{}));
    };
    // INT8 decode tile step: one page-table read per tile, scale loads issued
    // before the K loads, per-block K/V surfaces built once per tile, V of this
    // tile and K of the next tile prefetched through per-block surfaces.
    auto int8_k_surface = [&](int pblk) {
      return make_tensor(
          K_2D.data() + int64_t(pblk) * params.page_stride_bytes,
          make_layout(
              make_shape(params.page_size, get<1>(K_2D.shape())),
              K_2D.stride()));
    };
    auto int8_v_surface = [&](int pblk) {
      return make_tensor(
          V_2D.data() + int64_t(pblk) * params.page_stride_bytes,
          make_layout(
              make_shape(get<0>(V_2D.shape()), params.page_size),
              V_2D.stride()));
    };
    auto int8_scales = [&](void* cache, int k_tile, int pblk) {
      FragSCol sc;
      const float* base = static_cast<const float*>(cache) +
                          int64_t(pblk) * params.scale_stride_block +
                          int64_t(head_kv) * params.scale_stride_head;
      int k =
          get<0>(tKgK(0, 0, 0, k_tile, 0)) + get_sub_group().get_local_id()[0];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < sc.size(); i++, k += intel::sg_size) {
        sc(i) = (k < seq_len) ? ElementS(
                                    base
                                        [int64_t(k % params.page_size) *
                                         params.scale_stride_slot])
                              : ElementS(0);
      }
      return sc;
    };
    auto int8_prefetch_k = [&](int k_tile) {
      if (k_tile >= blk_k1) return;
      auto K_blk = int8_k_surface(int8_phys_block(k_tile));
      TiledCopyK copy_k_blk{K_blk};
      auto pf_k = make_block_2d_prefetch(copy_k_blk);
      auto pKgK_blk = pf_k.get_slice(thr_id).partition_S(local_tile(
          make_identity_tensor(K_blk.shape()),
          TileShapeQK{},
          make_coord(_, _, _),
          Step<X, _1, _1>{}));
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(pKgK_blk); D++) {
        prefetch(pf_k, pKgK_blk(_, _, _, int8_tile_in_block(k_tile), D));
      }
    };
    auto int8_decode_tile = [&](int K) {
      const int pblk = int8_phys_block(K);
      const int t = int8_tile_in_block(K);
      FragSCol ksc = int8_scales(params.scale_k, K, pblk);
      FragSCol vsc = int8_scales(params.scale_v, K, pblk);

      auto K_blk = int8_k_surface(pblk);
      TiledCopyK copy_k_blk{K_blk};
      auto tKgK_blk = thr_copy_k.partition_S(local_tile(
          make_identity_tensor(K_blk.shape()),
          TileShapeQK{},
          make_coord(_, _, _),
          Step<X, _1, _1>{}));
      auto V_blk = int8_v_surface(pblk);
      TiledCopyV copy_v_blk{V_blk};
      auto gV_blk = local_tile(
          make_identity_tensor(V_blk.shape()),
          tile_shape_v,
          make_coord(get<1>(blk_qv), _));
      auto tVgV_blk = thr_copy_v.partition_S(local_tile(
          gV_blk, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{}));
      auto pf_v = make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_blk);
      auto pVgV_blk = pf_v.get_slice(thr_id).partition_S(gV_blk);

      /* GEMM 1: S = K * Q */
      clear(tSrS);
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k_blk, tKgK_blk(_, _, _, t, D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);
        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }
      prefetch(pf_v, pVgV_blk(_, _, _, t));
      int8_prefetch_k(K + 1);

      // S(:, j) *= k_scale[j]: per-token K dequantization (before masking).
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tSrS.size(); i++) {
        tSrS(i) *= broadcast<1>(ksc, tSrS, i);
      }

      /* Local/sliding window masking */
      if constexpr (LocalMask) {
        int decode_row = seq_len - 1 - full_tile_offset;
        Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
        Tensor gP = local_tile(
            cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
        auto cS_thread = thr_mma_qk.partition_C(gP);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
          bool left_mask = col_idx < decode_row - params.window_size_left;
          bool right_mask = col_idx > decode_row + params.window_size_right;
          if (left_mask || right_mask) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        int k = get<0>(tKgK(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) =
              (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }

      softmax(effective_scale, K == blk_k0, tSrS, tA_max, tA_sum, tArA);
      // P(:, j) *= v_scale[j]: per-token V dequantization, applied after the
      // softmax row sums so the normalizer is unaffected.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tSrS.size(); i++) {
        tSrS(i) *= broadcast<1>(vsc, tSrS, i);
      }
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v_blk, tVgV_blk(_, _, _, VV, t), tVrV);
        reorder(tVrV, tArV);
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }
      barrier();
    };

    if constexpr (Int8Dot) {
      // INT8 Q*K walk: q8 / row_scale come from the prologue above. Each KV
      // tile's K codes go into the s8 x s8 -> s32 DPAS as loaded (block 2D
      // copy built for the s8 MMA, no widening); the s32 dot is exact
      // (|sum| <= 256 * 127 * 127 < 2^31).
      const int d_chunks = size<4>(tKgK);
      using TiledCopyK8 =
          decltype(make_block_2d_copy_B(TiledMMAQK8{}, TensorK2D{}));
      TiledMMAQK8 mma_qk8{};
      TiledCopyK8 copy_k8{K_2D};
      auto thr_copy_k8 = copy_k8.get_slice(thr_id);
      auto tKrK8 = thr_copy_k8.partition_sg_fragment_D(gK(_, _, 0, 0));
      auto tSrK8 = thr_mma_qk8.partition_sg_fragment_B(gK(_, _, 0, 0));
      auto tSrQ8 = thr_mma_qk8.partition_sg_fragment_A(gQ(_, _, 0));
      static_assert(decltype(size(tSrQ8.tensor()))::value == kQ8LaneBytes);
      using FragS8 = decltype(thr_mma_qk8.partition_sg_fragment_C(cP));
      static_assert(
          is_same_v<decltype(FragS8{}.layout()), decltype(tSrS.layout())>,
          "s32 and f32 S fragments must share one layout");
      static_assert(
          is_same_v<decltype(FragS8{}.tv_layout()), decltype(tSrS.tv_layout())>,
          "s32 and f32 S fragments must share one TV layout");
      // The s32 accumulator lives in the f32 S registers (same layout, same
      // width) and is converted in place: 8 GRF fewer in the KV loop.
      auto tSrS8 = make_subgroup_tensor(
          make_tensor(recast_ptr<int32_t>(tSrS.data()), tSrS.layout()),
          tSrS.tv_layout());
      const int lane = get_sub_group().get_local_id()[0];
      auto stv = tSrS.tv_layout();  // (lane, value) -> (q, k) of the S tile
      constexpr int kSVals = decltype(size(tSrS.tensor()))::value;

      auto int8_dot_tile = [&](int K) {
        const int pblk = int8_phys_block(K);
        const int t = int8_tile_in_block(K);
        FragSCol ksc = int8_scales(params.scale_k, K, pblk);
        FragSCol vsc = int8_scales(params.scale_v, K, pblk);

        auto K_blk = int8_k_surface(pblk);
        TiledCopyK8 copy_k8_blk{K_blk};
        auto tKgK8_blk = thr_copy_k8.partition_S(local_tile(
            make_identity_tensor(K_blk.shape()),
            TileShapeQK{},
            make_coord(_, _, _),
            Step<X, _1, _1>{}));
        auto V_blk = int8_v_surface(pblk);
        TiledCopyV copy_v_blk{V_blk};
        auto gV_blk = local_tile(
            make_identity_tensor(V_blk.shape()),
            tile_shape_v,
            make_coord(get<1>(blk_qv), _));
        auto tVgV_blk = thr_copy_v.partition_S(local_tile(
            gV_blk, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{}));
        auto pf_v = make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_blk);
        auto pVgV_blk = pf_v.get_slice(thr_id).partition_S(gV_blk);

        /* GEMM 1: S = K * Q, s8 x s8 -> s32 */
        clear(tSrS8);
        for (int D = 0; D < d_chunks; D++) {
          copy(copy_k8_blk, tKgK8_blk(_, _, _, t, D), tKrK8);
          const int8_t* slot = smem_->q8[thr_id / intel::sg_size][D][lane];
          CUTLASS_PRAGMA_UNROLL
          for (int b = 0; b < kQ8LaneBytes; b++) {
            tSrQ8.data()[b] = slot[b];
          }
          reorder(tKrK8, tSrK8);
          cute::gemm(mma_qk8, tSrQ8, tSrK8, tSrS8);
        }
        prefetch(pf_v, pVgV_blk(_, _, _, t));
        int8_prefetch_k(K + 1);

        // S = s32 * q_scale[row] * k_scale[col] in fp32 (before masking).
        // Columns past seq_len have k_scale 0 (finite stale codes) and are
        // masked below.
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kSVals; v++) {
          const int r = get<0>(stv(lane, v));
          float qs = row_scale[0];
          CUTLASS_PRAGMA_UNROLL
          for (int rr = 1; rr < kQRows; rr++) {
            qs = (r == rr) ? row_scale[rr] : qs;
          }
          tSrS.data()[v] = static_cast<ElementS>(tSrS8.data()[v]) * qs;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) *= broadcast<1>(ksc, tSrS, i);
        }

        /* Local/sliding window masking */
        if constexpr (LocalMask) {
          int decode_row = seq_len - 1 - full_tile_offset;
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(
              cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
            bool left_mask = col_idx < decode_row - params.window_size_left;
            bool right_mask = col_idx > decode_row + params.window_size_right;
            if (left_mask || right_mask) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }

        /* k masking for remainder tiles */
        if (check_remainder_k && K == blk_k1 - 1) {
          FragSCol k_rem_mask;
          int k =
              get<0>(tKgK(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
            k_rem_mask(i) =
                (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); i++) {
            tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
          }
        }

        softmax(effective_scale, K == blk_k0, tSrS, tA_max, tA_sum, tArA);
        // P(:, j) *= v_scale[j] after the row sums, as in int8_decode_tile.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) *= broadcast<1>(vsc, tSrS, i);
        }
        reorder(tSrS, tArP);

        /* GEMM 2: A += P * V, split in v dimension */
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          copy(copy_v_blk, tVgV_blk(_, _, _, VV, t), tVrV);
          reorder(tVrV, tArV);
          cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
        }
        barrier();
      };

      int8_prefetch_k(blk_k0);
      for (int K = blk_k0; K < blk_k1; K++) {
        int8_dot_tile(K);
      }
      return;
    }

    if constexpr (Int8KV) {
      int8_prefetch_k(blk_k0);
    }

    /* Main loop, blocked in k. */
    int next_tile_idx;
    for (int K = blk_k0; K < blk_k1; K++) {
      if constexpr (Int8KV) {
        int8_decode_tile(K);
        continue;
      }
      /* Split barrier to keep threads together */
      // barrier_arrive(ScopeWorkgroup);

      auto tKgK_cache =
          PagedKV ? tKgK(_, _, _, tile_idx, _) : tKgK(_, _, _, K, _);
      auto tVgV_cache =
          PagedKV ? tVgV(_, _, _, _, tile_idx) : tVgV(_, _, _, _, K);

      /* GEMM 1: S = K * Q */
      clear(tSrS); /* TODO: fuse w/ initial gemm call */
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k, tKgK_cache(_, _, _, D), tKrK);

        reorder(tQrQ, tSrQ);
        // reorder() performs the (vectorized) fp8 -> ElementQ cast; the
        // per-tensor scale_k has been folded into effective_scale above.
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* V prefetch for GEMM 2 */
      prefetch(prefetch_v, pVgV(_, _, _, tile_idx));

      /* Causal masking */
      // No Causal masking in decoding
      // if constexpr (CausalMask) {
      //   if (K == blk_k1 - 1) {
      //     // Need to get global col and row indices to mask the elements
      //     Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
      //     Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}),
      //     make_coord(get<0>(blk_qv), K)); auto cS_thread =
      //     thr_mma_qk.partition_C(gP); CUTLASS_PRAGMA_UNROLL for (int i = 0; i
      //     < tSrS.size(); ++i) {
      //       int row_idx = get<0>(cS_thread(i));
      //       int col_idx = get<1>(cS_thread(i));
      //       if (col_idx - full_tile_offset > row_idx - discard_seq_coord) {
      //         tSrS(i) = ElementS(-INFINITY);
      //       }
      //     }
      //   }
      // }

      /* Local/sliding window masking */
      if constexpr (LocalMask) {
        // For decode, all packed GQA heads share the same KV position
        // (seq_len_kv - 1). Use a fixed decode row for all elements.
        int decode_row = seq_len - 1 - full_tile_offset;
        Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
        Tensor gP = local_tile(
            cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
        auto cS_thread = thr_mma_qk.partition_C(gP);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
          bool left_mask = col_idx < decode_row - params.window_size_left;
          bool right_mask = col_idx > decode_row + params.window_size_right;
          if (left_mask || right_mask) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        int k = get<0>(tKgK(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) =
              (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }

      /* Apply softmax and scaling */
      softmax(effective_scale, K == blk_k0, tSrS, tA_max, tA_sum, tArA);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV_cache(_, _, _, VV), tVrV);
        // reorder() performs the (vectorized) fp8 -> ElementQ cast; the
        // per-tensor scale_v is applied once after the K loop below.
        reorder(tVrV, tArV);
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      barrier();

      // next tile_idx
      next_tile_idx = K + 1;
      if constexpr (PagedKV) {
        int next_page_local_idx =
            next_tile_idx * get<1>(TileShapeQK{}) / params.page_size;
        if (next_page_local_idx < params.max_pages_per_seq) {
          int next_block_idx =
              params.ptr_page_table[b_offset + next_page_local_idx];
          next_tile_idx = next_block_idx * page_stride_tiles +
                          next_tile_idx % tiles_per_page;
        } else {
          // set to last page
          next_tile_idx = params.max_pages_per_seq * tiles_per_page - 1;
        }
      }
      tile_idx = next_tile_idx;

      if constexpr (has_large_surface) {
        update_surface_info(tile_idx);
      }

      /* K prefetch */
      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
      }

      // barrier_wait(ScopeWorkgroup);
    }

    // Apply the per-tensor V scale once after the K loop.
    if constexpr (Fp8KV) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tArA.size(); ++i) {
        tArA(i) *= ElementA(scale_v);
      }
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  void softmax(
      ElementS scale,    // Effective softmax scale (fp8 K scale folded in)
      bool first_block,  // First softmax block?
      FragS& tS,         // Softmax src/dst block
      FragSRow& tS_max,  // Softmax row-wise max accumulator
      FragSRow& tS_sum,  // Softmax row-wise sum accumulator
      FragA& tA) {       // O accumulator (for rescaling)

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    /* Update (scaled) maxima */
    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), scale * tS_bmax(i));
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums and O accumulator */
    if (!first_block) {
      FragSRow rescale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_prev_max(i) - tS_max(i));
        tS_sum(i) *= rescale(i);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA.size(); i++)
        tA(i) *= broadcast<0>(rescale, tA, i);
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
  }
};

template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE constexpr auto get_sg_layout_pv(SGLayoutQK const&) {
  return make_layout(
      get<0>(SGLayoutQK{}), Layout<_1, _0>{}, get<1>(SGLayoutQK{}));
}

// Get a P*V TiledMMA given K*Q tile size and SG configuration, for mainloops
//   not supporting S data interchange among subgroups (e.g. XeDefault).
template <
    typename MMAOp,
    typename WGTileQK,
    typename SGLayoutQK,
    typename TileV>
CUTLASS_HOST_DEVICE constexpr auto get_tiled_mma_pv(
    MMAOp const&,
    WGTileQK const& wg_tile_qk,
    SGLayoutQK const& sg_layout_qk,
    TileV const&) {
  using TileQ = decltype(get<0>(wg_tile_qk));
  using TileK = decltype(get<1>(wg_tile_qk));

  using WGTilePV = Shape<TileQ, TileV, TileK>;
  using SGLayoutPV = decltype(get_sg_layout_pv(sg_layout_qk));

  static_assert(
      size(SGLayoutPV{}) == size(SGLayoutQK{}),
      "Q*K cannot be parallelized in the head size dimension");

  return TiledMMAHelper<MMAOp, WGTilePV, SGLayoutPV>{};
}

}  // namespace vllm_xpu_xe2

}  // namespace cutlass::fmha::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
