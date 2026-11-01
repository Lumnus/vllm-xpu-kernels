// B70 W4A8 grouped GEMM launchers (Xe2 / BMG), shared by the per-variant
// translation units (w4a8_inst_*.cpp): each TU instantiates one
// (prefetch distance, tile K, activation-scale mode) slice of the kernels so
// the AOT device link compiles them in parallel modules.
#pragma once

#include <torch/all.h>

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"

#include "gemm_w4a8_xe2.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoEW4A8 {
using namespace cute;

using bf16 = cute::bfloat16_t;

// WG tile (M, N, K) and SG layout. K is a multiple of the s8 x s4 atom's 32.
template <int TM, int TN, int TK, int SGM, int SGN>
struct w4a8_policy {
  static constexpr int tile_k = TK;
  using WGTile = Shape<Int<TM>, Int<TN>, Int<TK>>;
  using SGLayout =
      Layout<Shape<Int<SGM>, Int<SGN>, _1>, Stride<Int<SGN>, _1, _0>>;
  using GmemTiledCopyD = void;
};

// Default tiles (IP-L): large M = the W4A16 128 x 256, 4 x 8 SGs.
template <int TK>
using w4a8_policy_big = w4a8_policy<128, 256, TK, 4, 8>;
template <int TK>
using w4a8_policy_m_32 = w4a8_policy<32, 64, TK, 1, 4>;
template <int TK>
using w4a8_policy_m_16 = w4a8_policy<16, 64, TK, 1, 4>;
template <int TK>
using w4a8_policy_m_8 = w4a8_policy<8, 64, TK, 1, 4>;
// Small-M variants (B70_W4A8_SMALL_TILE): 1 = N 128 over 8 SGs (same work
// per SG, half the WGs); 2 = N 128 over 4 SGs (two columns per lane).
template <int TM, int TK>
using w4a8_policy_s1 = w4a8_policy<TM, 128, TK, 1, 8>;
template <int TM, int TK>
using w4a8_policy_s2 = w4a8_policy<TM, 128, TK, 1, 4>;

// Fused gemm1 (silu-and-mul + int8 quant) tiles, N in activation columns.
// 128 activation columns per WG tile (one activation-scale group), 8 SGs.
template <int TK>
using glu_policy_big = w4a8_policy<128, 128, TK, 4, 8>;
template <int TM, int TK>
using glu_policy_small = w4a8_policy<TM, 128, TK, 1, 8>;

// Activation-scale group of gemm2's A = the weight group (128): gemm2 folds
// its int32 partial at the same points as with per-row scales.
static constexpr int kActGroup = 128;

template <typename, typename, class, int, int, int>
class W4A8GemmName;
template <class, int, int>
class W4A8GluName;

struct PlainArgs {
  sycl::queue* q;
  const int8_t* a;
  const float* a_scale;
  const uint8_t* b;
  const bf16* b_scale;
  const bf16* bias;
  bf16* d;
  int N, K, E;
  const int* rows;
  int32_t* atomic;
  int avg_m, t8, t16, t32, small_tile;
};

struct GluArgs {
  sycl::queue* q;
  const int8_t* a;
  const float* a_scale;
  const uint8_t* b;      // w13 [E, 2*inter, K/2]
  const bf16* b_scale;   // [E, 2*inter, K/128]
  const bf16* bias;      // [E, 2*inter] or nullptr
  int8_t* qout;          // [M, inter]
  float* qscale;         // [M, inter / kActGroup]
  int inter, K, E;
  const int* rows;
  int32_t* atomic;
  int avg_m, t8, t16, t32;
};

template <
    class GmemTiledCopyD,
    int GroupSize,
    int AG,
    int PF,
    class TiledMMA,
    class TiledMMAF,
    typename ElementS,
    typename ElementD>
CUTE_DEVICE void MoEGEMMW4A8(
    const int8_t* Activations,
    const float* ActScales,
    const uint8_t* Weights,
    const ElementS* Scales,
    const ElementD* Bias,
    ElementD* Outputs,
    TiledMMA const& mma,
    TiledMMAF const& mma_f,
    const int* rows_per_expert,
    const int32_t num_experts,
    const int32_t gemm_n,
    const int32_t gemm_k,
    int32_t* atomic_buffer,
    const sycl::local_accessor<int32_t, 1>& slm_mem_const,
    const sycl::local_accessor<float, 1>& slm_as_acc) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_tile = mma.tile_mnk();
  auto wg_tile_m = get<0>(wg_tile);
  auto wg_tile_n = get<1>(wg_tile);
  float* slm_as = static_cast<float*>(
      slm_as_acc.template get_multi_ptr<sycl::access::decorated::no>().get());

  int group_id = item.get_group_linear_id();
  int gemm_n_pad = (gemm_n + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
  int group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
  int group_range = item.get_group_range(1);
  int local_id = item.get_local_linear_id();
  const int group_num = gemm_k / GroupSize;
  const int a_scale_pitch = AG > 0 ? gemm_k / AG : 1;

  int pre_rows = 0;
  int pre_tiles = 0;

  int32_t* slm_mem = static_cast<int32_t*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());

  for (int i = 0; i < num_experts; ++i) {
    int gemm_m = rows_per_expert[i];
    int cumsum_rows_for_experts = pre_rows + gemm_m;
    int cumsum_tiles_for_experts =
        (gemm_m + wg_tile_m - 1) / wg_tile_m + pre_tiles;

    if (group_m_id >= cumsum_tiles_for_experts) {
      pre_rows = cumsum_rows_for_experts;
      pre_tiles = cumsum_tiles_for_experts;
      continue;
    }

    const int64_t B_offset = static_cast<int64_t>(i) *
                             static_cast<int64_t>(gemm_n) *
                             static_cast<int64_t>(gemm_k) / 2;
    const int8_t* ptr_A = Activations + static_cast<int64_t>(pre_rows) * gemm_k;
    const float* ptr_AS =
        ActScales + static_cast<int64_t>(pre_rows) * a_scale_pitch;
    const uint8_t* ptr_B = Weights + B_offset;
    const ElementS* ptr_S = Scales + static_cast<int64_t>(i) * gemm_n * group_num;
    const ElementD* ptr_Bias =
        Bias != nullptr ? Bias + static_cast<int64_t>(i) * gemm_n : nullptr;
    ElementD* ptr_D = Outputs + static_cast<int64_t>(pre_rows) * gemm_n;

    auto A_tensor = make_tensor(
        make_gmem_ptr(const_cast<int8_t*>(ptr_A)),
        make_layout(make_shape(gemm_m, gemm_k), make_stride(gemm_k, _1{})));
    auto B_tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<int4_t*>(const_cast<uint8_t*>(ptr_B))),
        make_layout(make_shape(gemm_n, gemm_k), make_stride(gemm_k, _1{})));
    auto D_tensor = make_tensor(
        make_gmem_ptr(ptr_D),
        make_layout(make_shape(gemm_m, gemm_n), make_stride(gemm_n, _1{})));

    while (group_m_id < cumsum_tiles_for_experts) {
      int n_coord = (group_id * wg_tile_n) % gemm_n_pad / wg_tile_n;
      int m_coord = (group_m_id - pre_tiles);
      auto tile_coord = make_coord(m_coord, n_coord, _, 0);

      xe_gemm_w4a8<GroupSize, AG, PF, GmemTiledCopyD>(
          A_tensor,
          B_tensor,
          ptr_S,
          ptr_AS,
          ptr_Bias,
          D_tensor,
          tile_coord,
          mma,
          mma_f,
          slm_as);

      if (local_id == 0) {
        slm_mem[0] = cutlass::atomicAdd(atomic_buffer, 1);
      }
      sycl::group_barrier(item.get_group());
      group_id = group_range + slm_mem[0];
      group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
    }
    pre_rows = cumsum_rows_for_experts;
    pre_tiles = cumsum_tiles_for_experts;
  }
}

template <
    int GroupSize,
    int AG,
    int PF,
    class TiledMMA,
    class TiledMMAF,
    typename ElementS,
    typename ElementBI>
CUTE_DEVICE void MoEGEMMW4A8GLU(
    const int8_t* Activations,
    const float* ActScales,
    const uint8_t* Weights,
    const ElementS* Scales,
    const ElementBI* Bias,
    int8_t* QOut,
    float* QScale,
    TiledMMA const& mma,
    TiledMMAF const& mma_f,
    const int* rows_per_expert,
    const int32_t num_experts,
    const int32_t inter,
    const int32_t gemm_k,
    int32_t* atomic_buffer,
    const sycl::local_accessor<int32_t, 1>& slm_mem_const,
    const sycl::local_accessor<float, 1>& slm_red) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_tile = mma.tile_mnk();
  auto wg_tile_m = get<0>(wg_tile);
  auto wg_tile_n = get<1>(wg_tile);

  int group_id = item.get_group_linear_id();
  int n_pad = (inter + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
  int group_m_id = (group_id * wg_tile_n) / n_pad;
  int group_range = item.get_group_range(1);
  int local_id = item.get_local_linear_id();
  const int group_num = gemm_k / GroupSize;
  const int n13 = 2 * inter;

  int pre_rows = 0;
  int pre_tiles = 0;

  int32_t* slm_mem = static_cast<int32_t*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());
  float* red = static_cast<float*>(
      slm_red.template get_multi_ptr<sycl::access::decorated::no>().get());

  for (int i = 0; i < num_experts; ++i) {
    int gemm_m = rows_per_expert[i];
    int cumsum_rows_for_experts = pre_rows + gemm_m;
    int cumsum_tiles_for_experts =
        (gemm_m + wg_tile_m - 1) / wg_tile_m + pre_tiles;

    if (group_m_id >= cumsum_tiles_for_experts) {
      pre_rows = cumsum_rows_for_experts;
      pre_tiles = cumsum_tiles_for_experts;
      continue;
    }

    const int64_t B_offset = static_cast<int64_t>(i) * n13 *
                             static_cast<int64_t>(gemm_k) / 2;
    const int64_t B_up = static_cast<int64_t>(inter) * gemm_k / 2;
    const int8_t* ptr_A = Activations + static_cast<int64_t>(pre_rows) * gemm_k;
    const float* ptr_AS = ActScales + pre_rows;
    const uint8_t* ptr_Bg = Weights + B_offset;
    const uint8_t* ptr_Bu = ptr_Bg + B_up;
    const ElementS* ptr_Sg = Scales + static_cast<int64_t>(i) * n13 * group_num;
    const ElementS* ptr_Su = ptr_Sg + static_cast<int64_t>(inter) * group_num;
    const ElementBI* ptr_Bias_g =
        Bias != nullptr ? Bias + static_cast<int64_t>(i) * n13 : nullptr;
    const ElementBI* ptr_Bias_u =
        Bias != nullptr ? ptr_Bias_g + inter : nullptr;
    int8_t* ptr_Q = QOut + static_cast<int64_t>(pre_rows) * inter;
    float* ptr_QS = QScale + static_cast<int64_t>(pre_rows) * (inter / AG);

    auto A_tensor = make_tensor(
        make_gmem_ptr(const_cast<int8_t*>(ptr_A)),
        make_layout(make_shape(gemm_m, gemm_k), make_stride(gemm_k, _1{})));
    auto Bg_tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<int4_t*>(const_cast<uint8_t*>(ptr_Bg))),
        make_layout(make_shape(inter, gemm_k), make_stride(gemm_k, _1{})));
    auto Bu_tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<int4_t*>(const_cast<uint8_t*>(ptr_Bu))),
        make_layout(make_shape(inter, gemm_k), make_stride(gemm_k, _1{})));

    while (group_m_id < cumsum_tiles_for_experts) {
      int n_coord = (group_id * wg_tile_n) % n_pad / wg_tile_n;
      int m_coord = (group_m_id - pre_tiles);
      auto tile_coord = make_coord(m_coord, n_coord, _, 0);

      xe_gemm_w4a8_glu<GroupSize, AG, PF>(
          A_tensor,
          Bg_tensor,
          Bu_tensor,
          ptr_Sg,
          ptr_Su,
          ptr_AS,
          ptr_Bias_g,
          ptr_Bias_u,
          ptr_Q,
          ptr_QS,
          tile_coord,
          mma,
          mma_f,
          red);

      if (local_id == 0) {
        slm_mem[0] = cutlass::atomicAdd(atomic_buffer, 1);
      }
      sycl::group_barrier(item.get_group());
      group_id = group_range + slm_mem[0];
      group_m_id = (group_id * wg_tile_n) / n_pad;
    }
    pre_rows = cumsum_rows_for_experts;
    pre_tiles = cumsum_tiles_for_experts;
  }
}

template <class policy, typename ElementD>
struct W4A8MMAs {
  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  // s8 x s4 -> s32 (dpas.s4.s8), and its 16-bit twin for the f32 C layout.
  using MMA = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, int32_t, int8_t, int4_t>>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;
  using MMAF = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, ElementD>>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;
  static_assert(
      is_same_v<decltype(MMA{}.tile_mnk()), decltype(MMAF{}.tile_mnk())>,
      "s8 and 16-bit TiledMMA must have the same WG tile");
  using CS32 = decltype(MMA{}
                            .get_slice(0)
                            .partition_sg_fragment_C(make_identity_tensor(
                                select<0, 1>(WGTile{})))
                            .layout());
  using CF32 = decltype(MMAF{}
                            .get_slice(0)
                            .partition_sg_fragment_C(make_identity_tensor(
                                select<0, 1>(WGTile{})))
                            .layout());
  static_assert(
      is_same_v<CS32, CF32>,
      "s32 and f32 C fragments must share one layout");
};

inline sycl::range<3> w4a8_global(int threads_per_wg) {
  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  static constexpr int MaxThreadsPerSM = 512;
  TORCH_CHECK(
      MaxThreadsPerSM % threads_per_wg == 0,
      "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup");
  return sycl::range<3>(1, sm_count * MaxThreadsPerSM / threads_per_wg, 1);
}

template <class policy, int GroupSize, int AG, int PF>
void launch(const PlainArgs& p) {
  using M2 = W4A8MMAs<policy, bf16>;
  auto mma = typename M2::MMA{};
  auto mma_f = typename M2::MMAF{};
  const int threads = size(mma);
  sycl::range<3> local(1, 1, threads);
  sycl::range<3> global = w4a8_global(threads);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};

  using GmemTiledCopyD = typename policy::GmemTiledCopyD;
  const PlainArgs a = p;

  p.q->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    static constexpr int TM = get<0>(typename policy::WGTile{});
    sycl::local_accessor<float, 1> slm_as(
        sycl::range<1>(AG > 0 ? TM * (a.K / AG) : 1), cgh);
    cgh.parallel_for<W4A8GemmName<bf16, bf16, policy, GroupSize, AG, PF>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoEGEMMW4A8<GmemTiledCopyD, GroupSize, AG, PF>(
              a.a, a.a_scale, a.b, a.b_scale, a.bias, a.d, mma, mma_f,
              a.rows, a.E, a.N, a.K, a.atomic, local_mem, slm_as);
        });
  });
}

template <class policy, int GroupSize, int PF>
void launch_glu(const GluArgs& p) {
  using M2 = W4A8MMAs<policy, bf16>;
  auto mma = typename M2::MMA{};
  auto mma_f = typename M2::MMAF{};
  const int threads = size(mma);
  sycl::range<3> local(1, 1, threads);
  sycl::range<3> global = w4a8_global(threads);
  static constexpr int ATOM_N =
      get<2>(typename M2::MMA::ThrLayoutVMNK{}.shape());
  static constexpr int TM = get<0>(typename policy::WGTile{});

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};
  const GluArgs a = p;

  p.q->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    sycl::local_accessor<float, 1> red(sycl::range<1>(ATOM_N * TM), cgh);
    cgh.parallel_for<W4A8GluName<policy, GroupSize, PF>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoEGEMMW4A8GLU<GroupSize, kActGroup, PF>(
              a.a, a.a_scale, a.b, a.b_scale, a.bias, a.qout, a.qscale, mma,
              mma_f, a.rows, a.E, a.inter, a.K, a.atomic, local_mem, red);
        });
  });
}

// One (PF, TK, AG) slice of the plain kernels; M tile by avg rows per expert.
template <int PF, int TK, int AG>
void dispatch_plain(const PlainArgs& p) {
  constexpr int GS = 128;  // only g128 is built (Intel AutoRound / GPTQ)
  if (p.avg_m <= p.t8) {
    if (p.small_tile == 1)
      launch<w4a8_policy_s1<8, TK>, GS, AG, PF>(p);
    else if (p.small_tile == 2)
      launch<w4a8_policy_s2<8, TK>, GS, AG, PF>(p);
    else
      launch<w4a8_policy_m_8<TK>, GS, AG, PF>(p);
  } else if (p.avg_m <= p.t16) {
    if (p.small_tile == 1)
      launch<w4a8_policy_s1<16, TK>, GS, AG, PF>(p);
    else if (p.small_tile == 2)
      launch<w4a8_policy_s2<16, TK>, GS, AG, PF>(p);
    else
      launch<w4a8_policy_m_16<TK>, GS, AG, PF>(p);
  } else if (p.avg_m <= p.t32) {
    if (p.small_tile == 1)
      launch<w4a8_policy_s1<32, TK>, GS, AG, PF>(p);
    else if (p.small_tile == 2)
      launch<w4a8_policy_s2<32, TK>, GS, AG, PF>(p);
    else
      launch<w4a8_policy_m_32<TK>, GS, AG, PF>(p);
  } else {
    launch<w4a8_policy_big<TK>, GS, AG, PF>(p);
  }
}

template <int PF, int TK>
void dispatch_glu(const GluArgs& p) {
  constexpr int GS = 128;
  if (p.avg_m <= p.t8) {
    launch_glu<glu_policy_small<8, TK>, GS, PF>(p);
  } else if (p.avg_m <= p.t16) {
    launch_glu<glu_policy_small<16, TK>, GS, PF>(p);
  } else if (p.avg_m <= p.t32) {
    launch_glu<glu_policy_small<32, TK>, GS, PF>(p);
  } else {
    launch_glu<glu_policy_big<TK>, GS, PF>(p);
  }
}

#define W4A8_EXTERN_SLICES(EXT)                         \
  EXT template void dispatch_plain<4, 32, 0>(const PlainArgs&);  \
  EXT template void dispatch_plain<4, 64, 0>(const PlainArgs&);  \
  EXT template void dispatch_plain<6, 32, 0>(const PlainArgs&);  \
  EXT template void dispatch_plain<6, 64, 0>(const PlainArgs&);  \
  EXT template void dispatch_plain<4, 32, 64>(const PlainArgs&); \
  EXT template void dispatch_plain<4, 64, 64>(const PlainArgs&); \
  EXT template void dispatch_plain<6, 32, 64>(const PlainArgs&); \
  EXT template void dispatch_plain<6, 64, 64>(const PlainArgs&); \
  EXT template void dispatch_glu<4, 32>(const GluArgs&);         \
  EXT template void dispatch_glu<4, 64>(const GluArgs&);         \
  EXT template void dispatch_glu<6, 32>(const GluArgs&);         \
  EXT template void dispatch_glu<6, 64>(const GluArgs&);

}  // namespace MoEW4A8
