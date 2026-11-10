// B70 W4A8 grouped GEMM (Xe2 / BMG): int8 per-token activations x the int4
// expert weights XpuFusedMoe holds, s8 x s4 -> s32 DPAS, per-group weight
// scale and per-row activation scale in fp32, bf16/fp16 out. Same grouping
// contract as cutlass_grouped_gemm_xe2 (rows of A sorted by expert,
// rows_per_expert[E], persistent work-groups with an atomic tile counter).
#include <torch/all.h>
#include <c10/xpu/XPUStream.h>
#include <cstdlib>
#include <cstring>

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"

#include "gemm_w4a8_xe2.hpp"
#include "grouped_gemm_w4a8_xe2.h"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoEW4A8 {
using namespace cute;

// WG tile (M, N, K) and SG layout. K is a multiple of the s8 x s4 atom's 32.
template <int TM, int TN, int TK, int SGM, int SGN>
struct w4a8_policy {
  static constexpr int tile_k = TK;
  using WGTile = Shape<Int<TM>, Int<TN>, Int<TK>>;
  using SGLayout =
      Layout<Shape<Int<SGM>, Int<SGN>, _1>, Stride<Int<SGN>, _1, _0>>;
  using GmemTiledCopyD = void;
};

// Large-M tile: the W4A16 w4a16_policy shape (128 x 256, 4 x 8 SGs).
template <int TK>
using w4a8_policy_big = w4a8_policy<128, 256, TK, 4, 8>;
template <int TK>
using w4a8_policy_m_32 = w4a8_policy<32, 64, TK, 1, 4>;
template <int TK>
using w4a8_policy_m_16 = w4a8_policy<16, 64, TK, 1, 4>;
template <int TK>
using w4a8_policy_m_8 = w4a8_policy<8, 64, TK, 1, 4>;

template <typename, typename, class, int>
class W4A8GemmName;

template <
    class GmemTiledCopyD,
    int GroupSize,
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
    const sycl::local_accessor<int32_t, 1>& slm_mem_const) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_tile = mma.tile_mnk();
  auto wg_tile_m = get<0>(wg_tile);
  auto wg_tile_n = get<1>(wg_tile);

  int group_id = item.get_group_linear_id();
  int gemm_n_pad = (gemm_n + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
  int group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
  int group_range = item.get_group_range(1);
  int local_id = item.get_local_linear_id();
  const int group_num = gemm_k / GroupSize;

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
    const float* ptr_AS = ActScales + pre_rows;
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

      xe_gemm_w4a8<GroupSize, GmemTiledCopyD>(
          A_tensor,
          B_tensor,
          ptr_S,
          ptr_AS,
          ptr_Bias,
          D_tensor,
          tile_coord,
          mma,
          mma_f);

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

template <class policy, int GroupSize, typename ElementS, typename ElementD>
void launch(
    sycl::queue& stream,
    const int8_t* activations,
    const float* act_scales,
    const uint8_t* weights,
    const ElementS* scales,
    const ElementD* bias,
    ElementD* outputs,
    int gemm_n,
    int gemm_k,
    const int* rows_per_expert,
    int num_experts,
    int32_t* atomic_buffer) {
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

  auto mma = MMA{};
  auto mma_f = MMAF{};

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);
  static constexpr int MaxThreadsPerSM = 512;
  TORCH_CHECK(
      MaxThreadsPerSM % MaxThreadsPerWorkgroup == 0,
      "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup");

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(
      1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};

  using GmemTiledCopyD = typename policy::GmemTiledCopyD;

  stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<W4A8GemmName<ElementS, ElementD, policy, GroupSize>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoEGEMMW4A8<GmemTiledCopyD, GroupSize>(
              activations,
              act_scales,
              weights,
              scales,
              bias,
              outputs,
              mma,
              mma_f,
              rows_per_expert,
              num_experts,
              gemm_n,
              gemm_k,
              atomic_buffer,
              local_mem);
        });
  });
}

static int env_int(const char* name, int dflt) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? std::atoi(v) : dflt;
}

template <typename ElementS, typename ElementD>
void dispatch(
    sycl::queue& q,
    at::Tensor& A,
    at::Tensor& A_scale,
    at::Tensor& B,
    at::Tensor& B_scale,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& D,
    at::Tensor& rows_per_expert,
    int N,
    int K,
    int E,
    int group_size,
    int avg_m,
    at::Tensor& atomic_buffer) {
  // Tile K: 64 by default (two s8 x s4 atoms deep, the A bytes per k-tile of
  // the bf16 path); B70_W4A8_TILE_K=32 for one atom.
  const int tk = env_int("B70_W4A8_TILE_K", 64) == 32 ? 32 : 64;
  // M-tile selection mirrors W4A16 (avg rows per expert); the thresholds can
  // be moved with B70_W4A8_M8 / _M16 / _M32 (upper bounds, inclusive).
  const int t8 = env_int("B70_W4A8_M8", 4);
  const int t16 = env_int("B70_W4A8_M16", 8);
  const int t32 = env_int("B70_W4A8_M32", 128);

  auto* pa = reinterpret_cast<const int8_t*>(A.data_ptr());
  auto* pas = reinterpret_cast<const float*>(A_scale.data_ptr());
  auto* pb = reinterpret_cast<const uint8_t*>(B.data_ptr());
  auto* ps = reinterpret_cast<const ElementS*>(B_scale.data_ptr());
  auto* pbi = bias.has_value()
                  ? reinterpret_cast<const ElementD*>(bias->data_ptr())
                  : static_cast<const ElementD*>(nullptr);
  auto* pd = reinterpret_cast<ElementD*>(D.data_ptr());
  auto* prow = reinterpret_cast<const int*>(rows_per_expert.data_ptr());
  auto* patom = static_cast<int32_t*>(atomic_buffer.data_ptr());

#define W4A8_LAUNCH(POLICY, GS)                                             \
  launch<POLICY, GS, ElementS, ElementD>(                                   \
      q, pa, pas, pb, ps, pbi, pd, N, K, prow, E, patom)

  // Only g128 is instantiated (Intel AutoRound / GPTQ g128); add more group
  // sizes here and in the host check below when a checkpoint needs them.
#define W4A8_BY_GS(POLICY) W4A8_LAUNCH(POLICY, 128);

#define W4A8_BY_M(TK)                                                       \
  if (avg_m <= t8) {                                                        \
    W4A8_BY_GS(w4a8_policy_m_8<TK>)                                         \
  } else if (avg_m <= t16) {                                                \
    W4A8_BY_GS(w4a8_policy_m_16<TK>)                                        \
  } else if (avg_m <= t32) {                                                \
    W4A8_BY_GS(w4a8_policy_m_32<TK>)                                        \
  } else {                                                                  \
    W4A8_BY_GS(w4a8_policy_big<TK>)                                         \
  }

  if (tk == 32) {
    W4A8_BY_M(32)
  } else {
    W4A8_BY_M(64)
  }
#undef W4A8_BY_M
#undef W4A8_BY_GS
#undef W4A8_LAUNCH
}

}  // namespace MoEW4A8

at::Tensor cutlass_grouped_gemm_w4a8_xe2(
    at::Tensor ptr_A,
    at::Tensor ptr_A_scale,
    at::Tensor ptr_B,
    at::Tensor ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor ptr_D,
    at::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts) {
  TORCH_CHECK(ptr_A.scalar_type() == at::kChar, "w4a8: A must be int8");
  TORCH_CHECK(
      ptr_A.dim() == 2 && ptr_A.size(1) == K && ptr_A.is_contiguous(),
      "w4a8: A must be contiguous [M, K]");
  const int64_t M = ptr_A.size(0);
  TORCH_CHECK(
      ptr_A_scale.scalar_type() == at::kFloat && ptr_A_scale.is_contiguous() &&
          ptr_A_scale.numel() == M,
      "w4a8: A_scale must be contiguous float32 with one entry per row of A");
  TORCH_CHECK(
      ptr_B.scalar_type() == at::kChar || ptr_B.scalar_type() == at::kByte,
      "w4a8: B must be packed int4 bytes (int8/uint8)");
  TORCH_CHECK(
      ptr_B.dim() == 3 && ptr_B.size(0) == num_experts && ptr_B.size(1) == N &&
          ptr_B.size(2) * 2 == K && ptr_B.is_contiguous(),
      "w4a8: B must be contiguous [E, N, K/2]");
  TORCH_CHECK(
      ptr_B_scale.dim() == 3 && ptr_B_scale.size(0) == num_experts &&
          ptr_B_scale.size(1) == N && ptr_B_scale.is_contiguous() &&
          K % ptr_B_scale.size(2) == 0,
      "w4a8: B_scale must be contiguous [E, N, K/group_size]");
  const int group_size = static_cast<int>(K / ptr_B_scale.size(2));
  TORCH_CHECK(group_size == 128, "w4a8: only group_size 128 is built");
  TORCH_CHECK(K % 64 == 0, "w4a8: K must be a multiple of 64");
  TORCH_CHECK(N % 64 == 0, "w4a8: N must be a multiple of 64");
  TORCH_CHECK(
      ptr_D.dim() == 2 && ptr_D.size(0) == M && ptr_D.size(1) == N &&
          ptr_D.is_contiguous(),
      "w4a8: D must be contiguous [M, N]");
  TORCH_CHECK(
      ptr_D.scalar_type() == at::kBFloat16, "w4a8: D must be bf16 (built)");
  TORCH_CHECK(
      ptr_B_scale.scalar_type() == ptr_D.scalar_type(),
      "w4a8: B_scale dtype must match D (bf16/fp16)");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(
        ptr_bias->dim() == 2 && ptr_bias->size(0) == num_experts &&
            ptr_bias->size(1) == N && ptr_bias->is_contiguous() &&
            ptr_bias->scalar_type() == ptr_D.scalar_type(),
        "w4a8: bias must be contiguous [E, N] in the D dtype");
  }
  TORCH_CHECK(
      rows_per_expert.scalar_type() == at::kInt &&
          rows_per_expert.numel() == num_experts,
      "w4a8: rows_per_expert must be int32 [E]");
  if (M == 0) return ptr_D;

  auto& q = at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  at::Tensor atomic_buffer =
      at::zeros({1}, ptr_A.options().dtype(at::kInt));
  const int avg_m = static_cast<int>(M / num_experts);

  MoEW4A8::dispatch<cute::bfloat16_t, cute::bfloat16_t>(
      q, ptr_A, ptr_A_scale, ptr_B, ptr_B_scale, ptr_bias, ptr_D,
      rows_per_expert, static_cast<int>(N), static_cast<int>(K),
      static_cast<int>(num_experts), group_size, avg_m, atomic_buffer);
  return ptr_D;
}
