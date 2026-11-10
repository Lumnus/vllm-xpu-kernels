// W4A8 grouped GEMM (Xe2 / BMG): int8 activations x the int4 expert
// weights XpuFusedMoe holds, s8 x s4 -> s32 DPAS, per-group weight scale and
// per-row (or per row-and-128-column-group) activation scale in fp32, bf16 out.
// Same grouping contract as cutlass_grouped_gemm_xe2 (rows of A sorted by
// expert, rows_per_expert[E], persistent work-groups with an atomic tile
// counter). The kernels live in w4a8_inst_*.cpp (one slice per TU).
//
// Tuning (fixed): tile K 64, prefetch 4, 64-wide small-M tiles (1 x 4 SGs),
// average-rows-per-expert bounds 4 / 8 / 128 for the 8 / 16 / 32-row tiles.
// The other instantiated variants (tile K 32, prefetch 6, 128-wide small
// tiles) stay compiled and are not selected.
#include <torch/all.h>
#include <c10/xpu/XPUStream.h>

#include "w4a8_launch.hpp"
#include "grouped_gemm_w4a8_xe2.h"

namespace MoEW4A8 {
W4A8_EXTERN_SLICES(extern)

static constexpr int kTileK = 64;
static constexpr int kPrefetch = 4;
static constexpr int kRowsM8 = 4;
static constexpr int kRowsM16 = 8;
static constexpr int kRowsM32 = 128;
static constexpr int kSmallTile = 0;

static int tile_k() { return kTileK; }
static int prefetch() { return kPrefetch; }

static void run_plain(PlainArgs& p, bool ag) {
  p.t8 = kRowsM8;
  p.t16 = kRowsM16;
  p.t32 = kRowsM32;
  p.small_tile = kSmallTile;
  const int tk = tile_k(), pf = prefetch();
#define W4A8_PLAIN(PF, TK)                                         \
  if (pf == PF && tk == TK) {                                      \
    if (ag)                                                        \
      dispatch_plain<PF, TK, kActGroup>(p);                        \
    else                                                           \
      dispatch_plain<PF, TK, 0>(p);                                \
    return;                                                        \
  }
  W4A8_PLAIN(4, 64)
  W4A8_PLAIN(4, 32)
  W4A8_PLAIN(6, 64)
  W4A8_PLAIN(6, 32)
#undef W4A8_PLAIN
}

static void run_glu(GluArgs& p) {
  p.t8 = kRowsM8;
  p.t16 = kRowsM16;
  p.t32 = kRowsM32;
  const int tk = tile_k(), pf = prefetch();
  if (pf == 4 && tk == 64) return dispatch_glu<4, 64>(p);
  if (pf == 4 && tk == 32) return dispatch_glu<4, 32>(p);
  if (pf == 6 && tk == 64) return dispatch_glu<6, 64>(p);
  dispatch_glu<6, 32>(p);
}

}  // namespace MoEW4A8

static void check_b(
    const at::Tensor& ptr_B,
    const at::Tensor& ptr_B_scale,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    const char* who) {
  TORCH_CHECK(
      ptr_B.scalar_type() == at::kChar || ptr_B.scalar_type() == at::kByte,
      who, ": B must be packed int4 bytes (int8/uint8)");
  TORCH_CHECK(
      ptr_B.dim() == 3 && ptr_B.size(0) == num_experts && ptr_B.size(1) == N &&
          ptr_B.size(2) * 2 == K && ptr_B.is_contiguous(),
      who, ": B must be contiguous [E, N, K/2]");
  TORCH_CHECK(
      ptr_B_scale.dim() == 3 && ptr_B_scale.size(0) == num_experts &&
          ptr_B_scale.size(1) == N && ptr_B_scale.is_contiguous() &&
          K % ptr_B_scale.size(2) == 0,
      who, ": B_scale must be contiguous [E, N, K/group_size]");
  TORCH_CHECK(
      K / ptr_B_scale.size(2) == 128, who, ": only group_size 128 is built");
  TORCH_CHECK(
      ptr_B_scale.scalar_type() == at::kBFloat16, who, ": B_scale must be bf16");
  TORCH_CHECK(K % 64 == 0, who, ": K must be a multiple of 64");
}

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
  // A_scale: [M] / [M, 1] (one per row), or [M, K/128] (one per row and
  // 128-column group, as the fused gemm1 writes it).
  TORCH_CHECK(
      ptr_A_scale.scalar_type() == at::kFloat && ptr_A_scale.is_contiguous(),
      "w4a8: A_scale must be contiguous float32");
  const bool ag = ptr_A_scale.dim() == 2 && K / MoEW4A8::kActGroup > 1 &&
                  ptr_A_scale.size(1) == K / MoEW4A8::kActGroup;
  TORCH_CHECK(
      ag ? ptr_A_scale.size(0) == M : ptr_A_scale.numel() == M,
      "w4a8: A_scale must be [M], [M, 1] or [M, K/128]");
  check_b(ptr_B, ptr_B_scale, N, K, num_experts, "w4a8");
  TORCH_CHECK(N % 64 == 0, "w4a8: N must be a multiple of 64");
  TORCH_CHECK(
      ptr_D.dim() == 2 && ptr_D.size(0) == M && ptr_D.size(1) == N &&
          ptr_D.is_contiguous(),
      "w4a8: D must be contiguous [M, N]");
  TORCH_CHECK(
      ptr_D.scalar_type() == at::kBFloat16, "w4a8: D must be bf16 (built)");
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
  at::Tensor atomic_buffer = at::zeros({1}, ptr_A.options().dtype(at::kInt));

  MoEW4A8::PlainArgs p{};
  p.q = &q;
  p.a = reinterpret_cast<const int8_t*>(ptr_A.data_ptr());
  p.a_scale = reinterpret_cast<const float*>(ptr_A_scale.data_ptr());
  p.b = reinterpret_cast<const uint8_t*>(ptr_B.data_ptr());
  p.b_scale = reinterpret_cast<const MoEW4A8::bf16*>(ptr_B_scale.data_ptr());
  p.bias = ptr_bias.has_value()
               ? reinterpret_cast<const MoEW4A8::bf16*>(ptr_bias->data_ptr())
               : nullptr;
  p.d = reinterpret_cast<MoEW4A8::bf16*>(ptr_D.data_ptr());
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);
  p.E = static_cast<int>(num_experts);
  p.rows = reinterpret_cast<const int*>(rows_per_expert.data_ptr());
  p.atomic = static_cast<int32_t*>(atomic_buffer.data_ptr());
  p.avg_m = static_cast<int>(M / num_experts);
  MoEW4A8::run_plain(p, ag);
  return ptr_D;
}

void cutlass_grouped_gemm_w4a8_silu_quant_xe2(
    at::Tensor ptr_A,
    at::Tensor ptr_A_scale,
    at::Tensor ptr_B,
    at::Tensor ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor q_out,
    at::Tensor q_scale,
    at::Tensor rows_per_expert,
    int64_t inter,
    int64_t K,
    int64_t num_experts) {
  constexpr int AG = MoEW4A8::kActGroup;
  TORCH_CHECK(ptr_A.scalar_type() == at::kChar, "w4a8 glu: A must be int8");
  TORCH_CHECK(
      ptr_A.dim() == 2 && ptr_A.size(1) == K && ptr_A.is_contiguous(),
      "w4a8 glu: A must be contiguous [M, K]");
  const int64_t M = ptr_A.size(0);
  TORCH_CHECK(
      ptr_A_scale.scalar_type() == at::kFloat && ptr_A_scale.is_contiguous() &&
          ptr_A_scale.numel() == M,
      "w4a8 glu: A_scale must be contiguous float32, one per row");
  check_b(ptr_B, ptr_B_scale, 2 * inter, K, num_experts, "w4a8 glu");
  TORCH_CHECK(inter % AG == 0, "w4a8 glu: inter must be a multiple of 128");
  TORCH_CHECK(
      q_out.scalar_type() == at::kChar && q_out.dim() == 2 &&
          q_out.size(0) == M && q_out.size(1) == inter &&
          q_out.is_contiguous(),
      "w4a8 glu: q_out must be contiguous int8 [M, inter]");
  TORCH_CHECK(
      q_scale.scalar_type() == at::kFloat && q_scale.dim() == 2 &&
          q_scale.size(0) == M && q_scale.size(1) == inter / AG &&
          q_scale.is_contiguous(),
      "w4a8 glu: q_scale must be contiguous float32 [M, inter/128]");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(
        ptr_bias->dim() == 2 && ptr_bias->size(0) == num_experts &&
            ptr_bias->size(1) == 2 * inter && ptr_bias->is_contiguous() &&
            ptr_bias->scalar_type() == at::kBFloat16,
        "w4a8 glu: bias must be contiguous bf16 [E, 2*inter]");
  }
  TORCH_CHECK(
      rows_per_expert.scalar_type() == at::kInt &&
          rows_per_expert.numel() == num_experts,
      "w4a8 glu: rows_per_expert must be int32 [E]");
  if (M == 0) return;

  auto& q = at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  at::Tensor atomic_buffer = at::zeros({1}, ptr_A.options().dtype(at::kInt));

  MoEW4A8::GluArgs p{};
  p.q = &q;
  p.a = reinterpret_cast<const int8_t*>(ptr_A.data_ptr());
  p.a_scale = reinterpret_cast<const float*>(ptr_A_scale.data_ptr());
  p.b = reinterpret_cast<const uint8_t*>(ptr_B.data_ptr());
  p.b_scale = reinterpret_cast<const MoEW4A8::bf16*>(ptr_B_scale.data_ptr());
  p.bias = ptr_bias.has_value()
               ? reinterpret_cast<const MoEW4A8::bf16*>(ptr_bias->data_ptr())
               : nullptr;
  p.qout = reinterpret_cast<int8_t*>(q_out.data_ptr());
  p.qscale = reinterpret_cast<float*>(q_scale.data_ptr());
  p.inter = static_cast<int>(inter);
  p.K = static_cast<int>(K);
  p.E = static_cast<int>(num_experts);
  p.rows = reinterpret_cast<const int*>(rows_per_expert.data_ptr());
  p.atomic = static_cast<int32_t*>(atomic_buffer.data_ptr());
  p.avg_m = static_cast<int>(M / num_experts);
  MoEW4A8::run_glu(p);
}
