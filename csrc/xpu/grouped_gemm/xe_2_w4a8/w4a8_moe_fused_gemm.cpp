// Fused W4A8 MoE decode: the grouped GEMM over the map's tiles.
//
// One work-group per (tile, N tile) of a grid sized for the most tiles the
// routing can need (tile_ws's umt); work-groups past the device-side tile
// count tile_ws[0] leave at once. Each tile is one expert's rows
// [row0, row0 + rows) with rows <= tile_m, so its A, A-scale and D views
// start at row0 and the WG works at M coordinate 0. The mainloop is
// xe_gemm_w4a8 unchanged, with the configuration the unfused dispatch runs
// (tile K 64, prefetch 4, one activation scale per row) and its 8, 16 or
// 32-row policy: the same k order and the same epilogue per element.
// This replaces the persistent work-groups, their walk over rows_per_expert
// and the atomic tile counter (and its zero fill) of MoEGEMMW4A8, as kernels
// #670 does for its fp8 decode GEMMs.
#include <torch/all.h>
#include <c10/xpu/XPUStream.h>

#include "w4a8_launch.hpp"
#include "w4a8_moe_fused.h"

namespace MoEW4A8 {

template <class>
class W4A8TileMapGemmName;

struct TileMapArgs {
  const int8_t* a;
  const float* a_scale;
  const uint8_t* b;
  const bf16* b_scale;
  const bf16* bias;
  bf16* d;
  const int32_t* ws;
  int umt, N, K;
};

template <class policy>
void launch_tile_map(sycl::queue& q, const TileMapArgs& p) {
  constexpr int GS = 128;
  constexpr int TK = policy::tile_k;
  static_assert(TK == 64, "the unfused dispatch runs tile K 64");
  using M2 = W4A8MMAs<policy, bf16>;
  auto mma = typename M2::MMA{};
  auto mma_f = typename M2::MMAF{};
  const int threads = size(mma);
  constexpr int TN = get<1>(typename policy::WGTile{});
  const int n_tiles = (p.N + TN - 1) / TN;
  const size_t wgs = static_cast<size_t>(p.umt) * n_tiles;
  sycl::range<3> local(1, 1, threads);
  sycl::range<3> global(1, wgs, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};
  const TileMapArgs a = p;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<W4A8TileMapGemmName<policy>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
          const int tid = item.get_group_linear_id();
          const int m_tile = tid / n_tiles;
          const int n_tile = tid % n_tiles;
          if (m_tile >= a.ws[0]) return;  // whole work-group: surplus tile
          const int32_t* mt_expert = a.ws + 2;
          const int e = mt_expert[m_tile];
          const int row0 = mt_expert[a.umt + m_tile];
          const int rows = mt_expert[2 * a.umt + m_tile];
          const int64_t group_num = a.K / GS;

          const int8_t* ptr_A = a.a + static_cast<int64_t>(row0) * a.K;
          const uint8_t* ptr_B =
              a.b + static_cast<int64_t>(e) * a.N * a.K / 2;
          bf16* ptr_D = a.d + static_cast<int64_t>(row0) * a.N;
          auto A_tensor = make_tensor(
              make_gmem_ptr(const_cast<int8_t*>(ptr_A)),
              make_layout(make_shape(rows, a.K), make_stride(a.K, _1{})));
          auto B_tensor = make_tensor(
              make_gmem_ptr(
                  reinterpret_cast<int4_t*>(const_cast<uint8_t*>(ptr_B))),
              make_layout(make_shape(a.N, a.K), make_stride(a.K, _1{})));
          auto D_tensor = make_tensor(
              make_gmem_ptr(ptr_D),
              make_layout(make_shape(rows, a.N), make_stride(a.N, _1{})));

          xe_gemm_w4a8<GS, 0, 4, typename policy::GmemTiledCopyD>(
              A_tensor,
              B_tensor,
              a.b_scale + static_cast<int64_t>(e) * a.N * group_num,
              a.a_scale + row0,
              a.bias != nullptr ? a.bias + static_cast<int64_t>(e) * a.N
                                : static_cast<const bf16*>(nullptr),
              D_tensor,
              make_coord(0, n_tile, _, 0),
              mma,
              mma_f);
        });
  });
}

}  // namespace MoEW4A8

void moe_w4a8_tile_gemm_xe2(
    const at::Tensor& a,
    const at::Tensor& a_scale,
    const at::Tensor& b,
    const at::Tensor& b_scale,
    const std::optional<at::Tensor>& bias,
    at::Tensor& d,
    const at::Tensor& tile_ws,
    int64_t N,
    int64_t K,
    int64_t tile_m) {
  using namespace MoEW4A8;
  const char* who = "moe_w4a8_tile_gemm";
  TORCH_CHECK(
      a.is_xpu() && a.scalar_type() == at::kChar && a.dim() == 2 &&
          a.size(1) == K && a.is_contiguous(),
      who, ": A must be contiguous int8 [R, K] on XPU");
  const int64_t R = a.size(0);
  TORCH_CHECK(
      a_scale.scalar_type() == at::kFloat && a_scale.is_contiguous() &&
          a_scale.numel() == R,
      who, ": A_scale must be contiguous fp32 [R] or [R, 1]");
  TORCH_CHECK(
      b.dim() == 3 && b.size(1) == N && b.size(2) * 2 == K && b.is_contiguous() &&
          (b.scalar_type() == at::kChar || b.scalar_type() == at::kByte),
      who, ": B must be contiguous packed int4 [E, N, K/2]");
  const int64_t E = b.size(0);
  TORCH_CHECK(
      b_scale.scalar_type() == at::kBFloat16 && b_scale.is_contiguous() &&
          b_scale.dim() == 3 && b_scale.size(0) == E && b_scale.size(1) == N &&
          b_scale.size(2) * 128 == K,
      who, ": B_scale must be contiguous bf16 [E, N, K/128] (g128 only)");
  TORCH_CHECK(K % 128 == 0 && N % 64 == 0, who, ": K % 128 and N % 64 must be 0");
  TORCH_CHECK(
      d.scalar_type() == at::kBFloat16 && d.is_contiguous() && d.dim() == 2 &&
          d.size(0) == R && d.size(1) == N,
      who, ": D must be contiguous bf16 [R, N]");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->scalar_type() == at::kBFloat16 && bias->is_contiguous() &&
            bias->dim() == 2 && bias->size(0) == E && bias->size(1) == N,
        who, ": bias must be contiguous bf16 [E, N]");
  }
  TORCH_CHECK(
      tile_ws.scalar_type() == at::kInt && tile_ws.is_contiguous() &&
          tile_ws.numel() >= 2 && (tile_ws.numel() - 2) % 3 == 0,
      who, ": tile_ws must be the map's int32 [2 + 3 * umt]");
  for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&a_scale, &b, &b_scale, &d, &tile_ws}) {
    TORCH_CHECK(t->device() == a.device(), who, ": all tensors on A's device");
  }
  const int umt = static_cast<int>((tile_ws.numel() - 2) / 3);
  if (R == 0 || umt == 0) return;

  auto& q = c10::xpu::getCurrentXPUStream(a.device().index()).queue();
  TileMapArgs p{};
  p.a = a.data_ptr<int8_t>();
  p.a_scale = a_scale.data_ptr<float>();
  p.b = reinterpret_cast<const uint8_t*>(b.data_ptr());
  p.b_scale = reinterpret_cast<const bf16*>(b_scale.data_ptr());
  p.bias = bias.has_value() ? reinterpret_cast<const bf16*>(bias->data_ptr())
                            : nullptr;
  p.d = reinterpret_cast<bf16*>(d.data_ptr());
  p.ws = tile_ws.data_ptr<int32_t>();
  p.umt = umt;
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);
  if (tile_m == 8) {
    launch_tile_map<w4a8_policy_m_8<64>>(q, p);
  } else if (tile_m == 16) {
    launch_tile_map<w4a8_policy_m_16<64>>(q, p);
  } else {
    TORCH_CHECK(tile_m == 32, who, ": tile_m must be 8, 16 or 32");
    launch_tile_map<w4a8_policy_m_32<64>>(q, p);
  }
}
