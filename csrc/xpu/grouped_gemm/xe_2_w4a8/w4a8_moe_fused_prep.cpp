// Fused W4A8 MoE decode: the routing map, the per-token int8 quant with the
// gather to expert rows, and silu_and_mul with the per-token int8 quant of
// its output (see w4a8_moe_fused.h for the whole layer).
//
// This TU is compiled with -ffp-model=precise -foffload-fp32-prec-div
// (CMakeLists.txt), the flags of csrc/quantization/int8/int8_quant.cpp, and
// both quant kernels run that file's kernel shape: the same work-group size
// rule, the same vector width and per-lane order for the absmax, the same
// group reduction, and the int8q:: math of int8_quant_math.h. So a token row
// gets the bits _C.dynamic_per_token_scaled_int8_quant gives it, NaN rows
// included (whatever the op does with them, this does the same).
#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <cstdint>

#include "csrc/quantization/int8/int8_quant_math.h"
#include "w4a8_moe_fused.h"

namespace w4a8f {

using scalar_t = c10::BFloat16;
namespace int8q = vllm::int8q;

constexpr int kMapWG = 256;
constexpr int kMaxLocal = 512;
constexpr int kMaxRoutes = 2048;
constexpr int kRoutesPerLane = kMaxRoutes / kMapWG;
constexpr int kMaxTopK = 16;
constexpr int kVec = 16 / sizeof(scalar_t);  // the int8 quant op's VEC

// remap_hidden_states.cpp to_local_expert_id, word for word; the id is read
// as int, as RowsPerExpertCount reads it.
inline int to_local(int g, const int32_t* emap, int total, int local) {
  if (g < 0 || g >= total) return -1;
  const int l = emap == nullptr ? g : emap[g];
  return (l >= 0 && l < local) ? l : -1;
}

// launch_int8_quant's work-group size for a row of `cols` elements.
inline int quant_wg(int cols) {
  int wg = 256;
  const int work = cols / kVec;
  while (wg > 32 && wg / 2 >= work) wg /= 2;
  return wg;
}

struct alignas(sizeof(scalar_t) * kVec) in_vec_t {
  scalar_t v[kVec];
};
struct alignas(sizeof(uint16_t) * kVec) bits_vec_t {
  uint16_t v[kVec];
};
struct alignas(kVec) out_vec_t {
  int8_t v[kVec];
};

template <typename IDT>
class W4A8FMapName;
class W4A8FQuantGatherName;
class W4A8FSiluQuantName;

// ---------------------------------------------------------------------------
// 1. map: one work-group of kMapWG lanes. Each lane takes up to
// kRoutesPerLane routes and counts the local ones into SLM (an atomic gives
// the route its slot inside its expert, as RowsPerExpertCount does; the slot
// order inside an expert enters no value). One exclusive scan over the
// lanes' expert ranges gives each expert its first row and first tile; the
// lanes write rows_per_expert and the tile map of their experts, then every
// route its row.
// Tiles: an expert with c rows takes ceil(c / tm) tiles, at most c / tm + 1
// for c > 0, so a routing of R routes needs at most R / tm + min(R, local)
// tiles, and never more than R. The host sizes umt by that bound.
// ---------------------------------------------------------------------------
template <typename IDT>
void launch_map(
    sycl::queue& q,
    const IDT* ids,
    const int32_t* emap,
    int R,
    int total,
    int local,
    int tm_shift,
    int umt,
    int32_t* perm_pos,
    int32_t* rows_per_expert,
    int32_t* ws) {
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> slm(sycl::range<1>(2 * kMaxLocal), cgh);
    cgh.parallel_for<W4A8FMapName<IDT>>(
        sycl::nd_range<1>(sycl::range<1>(kMapWG), sycl::range<1>(kMapWG)),
        [=](sycl::nd_item<1> it) {
          auto grp = it.get_group();
          int32_t* cnt = slm.get_multi_ptr<sycl::access::decorated::no>().get();
          int32_t* offs = cnt + kMaxLocal;
          int32_t* mt_expert = ws + 2;
          int32_t* mt_row_base = mt_expert + umt;
          int32_t* mt_rows = mt_row_base + umt;
          const int lane = it.get_local_linear_id();
          const int tm = 1 << tm_shift;

          for (int e = lane; e < local; e += kMapWG) cnt[e] = 0;
          sycl::group_barrier(grp);

          int my_loc[kRoutesPerLane];
          int my_slot[kRoutesPerLane];
#pragma unroll
          for (int i = 0; i < kRoutesPerLane; ++i) {
            const int r = lane + i * kMapWG;
            my_loc[i] = -1;
            my_slot[i] = 0;
            if (r < R) {
              const int l = to_local(static_cast<int>(ids[r]), emap, total, local);
              my_loc[i] = l;
              if (l >= 0) {
                sycl::atomic_ref<
                    int,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::work_group,
                    sycl::access::address_space::local_space>
                    c(cnt[l]);
                my_slot[i] = c.fetch_add(1);
              }
            }
          }
          sycl::group_barrier(grp);

          const int epl = (local + kMapWG - 1) / kMapWG;
          const int e_lo = sycl::min(lane * epl, local);
          const int e_hi = sycl::min(e_lo + epl, local);
          int rows_sum = 0, tiles_sum = 0;
          for (int e = e_lo; e < e_hi; ++e) {
            rows_sum += cnt[e];
            tiles_sum += (cnt[e] + tm - 1) >> tm_shift;
          }
          int row_off =
              sycl::exclusive_scan_over_group(grp, rows_sum, sycl::plus<int>());
          int tile_off =
              sycl::exclusive_scan_over_group(grp, tiles_sum, sycl::plus<int>());
          if (lane == kMapWG - 1) {
            ws[0] = tile_off + tiles_sum;
            ws[1] = row_off + rows_sum;
          }
          for (int e = e_lo; e < e_hi; ++e) {
            const int c = cnt[e];
            offs[e] = row_off;
            rows_per_expert[e] = c;
            const int nt = (c + tm - 1) >> tm_shift;
            for (int t = 0; t < nt; ++t) {
              const int slot = tile_off + t;
              if (slot < umt) {  // the host bound makes this always true
                const int rem = c - (t << tm_shift);
                mt_expert[slot] = e;
                mt_row_base[slot] = row_off + (t << tm_shift);
                mt_rows[slot] = rem < tm ? rem : tm;
              }
            }
            row_off += c;
            tile_off += nt;
          }
          sycl::group_barrier(grp);

#pragma unroll
          for (int i = 0; i < kRoutesPerLane; ++i) {
            const int r = lane + i * kMapWG;
            if (r < R) {
              perm_pos[r] = my_loc[i] >= 0 ? offs[my_loc[i]] + my_slot[i] : -1;
            }
          }
        });
  });
}

// ---------------------------------------------------------------------------
// 2. quant + gather: one work-group per token, the int8 quant op's kernel
// (dynamic_per_token_int8_quant_kernel, vector path) with its row written to
// every expert row of the token (perm_pos >= 0) and the scale beside it.
// ---------------------------------------------------------------------------
void launch_quant_gather(
    sycl::queue& q,
    const scalar_t* x,
    int64_t x_stride,
    int T,
    int H,
    int topk,
    const int32_t* perm_pos,
    int8_t* a_q,
    float* a_scale) {
  const int wg = quant_wg(H);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<W4A8FQuantGatherName>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(T) * wg),
                          sycl::range<1>(wg)),
        [=](sycl::nd_item<1> item) {
          const int tid = item.get_local_id(0);
          const int nthr = item.get_local_range(0);
          const int64_t row = item.get_group(0);
          const int nvec = H / kVec;
          auto const* vin = reinterpret_cast<in_vec_t const*>(x + row * x_stride);

          float absmax = 0.0f;
          for (int i = tid; i < nvec; i += nthr) {
            in_vec_t xv = vin[i];
#pragma unroll
            for (int j = 0; j < kVec; ++j) {
              absmax = int8q::max_f(
                  absmax, int8q::abs_f(static_cast<float>(xv.v[j])));
            }
          }
          absmax = sycl::reduce_over_group(
              item.get_group(), absmax, sycl::maximum<float>());
          const float s = int8q::token_scale_from_absmax<scalar_t>(absmax);

          int dst[kMaxTopK];
#pragma unroll
          for (int k = 0; k < kMaxTopK; ++k) {
            dst[k] = k < topk ? perm_pos[row * topk + k] : -1;
          }
          if (tid == 0) {
#pragma unroll
            for (int k = 0; k < kMaxTopK; ++k) {
              if (dst[k] >= 0) a_scale[dst[k]] = s;
            }
          }
          for (int i = tid; i < nvec; i += nthr) {
            in_vec_t xv = vin[i];
            out_vec_t qv;
#pragma unroll
            for (int j = 0; j < kVec; ++j) {
              qv.v[j] =
                  int8q::quantize_one<scalar_t>(static_cast<float>(xv.v[j]), s);
            }
#pragma unroll
            for (int k = 0; k < kMaxTopK; ++k) {
              if (dst[k] >= 0) {
                reinterpret_cast<out_vec_t*>(
                    a_q + static_cast<int64_t>(dst[k]) * H)[i] = qv;
              }
            }
          }
        });
  });
}

// ---------------------------------------------------------------------------
// 4. silu_and_mul + quant: one work-group per expert row below ws[1] (the
// rows past it hold no route). The element is silu_and_mul's
// (w4a8f_silu_mul_bits, compiled under activation.cpp's model); the quant is
// the int8 quant op's kernel over the bf16 values, which it would have read
// back from silu_and_mul's output. The second pass recomputes the element
// instead of keeping it, as the op re-reads its row.
// ---------------------------------------------------------------------------
inline float silu_mul_f(uint16_t g, uint16_t u, scalar_t& v) {
  v = scalar_t(w4a8f_silu_mul_bits(g, u), scalar_t::from_bits());
  return static_cast<float>(v);
}

void launch_silu_quant(
    sycl::queue& q,
    const uint16_t* g1,
    int R_cap,
    int I,
    const int32_t* ws,
    int8_t* qo,
    float* qs) {
  const int wg = quant_wg(I);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<W4A8FSiluQuantName>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(R_cap) * wg),
                          sycl::range<1>(wg)),
        [=](sycl::nd_item<1> item) {
          const int64_t row = item.get_group(0);
          if (row >= ws[1]) return;  // whole work-group: no route here
          const int tid = item.get_local_id(0);
          const int nthr = item.get_local_range(0);
          const int nvec = I / kVec;
          auto const* gv = reinterpret_cast<bits_vec_t const*>(g1 + row * 2 * I);
          auto const* uv = gv + nvec;

          float absmax = 0.0f;
          for (int i = tid; i < nvec; i += nthr) {
            bits_vec_t g = gv[i], u = uv[i];
#pragma unroll
            for (int j = 0; j < kVec; ++j) {
              scalar_t v;
              absmax = int8q::max_f(absmax, int8q::abs_f(silu_mul_f(g.v[j], u.v[j], v)));
            }
          }
          absmax = sycl::reduce_over_group(
              item.get_group(), absmax, sycl::maximum<float>());
          const float s = int8q::token_scale_from_absmax<scalar_t>(absmax);
          if (tid == 0) qs[row] = s;

          auto* vout = reinterpret_cast<out_vec_t*>(qo + row * I);
          for (int i = tid; i < nvec; i += nthr) {
            bits_vec_t g = gv[i], u = uv[i];
            out_vec_t qv;
#pragma unroll
            for (int j = 0; j < kVec; ++j) {
              scalar_t v;
              qv.v[j] = int8q::quantize_one<scalar_t>(silu_mul_f(g.v[j], u.v[j], v), s);
            }
            vout[i] = qv;
          }
        });
  });
}

inline int tile_shift(int64_t tile_m) {
  TORCH_CHECK(
      tile_m == 8 || tile_m == 16 || tile_m == 32,
      "w4a8 fused MoE: tile_m must be 8, 16 or 32");
  return tile_m == 8 ? 3 : (tile_m == 16 ? 4 : 5);
}

}  // namespace w4a8f

void moe_w4a8_route_quant_xe2(
    const at::Tensor& x,
    const at::Tensor& topk_ids,
    const std::optional<at::Tensor>& expert_map,
    at::Tensor& a_q,
    at::Tensor& a_scale,
    at::Tensor& perm_pos,
    at::Tensor& rows_per_expert,
    at::Tensor& tile_ws,
    int64_t total_experts,
    int64_t local_experts,
    int64_t tile_m) {
  using namespace w4a8f;
  const char* who = "moe_w4a8_route_quant";
  TORCH_CHECK(x.is_xpu() && x.scalar_type() == at::kBFloat16, who, ": x must be bf16 on XPU");
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, who, ": x must be [T, H], H contiguous");
  const int64_t T = x.size(0), H = x.size(1);
  TORCH_CHECK(
      topk_ids.dim() == 2 && topk_ids.size(0) == T && topk_ids.is_contiguous() &&
          (topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong),
      who, ": topk_ids must be contiguous int32/int64 [T, top_k]");
  const int64_t topk = topk_ids.size(1);
  const int64_t R = T * topk;
  TORCH_CHECK(topk >= 1 && topk <= kMaxTopK, who, ": top_k must be 1..", kMaxTopK);
  TORCH_CHECK(R <= kMaxRoutes, who, ": T * top_k must be <= ", kMaxRoutes);
  TORCH_CHECK(
      local_experts >= 1 && local_experts <= kMaxLocal && total_experts >= 1,
      who, ": 1 <= local_experts <= ", kMaxLocal);
  TORCH_CHECK(
      H % kVec == 0 && x.stride(0) % kVec == 0 &&
          reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
      who, ": H and the row stride must be multiples of ", kVec, ", x 16-byte aligned");
  TORCH_CHECK(
      a_q.scalar_type() == at::kChar && a_q.is_contiguous() && a_q.dim() == 2 &&
          a_q.size(0) == R && a_q.size(1) == H,
      who, ": a_q must be contiguous int8 [T * top_k, H]");
  TORCH_CHECK(
      a_scale.scalar_type() == at::kFloat && a_scale.is_contiguous() &&
          a_scale.numel() == R,
      who, ": a_scale must be contiguous fp32 [T * top_k]");
  TORCH_CHECK(
      perm_pos.scalar_type() == at::kInt && perm_pos.is_contiguous() &&
          perm_pos.numel() == R,
      who, ": perm_pos must be contiguous int32 [T, top_k]");
  TORCH_CHECK(
      rows_per_expert.scalar_type() == at::kInt && rows_per_expert.is_contiguous() &&
          rows_per_expert.numel() == local_experts,
      who, ": rows_per_expert must be contiguous int32 [local_experts]");
  TORCH_CHECK(
      tile_ws.scalar_type() == at::kInt && tile_ws.is_contiguous() &&
          tile_ws.numel() >= 2 && (tile_ws.numel() - 2) % 3 == 0,
      who, ": tile_ws must be contiguous int32 [2 + 3 * umt]");
  const int shift = tile_shift(tile_m);
  const int64_t umt = (tile_ws.numel() - 2) / 3;
  const int64_t need =
      std::min<int64_t>(R, R / tile_m + std::min<int64_t>(R, local_experts));
  TORCH_CHECK(umt >= need, who, ": tile_ws holds ", umt, " tiles, routing may need ", need);
  const int32_t* emap = nullptr;
  if (expert_map.has_value()) {
    TORCH_CHECK(
        expert_map->scalar_type() == at::kInt && expert_map->is_contiguous() &&
            expert_map->numel() == total_experts,
        who, ": expert_map must be contiguous int32 [total_experts]");
    emap = expert_map->data_ptr<int32_t>();
  }
  for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&topk_ids, &a_q, &a_scale, &perm_pos, &rows_per_expert, &tile_ws}) {
    TORCH_CHECK(t->device() == x.device(), who, ": all tensors on x's device");
  }

  const at::DeviceGuard guard(x.device());
  auto& q = c10::xpu::getCurrentXPUStream(x.device().index()).queue();
  int32_t* ws = tile_ws.data_ptr<int32_t>();
  if (topk_ids.scalar_type() == at::kInt) {
    launch_map<int32_t>(
        q, topk_ids.data_ptr<int32_t>(), emap, R, total_experts, local_experts,
        shift, umt, perm_pos.data_ptr<int32_t>(), rows_per_expert.data_ptr<int32_t>(), ws);
  } else {
    launch_map<int64_t>(
        q, topk_ids.data_ptr<int64_t>(), emap, R, total_experts, local_experts,
        shift, umt, perm_pos.data_ptr<int32_t>(), rows_per_expert.data_ptr<int32_t>(), ws);
  }
  if (T == 0) return;
  launch_quant_gather(
      q, reinterpret_cast<const scalar_t*>(x.data_ptr()), x.stride(0), T, H, topk,
      perm_pos.data_ptr<int32_t>(), a_q.data_ptr<int8_t>(), a_scale.data_ptr<float>());
}

void moe_w4a8_silu_mul_quant_xe2(
    const at::Tensor& g1,
    at::Tensor& q_out,
    at::Tensor& q_scale,
    const at::Tensor& tile_ws) {
  using namespace w4a8f;
  const char* who = "moe_w4a8_silu_mul_quant";
  TORCH_CHECK(
      g1.is_xpu() && g1.scalar_type() == at::kBFloat16 && g1.dim() == 2 &&
          g1.is_contiguous() && g1.size(1) % 2 == 0,
      who, ": g1 must be contiguous bf16 [R, 2 * I] on XPU");
  const int64_t R = g1.size(0), I = g1.size(1) / 2;
  TORCH_CHECK(I % kVec == 0, who, ": I must be a multiple of ", kVec);
  TORCH_CHECK(
      q_out.scalar_type() == at::kChar && q_out.is_contiguous() && q_out.dim() == 2 &&
          q_out.size(0) == R && q_out.size(1) == I,
      who, ": q_out must be contiguous int8 [R, I]");
  TORCH_CHECK(
      q_scale.scalar_type() == at::kFloat && q_scale.is_contiguous() &&
          q_scale.numel() == R,
      who, ": q_scale must be contiguous fp32 [R]");
  TORCH_CHECK(
      tile_ws.scalar_type() == at::kInt && tile_ws.numel() >= 2,
      who, ": tile_ws must be the map's int32 workspace");
  for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&q_out, &q_scale, &tile_ws}) {
    TORCH_CHECK(t->device() == g1.device(), who, ": all tensors on g1's device");
  }
  if (R == 0) return;
  const at::DeviceGuard guard(g1.device());
  auto& q = c10::xpu::getCurrentXPUStream(g1.device().index()).queue();
  launch_silu_quant(
      q, reinterpret_cast<const uint16_t*>(g1.data_ptr()), R, I,
      tile_ws.data_ptr<int32_t>(), q_out.data_ptr<int8_t>(), q_scale.data_ptr<float>());
}
