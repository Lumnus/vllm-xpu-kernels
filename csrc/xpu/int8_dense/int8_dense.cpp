// small-M int8 dense GEMMs over the weights the int8 linear path already holds
// (int8 [N, K] symmetric per output row, RTN, fp32 scale [N] / [N, 1]).
//
// Decode on the D8 shapes runs M = 8..64 rows, where W1's path (a separate
// scaled_int8_quant kernel, then Triton scaled_mm) costs 1.3-1.5x BF16 on the
// small shapes: fixed cost, not math. Two kernels replace it:
//
// int8_dense_w8a16_gemm (weight-only, M <= ~16): y[m, n] = ws[n] * sum_k x[m,k] w[n,k]
//   bf16 x, int8 w, fp32 FMA. No activation quant pass, half the weight bytes
//   of BF16. A work-group = KS sub-groups over the K slices of NR output rows;
//   each lane streams 16 int8 weights (one 128-bit load) per row and step and
//   reuses each activation vector across the NR rows; M is tiled by MC rows in
//   registers. Reduction: sub-group, then SLM over the KS slices.
//
// int8_dense_w8a8_fused_gemm (any M; tuned for M <= 64): a per-token quant kernel (the scaled_int8_quant
//   rule), then the cutlass s8 x s8 streaming GEMM in
//   xe_2_w4a8/w8a8_dense_xe2.cpp, whose WG tile spans all tokens so the weights
//   stream from memory once; narrow N splits K over WGs. int8_dense_w8a8_gemm is the
//   GEMM alone, for already-quantized activations.
//
// Both are graph-capturable (no host sync; outputs and the int8 staging
// buffer are the only allocations).
//
// Knobs (per call): B70_W8A16_NR, B70_W8A16_KS override the w8a16 heuristic;
// B70_I8D_TN (64|128), B70_I8D_KSPLIT, B70_I8D_MIN_WGS the W8A8 GEMM's.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdlib>

#include "csrc/utils.h"
#include "xpu/ops.h"
#if defined(VLLM_XPU_ENABLE_XE2) && defined(VLLM_MOE_ENABLED)
  #include "xpu/grouped_gemm/xe_2_w4a8/grouped_gemm_w4a8_xe2.h"
#endif

namespace int8_dense {

constexpr int kSG = 16;
namespace jm = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

static int env_int(const char* name, int dflt) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? std::atoi(v) : dflt;
}

static int pow2_floor(int v) {
  int p = 1;
  while (p * 2 <= v) p *= 2;
  return p;
}

// ------------------------------------------------------------------ W8A16
template <typename T, int NR, int MC>
void w8a16_launch(
    sycl::queue& q,
    const T* X,
    const int8_t* W,
    const float* WS,
    const T* Bias,
    T* Y,
    int M,
    int N,
    int K,
    int KS) {
  const int n_wg = (N + NR - 1) / NR;
  const int chunks = K / 16;  // 16 int8 weights per lane and step
  sycl::range<1> local(kSG * KS);
  sycl::range<1> global(static_cast<size_t>(n_wg) * kSG * KS);
  syclex::properties props{syclex::sub_group_size<kSG>, intelex::grf_size<256>};

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> red(sycl::range<1>(KS * NR * MC), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(global, local),
        props,
        [=](sycl::nd_item<1> it) {
          auto sg = it.get_sub_group();
          const int s = sg.get_group_linear_id();
          const int lane = sg.get_local_linear_id();
          const int n0 = static_cast<int>(it.get_group(0)) * NR;
          // This SG's slice of the 16-byte chunks of K.
          const int c_beg = static_cast<int>(
              (static_cast<int64_t>(chunks) * s) / KS);
          const int c_end = static_cast<int>(
              (static_cast<int64_t>(chunks) * (s + 1)) / KS);
          using WV = sycl::vec<int8_t, 16>;
          using XV = sycl::vec<T, 8>;

          for (int m0 = 0; m0 < M; m0 += MC) {
            const int mc = sycl::min(MC, M - m0);
            float acc[NR][MC];
#pragma unroll
            for (int r = 0; r < NR; ++r)
#pragma unroll
              for (int m = 0; m < MC; ++m) acc[r][m] = 0.f;

            for (int c = c_beg + lane; c < c_end; c += kSG) {
              const int k = c * 16;
              // Activations stay packed (MC x 16 x 16-bit) and convert per
              // element: keeps the live set ~MC*8 + NR*MC + NR*4 GRFs.
              XV xv[MC][2];
#pragma unroll
              for (int m = 0; m < MC; ++m) {
                if (m < mc) {
                  const XV* xp = reinterpret_cast<const XV*>(
                      X + static_cast<int64_t>(m0 + m) * K + k);
                  xv[m][0] = xp[0];
                  xv[m][1] = xp[1];
                } else {
                  xv[m][0] = XV(static_cast<T>(0.f));
                  xv[m][1] = XV(static_cast<T>(0.f));
                }
              }
              WV wv[NR];
#pragma unroll
              for (int r = 0; r < NR; ++r) {
                const int n = n0 + r;
                wv[r] = n < N ? *reinterpret_cast<const WV*>(
                                    W + static_cast<int64_t>(n) * K + k)
                              : WV(static_cast<int8_t>(0));
              }
#pragma unroll
              for (int j = 0; j < 16; ++j) {
                float wf[NR];
#pragma unroll
                for (int r = 0; r < NR; ++r) wf[r] = static_cast<float>(wv[r][j]);
#pragma unroll
                for (int m = 0; m < MC; ++m) {
                  const float xf = static_cast<float>(xv[m][j >> 3][j & 7]);
#pragma unroll
                  for (int r = 0; r < NR; ++r) acc[r][m] += wf[r] * xf;
                }
              }
            }

            // Sub-group reduce, then across the KS slices through SLM.
#pragma unroll
            for (int r = 0; r < NR; ++r)
#pragma unroll
              for (int m = 0; m < MC; ++m) {
                const float v =
                    sycl::reduce_over_group(sg, acc[r][m], sycl::plus<float>());
                if (lane == 0) red[(s * NR + r) * MC + m] = v;
              }
            sycl::group_barrier(it.get_group());
            const int lid = static_cast<int>(it.get_local_id(0));
            for (int idx = lid; idx < NR * MC; idx += kSG * KS) {
              const int m = idx / NR;  // r fastest: adjacent n per lane
              const int r = idx % NR;
              const int n = n0 + r;
              if (m < mc && n < N) {
                float v = 0.f;
                for (int t = 0; t < KS; ++t) v += red[(t * NR + r) * MC + m];
                v *= WS[n];
                if (Bias != nullptr) v += static_cast<float>(Bias[n]);
                Y[static_cast<int64_t>(m0 + m) * N + n] = static_cast<T>(v);
              }
            }
            sycl::group_barrier(it.get_group());
          }
        });
  });
}

template <typename T>
void w8a16_dispatch(
    sycl::queue& q,
    const T* X,
    const int8_t* W,
    const float* WS,
    const T* B,
    T* Y,
    int M,
    int N,
    int K) {
  int nr = env_int("B70_W8A16_NR", 0);
  if (nr != 1 && nr != 2 && nr != 4) nr = N >= 1024 ? 4 : (N >= 256 ? 2 : 1);
  int ks = env_int("B70_W8A16_KS", 0);
  if (ks != 1 && ks != 2 && ks != 4 && ks != 8) {
    // Enough sub-groups to fill the card (~1024+), at least one chunk/lane.
    const int n_wg = (N + nr - 1) / nr;
    ks = pow2_floor(std::max(1, std::min(8, (1024 + n_wg - 1) / n_wg)));
    while (ks > 1 && (K / 16) / ks < kSG) ks /= 2;
  }
  const int mc = M <= 1 ? 1 : (M <= 2 ? 2 : (M <= 4 ? 4 : 8));
#define W8A16_NR(NR)                                                        \
  if (nr == NR) {                                                           \
    if (mc == 1) return w8a16_launch<T, NR, 1>(q, X, W, WS, B, Y, M, N, K, ks); \
    if (mc == 2) return w8a16_launch<T, NR, 2>(q, X, W, WS, B, Y, M, N, K, ks); \
    if (mc == 4) return w8a16_launch<T, NR, 4>(q, X, W, WS, B, Y, M, N, K, ks); \
    return w8a16_launch<T, NR, 8>(q, X, W, WS, B, Y, M, N, K, ks);          \
  }
  W8A16_NR(1)
  W8A16_NR(2)
  W8A16_NR(4)
#undef W8A16_NR
}

// ------------------------------------------------------- W8A8 (v2)
// Per-token symmetric int8 quant (the scaled_int8_quant rule: scale = amax / 127,
// q = rint(x * 127 / amax)); one work-group per token row. The GEMM is the
// cutlass s8 x s8 streaming kernel in xe_2_w4a8/w8a8_dense_xe2.cpp.
template <typename T>
void quant_rows_launch(
    sycl::queue& q, const T* X, int8_t* XQ, float* XS, int M, int K) {
  constexpr int WG = 256;
  sycl::range<1> local(WG);
  sycl::range<1> global(static_cast<size_t>(M) * WG);
  syclex::properties props{syclex::sub_group_size<kSG>};
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(global, local), props, [=](sycl::nd_item<1> it) {
          using XV = sycl::vec<T, 8>;
          const int m = static_cast<int>(it.get_group(0));
          const int lid = static_cast<int>(it.get_local_id(0));
          const T* xr = X + static_cast<int64_t>(m) * K;
          float mx = 0.f;
          for (int k = lid * 8; k < K; k += WG * 8) {
            const XV v = *reinterpret_cast<const XV*>(xr + k);
#pragma unroll
            for (int j = 0; j < 8; ++j)
              mx = sycl::fmax(mx, sycl::fabs(static_cast<float>(v[j])));
          }
          mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
          const float inv = mx > 0.f ? 127.0f / mx : 0.f;
          if (lid == 0) XS[m] = mx * (1.0f / 127.0f);
          int8_t* qr = XQ + static_cast<int64_t>(m) * K;
          for (int k = lid * 8; k < K; k += WG * 8) {
            const XV v = *reinterpret_cast<const XV*>(xr + k);
            sycl::vec<int8_t, 8> o;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              float f = sycl::rint(static_cast<float>(v[j]) * inv);
              f = sycl::fmin(127.f, sycl::fmax(-127.f, f));
              o[j] = static_cast<int8_t>(f);
            }
            *reinterpret_cast<sycl::vec<int8_t, 8>*>(qr + k) = o;
          }
        });
  });
}

static void check_args(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& w_scale,
    const std::optional<torch::Tensor>& bias,
    const char* who) {
  TORCH_CHECK(
      x.dim() == 2 && x.is_contiguous() &&
          (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf),
      who, ": x must be contiguous bf16/fp16 [M, K]");
  TORCH_CHECK(
      w.dim() == 2 && w.is_contiguous() && w.scalar_type() == at::kChar &&
          w.size(1) == x.size(1),
      who, ": w must be contiguous int8 [N, K]");
  TORCH_CHECK(
      w_scale.scalar_type() == at::kFloat && w_scale.is_contiguous() &&
          w_scale.numel() == w.size(0),
      who, ": w_scale must be contiguous float32 [N] or [N, 1]");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->is_contiguous() && bias->numel() == w.size(0) &&
            bias->scalar_type() == x.scalar_type(),
        who, ": bias must be contiguous [N] in the dtype of x");
  }
  TORCH_CHECK(
      x.device() == w.device() && x.device() == w_scale.device(),
      who, ": tensors must be on one device");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
      who, ": x and w must be 16-byte aligned (vector loads)");
}

}  // namespace int8_dense

torch::Tensor int8_dense_w8a16_gemm(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& w_scale,
    const std::optional<torch::Tensor>& bias) {
  int8_dense::check_args(x, w, w_scale, bias, "int8_dense_w8a16_gemm");
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(K % 16 == 0, "int8_dense_w8a16_gemm: K must be a multiple of 16");
  auto y = torch::empty({M, N}, x.options());
  if (M == 0 || N == 0) return y;
  auto& q = vllm::xpu::vllmGetQueue(x.device().index());
  const float* ws = w_scale.data_ptr<float>();
  const int8_t* wq = reinterpret_cast<const int8_t*>(w.data_ptr());
  if (x.scalar_type() == at::kBFloat16) {
    using T = vllm::xpu::SyclTypeTrait<c10::BFloat16>::Type;
    int8_dense::w8a16_dispatch<T>(
        q, reinterpret_cast<const T*>(x.data_ptr()), wq, ws,
        bias.has_value() ? reinterpret_cast<const T*>(bias->data_ptr())
                         : nullptr,
        reinterpret_cast<T*>(y.data_ptr()), M, N, K);
  } else {
    using T = vllm::xpu::SyclTypeTrait<c10::Half>::Type;
    int8_dense::w8a16_dispatch<T>(
        q, reinterpret_cast<const T*>(x.data_ptr()), wq, ws,
        bias.has_value() ? reinterpret_cast<const T*>(bias->data_ptr())
                         : nullptr,
        reinterpret_cast<T*>(y.data_ptr()), M, N, K);
  }
  return y;
}

torch::Tensor int8_dense_w8a8_gemm(
    const torch::Tensor& xq,
    const torch::Tensor& xs,
    const torch::Tensor& w,
    const torch::Tensor& w_scale,
    const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(
      xq.dim() == 2 && xq.is_contiguous() && xq.scalar_type() == at::kChar,
      "int8_dense_w8a8_gemm: xq must be contiguous int8 [M, K]");
  TORCH_CHECK(
      xs.scalar_type() == at::kFloat && xs.is_contiguous() &&
          xs.numel() == xq.size(0),
      "int8_dense_w8a8_gemm: xs must be contiguous float32 [M] / [M, 1]");
  TORCH_CHECK(
      w.dim() == 2 && w.is_contiguous() && w.scalar_type() == at::kChar &&
          w.size(1) == xq.size(1),
      "int8_dense_w8a8_gemm: w must be contiguous int8 [N, K]");
  TORCH_CHECK(
      w_scale.scalar_type() == at::kFloat && w_scale.is_contiguous() &&
          w_scale.numel() == w.size(0),
      "int8_dense_w8a8_gemm: w_scale must be contiguous float32 [N] / [N, 1]");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->is_contiguous() && bias->numel() == w.size(0) &&
            bias->scalar_type() == at::kBFloat16,
        "int8_dense_w8a8_gemm: bias must be contiguous bf16 [N]");
  }
  auto y = torch::empty(
      {xq.size(0), w.size(0)}, xq.options().dtype(at::kBFloat16));
#if defined(VLLM_XPU_ENABLE_XE2) && defined(VLLM_MOE_ENABLED)
  w8a8_dense_xe2(xq, xs, w, w_scale, bias, y);
#else
  TORCH_CHECK(false, "int8_dense_w8a8_gemm needs the Xe2 MoE kernel build");
#endif
  return y;
}

torch::Tensor int8_dense_w8a8_fused_gemm(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& w_scale,
    const std::optional<torch::Tensor>& bias) {
  int8_dense::check_args(x, w, w_scale, bias, "int8_dense_w8a8_fused_gemm");
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(
      x.scalar_type() == at::kBFloat16, "int8_dense_w8a8_fused_gemm: x must be bf16");
  TORCH_CHECK(
      K % 32 == 0 && N % 16 == 0 && N >= 32,
      "int8_dense_w8a8_fused_gemm: needs K % 32 == 0, N % 16 == 0 and N >= 32");
  auto xq = torch::empty({M, K}, x.options().dtype(at::kChar));
  auto xs = torch::empty({M}, x.options().dtype(at::kFloat));
  if (M == 0) return torch::empty({0, N}, x.options());
  auto& q = vllm::xpu::vllmGetQueue(x.device().index());
  using T = vllm::xpu::SyclTypeTrait<c10::BFloat16>::Type;
  int8_dense::quant_rows_launch<T>(
      q, reinterpret_cast<const T*>(x.data_ptr()),
      reinterpret_cast<int8_t*>(xq.data_ptr()), xs.data_ptr<float>(), M, K);
  return int8_dense_w8a8_gemm(xq, xs, w, w_scale, bias);
}
