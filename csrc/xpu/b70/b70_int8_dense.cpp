// B70 K1: small-M int8 dense GEMMs over the weights B70-IP-W1 already holds
// (int8 [N, K] symmetric per output row, RTN, fp32 scale [N] / [N, 1]).
//
// Decode on the D8 shapes runs M = 8..64 rows, where W1's path (a separate
// scaled_int8_quant kernel, then Triton scaled_mm) costs 1.3-1.5x BF16 on the
// small shapes: fixed cost, not math. Two kernels replace it:
//
// b70_w8a16_gemm (weight-only, M <= ~16): y[m, n] = ws[n] * sum_k x[m,k] w[n,k]
//   bf16 x, int8 w, fp32 FMA. No activation quant pass, half the weight bytes
//   of BF16. A work-group = KS sub-groups over the K slices of NR output rows;
//   each lane streams 16 int8 weights (one 128-bit load) per row and step and
//   reuses each activation vector across the NR rows; M is tiled by MC rows in
//   registers. Reduction: sub-group, then SLM over the KS slices.
//
// b70_w8a8_fused_gemm (M <= 64): the per-token activation quant runs inside
//   the GEMM. It computes out^T = W [N, K] x Xq^T [K, M], so the weight rows
//   are the DPAS A operand (8 rows x 32 int8, row-major, straight from global
//   memory) and the tokens are the B operand (32 x 16 int8, from SLM):
//   joint_matrix s8 x s8 -> s32, which IGC lowers to dpas.8x8 s8 on Xe2.
//   Prologue: every work-group reads its tokens' rows once for the absmax
//   (scale = amax / 127, the same as scaled_int8_quant), then the main loop
//   quantizes each K step of the tokens into SLM ([MB][KSTEP][16] int8) and
//   the S sub-groups each run their KS-wide part of that step on NT 8-row
//   weight tiles x MB 16-token tiles. Accumulation is int32 over the whole K
//   (K <= 131072 cannot overflow), then out = acc * xs[m] * ws[n] (+ bias).
//
// Both read weights exactly once per work-group; both are graph-capturable
// (no host sync, no allocation besides the output).
//
// Knobs (per call): B70_W8A16_NR, B70_W8A16_KS, B70_W8A8F_S, B70_W8A8F_NT
// override the shape heuristics below (0 / unset = heuristic).

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdlib>

#include "csrc/utils.h"
#include "xpu/ops.h"

namespace b70i8 {

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

// ------------------------------------------------------------ W8A8 fused
constexpr int kKS = 64;  // K per sub-group per main-loop step (2 DPAS deep)

template <typename T, int MB, int NT>
void w8a8f_launch(
    sycl::queue& q,
    const T* X,
    const int8_t* W,
    const float* WS,
    const T* Bias,
    T* Y,
    int M,
    int N,
    int K,
    int S) {
  constexpr int MT = MB * 16;  // padded tokens
  constexpr int RW = NT * 8;   // weight rows per WG
  const int KSTEP = S * kKS;
  const int n_wg = (N + RW - 1) / RW;
  sycl::range<1> local(kSG * S);
  sycl::range<1> global(static_cast<size_t>(n_wg) * kSG * S);
  syclex::properties props{syclex::sub_group_size<kSG>, intelex::grf_size<256>};

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> xs(sycl::range<1>(MT), cgh);
    sycl::local_accessor<float, 1> xinv(sycl::range<1>(MT), cgh);
    sycl::local_accessor<int8_t, 1> qbuf(
        sycl::range<1>(static_cast<size_t>(MB) * KSTEP * 16), cgh);
    sycl::local_accessor<int32_t, 1> cbuf(
        sycl::range<1>(static_cast<size_t>(S) * RW * MT), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(global, local),
        props,
        [=](sycl::nd_item<1> it) {
          auto sg = it.get_sub_group();
          auto grp = it.get_group();
          const int s = sg.get_group_linear_id();
          const int lane = sg.get_local_linear_id();
          const int lid = static_cast<int>(it.get_local_id(0));
          const int lsz = kSG * S;
          const int n_base = static_cast<int>(it.get_group(0)) * RW;
          using XV = sycl::vec<T, 8>;

          // 1. Per-token absmax over the whole K (scale = amax / 127).
          for (int m = s; m < MT; m += S) {
            float mx = 0.f;
            if (m < M) {
              const T* xr = X + static_cast<int64_t>(m) * K;
              for (int k = lane * 8; k < K; k += kSG * 8) {
                const XV v = *reinterpret_cast<const XV*>(xr + k);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                  mx = sycl::fmax(mx, sycl::fabs(static_cast<float>(v[j])));
              }
            }
            mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
            if (lane == 0) {
              xs[m] = mx * (1.0f / 127.0f);
              xinv[m] = mx > 0.f ? 127.0f / mx : 0.f;
            }
          }
          sycl::group_barrier(grp);

          jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8,
                           16>
              acc[NT][MB];
#pragma unroll
          for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int b = 0; b < MB; ++b) jm::joint_matrix_fill(sg, acc[t][b], 0);

          int8_t* qb = qbuf.template get_multi_ptr<sycl::access::decorated::no>()
                           .get();
          for (int k0 = 0; k0 < K; k0 += KSTEP) {
            // 2. Quantize tokens x [0, MT) x [k0, k0 + KSTEP) into SLM,
            //    B layout: qb[(mb * KSTEP + kk) * 16 + (m % 16)].
            const int items = MT * (KSTEP / 8);
            for (int i = lid; i < items; i += lsz) {
              const int m = i / (KSTEP / 8);
              const int kk = (i % (KSTEP / 8)) * 8;
              const int k = k0 + kk;
              int8_t qv[8];
              if (m < M && k < K) {
                const XV v = *reinterpret_cast<const XV*>(
                    X + static_cast<int64_t>(m) * K + k);
                const float inv = xinv[m];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                  float f = sycl::rint(static_cast<float>(v[j]) * inv);
                  f = sycl::fmin(127.f, sycl::fmax(-127.f, f));
                  qv[j] = static_cast<int8_t>(f);
                }
              } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) qv[j] = 0;
              }
              int8_t* dst = qb + (static_cast<int>(m / 16) * KSTEP + kk) * 16 +
                            (m % 16);
#pragma unroll
              for (int j = 0; j < 8; ++j) dst[j * 16] = qv[j];
            }
            sycl::group_barrier(grp);

            // 3. This SG's KS-wide part of the step.
            const int ks0 = k0 + s * kKS;
#pragma unroll
            for (int u = 0; u < kKS; u += 32) {
              const int k = ks0 + u;
              if (k < K) {
                jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, 16,
                                 jm::layout::row_major>
                    bm[MB];
#pragma unroll
                for (int b = 0; b < MB; ++b) {
                  jm::joint_matrix_load(
                      sg, bm[b],
                      qbuf.template get_multi_ptr<sycl::access::decorated::no>() +
                          (b * KSTEP + (k - k0)) * 16,
                      16);
                }
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                  const int n = n_base + t * 8;
                  if (n < N) {
                    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32,
                                     jm::layout::row_major>
                        am;
                    jm::joint_matrix_load(
                        sg, am,
                        sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(const_cast<int8_t*>(
                            W + static_cast<int64_t>(n) * K + k)),
                        K);
#pragma unroll
                    for (int b = 0; b < MB; ++b)
                      jm::joint_matrix_mad(sg, acc[t][b], am, bm[b], acc[t][b]);
                  }
                }
              }
            }
            sycl::group_barrier(grp);
          }

          // 4. Partial sums per SG to SLM [S][RW][MT], reduce, rescale, store.
#pragma unroll
          for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int b = 0; b < MB; ++b)
              jm::joint_matrix_store(
                  sg, acc[t][b],
                  cbuf.template get_multi_ptr<sycl::access::decorated::no>() +
                      (s * RW + t * 8) * MT + b * 16,
                  MT, jm::layout::row_major);
          sycl::group_barrier(grp);
          for (int i = lid; i < RW * MT; i += lsz) {
            const int m = i / RW;  // r fastest: adjacent n per lane
            const int r = i % RW;
            const int n = n_base + r;
            if (m < M && n < N) {
              int32_t sum = 0;
              for (int t = 0; t < S; ++t) sum += cbuf[(t * RW + r) * MT + m];
              float v = static_cast<float>(sum) * xs[m] * WS[n];
              if (Bias != nullptr) v += static_cast<float>(Bias[n]);
              Y[static_cast<int64_t>(m) * N + n] = static_cast<T>(v);
            }
          }
        });
  });
}

template <typename T>
void w8a8f_dispatch(
    sycl::queue& q,
    const T* X,
    const int8_t* W,
    const float* WS,
    const T* B,
    T* Y,
    int M,
    int N,
    int K) {
  int S = env_int("B70_W8A8F_S", 0);
  if (S != 1 && S != 2 && S != 4 && S != 8)
    S = pow2_floor(std::max(1, std::min(8, K / 256)));
  int nt = env_int("B70_W8A8F_NT", 0);
  if (nt != 1 && nt != 2) nt = N >= 2048 ? 2 : 1;
  const int mb = M <= 16 ? 1 : (M <= 32 ? 2 : 4);
#define W8A8F_NT(NT)                                                         \
  if (nt == NT) {                                                            \
    if (mb == 1) return w8a8f_launch<T, 1, NT>(q, X, W, WS, B, Y, M, N, K, S); \
    if (mb == 2) return w8a8f_launch<T, 2, NT>(q, X, W, WS, B, Y, M, N, K, S); \
    return w8a8f_launch<T, 4, NT>(q, X, W, WS, B, Y, M, N, K, S);            \
  }
  W8A8F_NT(1)
  W8A8F_NT(2)
#undef W8A8F_NT
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

}  // namespace b70i8

torch::Tensor b70_w8a16_gemm(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& w_scale,
    const std::optional<torch::Tensor>& bias) {
  b70i8::check_args(x, w, w_scale, bias, "b70_w8a16_gemm");
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(K % 16 == 0, "b70_w8a16_gemm: K must be a multiple of 16");
  auto y = torch::empty({M, N}, x.options());
  if (M == 0 || N == 0) return y;
  auto& q = vllm::xpu::vllmGetQueue(x.device().index());
  const float* ws = w_scale.data_ptr<float>();
  const int8_t* wq = reinterpret_cast<const int8_t*>(w.data_ptr());
  if (x.scalar_type() == at::kBFloat16) {
    using T = vllm::xpu::SyclTypeTrait<c10::BFloat16>::Type;
    b70i8::w8a16_dispatch<T>(
        q, reinterpret_cast<const T*>(x.data_ptr()), wq, ws,
        bias.has_value() ? reinterpret_cast<const T*>(bias->data_ptr())
                         : nullptr,
        reinterpret_cast<T*>(y.data_ptr()), M, N, K);
  } else {
    using T = vllm::xpu::SyclTypeTrait<c10::Half>::Type;
    b70i8::w8a16_dispatch<T>(
        q, reinterpret_cast<const T*>(x.data_ptr()), wq, ws,
        bias.has_value() ? reinterpret_cast<const T*>(bias->data_ptr())
                         : nullptr,
        reinterpret_cast<T*>(y.data_ptr()), M, N, K);
  }
  return y;
}

torch::Tensor b70_w8a8_fused_gemm(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& w_scale,
    const std::optional<torch::Tensor>& bias) {
  b70i8::check_args(x, w, w_scale, bias, "b70_w8a8_fused_gemm");
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(M <= 64, "b70_w8a8_fused_gemm: M must be <= 64");
  TORCH_CHECK(
      K % 32 == 0 && N % 8 == 0,
      "b70_w8a8_fused_gemm: needs K % 32 == 0 and N % 8 == 0");
  auto y = torch::empty({M, N}, x.options());
  if (M == 0 || N == 0) return y;
  auto& q = vllm::xpu::vllmGetQueue(x.device().index());
  const float* ws = w_scale.data_ptr<float>();
  const int8_t* wq = reinterpret_cast<const int8_t*>(w.data_ptr());
  if (x.scalar_type() == at::kBFloat16) {
    using T = vllm::xpu::SyclTypeTrait<c10::BFloat16>::Type;
    b70i8::w8a8f_dispatch<T>(
        q, reinterpret_cast<const T*>(x.data_ptr()), wq, ws,
        bias.has_value() ? reinterpret_cast<const T*>(bias->data_ptr())
                         : nullptr,
        reinterpret_cast<T*>(y.data_ptr()), M, N, K);
  } else {
    using T = vllm::xpu::SyclTypeTrait<c10::Half>::Type;
    b70i8::w8a8f_dispatch<T>(
        q, reinterpret_cast<const T*>(x.data_ptr()), wq, ws,
        bias.has_value() ? reinterpret_cast<const T*>(bias->data_ptr())
                         : nullptr,
        reinterpret_cast<T*>(y.data_ptr()), M, N, K);
  }
  return y;
}
