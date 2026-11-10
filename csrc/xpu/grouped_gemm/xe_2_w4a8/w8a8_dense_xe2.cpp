// small-M dense W8A8 GEMM on the Xe2 cutlass mainloop (s8 x s8 -> s32 DPAS).
//
//   Y[m, n] = xs[m] * ws[n] * sum_k Xq[m, k] * W[n, k]  (+ bias[n]),  bf16 out
//
//   Xq  int8 [M, K] row-major, per-token symmetric (int8_dense.cpp's quant kernel, or any caller)
//   W   int8 [N, K] K-contiguous with an fp32 per-row scale (the int8 linear weights, as stored)
//
// Decode M is 1..64: the WG tile spans ALL tokens (TM = M rounded up to 8/16/32/64), so the weights stream
// from memory once and the token rows are what each WG re-reads (L2/L1). Larger M runs 64-row tiles.
// The int32 partial is never flushed inside K (per-row weight scale, per-token act scale): one convert
// and 2 multiplies per value at the end.
//
// Tiles (B70_I8D_TILE): 64x64 (TN x TK), 128x64, 32x128, 64x128. "auto" takes 32x128 for N >= 2048 (the
// shape vLLM's Triton scaled_mm uses on XPU for M <= 64: BLOCK_M = M, BLOCK_N 32, BLOCK_K 128, no split)
// and 64x64 with split-K otherwise.
//
// Split K (narrow N, deep K: HC down K10240 N336, shared gate_up K2560 N320): the WGs of a tile add their
// int32 partials into an int32 workspace with integer atomics, and the last WG of the tile (a counter, no
// spin; the non-spinning variant of sycl-tla's deterministic stream-K fixup, which waits on a lock per
// split) converts once, writes bf16 and re-zeroes the workspace. Integer addition is exact and
// associative, so the result is bit-identical for every split count, arrival order and batch
// composition: a row's output depends only on that row (per-row quant, exact int32 dot, one fp32
// epilogue). That is the determinism the ordered HC down path needs.
#include <torch/all.h>
#include <c10/xpu/XPUStream.h>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"

#include "grouped_gemm_w4a8_xe2.h"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace W8A8Dense {
using namespace cute;
using bf16 = cute::bfloat16_t;

template <int TM, int TN, int TK, int SGM, int SGN>
struct policy {
  using WGTile = Shape<Int<TM>, Int<TN>, Int<TK>>;
  using SGLayout =
      Layout<Shape<Int<SGM>, Int<SGN>, _1>, Stride<Int<SGN>, _1, _0>>;
};

struct Args {
  const int8_t* xq;
  const float* xs;
  const int8_t* w;
  const float* ws;
  const bf16* bias;
  bf16* y;
  int* wsp;     // split-K workspace [M, N] int32 (zero between calls) or nullptr
  int* cnt;     // split-K tile counters (zero between calls) or nullptr
  int M, N, K;
  int n_tiles, ksplit, kt_per;
};

template <class> class W8A8DenseName;

template <int PF, class TiledMMA, class TiledMMAF>
CUTE_DEVICE void w8a8_tile(
    const Args& p,
    int wg_m,
    int wg_n,
    int ks,
    TiledMMA const& mma,
    TiledMMAF const& mma_f,
    int* slm_flag) {
  static constexpr int sg_local_range = 16;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int local_id = item.get_local_linear_id();

  auto A = make_tensor(
      make_gmem_ptr(const_cast<int8_t*>(p.xq)),
      make_layout(make_shape(p.M, p.K), make_stride(p.K, _1{})));
  auto B = make_tensor(
      make_gmem_ptr(const_cast<int8_t*>(p.w)),
      make_layout(make_shape(p.N, p.K), make_stride(p.K, _1{})));
  auto C = make_tensor(
      make_gmem_ptr(p.y),
      make_layout(make_shape(p.M, p.N), make_stride(p.N, _1{})));

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gC = local_tile(cC, wg_tile, make_coord(wg_m, wg_n, 0), Step<_1, _1, X>{});

  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = get_block_2d_copy_D<void>(mma_f, C);

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

  auto tCrC = thr_mma.partition_sg_fragment_C(gC);
  auto tCrF = thr_mma_f.partition_sg_fragment_C(gC);
  auto tCcC = thr_mma_f.partition_C(gC);
  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);
  auto tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);
  auto pAgA = prefetch_a.get_slice(local_id).partition_S(gA);
  auto pBgB = prefetch_b.get_slice(local_id).partition_S(gB);

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
  static constexpr int tile_m = get<0>(wg_tile);
  static constexpr int tile_n = get<1>(wg_tile);
  static constexpr int tile_k = get<2>(wg_tile);

  const int k_tiles = ceil_div(p.K, tile_k);
  const int kt_beg = ks * p.kt_per;
  const int kt_end = sycl::min(k_tiles, kt_beg + p.kt_per);

  clear(tCrC);
  int kp = kt_beg;
  CUTE_UNROLL
  for (int i = 0; i < PF; ++i, ++kp) {
    if (kp < kt_end) {
      prefetch(prefetch_a, pAgA(_, _, _, kp));
      prefetch(prefetch_b, pBgB(_, _, _, kp));
    }
  }
  for (int kt = kt_beg; kt < kt_end; ++kt, ++kp) {
    barrier_arrive(barrier_scope);
    copy(copy_a, tAgA(_, _, _, kt), tArA);
    copy(copy_b, tBgB(_, _, _, kt), tBrB);
    if (kp < kt_end) {
      prefetch(prefetch_a, pAgA(_, _, _, kp));
      prefetch(prefetch_b, pBgB(_, _, _, kp));
    }
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);
    cute::gemm(mma, tCrA, tCrB, tCrC);
    barrier_wait(barrier_scope);
  }

  if (p.ksplit == 1) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrF.size(); ++i) {
      const int m = get<0>(tCcC(i));
      const int n = get<1>(tCcC(i));
      const bool in = m < p.M && n < p.N;
      float v = in ? static_cast<float>(tCrC(i)) * p.xs[m] * p.ws[n] : 0.0f;
      if (in && p.bias != nullptr) v += static_cast<float>(p.bias[n]);
      tCrF(i) = v;
    }
    reorder(tCrF, tCrC_out);
    copy(copy_c, tCrC_out, tCgC);
    return;
  }

  // Split K: exact int32 atomics into the workspace, then the last WG of the tile finishes it.
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < tCrF.size(); ++i) {
    const int m = get<0>(tCcC(i));
    const int n = get<1>(tCcC(i));
    if (m < p.M && n < p.N && kt_beg < kt_end) {
      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          a(p.wsp[static_cast<int64_t>(m) * p.N + n]);
      a.fetch_add(static_cast<int>(tCrC(i)));
    }
  }
  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
  sycl::group_barrier(item.get_group());
  const int tile_id = wg_m * p.n_tiles + wg_n;
  if (local_id == 0) {
    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        c(p.cnt[tile_id]);
    slm_flag[0] = c.fetch_add(1) == p.ksplit - 1;
  }
  sycl::group_barrier(item.get_group());
  if (slm_flag[0] == 0) return;
  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
  const int lsz = item.get_local_range(2);
  for (int idx = local_id; idx < tile_m * tile_n; idx += lsz) {
    const int m = wg_m * tile_m + idx / tile_n;
    const int n = wg_n * tile_n + idx % tile_n;
    if (m < p.M && n < p.N) {
      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          a(p.wsp[static_cast<int64_t>(m) * p.N + n]);
      // Same expression and order as the ksplit == 1 epilogue.
      float v = static_cast<float>(a.exchange(0)) * p.xs[m] * p.ws[n];
      if (p.bias != nullptr) v += static_cast<float>(p.bias[n]);
      p.y[static_cast<int64_t>(m) * p.N + n] = static_cast<bf16>(v);
    }
  }
  if (local_id == 0) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        c(p.cnt[tile_id]);
    c.store(0);
  }
}

template <class pol, int PF>
void launch(sycl::queue& q, const Args& a) {
  using WGTile = typename pol::WGTile;
  using SGLayout = typename pol::SGLayout;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, int32_t, int8_t>>, Layout<WGTile>, SGLayout>::TiledMMA;
  using MMAF = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, bf16>>, Layout<WGTile>, SGLayout>::TiledMMA;
  using CS32 = decltype(MMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(WGTile{}))).layout());
  using CF32 = decltype(MMAF{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(WGTile{}))).layout());
  static_assert(is_same_v<CS32, CF32>, "s32 and f32 C fragments must share one layout");
  auto mma = MMA{};
  auto mma_f = MMAF{};
  static constexpr int TM = get<0>(WGTile{});
  const int threads = size(mma);
  const int m_tiles = (a.M + TM - 1) / TM;
  sycl::range<3> local(1, 1, threads);
  sycl::range<3> global(1, static_cast<size_t>(m_tiles) * a.n_tiles * a.ksplit, 1);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
  const Args p = a;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int, 1> flag(sycl::range<1>(1), cgh);
    cgh.parallel_for<W8A8DenseName<pol>>(
        sycl::nd_range<3>{global * local, local}, props, [=](auto) {
          auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
          int g = item.get_group(1);
          const int ks = g % p.ksplit;
          g /= p.ksplit;
          const int wg_n = g % p.n_tiles;
          const int wg_m = g / p.n_tiles;
          w8a8_tile<PF>(p, wg_m, wg_n, ks, mma, mma_f,
                        flag.template get_multi_ptr<sycl::access::decorated::no>().get());
        });
  });
}

static int env_int(const char* name, int dflt) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? std::atoi(v) : dflt;
}

// Self-cleaning split-K workspace per device: int32 [kWsFloats] then int counters [kWsTiles].
static constexpr int64_t kWsFloats = 64LL * 16384;
static constexpr int64_t kWsTiles = 4096;
static at::Tensor& workspace(const at::Device& dev) {
  static std::mutex mu;
  // Leaked on purpose: never destroyed after the XPU runtime at exit.
  static auto* per_dev = new std::vector<at::Tensor>();
  std::lock_guard<std::mutex> g(mu);
  const int i = dev.index() < 0 ? 0 : dev.index();
  if (static_cast<int>(per_dev->size()) <= i) per_dev->resize(i + 1);
  if (!(*per_dev)[i].defined()) {
    (*per_dev)[i] = at::zeros({kWsFloats + kWsTiles}, at::TensorOptions().dtype(at::kInt).device(dev));
  }
  return (*per_dev)[i];
}

}  // namespace W8A8Dense

void w8a8_dense_xe2(
    const at::Tensor& xq,
    const at::Tensor& xs,
    const at::Tensor& w,
    const at::Tensor& ws,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& y) {
  using namespace W8A8Dense;
  const int M = static_cast<int>(xq.size(0));
  const int K = static_cast<int>(xq.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(K % 32 == 0 && N % 16 == 0 && N >= 32,
              "w8a8_dense: needs K % 32 == 0, N % 16 == 0 and N >= 32");
  if (M == 0) return;
  auto& q = at::xpu::getCurrentXPUStream(xq.device().index()).queue();

  // Tile (TN x TK): B70_I8D_TILE = auto | 64x64 | 128x64 | 32x128 | 64x128.
  int tn = 64, tk = 64;
  const char* tv = std::getenv("B70_I8D_TILE");
  const std::string tile = (tv != nullptr && *tv != '\0') ? std::string(tv) : std::string("auto");
  if (tile == "128x64") {
    tn = 128;
  } else if (tile == "32x128") {
    tn = 32;
    tk = 128;
  } else if (tile == "64x128") {
    tk = 128;
  } else if (tile != "64x64") {  // auto
    if (N >= 2048) {
      tn = 32;
      tk = 128;
    }
  }
  if (env_int("B70_I8D_TN", 0) == 128) tn = 128;  // v2 knob, kept

  Args a{};
  a.xq = reinterpret_cast<const int8_t*>(xq.data_ptr());
  a.xs = xs.data_ptr<float>();
  a.w = reinterpret_cast<const int8_t*>(w.data_ptr());
  a.ws = ws.data_ptr<float>();
  a.bias = bias.has_value() ? reinterpret_cast<const bf16*>(bias->data_ptr()) : nullptr;
  a.y = reinterpret_cast<bf16*>(y.data_ptr());
  a.M = M;
  a.N = N;
  a.K = K;
  a.n_tiles = (N + tn - 1) / tn;
  const int tm = M <= 8 ? 8 : (M <= 16 ? 16 : (M <= 32 ? 32 : 64));
  const int m_tiles = (M + tm - 1) / tm;
  const int k_tiles = (K + tk - 1) / tk;
  // Split K until the card has ~min_wgs WGs, keeping >= 4 k-tiles per split. The output does not
  // depend on the split (exact int32), so this may follow M freely.
  const int min_wgs = env_int("B70_I8D_MIN_WGS", 64);
  int ks = env_int("B70_I8D_KSPLIT", 0);
  if (ks <= 0) {
    ks = 1;
    while (m_tiles * a.n_tiles * ks < min_wgs && k_tiles / (ks * 2) >= 4 && ks < 16) ks *= 2;
  }
  ks = std::max(1, std::min(ks, k_tiles));
  if (static_cast<int64_t>(M) * N > kWsFloats || static_cast<int64_t>(a.n_tiles) * m_tiles > kWsTiles) ks = 1;
  a.kt_per = (k_tiles + ks - 1) / ks;
  a.ksplit = (k_tiles + a.kt_per - 1) / a.kt_per;  // no empty splits
  if (a.ksplit > 1) {
    auto& wsp = workspace(xq.device());
    a.wsp = wsp.data_ptr<int>();
    a.cnt = a.wsp + kWsFloats;
  }

#define W8A8_TM(TM, SGM)                                                         \
  if (tn == 64 && tk == 64)                                                      \
    launch<policy<TM, 64, 64, SGM, 4>, 4>(q, a);                                 \
  else if (tn == 128)                                                            \
    launch<policy<TM, 128, 64, SGM, 8>, 4>(q, a);                                \
  else if (tn == 32)                                                             \
    launch<policy<TM, 32, 128, SGM, 2>, 4>(q, a);                                \
  else                                                                           \
    launch<policy<TM, 64, 128, SGM, 4>, 4>(q, a);
  if (tm == 8) {
    W8A8_TM(8, 1)
  } else if (tm == 16) {
    W8A8_TM(16, 1)
  } else if (tm == 32) {
    W8A8_TM(32, 1)
  } else {
    W8A8_TM(64, 2)
  }
#undef W8A8_TM
}
