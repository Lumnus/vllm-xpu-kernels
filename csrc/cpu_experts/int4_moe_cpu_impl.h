// Int4 MoE experts computed on the host CPU, in the layout the XPU kernels
// read (the expert store's unit): no torch and no device dependency, so the
// arithmetic can be tested on its own and the op can be built into any
// host library.
//
// One expert, in kernel layout (see implement_zp and moe_int4_gemv.cpp):
//   w13  [2I, H/2]  int8, two signed 4-bit values per byte, element 2j in the
//                   low nibble (two's complement)
//   w2   [H, I/2]   same
//   s13  [2I, H/G]  bf16, symmetric per-group scale
//   s2   [H, I/G]   bf16
//
// What a call computes is the card's W4A16 decode path, at the card's
// rounding points:
//   h   = bf16(x @ dequant(w13)^T)               fp32 accumulation
//   a   = bf16(bf16(silu(h[:, :I])) * h[:, I:])  silu in fp32, as
//                                                csrc/activation.cpp
//   y   = bf16(a @ dequant(w2)^T)                fp32 accumulation
//   out[t] = bf16(sum over the token's calls, in call order, of
//                 float(y) * weight)              as moe_gather
//
// Inside a dot product the scale is applied to the partial sum of each
// 32-element chunk (as the card's int4 GEMV does); a chunk's partial sum is
// formed in 8 lanes of 4 terms each, with fused multiply-adds, and the lanes
// are added in a fixed tree. The summation order inside a chunk and across
// chunks differs from the card's (16 work-items, a sub-group reduction, and
// a compiler allowed to contract and reassociate), so a result can differ
// from the card's by the rounding of fp32 sums: at bf16 outputs, a fraction
// of a bf16 step at the output's scale (measured by the self-test).
//
// Structure (what can be replaced without touching the rest):
//   1. rows(): the one inner function, "these output rows of one expert's
//      product for these tokens", dispatched by instruction set. Every
//      instruction set takes the scalar code's steps in its order, so all
//      agree bit for bit (the self-test and the op's start check prove it).
//      Built: scalar (std::fma, any host) and AVX2 + FMA (x86-64). The
//      dispatch is the place for further paths (AVX-512 holding two rows or
//      two tokens of 8 lanes per register, NEON/SVE with two 4-lane halves),
//      each obliged to the same lane order.
//   2. Pool: persistent worker threads, one fork/join per phase, optionally
//      pinned to a core list (set_cores), spinning a bounded time between
//      phases and sleeping between calls.
//   3. run_calls(): one layer call = four phases over ALL of the call's
//      experts at once (convert the inputs; GEMM1 with the activation fused,
//      split by blocks of gate/up row pairs; GEMM2 split by blocks of output
//      rows; the gather split by columns), so every thread streams a share of
//      every expert's weights whatever the number of tokens per expert.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #define INT4_MOE_CPU_X86 1
#else
  #define INT4_MOE_CPU_X86 0
#endif

#if defined(__linux__)
  #include <pthread.h>
  #include <sched.h>
#endif

#if defined(__clang__)
  // icpx and clang: no contraction to FMA, no reassociation in this file,
  // whatever the target's default floating-point model is.
  #pragma clang fp contract(off)
  #pragma clang fp reassociate(off)
#elif defined(__GNUC__)
  #pragma GCC optimize("fp-contract=off")
#endif

namespace cpu_moe {

// --------------------------------------------------------------------------
// bf16
// --------------------------------------------------------------------------

inline float bf16_to_f32(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Round to nearest even, as the device's conversion does; NaN stays NaN.
inline uint16_t f32_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

inline float round_bf16(float f) { return bf16_to_f32(f32_to_bf16(f)); }

// --------------------------------------------------------------------------
// The inner function: output rows [n0, n1) of a W4A16 product, for nt tokens.
//
// W  [N, K/2] bytes (row n at W + n * K/2), S [N, K/G] bf16.
// xe, xo: per token, the even and the odd input elements in fp32
//   (xe[t * K/2 + j] = x[t][2j], xo[...] = x[t][2j + 1]).
// out[t * ldo + n] = the fp32 dot product of row n with token t.
//
// The steps, per row and token: for each 32-element chunk c (16 bytes,
// j0 = 16c), lane l (0..7) forms
//   d_l = lo(j0+l) * xe[j0+l]                       (a product)
//   d_l = fma(hi(j0+l),   xo[j0+l],   d_l)
//   d_l = fma(lo(j0+8+l), xe[j0+8+l], d_l)
//   d_l = fma(hi(j0+8+l), xo[j0+8+l], d_l)
//   acc_l = fma(d_l, scale(c), acc_l)
// and the result is lane_sum(acc). An instruction set may vectorise across
// lanes, rows or tokens, never across this order.
// --------------------------------------------------------------------------

constexpr int kChunk = 32;  // elements per scaled partial sum (16 bytes)
constexpr int kTB = 4;      // tokens per pass over a row (scalar code)

// Fixed order for the 8 lanes: ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 +
// l7))
inline float lane_sum(const float* l) {
  const float a = l[0] + l[4];
  const float b = l[2] + l[6];
  const float c = l[1] + l[5];
  const float d = l[3] + l[7];
  return (a + b) + (c + d);
}

inline int nibble(uint8_t byte, int high) {
  const int n = high ? (byte >> 4) & 0xF : byte & 0xF;
  return n >= 8 ? n - 16 : n;
}

inline void rows_scalar(
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const float* xe,
    const float* xo,
    int nt,
    float* out,
    int ldo) {
  const int kb = K / 2;
  const int ng = K / G;
  for (int n = n0; n < n1; ++n) {
    const uint8_t* w = W + static_cast<int64_t>(n) * kb;
    const uint16_t* s = S + static_cast<int64_t>(n) * ng;
    for (int t0 = 0; t0 < nt; t0 += kTB) {
      const int tb = nt - t0 < kTB ? nt - t0 : kTB;
      float acc[kTB][8];
      for (int t = 0; t < tb; ++t)
        for (int l = 0; l < 8; ++l)
          acc[t][l] = 0.f;
      for (int c = 0; c < K / kChunk; ++c) {
        const float scale = bf16_to_f32(s[(c * kChunk) / G]);
        const int j0 = c * (kChunk / 2);  // first byte of the chunk
        for (int t = 0; t < tb; ++t) {
          const float* e = xe + static_cast<int64_t>(t0 + t) * kb;
          const float* o = xo + static_cast<int64_t>(t0 + t) * kb;
          // Lane l: bytes j0 + l and j0 + 8 + l, low nibble times the even
          // element, high nibble times the odd one, added in that order.
          float d[8];
          for (int h = 0; h < 2; ++h) {  // two halves of 8 bytes
            for (int l = 0; l < 8; ++l) {
              const int j = j0 + h * 8 + l;
              const float lo = static_cast<float>(nibble(w[j], 0));
              const float hi = static_cast<float>(nibble(w[j], 1));
              d[l] = h == 0 ? lo * e[j] : std::fma(lo, e[j], d[l]);
              d[l] = std::fma(hi, o[j], d[l]);
            }
          }
          for (int l = 0; l < 8; ++l)
            acc[t][l] = std::fma(d[l], scale, acc[t][l]);
        }
      }
      for (int t = 0; t < tb; ++t)
        out[static_cast<int64_t>(t0 + t) * ldo + n] = lane_sum(acc[t]);
    }
  }
}

#if INT4_MOE_CPU_X86
// 8 bytes -> 8 dwords (one per lane, zero-extended): a single
// vpmovzxbd from memory.
__attribute__((target("avx2,fma"))) inline __m256i load8_u32(const uint8_t* p) {
  return _mm256_cvtepu8_epi32(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}

// A nibble u (0..15, in the low bits of a dword, the rest zero) to the float
// of its signed value (u ^ 8) - 8, exactly: the bits 0x4B000000 | (u ^ 8)
// are the float 2^23 + (u ^ 8); subtracting 2^23 + 8 is exact (Sterbenz).
// The same value as nibble(): u < 8 -> u, u >= 8 -> u - 16.
__attribute__((target("avx2,fma"))) inline __m256 nib_to_ps(__m256i u) {
  const __m256i magic = _mm256_set1_epi32(0x4B000008);
  const __m256 bias = _mm256_set1_ps(8388616.0f);
  return _mm256_sub_ps(_mm256_castsi256_ps(_mm256_xor_si256(u, magic)), bias);
}

  // The 16 bytes of a chunk to its four lane vectors of signed nibble values
  // as floats: lo0/hi0 from bytes 0..7 (low, high nibble), lo1/hi1 from bytes
  // 8..15. Two ways, the same values (both exact):
  //   kUnpackLut: one vpshufb against the table {0..7, -8..-1} gives the
  //     signed bytes of all 16 low (and all 16 high) nibbles, then
  //     vpmovsxbd + vcvtdq2ps (the technique of ggml's x86 repack kernels,
  //     ggml-cpu/arch/x86/repack.cpp, MIT; re-implemented here);
  //   kUnpackMagic: vpmovzxbd straight from memory, mask/shift, and the float
  //     trick above (no shuffle-port work besides the two widening loads).
  // Rows ahead that the inner function prefetches (0: none).
  #ifndef INT4_MOE_CPU_PF_ROWS
    #define INT4_MOE_CPU_PF_ROWS 2
  #endif
constexpr int kPrefetchRows = INT4_MOE_CPU_PF_ROWS;

  #ifndef INT4_MOE_CPU_UNPACK
    #define INT4_MOE_CPU_UNPACK 1  // 0 LUT, 1 magic
  #endif

__attribute__((target("avx2,fma"), always_inline)) inline void unpack_chunk(
    const uint8_t* p, __m256& lo0, __m256& hi0, __m256& lo1, __m256& hi1) {
  #if INT4_MOE_CPU_UNPACK == 1
  const __m256i lo_mask = _mm256_set1_epi32(0x0F);
  const __m256i v0 = load8_u32(p);
  const __m256i v1 = load8_u32(p + 8);
  lo0 = nib_to_ps(_mm256_and_si256(v0, lo_mask));
  hi0 = nib_to_ps(_mm256_srli_epi32(v0, 4));
  lo1 = nib_to_ps(_mm256_and_si256(v1, lo_mask));
  hi1 = nib_to_ps(_mm256_srli_epi32(v1, 4));
  #else
  const __m128i lut =
      _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1);
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw, m4));
  const __m128i hi =
      _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw, 4), m4));
  lo0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
  hi0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
  lo1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_unpackhi_epi64(lo, lo)));
  hi1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_unpackhi_epi64(hi, hi)));
  #endif
}

// One 32-element chunk of R rows for NT tokens: each row's nibbles are
// unpacked once and used for every token.
template <int NT, int R>
__attribute__((target("avx2,fma"), always_inline)) inline void chunk_avx2(
    const uint8_t* const* w,
    const __m256* scale,
    int j0,
    int64_t kb,
    const float* xe,
    const float* xo,
    __m256 (*acc)[NT]) {
  for (int r = 0; r < R; ++r) {
    __m256 lo0, hi0, lo1, hi1;
    unpack_chunk(w[r] + j0, lo0, hi0, lo1, hi1);
    for (int t = 0; t < NT; ++t) {
      const float* e = xe + t * kb + j0;
      const float* o = xo + t * kb + j0;
      __m256 d = _mm256_mul_ps(lo0, _mm256_loadu_ps(e));
      d = _mm256_fmadd_ps(hi0, _mm256_loadu_ps(o), d);
      d = _mm256_fmadd_ps(lo1, _mm256_loadu_ps(e + 8), d);
      d = _mm256_fmadd_ps(hi1, _mm256_loadu_ps(o + 8), d);
      acc[r][t] = _mm256_fmadd_ps(d, scale[r], acc[r][t]);
    }
  }
}

// Groups [g0, g1) of rows [n0, n1), R rows side by side (two rows give the
// core independent chains to overlap; a decode token has little other work
// per row), NT tokens per pass; the accumulators come from and go back to
// acc[(n - n0) * NT + t] (so a row can be continued over the next groups).
template <int NT, int R>
__attribute__((target("avx2,fma"))) inline void panel_avx2(
    const uint8_t* W,
    const uint16_t* S,
    int64_t kb,
    int ng,
    int chunks_per_group,
    int n0,
    int n1,
    int g0,
    int g1,
    const float* xe,
    const float* xo,
    __m256* acc_io) {
  int n = n0;
  for (; n + R <= n1; n += R) {
    const uint8_t* w[R];
    const uint16_t* s[R];
    __m256 acc[R][NT];
    for (int r = 0; r < R; ++r) {
      w[r] = W + static_cast<int64_t>(n + r) * kb;
      s[r] = S + static_cast<int64_t>(n + r) * ng;
      for (int t = 0; t < NT; ++t)
        acc[r][t] = acc_io[(n + r - n0) * NT + t];
    }
    for (int g = g0; g < g1; ++g) {
      __m256 scale[R];
      for (int r = 0; r < R; ++r) {
        scale[r] = _mm256_set1_ps(bf16_to_f32(s[r][g]));
        // The same bytes kPrefetchRows rows further on (rows are
        // contiguous): a thread streams its rows from memory at a few GB/s
        // and the hardware prefetchers stop at page boundaries. A prefetch
        // past the expert's last row never faults.
        if (kPrefetchRows > 0)
          _mm_prefetch(
              reinterpret_cast<const char*>(
                  w[r] + kPrefetchRows * kb + g * (chunks_per_group * 16)),
              _MM_HINT_T0);
      }
      for (int cc = 0; cc < chunks_per_group; ++cc)
        chunk_avx2<NT, R>(
            w,
            scale,
            (g * chunks_per_group + cc) * (kChunk / 2),
            kb,
            xe,
            xo,
            acc);
    }
    for (int r = 0; r < R; ++r)
      for (int t = 0; t < NT; ++t)
        acc_io[(n + r - n0) * NT + t] = acc[r][t];
  }
  if (n < n1)
    panel_avx2<NT, 1>(
        W,
        S,
        kb,
        ng,
        chunks_per_group,
        n,
        n1,
        g0,
        g1,
        xe,
        xo,
        acc_io + (n - n0) * NT);
}

// Bytes of activations (both halves, all NT tokens) a panel of groups may
// span: they are read once per row, so they should stay in the L1 cache.
constexpr int kPanelBytes = 16 * 1024;
constexpr int kRowBlock = 16;  // rows whose accumulators wait between panels

template <int NT, int R>
__attribute__((target("avx2,fma"))) inline void rows_avx2_nt(
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const float* xe,
    const float* xo,
    float* out,
    int ldo) {
  const int64_t kb = K / 2;
  const int ng = K / G;
  const int chunks_per_group = G / kChunk;
  // groups per panel: activations of a group are G floats per token
  const int per_panel = std::max(1, kPanelBytes / (NT * G * 4));
  alignas(32) __m256 acc[kRowBlock * NT];
  for (int b0 = n0; b0 < n1; b0 += kRowBlock) {
    const int b1 = std::min(n1, b0 + kRowBlock);
    for (int i = 0; i < (b1 - b0) * NT; ++i)
      acc[i] = _mm256_setzero_ps();
    for (int g0 = 0; g0 < ng; g0 += per_panel)
      panel_avx2<NT, R>(
          W,
          S,
          kb,
          ng,
          chunks_per_group,
          b0,
          b1,
          g0,
          std::min(ng, g0 + per_panel),
          xe,
          xo,
          acc);
    for (int n = b0; n < b1; ++n)
      for (int t = 0; t < NT; ++t) {
        alignas(32) float l[8];
        _mm256_store_ps(l, acc[(n - b0) * NT + t]);
        out[static_cast<int64_t>(t) * ldo + n] = lane_sum(l);
      }
  }
}

constexpr int kTBAvx2 = 8;  // tokens per pass over a row (AVX2 code)
  #ifndef INT4_MOE_CPU_R1
    #define INT4_MOE_CPU_R1 2
  #endif
  #ifndef INT4_MOE_CPU_R2
    #define INT4_MOE_CPU_R2 2
  #endif
constexpr int kR1 = INT4_MOE_CPU_R1;  // rows side by side at 1 token
constexpr int kR2 = INT4_MOE_CPU_R2;  // rows side by side at 2 tokens

__attribute__((target("avx2,fma"))) inline void rows_avx2(
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const float* xe,
    const float* xo,
    int nt,
    float* out,
    int ldo) {
  const int64_t kb = K / 2;
  for (int t0 = 0; t0 < nt; t0 += kTBAvx2) {
    const int tb = nt - t0 < kTBAvx2 ? nt - t0 : kTBAvx2;
    const float* e = xe + t0 * kb;
    const float* o = xo + t0 * kb;
    float* y = out + static_cast<int64_t>(t0) * ldo;
    switch (tb) {
      case 1:
        rows_avx2_nt<1, kR1>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      case 2:
        rows_avx2_nt<2, kR2>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      case 3:
        rows_avx2_nt<3, 1>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      case 4:
        rows_avx2_nt<4, 1>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      case 5:
        rows_avx2_nt<5, 1>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      case 6:
        rows_avx2_nt<6, 1>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      case 7:
        rows_avx2_nt<7, 1>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
      default:
        rows_avx2_nt<8, 1>(W, S, K, G, n0, n1, e, o, y, ldo);
        break;
    }
  }
}

inline bool cpu_has_avx2() {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#else
inline bool cpu_has_avx2() { return false; }
#endif

// The instruction sets of rows(). New paths take the next numbers; auto
// picks the widest the host has.
enum class Isa : int { kAuto = 0, kScalar = 1, kAvx2 = 2 };

inline Isa resolve_isa(Isa want) {
  if (want == Isa::kScalar) return Isa::kScalar;
  if (want == Isa::kAvx2) {
    if (!cpu_has_avx2())
      throw std::runtime_error(
          "int4_moe_cpu: AVX2 and FMA asked for, not available");
    return Isa::kAvx2;
  }
  return cpu_has_avx2() ? Isa::kAvx2 : Isa::kScalar;
}

inline void rows(
    Isa isa,
    const uint8_t* W,
    const uint16_t* S,
    int K,
    int G,
    int n0,
    int n1,
    const float* xe,
    const float* xo,
    int nt,
    float* out,
    int ldo) {
#if INT4_MOE_CPU_X86
  if (isa == Isa::kAvx2) {
    rows_avx2(W, S, K, G, n0, n1, xe, xo, nt, out, ldo);
    return;
  }
#endif
  rows_scalar(W, S, K, G, n0, n1, xe, xo, nt, out, ldo);
}

// --------------------------------------------------------------------------
// A small persistent thread pool: one parallel phase at a time, items taken
// from a shared counter, the calling thread works too. Workers spin a
// bounded time after a phase (the phases of one call follow each other
// within microseconds), then sleep (between two layer calls the host waits
// for the card for a millisecond or more, and a spinning pool would take
// cores the other processes need).
// --------------------------------------------------------------------------

class Pool {
 public:
  explicit Pool(int threads) : n_(threads < 1 ? 1 : threads) {
    for (int i = 1; i < n_; ++i)
      workers_.emplace_back([this] { loop(); });
  }
  ~Pool() {
    stop_.store(true);
    epoch_.fetch_add(1);
    wake_all();
    for (auto& t : workers_)
      t.join();
  }
  int size() const { return n_; }

  // How long a worker spins after a phase before it sleeps.
  static constexpr int64_t kDefaultSpinUs = 100;
  void set_spin_us(int64_t us) {
    spin_us_.store(us < 0 ? 0 : us, std::memory_order_relaxed);
  }

  // Pin worker i (1..n-1) to cores[i % cores.size()]; the calling thread is
  // left where its process put it, so cores[0] is meant for it (the caller
  // pins it, or leaves it free). An empty list unpins (any core of the
  // process's mask). Returns false where the platform has no affinity call
  // or the kernel refused a core.
  bool set_cores(const std::vector<int>& cores) {
    bool ok = true;
#if defined(__linux__)
    cpu_set_t any;
    CPU_ZERO(&any);
    if (cores.empty() && sched_getaffinity(0, sizeof(any), &any) != 0)
      return false;
    for (size_t i = 0; i < workers_.size(); ++i) {
      cpu_set_t set;
      if (cores.empty()) {
        set = any;
      } else {
        CPU_ZERO(&set);
        CPU_SET(cores[(i + 1) % cores.size()], &set);
      }
      if (pthread_setaffinity_np(
              workers_[i].native_handle(), sizeof(set), &set) != 0)
        ok = false;
    }
#else
    ok = cores.empty();
#endif
    return ok;
  }

  // fn(i) for i in [0, n), across the pool; returns when all are done.
  // One caller at a time.
  void run(int64_t n, const std::function<void(int64_t)>& fn) {
    if (n <= 0) return;
    if (n_ == 1 || n == 1) {
      for (int64_t i = 0; i < n; ++i)
        fn(i);
      return;
    }
    fn_ = &fn;
    total_ = n;
    next_.store(0, std::memory_order_relaxed);
    pending_.store(n_ - 1, std::memory_order_relaxed);
    error_.clear();
    epoch_.fetch_add(1);  // publishes the phase (seq_cst)
    if (sleeping_.load() > 0) wake_all();
    work();
    // wait for the workers to leave this phase
    while (pending_.load(std::memory_order_acquire) > 0)
      relax();
    if (!error_.empty()) throw std::runtime_error(error_);
  }

 private:
  static void relax() {
#if INT4_MOE_CPU_X86
    _mm_pause();
#else
    std::this_thread::yield();
#endif
  }

  // A sleeping worker waits on the epoch itself (a futex where the library
  // has C++20 atomic waits): a wake-up takes no lock, so the workers of a
  // phase do not queue on a mutex one after the other.
  void wake_all() {
#if __cplusplus >= 202002L
    epoch_.notify_all();
#else
    {
      std::lock_guard<std::mutex> g(mu_);
    }
    cv_.notify_all();
#endif
  }

  void sleep_while(uint64_t seen) {
#if __cplusplus >= 202002L
    epoch_.wait(seen);
#else
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return epoch_.load() != seen; });
#endif
  }

  void work() {
    const auto& fn = *fn_;
    for (;;) {
      const int64_t i = next_.fetch_add(1, std::memory_order_relaxed);
      if (i >= total_) break;
      try {
        fn(i);
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> g(err_mu_);
        if (error_.empty()) error_ = e.what();
      }
    }
  }

  void loop() {
    uint64_t seen = 0;
    for (;;) {
      // Spin for a bounded time (not a count of pause instructions: a pause
      // takes a few cycles on some cores and over a hundred on others).
      const auto until =
          std::chrono::steady_clock::now() +
          std::chrono::microseconds(spin_us_.load(std::memory_order_relaxed));
      while (epoch_.load(std::memory_order_acquire) == seen) {
        for (int i = 0; i < 16; ++i)
          relax();
        if (std::chrono::steady_clock::now() >= until) {
          sleeping_.fetch_add(1);
          // Either run() sees this count and wakes us, or the epoch has
          // already moved and the wait returns at once.
          if (epoch_.load() == seen) sleep_while(seen);
          sleeping_.fetch_sub(1);
        }
      }
      seen = epoch_.load(std::memory_order_acquire);
      if (stop_.load()) return;
      work();
      pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  int n_;
  std::vector<std::thread> workers_;
  std::atomic<uint64_t> epoch_{0};
  std::atomic<int> sleeping_{0};
  std::atomic<bool> stop_{false};
  std::atomic<int64_t> spin_us_{kDefaultSpinUs};
  std::mutex mu_, err_mu_;
#if __cplusplus < 202002L
  std::condition_variable cv_;
#endif
  const std::function<void(int64_t)>* fn_ = nullptr;
  int64_t total_ = 0;
  std::atomic<int64_t> next_{0};
  std::atomic<int> pending_{0};
  std::string error_;
};

// --------------------------------------------------------------------------
// The call: a list of (unit, token, weight) triples, in the order the card's
// gather adds them for each token (token-major, top-k order inside a token).
// --------------------------------------------------------------------------

struct Layout {
  int hidden;          // H
  int inter;           // I
  int group;           // G
  int64_t unit_bytes;  // row stride of the units tensor
  int64_t off_w13, off_w2, off_s13, off_s2;
};

inline void check_layout(const Layout& L) {
  if (L.hidden % kChunk || L.inter % kChunk)
    throw std::runtime_error("int4_moe_cpu: H and I must be multiples of 32");
  if (L.group % kChunk || L.hidden % L.group || L.inter % L.group)
    throw std::runtime_error(
        "int4_moe_cpu: the group size must be a multiple of 32 dividing H "
        "and I");
  const int64_t H = L.hidden, I = L.inter;
  const int64_t need_w13 = 2 * I * H / 2, need_w2 = H * I / 2;
  const int64_t need_s13 = 2 * I * (H / L.group) * 2;
  const int64_t need_s2 = H * (I / L.group) * 2;
  if (L.off_w13 < 0 || L.off_w13 + need_w13 > L.unit_bytes || L.off_w2 < 0 ||
      L.off_w2 + need_w2 > L.unit_bytes || L.off_s13 < 0 ||
      L.off_s13 + need_s13 > L.unit_bytes || L.off_s2 < 0 ||
      L.off_s2 + need_s2 > L.unit_bytes || (L.off_s13 & 1) || (L.off_s2 & 1))
    throw std::runtime_error("int4_moe_cpu: tensor offsets outside the unit");
}

// One input row (bf16, H elements) to its even and odd halves in fp32.
inline void convert_row_scalar(const uint16_t* xt, float* e, float* o, int H) {
  for (int j = 0; j < H / 2; ++j) {
    e[j] = bf16_to_f32(xt[2 * j]);
    o[j] = bf16_to_f32(xt[2 * j + 1]);
  }
}

#if INT4_MOE_CPU_X86
// A dword holds x[2j] (low half) and x[2j+1] (high half): shifted left by
// 16 it is the float of x[2j], masked it is the float of x[2j+1].
__attribute__((target("avx2,fma"))) inline void
convert_row_avx2(const uint16_t* xt, float* e, float* o, int H) {
  const __m256i hi_mask = _mm256_set1_epi32(static_cast<int>(0xFFFF0000u));
  for (int j = 0; j < H / 2; j += 8) {
    const __m256i v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xt + 2 * j));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(e + j), _mm256_slli_epi32(v, 16));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(o + j), _mm256_and_si256(v, hi_mask));
  }
}
#endif

inline void
convert_row(Isa isa, const uint16_t* xt, float* e, float* o, int H) {
#if INT4_MOE_CPU_X86
  if (isa == Isa::kAvx2) {  // H is a multiple of 32
    convert_row_avx2(xt, e, o, H);
    return;
  }
#endif
  convert_row_scalar(xt, e, o, H);
}

// Input elements (calls x H) up to which the calling thread converts alone.
constexpr int64_t kSerialConvert = int64_t(1) << 18;

// fp32 scratch on 64-byte boundaries: every row of the activations starts on
// a cache line (row lengths are multiples of 16 floats), so no vector load
// of the inner function straddles two lines.
template <typename T, size_t A>
struct AlignedAllocator {
  using value_type = T;
  AlignedAllocator() = default;
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, A>&) {}
  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, A>;
  };
  T* allocate(size_t n) {
    void* p = ::operator new(n * sizeof(T), std::align_val_t(A));
    return static_cast<T*>(p);
  }
  void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t(A)); }
  bool operator==(const AlignedAllocator&) const { return true; }
  bool operator!=(const AlignedAllocator&) const { return false; }
};
using AlignedFloats = std::vector<float, AlignedAllocator<float, 64>>;

// Rows per work item: GEMM1 takes kRows1 gate rows and the kRows1 up rows
// that pair with them (so the activation of those columns is computed in the
// same item); GEMM2 takes kRows2 output rows. Small enough that a decode
// call (tens of experts, one token each) splits into hundreds of items per
// thread count, large enough that an item streams a few tens of KB.
constexpr int kRows1 = 16;
constexpr int kRows2 = 16;

// out [T, H] bf16 is overwritten: zero for tokens without calls.
inline void run_calls(
    Pool& pool,
    Isa isa,
    const Layout& L,
    uint16_t* out,
    const uint16_t* x,
    int64_t T,
    const uint8_t* units,
    int64_t num_units,
    const int32_t* call_unit,
    const int32_t* call_token,
    const float* call_weight,
    int64_t C) {
  check_layout(L);
  isa = resolve_isa(isa);
  const int H = L.hidden, I = L.inter, G = L.group;
  std::memset(out, 0, static_cast<size_t>(T) * H * sizeof(uint16_t));
  if (C == 0) return;
  for (int64_t c = 0; c < C; ++c) {
    if (call_unit[c] < 0 || call_unit[c] >= num_units)
      throw std::runtime_error(
          "int4_moe_cpu: call " + std::to_string(c) + " names unit " +
          std::to_string(call_unit[c]) + " of " + std::to_string(num_units));
    if (call_token[c] < 0 || call_token[c] >= T)
      throw std::runtime_error(
          "int4_moe_cpu: call " + std::to_string(c) + " names token " +
          std::to_string(call_token[c]) + " of " + std::to_string(T));
    if (c > 0 && call_token[c] < call_token[c - 1])
      throw std::runtime_error(
          "int4_moe_cpu: calls must be in token order (call " +
          std::to_string(c) + ")");
  }
  // Group the calls by unit, keeping call order inside a group.
  std::vector<int32_t> units_used;
  std::vector<std::vector<int64_t>> calls_of;
  {
    std::vector<int32_t> slot(static_cast<size_t>(num_units), -1);
    for (int64_t c = 0; c < C; ++c) {
      int32_t& s = slot[call_unit[c]];
      if (s < 0) {
        s = static_cast<int32_t>(units_used.size());
        units_used.push_back(call_unit[c]);
        calls_of.emplace_back();
      }
      calls_of[s].push_back(c);
    }
  }
  const int64_t E = static_cast<int64_t>(units_used.size());
  // Per call: its token's input split into even/odd (GEMM1), its activation
  // split the same way (GEMM2), its expert output.
  // Scratch kept per calling thread across calls (grown, never shrunk). The
  // workers reach it through these references: inside the lambdas a
  // thread_local name would be the worker's own.
  struct Scratch {
    AlignedFloats xe, xo, h, ae, ao, y;
  };
  static thread_local Scratch scratch;
  auto& xe = scratch.xe;
  auto& xo = scratch.xo;
  auto& h = scratch.h;
  auto& ae = scratch.ae;
  auto& ao = scratch.ao;
  auto& y = scratch.y;
  xe.resize(static_cast<size_t>(C) * (H / 2));
  xo.resize(static_cast<size_t>(C) * (H / 2));
  h.resize(static_cast<size_t>(C) * 2 * I);
  ae.resize(static_cast<size_t>(C) * (I / 2));
  ao.resize(static_cast<size_t>(C) * (I / 2));
  y.resize(static_cast<size_t>(C) * H);
  // Calls laid out by group: base[e] is the first call row of unit group e.
  std::vector<int64_t> base(E + 1, 0), row_of(C);
  for (int64_t e = 0; e < E; ++e) {
    base[e + 1] = base[e] + static_cast<int64_t>(calls_of[e].size());
    for (size_t i = 0; i < calls_of[e].size(); ++i)
      row_of[calls_of[e][i]] = base[e] + static_cast<int64_t>(i);
  }
  // Phase 1: the inputs, split into even and odd elements, in fp32. Cheap
  // (two vector operations per 16 elements): at decode sizes the calling
  // thread does it alone, saving a fork/join.
  auto convert = [&](int64_t c) {
    convert_row(
        isa,
        x + static_cast<int64_t>(call_token[c]) * H,
        xe.data() + row_of[c] * (H / 2),
        xo.data() + row_of[c] * (H / 2),
        H);
  };
  if (C * H <= kSerialConvert) {
    for (int64_t c = 0; c < C; ++c)
      convert(c);
  } else {
    pool.run(C, convert);
  }
  // Phase 2: GEMM1 [calls of e, H] x w13[e]^T -> h [calls, 2I], and in the
  // same item the activation of the columns it computed:
  // a = bf16(bf16(silu(bf16 gate)) * bf16 up).
  const int64_t blocks1 = (I + kRows1 - 1) / kRows1;
  pool.run(E * blocks1, [&](int64_t item) {
    const int64_t e = item / blocks1;
    const int i0 = static_cast<int>((item % blocks1) * kRows1);
    const int i1 = i0 + kRows1 < I ? i0 + kRows1 : I;
    const uint8_t* u =
        units + static_cast<int64_t>(units_used[e]) * L.unit_bytes;
    const int nt = static_cast<int>(base[e + 1] - base[e]);
    const uint8_t* w13 = u + L.off_w13;
    const auto* s13 = reinterpret_cast<const uint16_t*>(u + L.off_s13);
    const float* e_in = xe.data() + base[e] * (H / 2);
    const float* o_in = xo.data() + base[e] * (H / 2);
    float* hb = h.data() + base[e] * 2 * I;
    rows(isa, w13, s13, H, G, i0, i1, e_in, o_in, nt, hb, 2 * I);
    rows(isa, w13, s13, H, G, I + i0, I + i1, e_in, o_in, nt, hb, 2 * I);
    for (int64_t r = base[e]; r < base[e + 1]; ++r) {
      const float* hr = h.data() + r * 2 * I;
      float* ea = ae.data() + r * (I / 2);
      float* oa = ao.data() + r * (I / 2);
      for (int i = i0; i < i1; ++i) {
        const float g = round_bf16(hr[i]);
        const float up = round_bf16(hr[I + i]);
        const float s = round_bf16(g / (1.0f + std::exp(-g)));
        const float a = round_bf16(s * up);
        if (i & 1)
          oa[i / 2] = a;
        else
          ea[i / 2] = a;
      }
    }
  });
  // Phase 3: GEMM2 [calls of e, I] x w2[e]^T -> y [calls, H] and the gather
  // in one item per block of output columns, over every expert of the call:
  // out[t] = bf16(sum in call order of float(bf16 y) * weight).
  const int64_t blocks2 = (H + kRows2 - 1) / kRows2;
  pool.run(blocks2, [&](int64_t b) {
    const int n0 = static_cast<int>(b * kRows2);
    const int n1 = n0 + kRows2 < H ? n0 + kRows2 : H;
    for (int64_t e = 0; e < E; ++e) {
      const uint8_t* u =
          units + static_cast<int64_t>(units_used[e]) * L.unit_bytes;
      rows(
          isa,
          u + L.off_w2,
          reinterpret_cast<const uint16_t*>(u + L.off_s2),
          I,
          G,
          n0,
          n1,
          ae.data() + base[e] * (I / 2),
          ao.data() + base[e] * (I / 2),
          static_cast<int>(base[e + 1] - base[e]),
          y.data() + base[e] * H,
          H);
    }
    float acc[kRows2];
    int64_t c = 0;
    while (c < C) {
      const int32_t t = call_token[c];
      std::fill(acc, acc + (n1 - n0), 0.f);
      for (; c < C && call_token[c] == t; ++c) {
        const float* yr = y.data() + row_of[c] * H;
        const float w = call_weight[c];
        for (int j = n0; j < n1; ++j) {
          const float m = round_bf16(yr[j]) * w;
          acc[j - n0] = acc[j - n0] + m;
        }
      }
      uint16_t* ot = out + static_cast<int64_t>(t) * H;
      for (int j = n0; j < n1; ++j)
        ot[j] = f32_to_bf16(acc[j - n0]);
    }
  });
}

}  // namespace cpu_moe
