// Int4 MoE experts computed on the host CPU, in the layout the XPU kernels
// read (the expert store's unit): no torch dependency, so the arithmetic can
// be tested on its own.
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
// The card has two int4 kernels with different arithmetic inside a dot
// product, and the op computes either (`Arith`):
//   kChunkScaled  the int4 GEMV (moe_int4_gemv.cpp; the card takes it for a
//                 call of at most 8 rows): the scale is applied to the fp32
//                 partial sum of each 32-element chunk of q * x.
//   kBf16Weights  the grouped GEMM (grouped_gemm/xe_2/gemm_xe2.hpp, every
//                 larger call, so every call of a model with top-k above 8):
//                 each weight is dequantised to bf16(q * s) (apply_scale: a
//                 bf16 times fp32 multiply into bf16), then multiplied with
//                 the bf16 input and accumulated in fp32 (DPAS).
// The two differ in most bf16 outputs (the rounding of q * s to bf16 against
// none), so the op must take the one the card would take for the same call,
// per GEMM: the GEMV also needs K to be a multiple of 512, so with I = 640 a
// small call runs GEMM1 on the GEMV and GEMM2 on the grouped GEMM. The op's
// `arith` argument: bit 0 set, GEMM1 takes kBf16Weights; bit 1, GEMM2.
// Either way a dot product is formed in 8 lanes with fused multiply-adds and
// the lanes are added in a fixed tree. The summation order differs from the
// card's (work-items, a sub-group reduction, DPAS, and a compiler allowed to
// contract and reassociate), so a result can differ from the card's by the
// rounding of fp32 sums only: at bf16 outputs, a fraction of a bf16 step at
// the output's scale on a small share of the values (measured by the
// self-test). The scalar code (std::fma) and the AVX2 code (FMA3) take the
// same steps in the same order, so they agree bit for bit with each other on
// every host; the scalar one is the fallback for a host without AVX2 and
// FMA, and slow.

#pragma once

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
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

// The arithmetic inside a dot product (see the top of this file).
enum class Arith : int { kChunkScaled = 0, kBf16Weights = 1 };

// --------------------------------------------------------------------------
// One block of output rows of a W4A16 product, for up to kTB tokens.
//
// W  [N, K/2] bytes (row n at W + n * K/2), S [N, K/G] bf16.
// xe, xo: per token, the even and the odd input elements in fp32
//   (xe[t * K/2 + j] = x[t][2j], xo[...] = x[t][2j + 1]).
// out[t * ldo + n] = the fp32 dot product of row n with token t.
// --------------------------------------------------------------------------

constexpr int kChunk = 32;  // elements per scaled partial sum (16 bytes)
constexpr int kTB = 4;      // tokens per pass over a row

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
    Arith arith,
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
  if (arith == Arith::kBf16Weights) {
    // Lane l of a chunk: bytes j0 + l and j0 + 8 + l; per byte the weight
    // bf16(q * s) of the low nibble times the even element, then the high
    // nibble's times the odd one, each a fused multiply-add into the lane.
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
          const int j0 = c * (kChunk / 2);
          float wl[2][8], wh[2][8];
          for (int h = 0; h < 2; ++h)
            for (int l = 0; l < 8; ++l) {
              const int j = j0 + h * 8 + l;
              wl[h][l] =
                  round_bf16(static_cast<float>(nibble(w[j], 0)) * scale);
              wh[h][l] =
                  round_bf16(static_cast<float>(nibble(w[j], 1)) * scale);
            }
          for (int t = 0; t < tb; ++t) {
            const float* e = xe + static_cast<int64_t>(t0 + t) * kb;
            const float* o = xo + static_cast<int64_t>(t0 + t) * kb;
            for (int h = 0; h < 2; ++h)
              for (int l = 0; l < 8; ++l) {
                const int j = j0 + h * 8 + l;
                acc[t][l] = std::fma(wl[h][l], e[j], acc[t][l]);
                acc[t][l] = std::fma(wh[h][l], o[j], acc[t][l]);
              }
          }
        }
        for (int t = 0; t < tb; ++t)
          out[static_cast<int64_t>(t0 + t) * ldo + n] = lane_sum(acc[t]);
      }
    }
    return;
  }
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
__attribute__((target("avx2,fma"))) inline __m256
nibbles_to_ps(__m128i bytes8) {
  // 8 bytes (low half of bytes8) -> 8 int32 -> sign fix -> float
  const __m256i v = _mm256_cvtepu8_epi32(bytes8);
  const __m256i eight = _mm256_set1_epi32(8);
  const __m256i fixed = _mm256_sub_epi32(_mm256_xor_si256(v, eight), eight);
  return _mm256_cvtepi32_ps(fixed);
}

// 8 floats rounded to bf16 (to nearest even, as round_bf16), kept as floats.
// The products q * s it is used on are finite.
__attribute__((target("avx2,fma"))) inline __m256 round_bf16_ps(__m256 v) {
  const __m256i b = _mm256_castps_si256(v);
  const __m256i lsb =
      _mm256_and_si256(_mm256_srli_epi32(b, 16), _mm256_set1_epi32(1));
  const __m256i r =
      _mm256_add_epi32(b, _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7fff)));
  return _mm256_castsi256_ps(
      _mm256_and_si256(r, _mm256_set1_epi32(static_cast<int>(0xffff0000u))));
}

// Per lane, table[idx] for raw nibbles idx 0..15: lo holds 0..7, hi 8..15.
__attribute__((target("avx2,fma"))) inline __m256
lut_ps(__m256 lo, __m256 hi, __m256i idx) {
  const __m256 a = _mm256_permutevar8x32_ps(lo, idx);
  const __m256 b = _mm256_permutevar8x32_ps(hi, idx);
  return _mm256_blendv_ps(
      a, b, _mm256_castsi256_ps(_mm256_slli_epi32(idx, 28)));
}

__attribute__((target("avx2,fma"))) inline void rows_avx2(
    Arith arith,
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
  const __m128i lo_mask = _mm_set1_epi8(0x0F);
  const __m256 q_lo = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
  const __m256 q_hi = _mm256_setr_ps(-8, -7, -6, -5, -4, -3, -2, -1);
  for (int n = n0; n < n1; ++n) {
    const uint8_t* w = W + static_cast<int64_t>(n) * kb;
    const uint16_t* s = S + static_cast<int64_t>(n) * ng;
    for (int t0 = 0; t0 < nt; t0 += kTB) {
      const int tb = nt - t0 < kTB ? nt - t0 : kTB;
      __m256 acc[kTB];
      for (int t = 0; t < kTB; ++t)
        acc[t] = _mm256_setzero_ps();
      // kBf16Weights: bf16(q * s) of the current group for the nibbles 0..7
      // (q 0..7) and 8..15 (q -8..-1).
      __m256 lut_lo = _mm256_setzero_ps(), lut_hi = _mm256_setzero_ps();
      for (int c = 0; c < K / kChunk; ++c) {
        const __m256 scale = _mm256_set1_ps(bf16_to_f32(s[(c * kChunk) / G]));
        const int j0 = c * (kChunk / 2);
        const __m128i raw =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j0));
        const __m128i lo = _mm_and_si128(raw, lo_mask);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), lo_mask);
        if (arith == Arith::kBf16Weights) {
          // The grouped GEMM's weights bf16(q * s), looked up by nibble in
          // the group's table, fused into the lanes in the scalar code's
          // order (low then high nibble, first then second half).
          if ((c * kChunk) % G == 0) {
            lut_lo = round_bf16_ps(_mm256_mul_ps(q_lo, scale));
            lut_hi = round_bf16_ps(_mm256_mul_ps(q_hi, scale));
          }
          const __m256 lo0 = lut_ps(lut_lo, lut_hi, _mm256_cvtepu8_epi32(lo));
          const __m256 hi0 = lut_ps(lut_lo, lut_hi, _mm256_cvtepu8_epi32(hi));
          const __m256 lo1 = lut_ps(
              lut_lo, lut_hi, _mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)));
          const __m256 hi1 = lut_ps(
              lut_lo, lut_hi, _mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)));
          for (int t = 0; t < tb; ++t) {
            const float* e = xe + static_cast<int64_t>(t0 + t) * kb + j0;
            const float* o = xo + static_cast<int64_t>(t0 + t) * kb + j0;
            acc[t] = _mm256_fmadd_ps(lo0, _mm256_loadu_ps(e), acc[t]);
            acc[t] = _mm256_fmadd_ps(hi0, _mm256_loadu_ps(o), acc[t]);
            acc[t] = _mm256_fmadd_ps(lo1, _mm256_loadu_ps(e + 8), acc[t]);
            acc[t] = _mm256_fmadd_ps(hi1, _mm256_loadu_ps(o + 8), acc[t]);
          }
          continue;
        }
        const __m256 lo0 = nibbles_to_ps(lo);
        const __m256 hi0 = nibbles_to_ps(hi);
        const __m256 lo1 = nibbles_to_ps(_mm_srli_si128(lo, 8));
        const __m256 hi1 = nibbles_to_ps(_mm_srli_si128(hi, 8));
        for (int t = 0; t < tb; ++t) {
          const float* e = xe + static_cast<int64_t>(t0 + t) * kb + j0;
          const float* o = xo + static_cast<int64_t>(t0 + t) * kb + j0;
          __m256 d = _mm256_mul_ps(lo0, _mm256_loadu_ps(e));
          d = _mm256_fmadd_ps(hi0, _mm256_loadu_ps(o), d);
          d = _mm256_fmadd_ps(lo1, _mm256_loadu_ps(e + 8), d);
          d = _mm256_fmadd_ps(hi1, _mm256_loadu_ps(o + 8), d);
          acc[t] = _mm256_fmadd_ps(d, scale, acc[t]);
        }
      }
      for (int t = 0; t < tb; ++t) {
        alignas(32) float l[8];
        _mm256_store_ps(l, acc[t]);
        out[static_cast<int64_t>(t0 + t) * ldo + n] = lane_sum(l);
      }
    }
  }
}

inline bool cpu_has_avx2() {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#else
inline bool cpu_has_avx2() { return false; }
#endif

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
    Arith arith,
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
    rows_avx2(arith, W, S, K, G, n0, n1, xe, xo, nt, out, ldo);
    return;
  }
#endif
  rows_scalar(arith, W, S, K, G, n0, n1, xe, xo, nt, out, ldo);
}

// --------------------------------------------------------------------------
// A small persistent thread pool: one parallel phase at a time, items taken
// from a shared counter, the calling thread works too.
// --------------------------------------------------------------------------

class Pool {
 public:
  explicit Pool(int threads) : n_(threads < 1 ? 1 : threads) {
    for (int i = 1; i < n_; ++i)
      workers_.emplace_back([this] { loop(); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> g(mu_);
      stop_ = true;
      ++epoch_;
      epoch_atomic_.store(epoch_, std::memory_order_release);
    }
    cv_.notify_all();
    for (auto& t : workers_)
      t.join();
  }
  int size() const { return n_; }

  // fn(i) for i in [0, n), across the pool; returns when all are done.
  void run(int64_t n, const std::function<void(int64_t)>& fn) {
    if (n <= 0) return;
    if (n_ == 1 || n == 1) {
      for (int64_t i = 0; i < n; ++i)
        fn(i);
      return;
    }
    bool wake;
    {
      std::lock_guard<std::mutex> g(mu_);
      fn_ = &fn;
      total_ = n;
      next_.store(0);
      pending_.store(n_ - 1);
      error_.clear();
      ++epoch_;
      epoch_atomic_.store(epoch_, std::memory_order_release);
      wake = sleeping_ > 0;
    }
    if (wake) cv_.notify_all();
    work();
    // wait for the workers to leave this phase
    while (pending_.load(std::memory_order_acquire) > 0) {
#if INT4_MOE_CPU_X86
      _mm_pause();
#else
      std::this_thread::yield();
#endif
    }
    if (!error_.empty()) throw std::runtime_error(error_);
  }

 private:
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
      // Spin a while first: the phases of one call follow each other within
      // microseconds, and a sleeping worker costs a wake-up per phase.
      for (int i = 0;
           i < kSpin && epoch_atomic_.load(std::memory_order_acquire) == seen;
           ++i) {
#if INT4_MOE_CPU_X86
        _mm_pause();
#else
        std::this_thread::yield();
#endif
      }
      {
        std::unique_lock<std::mutex> lk(mu_);
        if (epoch_ == seen) {
          ++sleeping_;
          cv_.wait(lk, [&] { return epoch_ != seen; });
          --sleeping_;
        }
        seen = epoch_;
        if (stop_) return;
      }
      work();
      pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  static constexpr int kSpin = 20000;  // a fraction of a millisecond

  int n_;
  std::vector<std::thread> workers_;
  std::mutex mu_, err_mu_;
  std::condition_variable cv_;
  uint64_t epoch_ = 0;
  std::atomic<uint64_t> epoch_atomic_{0};
  int sleeping_ = 0;
  bool stop_ = false;
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
    int64_t C,
    int arith = 0) {
  check_layout(L);
  const Arith arith1 = (arith & 1) ? Arith::kBf16Weights : Arith::kChunkScaled;
  const Arith arith2 = (arith & 2) ? Arith::kBf16Weights : Arith::kChunkScaled;
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
    std::vector<float> xe, xo, h, ae, ao, y;
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
  pool.run(C, [&](int64_t c) {
    const uint16_t* xt = x + static_cast<int64_t>(call_token[c]) * H;
    float* e = xe.data() + row_of[c] * (H / 2);
    float* o = xo.data() + row_of[c] * (H / 2);
    for (int j = 0; j < H / 2; ++j) {
      e[j] = bf16_to_f32(xt[2 * j]);
      o[j] = bf16_to_f32(xt[2 * j + 1]);
    }
  });
  constexpr int kRows = 8;  // output rows per work item
  // GEMM1: [calls of e, H] x w13[e]^T -> h [calls, 2I], rounded to bf16.
  const int64_t blocks1 = (2 * I + kRows - 1) / kRows;
  pool.run(E * blocks1, [&](int64_t item) {
    const int64_t e = item / blocks1;
    const int n0 = static_cast<int>((item % blocks1) * kRows);
    const int n1 = n0 + kRows < 2 * I ? n0 + kRows : 2 * I;
    const uint8_t* u =
        units + static_cast<int64_t>(units_used[e]) * L.unit_bytes;
    const int nt = static_cast<int>(base[e + 1] - base[e]);
    rows(
        isa,
        arith1,
        u + L.off_w13,
        reinterpret_cast<const uint16_t*>(u + L.off_s13),
        H,
        G,
        n0,
        n1,
        xe.data() + base[e] * (H / 2),
        xo.data() + base[e] * (H / 2),
        nt,
        h.data() + base[e] * 2 * I,
        2 * I);
  });
  // Activation, per call row: a = bf16(bf16(silu(bf16 gate)) * bf16 up).
  pool.run(C, [&](int64_t r) {
    const float* hr = h.data() + r * 2 * I;
    float* e = ae.data() + r * (I / 2);
    float* o = ao.data() + r * (I / 2);
    for (int i = 0; i < I; ++i) {
      const float g = round_bf16(hr[i]);
      const float u = round_bf16(hr[I + i]);
      const float s = round_bf16(g / (1.0f + std::exp(-g)));
      const float a = round_bf16(s * u);
      if (i & 1)
        o[i / 2] = a;
      else
        e[i / 2] = a;
    }
  });
  // GEMM2: [calls of e, I] x w2[e]^T -> y [calls, H].
  const int64_t blocks2 = (H + kRows - 1) / kRows;
  pool.run(E * blocks2, [&](int64_t item) {
    const int64_t e = item / blocks2;
    const int n0 = static_cast<int>((item % blocks2) * kRows);
    const int n1 = n0 + kRows < H ? n0 + kRows : H;
    const uint8_t* u =
        units + static_cast<int64_t>(units_used[e]) * L.unit_bytes;
    const int nt = static_cast<int>(base[e + 1] - base[e]);
    rows(
        isa,
        arith2,
        u + L.off_w2,
        reinterpret_cast<const uint16_t*>(u + L.off_s2),
        I,
        G,
        n0,
        n1,
        ae.data() + base[e] * (I / 2),
        ao.data() + base[e] * (I / 2),
        nt,
        y.data() + base[e] * H,
        H);
  });
  // Gather: out[t] = bf16(sum in call order of float(bf16 y) * weight).
  // Calls are visited in their given order; columns split across threads.
  constexpr int kCols = 256;
  const int64_t col_blocks = (H + kCols - 1) / kCols;
  pool.run(col_blocks, [&](int64_t b) {
    const int j0 = static_cast<int>(b * kCols);
    const int j1 = j0 + kCols < H ? j0 + kCols : H;
    std::vector<float> acc(static_cast<size_t>(j1 - j0));
    int64_t c = 0;
    while (c < C) {
      const int32_t t = call_token[c];
      std::fill(acc.begin(), acc.end(), 0.f);
      for (; c < C && call_token[c] == t; ++c) {
        const float* yr = y.data() + row_of[c] * H;
        const float w = call_weight[c];
        for (int j = j0; j < j1; ++j) {
          const float m = round_bf16(yr[j]) * w;
          acc[j - j0] = acc[j - j0] + m;
        }
      }
      uint16_t* ot = out + static_cast<int64_t>(t) * H;
      for (int j = j0; j < j1; ++j)
        ot[j] = f32_to_bf16(acc[j - j0]);
    }
  });
}

}  // namespace cpu_moe
