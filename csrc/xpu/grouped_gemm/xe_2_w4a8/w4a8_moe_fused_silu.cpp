// silu_and_mul's element for the fused W4A8 MoE act + quant kernel.
//
// This TU is compiled with the library's default flags, the floating-point
// model csrc/activation.cpp is compiled with (icpx default, no per-file
// option), and spells the expression of activation.cpp's silu_kernel and
// compute<scalar_t, silu_kernel, act_first = true> on the type
// SyclTypeTrait<c10::BFloat16> gives it (sycl::ext::oneapi::bfloat16):
//   silu = bf16(x / (1 + exp(-x)))   in fp32, rounded to bf16
//   out  = silu * y                  bf16 x bf16: fp32 product, rounded
// The caller (w4a8_moe_fused_prep.cpp) is compiled with the int8 quant's
// precise model; keeping this element in its own TU keeps each half under
// the model of the kernel it replaces.
#include <sycl/sycl.hpp>

#include <cstdint>

#include "w4a8_moe_fused.h"

namespace {

using sycl_bf16 = sycl::ext::oneapi::bfloat16;

template <typename T>
inline T silu_kernel(const T& x) {
  // x * sigmoid(x), as csrc/activation.cpp
  return (T)(((float)x) / (1.0f + sycl::exp((float)-x)));
}

}  // namespace

SYCL_EXTERNAL uint16_t w4a8f_silu_mul_bits(uint16_t gate, uint16_t up) {
  const sycl_bf16 x = sycl::bit_cast<sycl_bf16>(gate);
  const sycl_bf16 y = sycl::bit_cast<sycl_bf16>(up);
  const sycl_bf16 out = silu_kernel(x) * y;
  return sycl::bit_cast<uint16_t>(out);
}
