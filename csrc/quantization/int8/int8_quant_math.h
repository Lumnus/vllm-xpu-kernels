// SPDX-License-Identifier: Apache-2.0
// B70 (Lumnus): scalar math of the dynamic per-token symmetric int8 quant.
//
// Bit-for-bit mirror of vllm._xpu_ops.xpu_ops.dynamic_per_token_int8_quant_ref
// (use_sym_quant=True, bits=8):
//   absmax = max(|min(row)|, |max(row)|)                 (exact in fp32)
//   scale  = clamp(absmax / 127, min=1e-5)               (fp32, IEEE div)
//   scale  = scale.to(input.dtype)                       (RNE to bf16/fp16)
//   q      = clamp(round_half_even(x_fp32 / scale_fp32), -128, 127)
// Header-only and free of SYCL so the same code is compiled into the device
// kernel and into the host-side CPU reference test.
#pragma once

#include <cstdint>

// icpx defaults to -ffp-model=fast; a reciprocal-multiply or a contracted
// op here would break the exact match with the reference.
#pragma float_control(precise, on, push)

namespace vllm {
namespace int8q {

template <typename scalar_t>
inline float token_scale_from_absmax(float absmax) {
  float s = absmax / 127.0f;
  // torch.clamp(min=1e-5): NaN stays NaN.
  s = (s < 1e-5f) ? 1e-5f : s;
  // scale.to(input.dtype): round to nearest even in the activation dtype.
  return static_cast<float>(static_cast<scalar_t>(s));
}

inline int8_t quantize_one(float x, float scale) {
  float q = __builtin_rintf(x / scale);  // torch.round = half to even
  q = q < -128.0f ? -128.0f : q;
  q = q > 127.0f ? 127.0f : q;
  return static_cast<int8_t>(q);
}

inline float abs_f(float x) { return __builtin_fabsf(x); }
inline float max_f(float a, float b) { return a > b ? a : b; }

}  // namespace int8q
}  // namespace vllm

#pragma float_control(pop)
