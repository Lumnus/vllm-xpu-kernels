#pragma once

#include <torch/all.h>

#include <vector>

// Int4 MoE experts computed on the host (see int4_moe_cpu_impl.h): for each
// call c, out[call_token[c]] gets call_weight[c] times expert
// units[call_unit[c]] applied to x[call_token[c]], summed per token in call
// order. Calls must be in token order. out is overwritten (zero where a
// token has no call).
//
//   out          [T, H] bf16, CPU, contiguous
//   x            [T, H] bf16, CPU, contiguous
//   units        [S, U] uint8, CPU, contiguous: one expert per row, in the
//                kernels' layout at the byte offsets `offsets` (w13, w2,
//                w13 scales, w2 scales)
//   call_unit    [C] int32, call_token [C] int32, call_weight [C] float32
//   inter        I (H is x's width), group: the scales' group size
//   num_threads  host threads for this call (a pool per count is kept)
//   isa          0 auto, 1 scalar, 2 AVX2 + FMA
void int4_moe_cpu(
    torch::Tensor& out,
    const torch::Tensor& x,
    const torch::Tensor& units,
    const torch::Tensor& call_unit,
    const torch::Tensor& call_token,
    const torch::Tensor& call_weight,
    int64_t inter,
    int64_t group,
    c10::IntArrayRef offsets,
    int64_t num_threads,
    int64_t isa);

// The instruction set `isa` (as above) resolves to on this host: 1 scalar,
// 2 AVX2 + FMA.
int64_t int4_moe_cpu_isa(int64_t isa);

// The schemas, for the extension's library block (and a test harness).
inline void register_int4_moe_cpu(torch::Library& m) {
  m.def(
      "int4_moe_cpu(Tensor(a!) out, Tensor x, Tensor units, Tensor call_unit, "
      "Tensor call_token, Tensor call_weight, int inter, int group, "
      "int[] offsets, int num_threads, int isa) -> ()");
  m.impl("int4_moe_cpu", torch::kCPU, &int4_moe_cpu);
  m.def("int4_moe_cpu_isa(int isa) -> int", &int4_moe_cpu_isa);
}
