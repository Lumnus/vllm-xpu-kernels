#pragma once

#include <torch/all.h>

// Int4 MoE experts with int8 activations (W4A8) computed on the host (see
// int4_moe_cpu_a8_impl.h). The call contract is int4_moe_cpu's: for each
// call c, out[call_token[c]] gets call_weight[c] times expert
// units[call_unit[c]] applied to x[call_token[c]], summed per token in call
// order; calls in token order; out overwritten (zero where a token has no
// call). The op quantizes the activations itself (x stays bf16).
//
//   out, x       [T, H] bf16, CPU, contiguous
//   units        [S, U] uint8: one expert per row in the kernels' layout at
//                the byte offsets `offsets` (w13, w2, w13 scales, w2 scales)
//   call_unit    [C] int32, call_token [C] int32, call_weight [C] float32
//   inter        I; group: the weight scales' group size
//   scheme       [] for the B70 card's W4A8 path, or [a1_rule, a1_group,
//                a2_rule, a2_group, acc, glu_fp32] (int4_moe_cpu_a8_impl.h)
//   num_threads  host threads for this call: int4_moe_cpu's pool of that
//                count (pinned and spinning as int4_moe_cpu_configure set)
//   isa          0 auto, 1 scalar, 2 AVX2 + FMA (int4_moe_cpu_isa resolves)
void int4_moe_cpu_a8(
    torch::Tensor& out,
    const torch::Tensor& x,
    const torch::Tensor& units,
    const torch::Tensor& call_unit,
    const torch::Tensor& call_token,
    const torch::Tensor& call_weight,
    int64_t inter,
    int64_t group,
    c10::IntArrayRef offsets,
    c10::IntArrayRef scheme,
    int64_t num_threads,
    int64_t isa);

// The schema, for the extension's library block (and a test harness).
inline void register_int4_moe_cpu_a8(torch::Library& m) {
  m.def(
      "int4_moe_cpu_a8(Tensor(a!) out, Tensor x, Tensor units, "
      "Tensor call_unit, Tensor call_token, Tensor call_weight, int inter, "
      "int group, int[] offsets, int[] scheme, int num_threads, int isa) -> "
      "()");
  m.impl("int4_moe_cpu_a8", torch::kCPU, &int4_moe_cpu_a8);
}
