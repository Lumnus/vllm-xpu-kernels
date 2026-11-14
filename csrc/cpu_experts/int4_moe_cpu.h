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
//   arith        the card kernel whose arithmetic each GEMM takes: bit 0
//                GEMM1, bit 1 GEMM2; clear: the int4 GEMV (scale per
//                32-element chunk sum; 0, the default, is the GEMV for
//                both), set: the grouped GEMM (weights bf16(q * s)). The
//                card takes the GEMV for a call of at most 8 rows whose K
//                is a multiple of 512, the grouped GEMM otherwise
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
    int64_t isa,
    int64_t arith);

// The instruction set `isa` (as above) resolves to on this host: 1 scalar,
// 2 AVX2 + FMA.
int64_t int4_moe_cpu_isa(int64_t isa);

// Settings of the process's host pools, now and for pools made later:
// worker i of a pool of n threads runs on cores[i % len] (i = 1..n-1; the
// calling thread is not moved, cores[0] is meant for it); an empty list
// unpins. spin_us: how long a worker spins after a phase before it sleeps
// (-1: the default, 100). Returns the number of cores. Raises when a worker
// cannot be pinned (a core outside the process's CPU mask).
int64_t int4_moe_cpu_configure(c10::IntArrayRef cores, int64_t spin_us);

namespace cpu_moe {
class Pool;
}  // namespace cpu_moe

// The process's host pool of `threads` threads, made on first use with the
// settings above; every host op of this library (int4_moe_cpu,
// int4_moe_cpu_a8) runs its phases on it.
cpu_moe::Pool& int4_moe_cpu_pool(int64_t threads);

// The schemas, for the extension's library block (and a test harness).
inline void register_int4_moe_cpu(torch::Library& m) {
  m.def(
      "int4_moe_cpu(Tensor(a!) out, Tensor x, Tensor units, Tensor call_unit, "
      "Tensor call_token, Tensor call_weight, int inter, int group, "
      "int[] offsets, int num_threads, int isa, int arith=0) -> ()");
  m.impl("int4_moe_cpu", torch::kCPU, &int4_moe_cpu);
  m.def("int4_moe_cpu_isa(int isa) -> int", &int4_moe_cpu_isa);
  m.def(
      "int4_moe_cpu_configure(int[] cores, int spin_us) -> int",
      &int4_moe_cpu_configure);
}
