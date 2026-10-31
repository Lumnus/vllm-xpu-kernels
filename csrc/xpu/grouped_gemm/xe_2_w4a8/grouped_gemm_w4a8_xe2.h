#pragma once
#include <torch/all.h>

// int8 A [M, K] (per-row fp32 scale [M]) x packed int4 B [E, N, K/2]
// (bf16 scales [E, N, K/128]) -> bf16 D [M, N]; rows of A grouped by expert.
at::Tensor cutlass_grouped_gemm_w4a8_xe2(
    at::Tensor ptr_A,
    at::Tensor ptr_A_scale,
    at::Tensor ptr_B,
    at::Tensor ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor ptr_D,
    at::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts);
