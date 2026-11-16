#pragma once
#include <torch/all.h>

#include <string>

// int8 A [M, K] (per-row fp32 scale [M]) x packed int4 B [E, N, K/2]
// (bf16 scales [E, N, K/128]) -> bf16 D [M, N]; rows of A grouped by expert.
// tile: an explicit work-group tile ("32x128s8", ...), else
// VLLM_XPU_W4A8_TILE, else the rows-per-expert rule (grouped_gemm_w4a8_xe2.cpp).
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
    int64_t num_experts,
    const c10::optional<std::string>& tile = c10::nullopt);

// The tile name of this thread's last plain W4A8 GEMM dispatch.
std::string cutlass_grouped_gemm_w4a8_last_tile_xe2();

// Fused gemm1 of the W4A8 MoE: int8 A [M, K] (fp32 per-row scale) x w13
// [E, 2*inter, K/2] (gate rows, then up rows), epilogue silu(gate) * up
// quantized to int8 q_out [M, inter] with fp32 scales q_scale [M, inter/128]
// (one per row and 128-column group): gemm2's A, fed to
// cutlass_grouped_gemm_w4a8 with A_scale = q_scale.
void cutlass_grouped_gemm_w4a8_silu_quant_xe2(
    at::Tensor ptr_A,
    at::Tensor ptr_A_scale,
    at::Tensor ptr_B,
    at::Tensor ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor q_out,
    at::Tensor q_scale,
    at::Tensor rows_per_expert,
    int64_t inter,
    int64_t K,
    int64_t num_experts);

// Dense W8A8, tuned for small M (any M runs): int8 xq [M, K] with fp32 xs [M] x int8 W [N, K]
// with fp32 ws [N] -> bf16 y [M, N] (+ bias [N]). Weights stream once (the WG tile spans all tokens up
// to 64); narrow N splits K over WGs with exact int32 partials, so a row's output depends only on that
// row (deterministic, batch-invariant). Needs K % 32 == 0, N % 16 == 0, N >= 32.
void w8a8_dense_xe2(
    const at::Tensor& xq,
    const at::Tensor& xs,
    const at::Tensor& w,
    const at::Tensor& ws,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& y);
