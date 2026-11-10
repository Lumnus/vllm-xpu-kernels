#include <torch/all.h>

torch::Tensor cutlass_grouped_gemm_interface(
    torch::Tensor ptr_A,
    const c10::optional<at::Tensor>& ptr_A_scale,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts);

// B70 W4A8: int8 A [M, K] with fp32 per-row scales x packed int4 B
// [E, N, K/2] with bf16 g128 scales -> bf16 D [M, N] (s8 x s4 DPAS).
torch::Tensor cutlass_grouped_gemm_w4a8(
    torch::Tensor ptr_A,
    torch::Tensor ptr_A_scale,
    torch::Tensor ptr_B,
    torch::Tensor ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts);

// B70 W4A8 fused gemm1: silu(gate) * up of int8 A x int4 w13 [E, 2*inter,
// K/2], written as int8 q_out [M, inter] + fp32 q_scale [M, inter/64].
void cutlass_grouped_gemm_w4a8_silu_quant(
    torch::Tensor ptr_A,
    torch::Tensor ptr_A_scale,
    torch::Tensor ptr_B,
    torch::Tensor ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor q_out,
    torch::Tensor q_scale,
    torch::Tensor rows_per_expert,
    int64_t inter,
    int64_t K,
    int64_t num_experts);
