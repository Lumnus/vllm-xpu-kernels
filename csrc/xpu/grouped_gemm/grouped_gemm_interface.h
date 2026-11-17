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
    int64_t num_experts,
    int64_t total_experts,
    const c10::optional<std::string>& tile);

std::string _cutlass_grouped_gemm_last_w4a16_tile();

// W4A8: int8 A [M, K] with fp32 per-row scales x packed int4 B
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
    int64_t num_experts,
    const c10::optional<std::string>& tile);

std::string _cutlass_grouped_gemm_last_w4a8_tile();

// Fused W4A8 MoE decode (xe_2_w4a8/w4a8_moe_fused.h): the routing map with
// the per-token int8 quant gathered to expert rows; the grouped GEMM over the
// map's tiles; silu_and_mul with the per-token int8 quant of its output.
void moe_w4a8_route_quant(
    torch::Tensor x,
    torch::Tensor topk_ids,
    const c10::optional<at::Tensor>& expert_map,
    torch::Tensor a_q,
    torch::Tensor a_scale,
    torch::Tensor perm_pos,
    torch::Tensor rows_per_expert,
    torch::Tensor tile_ws,
    int64_t total_experts,
    int64_t local_experts,
    int64_t tile_m);

void moe_w4a8_route_gather_q(
    torch::Tensor x_q,
    torch::Tensor x_scale,
    torch::Tensor topk_ids,
    const c10::optional<at::Tensor>& expert_map,
    torch::Tensor a_q,
    torch::Tensor a_scale,
    torch::Tensor perm_pos,
    torch::Tensor rows_per_expert,
    torch::Tensor tile_ws,
    int64_t total_experts,
    int64_t local_experts,
    int64_t tile_m);

void moe_w4a8_tile_gemm(
    torch::Tensor a,
    torch::Tensor a_scale,
    torch::Tensor b,
    torch::Tensor b_scale,
    const c10::optional<at::Tensor>& bias,
    torch::Tensor d,
    torch::Tensor tile_ws,
    int64_t N,
    int64_t K,
    int64_t tile_m);

void moe_w4a8_silu_mul_quant(
    torch::Tensor g1,
    torch::Tensor q_out,
    torch::Tensor q_scale,
    torch::Tensor tile_ws);

// W4A8 fused gemm1: silu(gate) * up of int8 A x int4 w13 [E, 2*inter,
// K/2], written as int8 q_out [M, inter] + fp32 q_scale [M, inter/128].
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
