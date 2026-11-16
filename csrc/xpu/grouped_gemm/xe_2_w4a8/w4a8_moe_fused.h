// Fused W4A8 MoE layer for decode (Xe2 / BMG): the routed experts of one MoE
// layer in five launches plus the stock moe_gather, against eleven on the
// unfused path (quant, remap clear / count / place, two atomic-counter fills,
// two grouped GEMMs, silu_and_mul, quant, gather).
//
//   1. map          one work-group: topk ids -> local expert (the remap's
//                   rule: negative, out of [0, total) or not owned -> -1),
//                   rows_per_expert, the route -> row map (moe_gather's
//                   unpermuted_row_to_permuted_row) and a per-tile map
//                   (expert, first row, rows) for the GEMMs.
//   2. quant+gather one work-group per token: the per-token int8 quant of
//                   _C.dynamic_per_token_scaled_int8_quant (same header,
//                   same compile flags), written to each of the token's
//                   expert rows with its scale.
//   3. gemm1        grid over (tile, N tile) read from the tile map; the
//                   W4A8 mainloop xe_gemm_w4a8 unchanged; bf16 out.
//   4. silu+quant   one work-group per expert row: silu_and_mul's expression
//                   (its compile model) and the per-token int8 quant.
//   5. gemm2        as 3.
//   then moe_gather (unchanged) with the route -> row map.
//
// Bits: every output element is computed by the same code as on the unfused
// path; only row positions and the GEMM scheduler change, and neither
// enters any value (a GEMM row depends on its own A row and its expert's
// weights; moe_gather reads rows through the map in fixed k order).
#pragma once

#include <torch/all.h>

#include <optional>
#include <string>

// tile_ws: int32 [2 + 3 * umt]: [0] tiles, [1] expert rows, then
// mt_expert[umt], mt_row_base[umt], mt_rows[umt]. umt >= the tiles any
// routing can need: min(R, R / tile_m + min(R, local_experts)).
void moe_w4a8_route_quant_xe2(
    const at::Tensor& x,                      // bf16 [T, H]
    const at::Tensor& topk_ids,               // int32 / int64 [T, top_k]
    const std::optional<at::Tensor>& expert_map,  // int32 [total] or none
    at::Tensor& a_q,                          // int8 [T * top_k, H]
    at::Tensor& a_scale,                      // fp32 [T * top_k]
    at::Tensor& perm_pos,                     // int32 [T, top_k]
    at::Tensor& rows_per_expert,              // int32 [local]
    at::Tensor& tile_ws,                      // int32 [2 + 3 * umt]
    int64_t total_experts,
    int64_t local_experts,
    int64_t tile_m);

// As moe_w4a8_route_quant_xe2, for rows quantized once at the MoE entry by
// _C.dynamic_per_token_scaled_int8_quant (the op 2. reproduces): the map,
// then the int8 rows and their scales copied to the expert rows.
void moe_w4a8_route_gather_q_xe2(
    const at::Tensor& x_q,                    // int8 [T, H]
    const at::Tensor& x_scale,                // fp32 [T] or [T, 1]
    const at::Tensor& topk_ids,               // int32 / int64 [T, top_k]
    const std::optional<at::Tensor>& expert_map,  // int32 [total] or none
    at::Tensor& a_q,                          // int8 [T * top_k, H]
    at::Tensor& a_scale,                      // fp32 [T * top_k]
    at::Tensor& perm_pos,                     // int32 [T, top_k]
    at::Tensor& rows_per_expert,              // int32 [local]
    at::Tensor& tile_ws,                      // int32 [2 + 3 * umt]
    int64_t total_experts,
    int64_t local_experts,
    int64_t tile_m);

void moe_w4a8_tile_gemm_xe2(
    const at::Tensor& a,           // int8 [R, K], rows in tile-map order
    const at::Tensor& a_scale,     // fp32 [R] (or [R, 1])
    const at::Tensor& b,           // packed int4 [E, N, K/2]
    const at::Tensor& b_scale,     // bf16 [E, N, K/128]
    const std::optional<at::Tensor>& bias,  // bf16 [E, N] or none
    at::Tensor& d,                 // bf16 [R, N]
    const at::Tensor& tile_ws,
    int64_t N,
    int64_t K,
    int64_t tile_m);

void moe_w4a8_silu_mul_quant_xe2(
    const at::Tensor& g1,       // bf16 [R, 2 * I]
    at::Tensor& q,              // int8 [R, I]
    at::Tensor& q_scale,        // fp32 [R]
    const at::Tensor& tile_ws);

// silu_and_mul's element (csrc/activation.cpp: silu_kernel then the bf16
// multiply of compute<..., act_first = true>) on bf16 bits. Defined in
// w4a8_moe_fused_silu.cpp, a TU with the compile model of activation.cpp;
// called from the act + quant kernel, whose TU has the quant's model.
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL uint16_t w4a8f_silu_mul_bits(uint16_t gate, uint16_t up);
#endif
