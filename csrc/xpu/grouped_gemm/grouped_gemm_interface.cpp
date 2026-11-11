#include "csrc/utils.h"
#include "grouped_gemm_interface.h"
#include <stdio.h>

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/grouped_gemm_xe2.h"
  #include "xe_2_w4a8/grouped_gemm_w4a8_xe2.h"
#endif
#ifdef VLLM_XPU_ENABLE_XE3P
  #include "xe_3/grouped_gemm_xe3.h"
#endif

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
    const c10::optional<std::string>& tile) {
  const bool force_xe3p = vllm::xpu::force_xe3p_kernel();
#ifdef VLLM_XPU_ENABLE_XE3P
  if (force_xe3p || vllm::xpu::is_xe3p_arch()) {
    // Xe3P consumes block-scaled activations (MXFP8 / MXFP4 / block-FP8)
    // natively, so ptr_A_scale is forwarded to the device kernel.
    return cutlass_grouped_gemm_xe3(
        ptr_A,
        ptr_A_scale,
        ptr_B,
        ptr_B_scale,
        ptr_bias,
        ptr_D,
        rows_per_expert,
        N,
        K,
        num_experts);
  }
#else
  TORCH_CHECK(
      !force_xe3p,
      "FORCE_XE3P_KERNEL is set, but XE3P grouped GEMM is not enabled in "
      "this build.");
#endif
#ifdef VLLM_XPU_ENABLE_XE2
  // BMG / PVC / LNL and the Xe3 client parts run the xe_2 kernel, which
  // consumes high-precision activations only.
  (void)ptr_A_scale;
  return cutlass_grouped_gemm_xe2(
      ptr_A,
      ptr_B,
      ptr_B_scale,
      ptr_bias,
      ptr_D,
      rows_per_expert,
      N,
      K,
      num_experts,
      total_experts,
      tile);
#else
  (void)total_experts;
  (void)tile;
  TORCH_CHECK(
      false, "No cutlass grouped GEMM kernel is enabled in this build.");
#endif
}

std::string _cutlass_grouped_gemm_last_w4a16_tile() {
#ifdef VLLM_XPU_ENABLE_XE2
  return cutlass_grouped_gemm_xe2_last_tile();
#else
  return "";
#endif
}

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
    int64_t num_experts) {
#ifdef VLLM_XPU_ENABLE_XE2
  TORCH_CHECK(
      vllm::xpu::is_xe2_arch(),
      "cutlass_grouped_gemm_w4a8 needs an Xe2 GPU (s8 x s4 DPAS)");
  return cutlass_grouped_gemm_w4a8_xe2(
      ptr_A,
      ptr_A_scale,
      ptr_B,
      ptr_B_scale,
      ptr_bias,
      ptr_D,
      rows_per_expert,
      N,
      K,
      num_experts);
#else
  TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
}

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
    int64_t num_experts) {
#ifdef VLLM_XPU_ENABLE_XE2
  TORCH_CHECK(
      vllm::xpu::is_xe2_arch(),
      "cutlass_grouped_gemm_w4a8_silu_quant needs an Xe2 GPU");
  cutlass_grouped_gemm_w4a8_silu_quant_xe2(
      ptr_A,
      ptr_A_scale,
      ptr_B,
      ptr_B_scale,
      ptr_bias,
      q_out,
      q_scale,
      rows_per_expert,
      inter,
      K,
      num_experts);
#else
  TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
}
