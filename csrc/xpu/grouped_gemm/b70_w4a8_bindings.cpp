// Standalone torch extension for the B70 W4A8 grouped GEMM, so it can be
// loaded next to an installed vllm_xpu_kernels wheel with
// torch.ops.load_library() without re-registering any _xpu_C op.
// Built only with -DB70_W4A8_STANDALONE=ON (target _b70_w4a8_C).
#include "core/registration.h"
#include "xpu/grouped_gemm/xe_2_w4a8/grouped_gemm_w4a8_xe2.h"

#include <torch/library.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def(
      "cutlass_grouped_gemm_w4a8(Tensor ptr_A, Tensor ptr_A_scale, "
      "Tensor ptr_B, Tensor ptr_B_scale, Tensor? ptr_bias, Tensor ptr_D, "
      "Tensor rows_per_expert, int N, int K, int num_experts) -> Tensor");
  m.impl(
      "cutlass_grouped_gemm_w4a8",
      torch::kXPU,
      &cutlass_grouped_gemm_w4a8_xe2);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
