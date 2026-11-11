#include <torch/all.h>

torch::Tensor cutlass_grouped_gemm_xe2(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    int64_t total_experts = -1,
    const c10::optional<std::string>& tile = c10::nullopt);

// Name of the W4A16 / MXFP4 tile of this thread's last dispatch (tests).
std::string cutlass_grouped_gemm_xe2_last_tile();
