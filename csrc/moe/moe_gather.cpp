#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

// SHARED: the shared expert's gated output is added in the same pass, with
// each rounding point of the unfused sequence kept:
//   out = T(float(T(acc)) + float(T(float(gate) * float(shared))))
// T(acc) is the stock store; T(gate * shared) is torch's T multiply of the
// gate column and the shared expert's output (a product of two bf16 values
// is exact in fp32, so only its rounding to T remains); the sum is torch's T
// add of the routed and the gated shared output. A row with no computed
// slot gathers zeros and still gets the shared term, as the add did.
// SINGLE_ROUND (tier 2, different bits): out = T(acc + gate * shared) in
// fp32 with one rounding to T; the two intermediate T roundings are dropped.
template <
    typename T,
    int TOPK,
    int ElemsPerItem,
    bool SHARED = false,
    bool SINGLE_ROUND = false>
class MoeGather {
 public:
  MoeGather(
      T* output,
      const T* moe_output,
      const float* topk_weights,
      const int* unpermuted_row_to_permuted_row,
      const int num_experts,
      const int num_tokens,
      const int hidden_size,
      const T* shared = nullptr,
      const T* shared_gate = nullptr,
      const int64_t shared_stride = 0,
      const int64_t gate_stride = 0)
      : output(output),
        moe_output(moe_output),
        topk_weights(topk_weights),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        num_experts(num_experts),
        num_tokens(num_tokens),
        hidden_size(hidden_size),
        shared(shared),
        shared_gate(shared_gate),
        shared_stride(shared_stride),
        gate_stride(gate_stride) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 32;
  static constexpr int Stride = GroupWorkItem * ElemsPerItem;

  static inline sycl::nd_range<1> get_nd_range(const int num_tokens) {
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(num_tokens);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto group_id_x = item.get_group(0);

    const int token_idx = group_id_x;

    int moe_ids[TOPK];
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
      moe_ids[i] = unpermuted_row_to_permuted_row[token_idx * TOPK + i];
    }

    float scores[TOPK];
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
      scores[i] = topk_weights[token_idx * TOPK + i];
    }

    const int loop_count = (hidden_size + Stride - 1) / Stride;

    for (int i = 0; i < loop_count; ++i) {
      const int hidden_idx = i * Stride + local_id_x * ElemsPerItem;
      if (hidden_idx < hidden_size) {
        float accum[ElemsPerItem];
#pragma unroll
        for (int e = 0; e < ElemsPerItem; ++e) {
          accum[e] = 0.0f;
        }
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          if (moe_ids[k] == -1) {
            // skip invalid index for EP
            continue;
          }
#pragma unroll
          for (int e = 0; e < ElemsPerItem; ++e) {
            accum[e] +=
                static_cast<float>(
                    moe_output[moe_ids[k] * hidden_size + hidden_idx + e]) *
                scores[k];
          }
        }
        if constexpr (SHARED) {
          const float gate =
              static_cast<float>(shared_gate[token_idx * gate_stride]);
          const T* srow = shared + token_idx * shared_stride + hidden_idx;
#pragma unroll
          for (int e = 0; e < ElemsPerItem; ++e) {
            if constexpr (SINGLE_ROUND) {
              output[token_idx * hidden_size + hidden_idx + e] =
                  static_cast<T>(accum[e] + gate * static_cast<float>(srow[e]));
              continue;
            }
            const T routed = static_cast<T>(accum[e]);
            const T gated = static_cast<T>(gate * static_cast<float>(srow[e]));
            output[token_idx * hidden_size + hidden_idx + e] = static_cast<T>(
                static_cast<float>(routed) + static_cast<float>(gated));
          }
        } else {
#pragma unroll
          for (int e = 0; e < ElemsPerItem; ++e) {
            output[token_idx * hidden_size + hidden_idx + e] = accum[e];
          }
        }
      }
    }
  }

 private:
  T* output;
  const T* moe_output;
  const float* topk_weights;
  const int* unpermuted_row_to_permuted_row;
  const int num_experts;
  const int num_tokens;
  const int hidden_size;
  const T* shared;
  const T* shared_gate;
  const int64_t shared_stride;
  const int64_t gate_stride;
};

template <typename T>
void MoeGatherLauncher(
    T* output,
    const T* moe_output,
    const float* topk_weights,
    const int* unpermuted_row_to_permuted_row,
    const int num_experts,
    const int num_tokens,
    const int topk,
    const int hidden_size,
    sycl::queue& queue,
    const T* shared = nullptr,
    const T* shared_gate = nullptr,
    const int64_t shared_stride = 0,
    const int64_t gate_stride = 0,
    const bool single_round = false) {
  int elems_per_item = sizeof(float) * 4 / sizeof(T);
  while (hidden_size % elems_per_item != 0) {
    elems_per_item /= 2;
  }
  // The stock kernel (SHARED = false) is the code it was; the shared variant
  // is a separate instantiation.
#define CASE_TOPK(TOPK, ElemsPerItem)                                       \
  case TOPK:                                                                \
    if (shared != nullptr && single_round) {                                \
      queue.submit([&](sycl::handler& cgh) {                                \
        cgh.parallel_for(                                                   \
            MoeGather<T, TOPK, ElemsPerItem, true, true>::get_nd_range(     \
                num_tokens),                                                \
            MoeGather<T, TOPK, ElemsPerItem, true, true>{                   \
                output,                                                     \
                moe_output,                                                 \
                topk_weights,                                               \
                unpermuted_row_to_permuted_row,                             \
                num_experts,                                                \
                num_tokens,                                                 \
                hidden_size,                                                \
                shared,                                                     \
                shared_gate,                                                \
                shared_stride,                                              \
                gate_stride});                                              \
      });                                                                   \
    } else if (shared != nullptr) {                                         \
      queue.submit([&](sycl::handler& cgh) {                                \
        cgh.parallel_for(                                                   \
            MoeGather<T, TOPK, ElemsPerItem, true>::get_nd_range(           \
                num_tokens),                                                \
            MoeGather<T, TOPK, ElemsPerItem, true>{                         \
                output,                                                     \
                moe_output,                                                 \
                topk_weights,                                               \
                unpermuted_row_to_permuted_row,                             \
                num_experts,                                                \
                num_tokens,                                                 \
                hidden_size,                                                \
                shared,                                                     \
                shared_gate,                                                \
                shared_stride,                                              \
                gate_stride});                                              \
      });                                                                   \
    } else {                                                                \
      queue.submit([&](sycl::handler& cgh) {                                \
        cgh.parallel_for(                                                   \
            MoeGather<T, TOPK, ElemsPerItem>::get_nd_range(num_tokens),     \
            MoeGather<T, TOPK, ElemsPerItem>{                               \
                output,                                                     \
                moe_output,                                                 \
                topk_weights,                                               \
                unpermuted_row_to_permuted_row,                             \
                num_experts,                                                \
                num_tokens,                                                 \
                hidden_size});                                              \
      });                                                                   \
    }                                                                       \
    break;

#define CASE_ElemsPerItem(TOPK, ElemsPerItem)                                  \
  case ElemsPerItem:                                                           \
    switch (TOPK) {                                                            \
      CASE_TOPK(1, ElemsPerItem)                                               \
      CASE_TOPK(2, ElemsPerItem)                                               \
      CASE_TOPK(3, ElemsPerItem)                                               \
      CASE_TOPK(4, ElemsPerItem)                                               \
      CASE_TOPK(6, ElemsPerItem)                                               \
      CASE_TOPK(7, ElemsPerItem)                                               \
      CASE_TOPK(8, ElemsPerItem)                                               \
      CASE_TOPK(10, ElemsPerItem)                                              \
      CASE_TOPK(12, ElemsPerItem)                                              \
      CASE_TOPK(16, ElemsPerItem)                                              \
      default:                                                                 \
        TORCH_CHECK(false, "error: not support TOPK=" + std::to_string(TOPK)); \
    }                                                                          \
    break;

  switch (elems_per_item) {
    CASE_ElemsPerItem(topk, 1) CASE_ElemsPerItem(topk, 2)
        CASE_ElemsPerItem(topk, 4) CASE_ElemsPerItem(topk, 8) default
        : TORCH_CHECK(
              false,
              "error: not support elems_per_item=" +
                  std::to_string(elems_per_item));
  }
#undef CASE_ElemsPerItem
#undef CASE_TOPK
}

}  // namespace moe
}  // namespace vllm

void moe_gather(
    torch::Tensor& output,              // [num_tokens, hidden_size]
    const torch::Tensor& moe_output,    // [num_tokens * topk, hidden_size]
    const torch::Tensor& topk_weights,  // [num_tokens, topk]
    const torch::Tensor& unpermuted_row_to_permuted_row,  // [num_tokens * topk]
    const int64_t num_experts,
    const std::optional<torch::Tensor>& shared,       // [num_tokens, hidden]
    const std::optional<torch::Tensor>& shared_gate,  // [num_tokens, 1]
    const bool single_round) {
  // Implementation of the gather operation
  const int num_tokens = topk_weights.size(0);
  const int topk = topk_weights.size(1);
  const int hidden_size = output.size(1);

  TORCH_CHECK(
      topk_weights.scalar_type() == torch::kFloat32,
      "topk_weights must be float32");
  TORCH_CHECK(
      shared.has_value() == shared_gate.has_value(),
      "moe_gather: shared and shared_gate come together");
  int64_t shared_stride = 0, gate_stride = 0;
  if (shared.has_value()) {
    TORCH_CHECK(
        output.scalar_type() == torch::kBFloat16 ||
            output.scalar_type() == torch::kFloat16,
        "moe_gather: the shared term needs a bf16 or fp16 output");
    TORCH_CHECK(
        shared->scalar_type() == output.scalar_type() &&
            shared_gate->scalar_type() == output.scalar_type(),
        "moe_gather: shared and shared_gate must have the output's dtype");
    TORCH_CHECK(
        shared->dim() == 2 && shared->size(0) == num_tokens &&
            shared->size(1) == hidden_size && shared->stride(1) == 1,
        "moe_gather: shared must be [num_tokens, hidden], hidden contiguous");
    TORCH_CHECK(
        shared_gate->numel() == num_tokens &&
            (shared_gate->dim() == 1 ||
             (shared_gate->dim() == 2 && shared_gate->size(1) == 1)),
        "moe_gather: shared_gate must be [num_tokens, 1]");
    TORCH_CHECK(
        output.is_contiguous(), "moe_gather: output must be contiguous");
    TORCH_CHECK(
        shared->device() == output.device() &&
            shared_gate->device() == output.device(),
        "moe_gather: shared tensors on the output's device");
    shared_stride = shared->stride(0);
    gate_stride = shared_gate->stride(0);
  }

  const at::DeviceGuard device_guard(output.device());
  auto& queue = vllm::xpu::vllmGetQueue();

#define LAUNCH_MOE_GATHER(T)                                             \
  vllm::moe::MoeGatherLauncher<T>(                                       \
      reinterpret_cast<T*>(output.data_ptr()),                           \
      reinterpret_cast<T*>(moe_output.data_ptr()),                       \
      reinterpret_cast<float*>(topk_weights.data_ptr()),                 \
      reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()), \
      num_experts,                                                       \
      num_tokens,                                                        \
      topk,                                                              \
      hidden_size,                                                       \
      queue,                                                             \
      shared.has_value() ? reinterpret_cast<const T*>(shared->data_ptr()) \
                         : nullptr,                                      \
      shared_gate.has_value()                                            \
          ? reinterpret_cast<const T*>(shared_gate->data_ptr())          \
          : nullptr,                                                     \
      shared_stride,                                                     \
      gate_stride,                                                       \
      single_round);

  if (output.scalar_type() == torch::kFloat16) {
    using scalar_t = sycl::half;
    LAUNCH_MOE_GATHER(scalar_t);
  } else if (output.scalar_type() == torch::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    LAUNCH_MOE_GATHER(scalar_t);
  } else if (output.scalar_type() == torch::kFloat32) {
    using scalar_t = float;
    LAUNCH_MOE_GATHER(scalar_t);
  } else {
    throw std::runtime_error("Unsupported data type in moe_gather");
  }
#undef LAUNCH_MOE_GATHER
}