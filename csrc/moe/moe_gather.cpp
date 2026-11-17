#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"
#include "sycl_launch.h"
#include "utils/dpcpp.h"

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
// GATE_LOGITS (tier 2, different bits): shared_gate holds the expert gate's
// logits and the gate is T(1 / (1 + exp(-logit))) in fp32 with the device's
// exp, rounded to T as torch's sigmoid output is; the separate sigmoid
// kernel goes away. The gate can differ from torch's by one T ulp where the
// two fp32 sigmoids round to different sides of a T boundary.
// The shared row is read element by element (its row stride need not keep
// the vector alignment of output / moe_output); the routed sum and the store
// are the stock kernel's.
template <
    typename T,
    int TOPK,
    int ElemsPerItem,
    bool SHARED = false,
    bool SINGLE_ROUND = false,
    bool GATE_LOGITS = false>
class MoeGather {
 public:
  using vec_t = vllm::xpu::aligned_vec<T, ElemsPerItem>;

  MoeGather(
      T* output,
      const T* moe_output,
      const float* topk_weights,
      const int* unpermuted_row_to_permuted_row,
      const int num_experts,
      const int num_tokens,
      const int hidden_size,
      const int lanes_per_token,
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
        lanes_per_token(lanes_per_token),
        shared(shared),
        shared_gate(shared_gate),
        shared_stride(shared_stride),
        gate_stride(gate_stride) {}

  static constexpr int MaxGroupWorkItem = 1024;
  static constexpr int WARP_SIZE = 16;

  // The compiler may give register-heavy instances (e.g. TOPK=16 with 16-bit
  // types on Xe3P) a large GRF, which lowers their work-group limit below
  // MaxGroupWorkItem.
  static inline int max_group_work_item() {
    static const int limit = static_cast<int>(std::min<int64_t>(
        MaxGroupWorkItem, vllm::xpu::getMaxWorkGroupSize<MoeGather>()));
    return limit;
  }

  static inline int get_lanes_per_token(const int hidden_size) {
    const int vecs = hidden_size / ElemsPerItem;
    const int max_wg = max_group_work_item();
    return vecs < max_wg ? vecs : max_wg;
  }

  static inline sycl::nd_range<1>
  get_nd_range(const int num_tokens, const int lanes_per_token) {
    const int tokens_per_wg = max_group_work_item() / lanes_per_token;
    const int num_wgs = (num_tokens + tokens_per_wg - 1) / tokens_per_wg;
    sycl::range<1> local(lanes_per_token * tokens_per_wg);
    sycl::range<1> group(num_wgs);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    const int local_id_x = item.get_local_id(0);
    const int tokens_per_wg = item.get_local_range(0) / lanes_per_token;
    const int token_idx =
        item.get_group(0) * tokens_per_wg + local_id_x / lanes_per_token;
    if (token_idx >= num_tokens) {
      return;
    }

    int moe_ids[TOPK];
    float scores[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      moe_ids[k] = unpermuted_row_to_permuted_row[token_idx * TOPK + k];
      scores[k] = topk_weights[token_idx * TOPK + k];
    }

    bool all_valid = true;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      all_valid &= moe_ids[k] != -1;
    }

    // Compiler batches top-k loads in branch-free path.
    if (all_valid) {
      gather_token<true>(token_idx, local_id_x, moe_ids, scores);
    } else {
      gather_token<false>(token_idx, local_id_x, moe_ids, scores);
    }
  }

 private:
  template <bool IsAllValid>
  inline void gather_token(
      const int token_idx,
      const int local_id_x,
      const int (&moe_ids)[TOPK],
      const float (&scores)[TOPK]) const {
    const int num_vecs = hidden_size / ElemsPerItem;
    float gate = 0.0f;
    if constexpr (SHARED) {
      gate = static_cast<float>(
          shared_gate[static_cast<int64_t>(token_idx) * gate_stride]);
      if constexpr (GATE_LOGITS) {
        gate = static_cast<float>(
            static_cast<T>(1.0f / (1.0f + sycl::exp(-gate))));
      }
    }
    for (int v = local_id_x % lanes_per_token; v < num_vecs;
         v += lanes_per_token) {
      const int64_t hidden_idx = static_cast<int64_t>(v) * ElemsPerItem;

      vec_t in[TOPK];
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        if (IsAllValid || moe_ids[k] != -1) {
          in[k] = *reinterpret_cast<const vec_t*>(
              moe_output + static_cast<int64_t>(moe_ids[k]) * hidden_size +
              hidden_idx);
        }
      }

      float accum[ElemsPerItem];
#pragma unroll
      for (int e = 0; e < ElemsPerItem; ++e) {
        accum[e] = 0.0f;
      }
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        if (!IsAllValid && moe_ids[k] == -1) {
          continue;
        }
#pragma unroll
        for (int e = 0; e < ElemsPerItem; ++e) {
          accum[e] += static_cast<float>(in[k][e]) * scores[k];
        }
      }

      vec_t out;
      if constexpr (SHARED) {
        const T* srow =
            shared + static_cast<int64_t>(token_idx) * shared_stride +
            hidden_idx;
#pragma unroll
        for (int e = 0; e < ElemsPerItem; ++e) {
          if constexpr (SINGLE_ROUND) {
            out[e] =
                static_cast<T>(accum[e] + gate * static_cast<float>(srow[e]));
          } else {
            const T routed = static_cast<T>(accum[e]);
            const T gated = static_cast<T>(gate * static_cast<float>(srow[e]));
            out[e] = static_cast<T>(
                static_cast<float>(routed) + static_cast<float>(gated));
          }
        }
      } else {
#pragma unroll
        for (int e = 0; e < ElemsPerItem; ++e) {
          out[e] = static_cast<T>(accum[e]);
        }
      }
      *reinterpret_cast<vec_t*>(
          output + static_cast<int64_t>(token_idx) * hidden_size + hidden_idx) =
          out;
    }
  }

  T* output;
  const T* moe_output;
  const float* topk_weights;
  const int* unpermuted_row_to_permuted_row;
  const int num_experts;
  const int num_tokens;
  const int hidden_size;
  const int lanes_per_token;
  const T* shared;
  const T* shared_gate;
  const int64_t shared_stride;
  const int64_t gate_stride;
};

template <
    typename T,
    int TOPK,
    int ElemsPerItem,
    bool SHARED,
    bool SINGLE,
    bool GATE_LOGITS = false>
inline void submit_moe_gather(
    sycl::queue& queue,
    T* output,
    const T* moe_output,
    const float* topk_weights,
    const int* unpermuted_row_to_permuted_row,
    const int num_experts,
    const int num_tokens,
    const int hidden_size,
    const T* shared,
    const T* shared_gate,
    const int64_t shared_stride,
    const int64_t gate_stride) {
  using K = MoeGather<T, TOPK, ElemsPerItem, SHARED, SINGLE, GATE_LOGITS>;
  const int lanes_per_token = K::get_lanes_per_token(hidden_size);
  vllm::xpu::sycl_kernel_submit(
      queue,
      K::get_nd_range(num_tokens, lanes_per_token),
      K{output,
        moe_output,
        topk_weights,
        unpermuted_row_to_permuted_row,
        num_experts,
        num_tokens,
        hidden_size,
        lanes_per_token,
        shared,
        shared_gate,
        shared_stride,
        gate_stride});
}

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
    const bool single_round = false,
    const bool gate_logits = false) {
  if (num_tokens == 0 || hidden_size == 0) {
    return;
  }
  // Start at 16-byte vectors; halve the width until hidden_size is a multiple
  // of it and output / moe_output are aligned to it.
  int elems_per_item = sizeof(float) * 4 / sizeof(T);
  while (
      hidden_size % elems_per_item != 0 ||
      reinterpret_cast<uintptr_t>(output) % (elems_per_item * sizeof(T)) != 0 ||
      reinterpret_cast<uintptr_t>(moe_output) % (elems_per_item * sizeof(T)) !=
          0) {
    elems_per_item /= 2;
  }
  // The stock kernel (SHARED = false) is upstream's; the shared variants are
  // separate instantiations of the same body.
#define GATHER_ARGS                                                    \
  queue, output, moe_output, topk_weights,                             \
      unpermuted_row_to_permuted_row, num_experts, num_tokens,         \
      hidden_size, shared, shared_gate, shared_stride, gate_stride
#define CASE_TOPK(TOPK, ElemsPerItem)                                       \
  case TOPK:                                                                \
    if (shared != nullptr && single_round && gate_logits) {                 \
      submit_moe_gather<T, TOPK, ElemsPerItem, true, true, true>(           \
          GATHER_ARGS);                                                     \
    } else if (shared != nullptr && single_round) {                         \
      submit_moe_gather<T, TOPK, ElemsPerItem, true, true>(GATHER_ARGS);    \
    } else if (shared != nullptr && gate_logits) {                          \
      submit_moe_gather<T, TOPK, ElemsPerItem, true, false, true>(          \
          GATHER_ARGS);                                                     \
    } else if (shared != nullptr) {                                         \
      submit_moe_gather<T, TOPK, ElemsPerItem, true, false>(GATHER_ARGS);   \
    } else {                                                                \
      submit_moe_gather<T, TOPK, ElemsPerItem, false, false>(GATHER_ARGS);  \
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
#undef GATHER_ARGS
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
    const bool single_round,
    const bool gate_logits) {
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
  TORCH_CHECK(
      !gate_logits || shared.has_value(),
      "moe_gather: gate_logits needs the shared term");
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

#define LAUNCH_MOE_GATHER(T)                                              \
  vllm::moe::MoeGatherLauncher<T>(                                        \
      reinterpret_cast<T*>(output.data_ptr()),                            \
      reinterpret_cast<T*>(moe_output.data_ptr()),                        \
      reinterpret_cast<float*>(topk_weights.data_ptr()),                  \
      reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()),  \
      num_experts,                                                        \
      num_tokens,                                                         \
      topk,                                                               \
      hidden_size,                                                        \
      queue,                                                              \
      shared.has_value() ? reinterpret_cast<const T*>(shared->data_ptr()) \
                         : nullptr,                                       \
      shared_gate.has_value()                                             \
          ? reinterpret_cast<const T*>(shared_gate->data_ptr())           \
          : nullptr,                                                      \
      shared_stride,                                                      \
      gate_stride,                                                        \
      single_round,                                                       \
      gate_logits);

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
