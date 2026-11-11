#include <vector>
#include "fp4_gemm_w4a4.h"
#include "fp8_gemm_w8a8.h"
#include "fp8_gemm_w8a16.h"
#include "int4_gemm_w4a16.h"
#include "int4_gemm_w4a8.h"
#include "int8_gemm_w8a8.h"

inline bool is_supported_fp8(at::ScalarType t) {
  return (t == at::ScalarType::Float8_e5m2) ||
         (t == at::ScalarType::Float8_e4m3fn);
}

inline bool is_supported_fp4(at::ScalarType t) {
  return t == at::ScalarType::Float4_e2m1fn_x2;
}

// If `out` is given, it is validated against A/B (device, dtype, and shape)
// and returned as-is so the result can be written in place into it;
// otherwise a new output tensor is allocated using `out_dtype` (defaulting
// to fp16).
torch::Tensor check_and_create_output_tensor(
    const torch::Tensor& A,
    const torch::Tensor& B,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& out = std::nullopt) {
  TORCH_CHECK(
      A.dim() == 2 || A.dim() == 3,
      "OneDNN Matmul only support 2D and 3D inputs!\n");
  TORCH_CHECK(
      B.dim() == 2 || B.dim() == 3,
      "OneDNN Matmul only support 2D and 3D weights!\n");
  if (B.dim() == 3) {
    TORCH_CHECK(
        A.dim() == 3,
        "OneDNN Matmul expects 3D input when using batched weights!\n");
    TORCH_CHECK(
        A.size(0) == B.size(0),
        "OneDNN Matmul expects input and weight batches to match, got ",
        A.size(0),
        " and ",
        B.size(0),
        ".");
  }

  if (B.scalar_type() == at::ScalarType::Int) {
    TORCH_CHECK(
        B.strides()[B.dim() - 2] == 1, "Int4 weight must be in NT format!\n");
  }

  std::vector<int64_t> result_shape;

  if (A.dim() == 2) {
    result_shape = {A.size(0), B.size(-1)};
    // src{m, k}, wei{k, n}, bias{n}, dst{m, n}
  } else {
    result_shape = {A.size(0), A.size(1), B.size(-1)};
    // src{b, m, k}, wei{k, n}, bias{n}, dst{b, m, n}
  }

  if (out.has_value()) {
    const auto& out_ = out.value();
    const c10::IntArrayRef expected_shape(result_shape);
    TORCH_CHECK(
        out_.device() == A.device(),
        "OneDNN Matmul expects out on device ",
        A.device(),
        ", got ",
        out_.device(),
        ".");
    TORCH_CHECK(
        out_.sizes() == expected_shape,
        "OneDNN Matmul expects out of shape ",
        expected_shape,
        ", got ",
        out_.sizes(),
        ".");
    TORCH_CHECK(
        !out_dtype.has_value() || out_.scalar_type() == out_dtype.value(),
        "OneDNN Matmul expects out of dtype ",
        out_dtype.value_or(out_.scalar_type()),
        ", got ",
        out_.scalar_type(),
        ".");
    return out_;
  }

  // deal with input shape [m, b, k] stride [k, m * k, 1]
  auto k = A.size(A.dim() - 1);
  auto n = result_shape.back();
  auto res_stride = A.strides().vec();
  for (int i = 0; i < res_stride.size() - 1; i++) {
    res_stride[i] = res_stride[i] / k * n;
  }

  // If out_dtype is not given, use fp16 as default
  const auto out_dtype_ = out_dtype.value_or(torch::kHalf);
  auto options = A.options().dtype(out_dtype_);
  return at::empty_strided(result_shape, res_stride, options);
}

static torch::Tensor fp8_gemm_common(
    const torch::Tensor& A,  // [b, m ,k]
    const torch::Tensor& B,  // [k, n]
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& A_scale_,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_,
    std::optional<torch::Tensor> out) {
  const at::DeviceGuard device_guard(A.device());
  // The weight B may be provided in a transposed (NT) layout, and A supports
  // strided layouts, so both are excluded from the contiguity check.
  TORCH_CHECK(
      !A_scale_.has_value() || A_scale_.value().is_contiguous(),
      "A_scale must be contiguous for fp8 matmul");
  TORCH_CHECK(
      !B_scale_.has_value() || B_scale_.value().is_contiguous(),
      "B_scale must be contiguous for fp8 matmul");
  TORCH_CHECK(
      !bias_.has_value() || bias_.value().is_contiguous(),
      "bias must be contiguous for fp8 matmul");
  // When `out` is given, it is validated and written in place; otherwise a
  // new output tensor is allocated.
  torch::Tensor result = check_and_create_output_tensor(A, B, out_dtype, out);
  auto a_st = A.scalar_type();
  auto b_st = B.scalar_type();
  TORCH_CHECK(
      is_supported_fp8(a_st) && is_supported_fp8(b_st) && a_st == b_st,
      "input and weight must be f8_e5m2 or f8_e4m3fn for fp8 matmul");
  TORCH_CHECK(
      result.scalar_type() == torch::kFloat16 ||
          result.scalar_type() == torch::kBFloat16,
      "output must be float16 or bfloat16 for fp8 matmul");
  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;

  torch::Tensor A_scale = A_scale_.value_or(at::ones({1}, torch::kFloat));
  torch::Tensor B_scale = B_scale_.value_or(at::ones({1}, torch::kFloat));
  oneDNN::dnnl_matmul_w8a8_fp8(result, A, B, is_nt, bias_, A_scale, B_scale);
  return result;
}

torch::Tensor fp8_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& A_scale_,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_) {
  return fp8_gemm_common(
      A, B, out_dtype, A_scale_, B_scale_, bias_, std::nullopt);
}

torch::Tensor fp8_gemm_out(
    torch::Tensor out,
    const torch::Tensor& A,
    const torch::Tensor& B,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& A_scale_,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_) {
  return fp8_gemm_common(
      A, B, out_dtype, A_scale_, B_scale_, bias_, std::move(out));
}

torch::Tensor fp8_bmm(
    const torch::Tensor& A,  // [b, m ,k]
    const torch::Tensor& B,  // [b, k, n]
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& A_scale_,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_) {
  const at::DeviceGuard device_guard(A.device());
  TORCH_CHECK(A.dim() == 3, "fp8_bmm expects A to be a 3D tensor");
  TORCH_CHECK(B.dim() == 3, "fp8_bmm expects B to be a 3D tensor");
  TORCH_CHECK(
      !A_scale_.has_value() || A_scale_.value().is_contiguous(),
      "A_scale must be contiguous for fp8 matmul");
  TORCH_CHECK(
      !B_scale_.has_value() || B_scale_.value().is_contiguous(),
      "B_scale must be contiguous for fp8 matmul");
  torch::Tensor result = check_and_create_output_tensor(A, B, out_dtype);
  auto a_st = A.scalar_type();
  auto b_st = B.scalar_type();
  TORCH_CHECK(
      is_supported_fp8(a_st) && is_supported_fp8(b_st) && a_st == b_st,
      "input and weight must be f8_e5m2 or f8_e4m3fn for fp8 matmul");
  TORCH_CHECK(
      result.scalar_type() == torch::kFloat16 ||
          result.scalar_type() == torch::kBFloat16,
      "output must be float16 or bfloat16 for fp8 matmul");
  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;

  torch::Tensor A_scale = A_scale_.value_or(at::ones({1}, torch::kFloat));
  torch::Tensor B_scale = B_scale_.value_or(at::ones({1}, torch::kFloat));
  oneDNN::dnnl_batch_matmul_w8a8_fp8(
      result, A, B, is_nt, bias_, A_scale, B_scale);
  return result;
}

torch::Tensor fp8_gemm_w8a16(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_) {
  const at::DeviceGuard device_guard(A.device());
  // The weight B may be provided in a transposed (NT) layout, and A supports
  // strided layouts, so both are excluded from the contiguity check.
  TORCH_CHECK(
      !B_scale_.has_value() || B_scale_.value().is_contiguous(),
      "B_scale must be contiguous for fp8 matmul");
  TORCH_CHECK(
      !bias_.has_value() || bias_.value().is_contiguous(),
      "bias must be contiguous for fp8 matmul");
  torch::Tensor result = check_and_create_output_tensor(A, B, A.scalar_type());
  TORCH_CHECK(
      is_supported_fp8(B.scalar_type()),
      "weight must be f8_e5m2 or f8_e4m3fn for fp8 matmul");
  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;

  torch::Tensor B_scale = B_scale_.has_value()
                              ? B_scale_.value()
                              : at::ones({1}, B.options().dtype(A.dtype()));
  oneDNN::dnnl_matmul_w8a16_fp8(result, A, B, is_nt, bias_, B_scale);
  return result;
}

torch::Tensor fp4_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& A_scale,
    const torch::Tensor& B_scale,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(A.device());
  TORCH_CHECK(
      A_scale.is_contiguous(), "A_scale must be contiguous for fp4 matmul");
  TORCH_CHECK(
      B_scale.is_contiguous(), "B_scale must be contiguous for fp4 matmul");
  torch::Tensor result = check_and_create_output_tensor(A, B, out_dtype);
  auto a_st = A.scalar_type();
  auto b_st = B.scalar_type();
  TORCH_CHECK(
      is_supported_fp4(a_st) && is_supported_fp4(b_st) && a_st == b_st,
      "input and weight must be f4_e2m1x2 or f4_e2m1x2 for fp4 matmul");
  TORCH_CHECK(
      result.scalar_type() == torch::kFloat16 ||
          result.scalar_type() == torch::kBFloat16,
      "output must be float16 or bfloat16 for fp4 matmul");
  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;
  oneDNN::dnnl_matmul_w4a4_fp4(result, A, B, is_nt, bias, A_scale, B_scale);
  return result;
}

torch::Tensor int4_gemm_w4a16(
    const torch::Tensor& A_,  // src, [b, m, k]
    const torch::Tensor& B,   // quantized weight, [k, n]
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& B_scale,  // [k/group_size, n]
    const torch::Tensor& B_zp,     // [k/group_size, n/8]
    int64_t group_size,
    const std::optional<torch::Tensor>& g_idx) {
  const at::DeviceGuard device_guard(A_.device());

  TORCH_CHECK(
      B_scale.is_contiguous(), "B_scale must be contiguous for int4 matmul");

  // For GPTQ with desc_act=True scenario
  auto A = g_idx.has_value() ? A_.index_select(-1, g_idx.value()) : A_;
  torch::Tensor result = check_and_create_output_tensor(A, B, A.scalar_type());

  oneDNN::dnnl_matmul_w4a16_int4(result, A, B, bias, B_scale, B_zp, group_size);
  return result;
}

torch::Tensor int8_gemm_w8a8(
    const torch::Tensor& A_,       // quantized activations, [m, k] int8
    const torch::Tensor& A_scale,  // per-token [m, 1] or per-tensor [1]
    const std::optional<torch::Tensor>& A_zp,  // as A_scale, int32; or none
    const torch::Tensor& B,        // quantized weight, [k, n] NT format int8
    const torch::Tensor& B_scale,  // per-channel [1, n] or per-group [k/gs, n]
    const std::optional<torch::Tensor>& B_zp,  // as B_scale, int32; or none
    int64_t group_size,                        // -1 for per-channel
    const std::optional<torch::Tensor>& bias,
    std::optional<c10::ScalarType> out_dtype) {  // float16 (default), bfloat16
  const at::DeviceGuard device_guard(A_.device());

  TORCH_CHECK(
      A_.scalar_type() == at::ScalarType::Char ||
          A_.scalar_type() == at::ScalarType::Byte,
      "int8_gemm_w8a8: activation must be int8 (signed/unsigned), got ",
      A_.scalar_type());
  TORCH_CHECK(
      B.scalar_type() == at::ScalarType::Char ||
          B.scalar_type() == at::ScalarType::Byte,
      "int8_gemm_w8a8: weight must be int8 (Char) or uint8 (Byte), got ",
      B.scalar_type());
  auto is_scale_dtype = [](at::ScalarType t) {
    return t == at::ScalarType::Float || t == at::ScalarType::Half ||
           t == at::ScalarType::BFloat16;
  };
  TORCH_CHECK(
      is_scale_dtype(A_scale.scalar_type()) &&
          is_scale_dtype(B_scale.scalar_type()),
      "int8_gemm_w8a8: scales must be float32, float16 or bfloat16, got "
      "A_scale=",
      A_scale.scalar_type(),
      " B_scale=",
      B_scale.scalar_type());
  for (const auto* zp : {&A_zp, &B_zp}) {
    TORCH_CHECK(
        !zp->has_value() || (*zp)->scalar_type() == at::ScalarType::Int,
        "int8_gemm_w8a8: zero points must be int32, got ",
        (*zp)->scalar_type());
  }

  // Validate quantization format for input A
  const bool per_token_A =
      (A_scale.dim() == 2 && A_scale.size(1) == 1 &&
       (!A_zp.has_value() || (A_zp->dim() == 2 && A_zp->size(1) == 1)));
  const bool per_tensor_A =
      (A_scale.numel() == 1 && (!A_zp.has_value() || A_zp->numel() == 1));
  TORCH_CHECK(
      per_token_A || per_tensor_A,
      "int8_gemm_w8a8: A_scale and A_zp must be per-token [m,1] or per-tensor "
      "[1]");
  // A per-token scale must have exactly one row per activation row (a [m',1]
  // scale with m' != m would otherwise be taken as per-tensor), and a zero
  // point must match its scale's shape.
  const int64_t m_rows = A_.numel() / A_.size(-1);
  TORCH_CHECK(
      A_scale.numel() == 1 || A_scale.numel() == m_rows,
      "int8_gemm_w8a8: A_scale must have 1 or m=",
      m_rows,
      " elements, got ",
      A_scale.sizes());
  TORCH_CHECK(
      !A_zp.has_value() || A_zp->sizes() == A_scale.sizes(),
      "int8_gemm_w8a8: A_zp must have A_scale's shape ",
      A_scale.sizes());
  TORCH_CHECK(
      !B_zp.has_value() || B_zp->sizes() == B_scale.sizes(),
      "int8_gemm_w8a8: B_zp must have B_scale's shape ",
      B_scale.sizes());
  TORCH_CHECK(
      A_.scalar_type() == at::ScalarType::Char || A_zp.has_value(),
      "int8_gemm_w8a8: uint8 activations need A_zp");
  for (const torch::Tensor* t : {&A_scale, &B, &B_scale}) {
    TORCH_CHECK(
        t->device() == A_.device(),
        "int8_gemm_w8a8: all inputs must be on ",
        A_.device());
  }
  for (const auto* t : {&A_zp, &B_zp, &bias}) {
    TORCH_CHECK(
        !t->has_value() || (*t)->device() == A_.device(),
        "int8_gemm_w8a8: all inputs must be on ",
        A_.device());
  }
  TORCH_CHECK(
      A_scale.is_contiguous() && B_scale.is_contiguous() &&
          (!A_zp.has_value() || A_zp->is_contiguous()) &&
          (!B_zp.has_value() || B_zp->is_contiguous()),
      "int8_gemm_w8a8: scales and zero points must be contiguous");

  const c10::ScalarType dst_dtype = out_dtype.value_or(torch::kHalf);
  TORCH_CHECK(
      dst_dtype == torch::kHalf || dst_dtype == torch::kBFloat16,
      "int8_gemm_w8a8: out_dtype must be float16 or bfloat16, got ",
      dst_dtype);
  torch::Tensor result = check_and_create_output_tensor(A_, B, dst_dtype);

  oneDNN::dnnl_matmul_w8a8_int8(
      result, A_, A_scale, A_zp, B, B_scale, B_zp, group_size, bias);
  return result;
}

torch::Tensor int4_gemm_w4a8(
    const torch::Tensor& A_,       // quantized inputs, [b, m, k]
    const torch::Tensor& A_scale,  // [b * m, 1] or [b]
    const torch::Tensor& A_zp,     // [b * m, 1] or [b]
    const torch::Tensor& B,        // quantized weight, [k, n]
    const torch::Tensor& B_scale,  // [k/group_size, n]
    const torch::Tensor& B_zp,     // [k/group_size, n/8]
    int64_t group_size,
    const std::optional<torch::Tensor>& g_idx,
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(A_.device());

  TORCH_CHECK(
      A_scale.is_contiguous(), "A_scale must be contiguous for int4 matmul");
  TORCH_CHECK(
      B_scale.is_contiguous(), "B_scale must be contiguous for int4 matmul");

  // Select indices if provided (for GPTQ with desc_act=True)
  const torch::Tensor& A =
      g_idx.has_value() ? A_.index_select(-1, g_idx.value()) : A_;

  // Validate quantization format for input A
  const bool per_token_A =
      (A_scale.dim() == 2 && A_scale.size(1) == 1 && A_zp.dim() == 2 &&
       A_zp.size(1) == 1);
  const bool per_tensor_A = (A_scale.dim() == 1 && A_zp.dim() == 1);
  TORCH_CHECK(
      per_token_A || per_tensor_A,
      "Int4-Int8 matmul expects quantized input A to be per-token ([b*m,1]) or "
      "per-tensor ([b]) quantized!");

  torch::Tensor result = check_and_create_output_tensor(A, B, torch::kHalf);

  oneDNN::dnnl_matmul_w4a8_int4(
      result, A, A_scale, A_zp, B, B_scale, B_zp, group_size, bias);

  return result;
}
