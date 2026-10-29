// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// B70 (Lumnus): W8A8 int8 GEMM through oneDNN, ported from
// vllm-project/vllm-xpu-kernels#196 (tianmu-li, int8_gemm_w8a8) and narrowed
// to what vLLM's symmetric Int8ScaledMM path needs:
//   out[m, n] = a_scale[m] * b_scale[n] * sum_k a[m, k] * b[k, n] + bias[n]
//   a: s8 [m, k] (row stride allowed), per-token [m] or per-tensor [1] f32 scale
//   b: s8 [k, n], either the NT view of a contiguous [n, k] (vLLM's w_q.t())
//      or contiguous [k, n]; per-channel [n] or per-tensor [1] f32 scale
//   out: f16 or bf16 [m, n], optional bias [n] (f32/f16/bf16)
//
// Per-token activation scale, two ways:
//   1. src_attr (default): oneDNN src scales with mask (M, K) and groups
//      {1, K}, i.e. one scale per row — the attr shape int4_gemm_w4a8 already
//      uses for its s8 per-token activations on this oneDNN.
//   2. post_op: weight scales via attr, then a binary_mul post-op by a {m, 1}
//      f32 tensor, then the bias as a binary_add post-op (oneDNN adds the
//      matmul bias before post-ops, so the bias must move behind the multiply).
// Mode 1 is tried first; if oneDNN reports it unimplemented the process
// switches to mode 2 for good (logged once). B70_INT8_GEMM_SCALE_MODE=postop
// forces mode 2.
#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

enum class w8a8_scale_mode : int { src_attr = 0, post_op = 1 };

static inline std::atomic<int>& w8a8_scale_mode_flag() {
  static std::atomic<int> flag([] {
    const char* e = std::getenv("B70_INT8_GEMM_SCALE_MODE");
    return (e && std::strcmp(e, "postop") == 0)
        ? static_cast<int>(w8a8_scale_mode::post_op)
        : static_cast<int>(w8a8_scale_mode::src_attr);
  }());
  return flag;
}

static inline void w8a8_log_once(const char* msg) {
  static std::atomic<bool> done{false};
  if (!done.exchange(true)) {
    std::fprintf(stderr, "%s\n", msg);
  }
}

static inline void dnnl_matmul_w8a8_int8(
    torch::Tensor& result,           // [m, n] f16/bf16, contiguous
    const torch::Tensor& mat1,       // [m, k] s8
    const torch::Tensor& m1_sc,      // f32 [m, 1] or [1]
    const torch::Tensor& mat2,       // [k, n] s8 (NT view or NN)
    const torch::Tensor& m2_sc,      // f32 [1, n] or [1]
    const std::optional<torch::Tensor>& bias) {
  const int m = mat1.size(0);
  const int k = mat1.size(1);
  const int n = mat2.size(1);

  joint_dtypes_t jd;
  if (result.scalar_type() == at::ScalarType::Half) {
    jd = joint_dtypes_t::int8;
  } else if (result.scalar_type() == at::ScalarType::BFloat16) {
    jd = joint_dtypes_t::int8_bf16;
  } else {
    TORCH_CHECK(false, "int8_scaled_mm: out dtype must be f16 or bf16");
  }

  trans_type_t tt;
  int64_t ldb;
  if (mat2.stride(0) == 1 && (n == 1 || mat2.stride(1) >= k)) {
    tt = trans_type_t::nt;  // [n, k] row-major viewed as [k, n]
    ldb = mat2.stride(1);
  } else {
    TORCH_CHECK(
        mat2.stride(1) == 1,
        "int8_scaled_mm: weight must be [k, n] with one unit stride");
    tt = trans_type_t::nn;
    ldb = mat2.stride(0);
  }
  const int64_t lda = mat1.stride(0);
  const int64_t ldc = result.stride(0);

  const bool per_token_a = m1_sc.numel() != 1;
  const bool per_channel_b = m2_sc.numel() != 1;
  const bias_type_t b_type = get_bias_type(bias, m, n);
  const bool has_bias = get_shape(b_type) != bias_shape_t::none;

  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device cur_device = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(cur_device);
  auto& strm = GpuStreamManager::Instance().get_stream();

  auto set_wei_scales = [&](primitive_attr& pattr) {
    pattr.set_scales(
        DNNL_ARG_WEIGHTS,
        per_channel_b ? (1 << 1) : 0,
        {},
        memory::data_type::f32);
  };

  const int sc_key = (per_token_a ? 1 : 0) | (per_channel_b ? 2 : 0);

  // ---- mode 1: per-token src scales as a oneDNN attribute -----------------
  auto run_src_attr = [&]() {
    auto f_attr = [&](primitive_attr& pattr) {
      pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      if (per_token_a) {
        pattr.set_scales(
            DNNL_ARG_SRC, (1 << 0) + (1 << 1), {1, k}, memory::data_type::f32);
      } else {
        pattr.set_scales(DNNL_ARG_SRC, 0, {}, memory::data_type::f32);
      }
      set_wei_scales(pattr);
    };
    auto& mm = matmul_primitive_create_and_cache(
        jd, tt, b_type, m, n, k, lda, ldb, ldc, dev_id, f_attr, sc_key, 0);
    int arg_off = 0;
    mm.set_attribute(
        arg_off++, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m1_sc.data_ptr(), [&]() {
          return make_onednn_memory(
              get_onednn_md(m1_sc), engine, m1_sc.data_ptr());
        });
    mm.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
        m2_sc.data_ptr(),
        [&]() {
          return make_onednn_memory(
              get_onednn_md(m2_sc), engine, m2_sc.data_ptr());
        });
    std::vector<std::pair<int, void*>> args;
    args.reserve(6);
    args.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
    args.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
    args.emplace_back(DNNL_ARG_DST, result.data_ptr());
    if (has_bias) args.emplace_back(DNNL_ARG_BIAS, bias.value().data_ptr());
    torch::Tensor scratch = at::empty(
        {static_cast<int64_t>(mm.get_scratchpad_size())},
        mat1.options().dtype(at::kByte));
    args.emplace_back(DNNL_ARG_SCRATCHPAD, scratch.data_ptr());
    mm.execute(strm, engine, std::move(args), arg_off);
  };

  // ---- mode 2: per-token scale (and bias) as post-ops ----------------------
  auto run_post_op = [&]() {
    const memory::desc row_sc_md({m, 1}, memory::data_type::f32, {1, 1});
    const memory::desc bias_md(
        {1, n}, get_onednn_bias_data_type(b_type), {n, 1});
    auto f_attr = [&](primitive_attr& pattr) {
      pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      post_ops po;
      po.append_binary(algorithm::binary_mul, row_sc_md);
      if (has_bias) po.append_binary(algorithm::binary_add, bias_md);
      pattr.set_post_ops(po);
      set_wei_scales(pattr);
    };
    // bias is a post-op here, so the matmul itself is built without one; the
    // bias type goes into the cache key instead.
    const bias_type_t no_bias = get_bias_type(std::nullopt, m, n);
    auto& mm = matmul_primitive_create_and_cache(
        jd, tt, no_bias, m, n, k, lda, ldb, ldc, dev_id, f_attr,
        sc_key, static_cast<int>(b_type));
    int arg_off = 0;
    mm.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
        m2_sc.data_ptr(),
        [&]() {
          return make_onednn_memory(
              get_onednn_md(m2_sc), engine, m2_sc.data_ptr());
        });
    mm.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
        m1_sc.data_ptr(),
        [&]() {
          return make_onednn_memory(row_sc_md, engine, m1_sc.data_ptr());
        });
    if (has_bias) {
      mm.set_attribute(
          arg_off++,
          DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
          bias.value().data_ptr(),
          [&]() {
            return make_onednn_memory(
                bias_md, engine, bias.value().data_ptr());
          });
    }
    std::vector<std::pair<int, void*>> args;
    args.reserve(4);
    args.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
    args.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
    args.emplace_back(DNNL_ARG_DST, result.data_ptr());
    torch::Tensor scratch = at::empty(
        {static_cast<int64_t>(mm.get_scratchpad_size())},
        mat1.options().dtype(at::kByte));
    args.emplace_back(DNNL_ARG_SCRATCHPAD, scratch.data_ptr());
    mm.execute(strm, engine, std::move(args), arg_off);
  };

  if (!per_token_a ||
      w8a8_scale_mode_flag().load() ==
          static_cast<int>(w8a8_scale_mode::src_attr)) {
    try {
      run_src_attr();
      return;
    } catch (const dnnl::error& e) {
      if (!per_token_a || e.status != dnnl_unimplemented) throw;
      w8a8_scale_mode_flag().store(static_cast<int>(w8a8_scale_mode::post_op));
      w8a8_log_once(
          "B70 int8 GEMM: oneDNN has no per-token src-scale kernel here; "
          "using the binary_mul post-op path");
    }
  }
  run_post_op();
}

}  // namespace oneDNN
