cmake_minimum_required(VERSION 3.18)

# W4A8 grouped GEMM (s8 x s4 DPAS) and the small-M dense W8A8 GEMM.
add_xe2_kernel_library(grouped_gemm_w4a8_xe_2 INCLUDE_CMAKE_SOURCE_DIR)

# The fused MoE decode's quant kernels must give the bits of
# _C.dynamic_per_token_scaled_int8_quant: the flags of
# csrc/quantization/int8/int8_quant.cpp (IEEE fp32 division, no fast-math).
# Its silu element stays in w4a8_moe_fused_silu.cpp under the library's
# default model, the model csrc/activation.cpp is built with.
set_source_files_properties(
  "w4a8_moe_fused_prep.cpp"
  PROPERTIES COMPILE_OPTIONS "-ffp-model=precise;-foffload-fp32-prec-div")
