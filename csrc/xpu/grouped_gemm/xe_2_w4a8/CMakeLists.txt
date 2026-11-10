cmake_minimum_required(VERSION 3.18)

# W4A8 grouped GEMM (s8 x s4 DPAS) and the small-M dense W8A8 GEMM.
add_xe2_kernel_library(grouped_gemm_w4a8_xe_2 INCLUDE_CMAKE_SOURCE_DIR)
