cmake_minimum_required(VERSION 3.18)

# B70 W4A8 grouped GEMM (s8 x s4 DPAS). Its own library so the standalone
# _b70_w4a8_C module can be built without the stock grouped_gemm_xe_2.
add_xe2_kernel_library(grouped_gemm_w4a8_xe_2 INCLUDE_CMAKE_SOURCE_DIR)
