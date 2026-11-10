// One slice of the B70 W4A8 fused gemm1 (silu-and-mul + int8) kernels.
#include "w4a8_launch.hpp"
namespace MoEW4A8 {
template void dispatch_glu<6, 32>(const GluArgs&);
}  // namespace MoEW4A8
