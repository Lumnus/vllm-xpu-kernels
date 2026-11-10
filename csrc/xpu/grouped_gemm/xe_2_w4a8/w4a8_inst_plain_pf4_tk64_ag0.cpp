// One slice of the B70 W4A8 grouped GEMM kernels (see w4a8_launch.hpp).
#include "w4a8_launch.hpp"
namespace MoEW4A8 {
template void dispatch_plain<4, 64, 0>(const PlainArgs&);
}  // namespace MoEW4A8
