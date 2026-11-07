#pragma once
#include <cstddef>
#include <cstdint>

namespace vllm {
namespace xpu {

enum xpuMemcpyKind { HostToDevice, DeviceToHost, DeviceToDevice };

/**
 * @brief Performs asynchronous memory copy between host and XPU device.
 *
 * Handles data transfer in three directions: host-to-device, device-to-host,
 * and device-to-device. For host-involved transfers, supports both pinned
 * (page-locked) and pageable host memory through the caching host allocator.
 *
 * @param dst        Destination memory address (device or host pointer)
 * @param src        Source memory address (device or host pointer)
 * @param n_bytes    Number of bytes to copy
 * @param kind       Direction of memory transfer (HostToDevice, DeviceToHost,
 *                   or DeviceToDevice)
 * @param hctx       Host context pointer from caching host allocator. Required
 *                   when is_pinned is true for optimized DMA transfer; nullptr
 *                   for non-pinned memory or D2D transfers
 * @param is_pinned  Whether the host pointer is page-locked (pinned) memory.
 *                   Must be true for hctx to be valid
 */
void xpuAsyncMemcpy(
    void* dst,
    const void* src,
    size_t n_bytes,
    xpuMemcpyKind kind,
    const void* hctx,
    bool is_pinned);

// True if [ptr, ptr + n_bytes) lies wholly inside one range registered with
// xpu_host_register (prepare_for_device_copy): DMA-capable host memory that
// sycl::get_pointer_type() reports as usm::alloc::unknown. A range only
// partly covered, or spanning two registered ranges, is reported false.
// Thread-safe; the answer holds at the time of the call only.
bool isHostRegistered(const void* ptr, size_t n_bytes);

/**
 * @brief Batch async memcpy: copies N independent (src, dst, size) triples
 *        in a single call, amortising per-copy overhead.
 *
 * The copy direction is auto-detected from the first non-zero entry's USM
 * pointer types.  All entries must share the same direction.
 *
 * For H2D: snapshots all source blocks through a single contiguous pinned
 *   staging buffer so the caller may safely mutate host memory immediately.
 *   With snapshot_src=false and every source in pinned memory (USM host, or
 *   a range registered with xpu_host_register), copies directly from the
 *   sources instead, in stream order; the caller must then not modify the
 *   sources until the copies have completed. A source counts as registered
 *   only if its whole [src, src + size) lies inside one registered range;
 *   if any non-empty source is not pinned, the whole batch is staged.
 *
 *   Registration vs. enqueue. The pinned check runs once, at call time,
 *   under the registry lock: a range registered or unregistered before the
 *   call is seen as such, and a range unregistered before the call is
 *   staged. Nothing is checked or held after the call returns: the copies
 *   read the sources when the stream reaches them. With snapshot_src=false
 *   the caller must therefore keep every source range registered (and
 *   mapped) until the copies have completed, i.e. synchronize the stream or
 *   wait on an event recorded after the call before xpu_host_unregister.
 *   Unregistering while a direct copy is pending is undefined behaviour.
 *   A register or unregister racing with the call on another thread gives
 *   either the staged or the direct outcome, both correct, provided the
 *   rule above holds for the direct one.
 * For D2H / D2D: direct async DMA without staging.
 *
 * @param src_ptrs   Array of N raw source addresses
 * @param dst_ptrs   Array of N raw destination addresses
 * @param sizes      Array of N byte counts
 * @param n          Number of entries
 * @param snapshot_src  For H2D from pinned sources: stage (true) or copy
 *                      directly (false). Pageable sources are always staged.
 */
void xpuAsyncMemcpyBatch(
    const uint64_t* src_ptrs,
    const uint64_t* dst_ptrs,
    const uint64_t* sizes,
    int64_t n,
    bool snapshot_src = true);

}  // namespace xpu
}  // namespace vllm
