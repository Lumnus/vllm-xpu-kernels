// SPDX-License-Identifier: Apache-2.0
//
// Mailbox all-reduce: tensor-parallel sum of a 16-bit tensor over 2 or 4
// cards, one kernel per call, no host round trip, graph-capturable.
//
// Schedule. Recursive doubling over peer-to-peer stores: in stage s every
// rank writes its current partial into the mailbox of one partner (the
// partner's memory, mapped here through Level Zero IPC) and polls its own
// mailbox for the partner's partial. 2 ranks: one stage. 4 ranks: two stages
// with partners r ^ a and r ^ b (a != b, both nonzero; the caller picks the
// pair). Every rank talks to one peer per stage: concurrent pushes to several
// peers collapse on PCIe (a one-shot exchange to three peers ran 30-80x
// slower than one flow on the four-card host this was written for).
//
// Wire format. A mailbox line is 16 bytes, written by one store and read by
// one load: three payload words and a tag word. Stage 1 carries six 16-bit
// input values per line; later stages carry three fp32 partials per line
// (fp32_partials) or six partials rounded to the input type. The tag is the
// call's epoch: a device counter that the kernel advances itself, so a
// launch has no per-call host argument and replays from a graph. A line
// whose tag does not match is not yet written (or is two calls old: each
// mailbox has two slots used alternately; a rank can only write slot k of
// call c+2 after its partner sent call c+1, i.e. after the partner finished
// reading call c). No peer atomics, no flags apart from the data, no fences.
// Platform assumption: a 16-byte aligned store lands, and a 16-byte aligned
// load reads, as a unit (the oneCCL LL256 protocol assumes the same of 64-byte
// lines on these cards).
//
// Summation contract. Every rank computes the same bits: each stage adds two
// fp32 values once, and fp32 addition is commutative, so a + b on one rank
// equals b + a on its partner. With fp32_partials the result is
//   round(((x_r + x_{r^a}) + (x_{r^b} + x_{r^a^b})))
// in fp32, rounded once to the input type (round to nearest even). Without
// it the stage-1 sums are rounded to the input type before stage 2 (two
// roundings in the tree, against three in oneCCL's ring). The bits differ
// from oneCCL's (another order, other roundings); they are equal across
// ranks, across calls and between eager and graph replay.
//
// Failure. A wait gives up after timeout_cycles of the sub-group clock
// (0 = never): the work-item writes the failing stage, tag, item and the tag
// it saw into the host words, sets FAILED (sticky), and the call writes NaN
// into every element it could not reduce. Any later wait that does not match
// at once reads FAILED after 256 polls and gives up too, so a dead peer
// cannot hang the queue. The host side watches the words and stops the
// process with a clear line.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/clock.hpp>

#include <ATen/ATen.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "mailbox_all_reduce.h"

namespace vllm::mailbox_ar {
namespace {

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;
using bf16 = sycl::ext::oneapi::bfloat16;
using u4 = sycl::vec<uint32_t, 4>;

constexpr int kEpw = 6;       // 16-bit elements per work-item = one line
constexpr int kWgSize = 256;  // 16 sub-groups of 16: 4 KB of lines per store
// One wave: 32 work-groups of 256 are resident at once on a 32-Xe-core card,
// so every work-item that waits has its partner's work-item running too.
// Larger tensors loop (grid stride), item by item in the same order on every
// rank.
constexpr int kMaxWgs = 32;
constexpr int kMaxStages = 2;

enum HostWord : int {
  H_FAILED = 0,  // sticky: a wait gave up
  H_STAGE,       // stage of the first recorded failure
  H_TAG,         // tag it waited for
  H_ITEM,        // line index
  H_SEEN,        // tag word it read last
  H_WORDS = 8
};

using host_ref = sycl::atomic_ref<
    int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;

inline uint32_t uc_load32(const uint32_t* p) {  // L1-uncached load
  syclex::annotated_ptr<
      const uint32_t,
      decltype(syclex::properties{intelex::read_hint<intelex::cache_control<
          intelex::cache_mode::uncached,
          syclex::cache_level::L1>>})>
      a(p);
  return *a;
}

typedef unsigned int lsc_u4 __attribute__((ext_vector_type(4)));
#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL void __builtin_IB_lsc_store_global_uint4(
    __attribute__((opencl_global)) lsc_u4* base,
    int off,
    lsc_u4 val,
    int cache);
extern "C" SYCL_EXTERNAL lsc_u4 __builtin_IB_lsc_load_global_uint4(
    __attribute__((opencl_global)) lsc_u4* base, int off, int cache);
#endif

// One 16-byte load, L1 uncached and L3 cached (LSC_LDCC_L1UC_L3C, the cache
// mode of oneCCL's LL256 polls on these cards). annotated_ptr's read hint is
// not applied to a 16-byte vector load (the ISA showed a default-cached
// load), hence the builtin.
inline u4 uc_load16(const uint32_t* p) {
#ifdef __SYCL_DEVICE_ONLY__
  auto* g =
      (__attribute__((opencl_global))
       lsc_u4*)(__attribute__((opencl_global)) void*)(const_cast<uint32_t*>(p));
  const lsc_u4 v = __builtin_IB_lsc_load_global_uint4(g, 0, 2);
  return u4(v[0], v[1], v[2], v[3]);
#else
  return u4(p[0], p[1], p[2], p[3]);
#endif
}

// One 16-byte store into the partner's mailbox, L1 and L3 uncached
// (LSC_STCC_L1UC_L3UC).
inline void
peer_store16(uint32_t* p, uint32_t a, uint32_t b, uint32_t c, uint32_t tag) {
#ifdef __SYCL_DEVICE_ONLY__
  auto* g = (__attribute__((opencl_global))
             lsc_u4*)(__attribute__((opencl_global)) void*)(p);
  lsc_u4 v = {a, b, c, tag};
  __builtin_IB_lsc_store_global_uint4(g, 0, v, 1);
#else
  p[0] = a;
  p[1] = b;
  p[2] = c;
  p[3] = tag;
#endif
}

template <typename T>
inline uint16_t bits16(T v) {
  return sycl::bit_cast<uint16_t>(v);
}
template <typename T>
inline T from16(uint32_t u) {
  return sycl::bit_cast<T>(static_cast<uint16_t>(u & 0xffffu));
}
inline uint32_t pack2(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// Waits until the line at p carries tag. False: gave up (timeout, or another
// wait of this device already failed).
inline bool wait_line(
    const uint32_t* p,
    uint32_t tag,
    u4& w,
    int* host,
    uint64_t timeout,
    int stage,
    int64_t item) {
  w = uc_load16(p);
  if (w[3] == tag) return true;
  const uint64_t t0 = syclex::clock<syclex::clock_scope::sub_group>();
  for (uint32_t k = 1;; ++k) {
    w = uc_load16(p);
    if (w[3] == tag) return true;
    if ((k & 255u) == 0) {
      const bool failed = host_ref(host[H_FAILED]).load() != 0;
      const bool late =
          timeout != 0 &&
          syclex::clock<syclex::clock_scope::sub_group>() - t0 > timeout;
      if (failed || late) {
        if (!failed) {
          host_ref(host[H_STAGE]).store(stage);
          host_ref(host[H_TAG]).store(static_cast<int>(tag));
          host_ref(host[H_ITEM]).store(static_cast<int>(item));
          host_ref(host[H_SEEN]).store(static_cast<int>(w[3]));
          host_ref(host[H_FAILED]).store(1);
        }
        return false;
      }
    }
  }
}

template <typename T, bool FP32P>
struct MailboxArgs {
  T* x;
  int64_t n;
  int64_t items;
  char* in[kMaxStages];
  char* out[kMaxStages];
  int nst;
  int64_t slot_bytes;
  uint32_t* state;
  int* host;
  uint64_t timeout;
  uint32_t nwg;
};

template <typename T, bool FP32P>
void run_item(const MailboxArgs<T, FP32P>& a, int64_t i, uint32_t tag) {
  const int64_t slot = static_cast<int64_t>(tag & 1u) * a.slot_bytes;
  const int64_t e0 = i * kEpw;
  float v[kEpw];
  uint16_t b[kEpw];
#pragma unroll
  for (int j = 0; j < kEpw; ++j) {
    const T t = (e0 + j < a.n) ? a.x[e0 + j] : T(0.0f);
    b[j] = bits16(t);
    v[j] = static_cast<float>(t);
  }
  bool ok = true;
  for (int s = 0; s < a.nst && ok; ++s) {
    char* ib = a.in[s] + slot;
    char* ob = a.out[s] + slot;
    if (s == 0 || !FP32P) {
      if (s > 0) {  // 16-bit partials: round the stage sum, keep it rounded
#pragma unroll
        for (int j = 0; j < kEpw; ++j) {
          const T r = T(v[j]);
          b[j] = bits16(r);
          v[j] = static_cast<float>(r);
        }
      }
      peer_store16(
          reinterpret_cast<uint32_t*>(ob + i * 16),
          pack2(b[0], b[1]),
          pack2(b[2], b[3]),
          pack2(b[4], b[5]),
          tag);
      u4 w;
      ok = wait_line(
          reinterpret_cast<const uint32_t*>(ib + i * 16),
          tag,
          w,
          a.host,
          a.timeout,
          s,
          i);
      if (ok) {
#pragma unroll
        for (int j = 0; j < kEpw; ++j) {
          const uint32_t word = w[j >> 1] >> ((j & 1) * 16);
          v[j] = v[j] + static_cast<float>(from16<T>(word));
        }
      }
    } else {  // fp32 partials: two lines per item
      peer_store16(
          reinterpret_cast<uint32_t*>(ob + (2 * i) * 16),
          sycl::bit_cast<uint32_t>(v[0]),
          sycl::bit_cast<uint32_t>(v[1]),
          sycl::bit_cast<uint32_t>(v[2]),
          tag);
      peer_store16(
          reinterpret_cast<uint32_t*>(ob + (2 * i + 1) * 16),
          sycl::bit_cast<uint32_t>(v[3]),
          sycl::bit_cast<uint32_t>(v[4]),
          sycl::bit_cast<uint32_t>(v[5]),
          tag);
      u4 w0, w1;
      ok = wait_line(
               reinterpret_cast<const uint32_t*>(ib + (2 * i) * 16),
               tag,
               w0,
               a.host,
               a.timeout,
               s,
               i) &&
           wait_line(
               reinterpret_cast<const uint32_t*>(ib + (2 * i + 1) * 16),
               tag,
               w1,
               a.host,
               a.timeout,
               s,
               i);
      if (ok) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          v[j] = v[j] + sycl::bit_cast<float>(static_cast<uint32_t>(w0[j]));
          v[3 + j] =
              v[3 + j] + sycl::bit_cast<float>(static_cast<uint32_t>(w1[j]));
        }
      }
    }
  }
#pragma unroll
  for (int j = 0; j < kEpw; ++j)
    if (e0 + j < a.n)
      a.x[e0 + j] = ok ? T(v[j]) : T(std::numeric_limits<float>::quiet_NaN());
}

template <typename T, bool FP32P>
void launch(sycl::queue& q, const MailboxArgs<T, FP32P>& a) {
  const int64_t wgs = (a.items + kWgSize - 1) / kWgSize;
  const uint32_t nwg =
      static_cast<uint32_t>(wgs < kMaxWgs ? (wgs > 0 ? wgs : 1) : kMaxWgs);
  MailboxArgs<T, FP32P> args = a;
  args.nwg = nwg;
  q.submit([&](sycl::handler& h) {
    h.parallel_for(
        sycl::nd_range<1>(nwg * kWgSize, kWgSize),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          // every work-group reads the epoch before any work-group of this
          // launch can publish the next one (that happens after all of them
          // passed the barrier below)
          const uint32_t epoch = uc_load32(args.state);
          uint32_t tag = epoch + 1u;
          if (tag == 0u) tag = 1u;
          const int64_t g = static_cast<int64_t>(it.get_global_range(0));
          for (int64_t i = static_cast<int64_t>(it.get_global_id(0));
               i < args.items;
               i += g)
            run_item<T, FP32P>(args, i, tag);
          sycl::group_barrier(it.get_group());
          if (it.get_local_linear_id() == 0) {
            sycl::atomic_ref<
                uint32_t,
                sycl::memory_order::acq_rel,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                done(args.state[1]);
            if (done.fetch_add(1u) == args.nwg - 1u) {
              done.store(0u);
              sycl::atomic_ref<
                  uint32_t,
                  sycl::memory_order::relaxed,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space>(args.state[0])
                  .store(tag);
            }
          }
        });
  });
}

template <typename T>
void dispatch_partials(
    sycl::queue& q, bool fp32p, const MailboxArgs<T, true>& a) {
  if (fp32p) {
    launch<T, true>(q, a);
  } else {
    MailboxArgs<T, false> b;
    b.x = a.x;
    b.n = a.n;
    b.items = a.items;
    for (int s = 0; s < kMaxStages; ++s) {
      b.in[s] = a.in[s];
      b.out[s] = a.out[s];
    }
    b.nst = a.nst;
    b.slot_bytes = a.slot_bytes;
    b.state = a.state;
    b.host = a.host;
    b.timeout = a.timeout;
    b.nwg = 0;
    launch<T, false>(q, b);
  }
}

}  // namespace

int64_t mailbox_ar_slot_bytes(int64_t numel, bool fp32_partials) {
  const int64_t items = (numel + kEpw - 1) / kEpw;
  return items * 16 * (fp32_partials ? 2 : 1);
}

void mailbox_all_reduce(
    torch::Tensor& x,
    std::vector<int64_t> inbox,
    std::vector<int64_t> outbox,
    int64_t state,
    int64_t host,
    int64_t slot_bytes,
    bool fp32_partials,
    int64_t timeout_cycles) {
  TORCH_CHECK(x.is_xpu(), "mailbox_all_reduce: tensor not on XPU");
  TORCH_CHECK(x.is_contiguous(), "mailbox_all_reduce: tensor not contiguous");
  TORCH_CHECK(
      x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
      "mailbox_all_reduce: bf16 or fp16 only, got ",
      x.scalar_type());
  const int nst = static_cast<int>(inbox.size());
  TORCH_CHECK(
      nst >= 1 && nst <= kMaxStages && outbox.size() == inbox.size(),
      "mailbox_all_reduce: 1 or 2 stages");
  TORCH_CHECK(state != 0 && host != 0, "mailbox_all_reduce: no state");
  const int64_t n = x.numel();
  if (n == 0) return;
  TORCH_CHECK(
      mailbox_ar_slot_bytes(n, fp32_partials) <= slot_bytes,
      "mailbox_all_reduce: ",
      n,
      " elements need ",
      mailbox_ar_slot_bytes(n, fp32_partials),
      " bytes per slot, the mailbox has ",
      slot_bytes);
  for (int s = 0; s < nst; ++s)
    TORCH_CHECK(
        inbox[s] != 0 && outbox[s] != 0 && inbox[s] % 16 == 0 &&
            outbox[s] % 16 == 0,
        "mailbox_all_reduce: mailbox pointers must be 16-byte aligned");
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  auto fill = [&](auto* xp, auto& a) {
    a.x = xp;
    a.n = n;
    a.items = (n + kEpw - 1) / kEpw;
    for (int s = 0; s < kMaxStages; ++s) {
      a.in[s] = s < nst ? reinterpret_cast<char*>(inbox[s]) : nullptr;
      a.out[s] = s < nst ? reinterpret_cast<char*>(outbox[s]) : nullptr;
    }
    a.nst = nst;
    a.slot_bytes = slot_bytes;
    a.state = reinterpret_cast<uint32_t*>(state);
    a.host = reinterpret_cast<int*>(host);
    a.timeout = static_cast<uint64_t>(timeout_cycles > 0 ? timeout_cycles : 0);
    a.nwg = 0;
  };
  if (x.scalar_type() == at::kBFloat16) {
    MailboxArgs<bf16, true> a;
    fill(reinterpret_cast<bf16*>(x.data_ptr()), a);
    dispatch_partials<bf16>(q, fp32_partials, a);
  } else {
    MailboxArgs<sycl::half, true> a;
    fill(reinterpret_cast<sycl::half*>(x.data_ptr()), a);
    dispatch_partials<sycl::half>(q, fp32_partials, a);
  }
}

int64_t mailbox_ar_host_alloc() {
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  int* w = sycl::malloc_host<int>(H_WORDS, q.get_context());
  TORCH_CHECK(w != nullptr, "mailbox_ar_host_alloc: malloc_host failed");
  std::memset(w, 0, H_WORDS * sizeof(int));
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(w));
}

void mailbox_ar_host_free(int64_t host) {
  if (host == 0) return;
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  sycl::free(reinterpret_cast<void*>(host), q.get_context());
}

std::vector<int64_t> mailbox_ar_host_read(int64_t host) {
  TORCH_CHECK(host != 0, "mailbox_ar_host_read: no host words");
  const volatile int* w = reinterpret_cast<const volatile int*>(host);
  std::vector<int64_t> out(H_WORDS);
  for (int i = 0; i < H_WORDS; ++i)
    out[i] = w[i];
  return out;
}

}  // namespace vllm::mailbox_ar
