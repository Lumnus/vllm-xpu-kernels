// What the host CPU offers the host MoE ops, read once from CPUID and the
// OS's enabled state (XCR0), and the AMX permission request. No torch, no
// device dependency.
//
// The dispatch levels (cpu_moe::Isa in int4_moe_cpu_impl.h) and what each
// requires:
//   2 AVX2 + FMA                       Haswell / Zen 1 and later
//   3 level 2 + AVX-VNNI (VEX)         Alder Lake / Sapphire Rapids / Zen 5
//   4 AVX-512 F, BW, DQ, VL            Skylake-SP, Zen 4 and later
//   5 level 4 + AVX-512 VNNI           Cascade Lake, Ice Lake-SP, Zen 4
//   6 level 5 + AVX-512 BF16           Cooper Lake, Sapphire Rapids, Zen 4
//   7 level 6 + AMX TILE, INT8, BF16   Sapphire / Emerald / Granite Rapids
// A level counts only when the OS saves its register state (XCR0: AVX for
// 2-3; the opmask and upper ZMM state for 4-6; the tile state for 7) and,
// for AMX on Linux, when the kernel grants this process the tile data
// (arch_prctl ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA: the permission is
// per process and covers every thread, present and later, so it is asked
// once; each thread still loads its own tile configuration before it uses
// the tiles).

#pragma once

#include <cstdint>
#include <mutex>

#if defined(__x86_64__) || defined(_M_X64)
  #include <cpuid.h>
  #define INT4_MOE_CPU_ISA_X86 1
#else
  #define INT4_MOE_CPU_ISA_X86 0
#endif

#if defined(__linux__)
  #include <sys/syscall.h>
  #include <unistd.h>
#endif

namespace cpu_moe {

struct CpuFeatures {
  bool avx2 = false, fma = false, avxvnni = false;
  bool avx512f = false, avx512bw = false, avx512dq = false, avx512vl = false;
  bool avx512vnni = false, avx512bf16 = false, avx512vbmi = false;
  bool amx_tile = false, amx_int8 = false, amx_bf16 = false;
  bool os_avx = false, os_avx512 = false, os_amx = false;
  bool amx_granted = false;  // the process may use the tile data (Linux)
};

namespace detail {

#if INT4_MOE_CPU_ISA_X86
inline uint64_t xgetbv0() {
  uint32_t lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

// Ask Linux for the AMX tile data once per process; true when granted.
inline bool request_amx() {
  #if defined(__linux__)
  constexpr long kGetPerm = 0x1022, kReqPerm = 0x1023;
  constexpr long kXtileData = 18;
  if (syscall(SYS_arch_prctl, kReqPerm, kXtileData) != 0) return false;
  unsigned long bits = 0;
  if (syscall(SYS_arch_prctl, kGetPerm, &bits) != 0) return false;
  return (bits & (1ul << kXtileData)) != 0;
  #else
  return false;
  #endif
}
#endif

inline CpuFeatures read_features() {
  CpuFeatures f;
#if INT4_MOE_CPU_ISA_X86
  unsigned a, b, c, d;
  if (!__get_cpuid(1, &a, &b, &c, &d)) return f;
  const bool osxsave = (c >> 27) & 1;
  f.fma = (c >> 12) & 1;
  const bool avx = (c >> 28) & 1;
  if (!osxsave || !avx) return f;
  const uint64_t xcr0 = xgetbv0();
  f.os_avx = (xcr0 & 0x6) == 0x6;                   // SSE, AVX
  f.os_avx512 = f.os_avx && (xcr0 & 0xE0) == 0xE0;  // opmask, ZMM
  f.os_amx = (xcr0 & 0x60000) == 0x60000;           // XTILECFG, XTILEDATA
  unsigned max_leaf = __get_cpuid_max(0, nullptr);
  if (max_leaf < 7) return f;
  __cpuid_count(7, 0, a, b, c, d);
  f.avx2 = (b >> 5) & 1;
  f.avx512f = (b >> 16) & 1;
  f.avx512dq = (b >> 17) & 1;
  f.avx512bw = (b >> 30) & 1;
  f.avx512vl = (b >> 31) & 1;
  f.avx512vbmi = (c >> 1) & 1;
  f.avx512vnni = (c >> 11) & 1;
  f.amx_bf16 = (d >> 22) & 1;
  f.amx_tile = (d >> 24) & 1;
  f.amx_int8 = (d >> 25) & 1;
  const unsigned subleaves = a;
  if (subleaves >= 1) {
    __cpuid_count(7, 1, a, b, c, d);
    f.avxvnni = (a >> 4) & 1;
    f.avx512bf16 = (a >> 5) & 1;
  }
  if (!f.os_avx) {
    f.avx2 = f.fma = f.avxvnni = false;
  }
  if (!f.os_avx512) {
    f.avx512f = f.avx512bw = f.avx512dq = f.avx512vl = false;
    f.avx512vnni = f.avx512bf16 = f.avx512vbmi = false;
  }
  if (f.os_amx && f.amx_tile && f.amx_int8 && f.amx_bf16)
    f.amx_granted = request_amx();
#endif
  return f;
}

}  // namespace detail

inline const CpuFeatures& cpu_features() {
  static const CpuFeatures f = detail::read_features();
  return f;
}

// Whether the host has what dispatch level `level` needs (see the top).
inline bool level_supported(int level) {
  const CpuFeatures& f = cpu_features();
  const bool l2 = f.avx2 && f.fma;
  const bool l4 = l2 && f.avx512f && f.avx512bw && f.avx512dq && f.avx512vl;
  switch (level) {
    case 1:
      return true;
    case 2:
      return l2;
    case 3:
      return l2 && f.avxvnni;
    case 4:
      return l4;
    case 5:
      return l4 && f.avx512vnni;
    case 6:
      return l4 && f.avx512vnni && f.avx512bf16;
    case 7:
    case 8:
      return l4 && f.avx512vnni && f.avx512bf16 && f.amx_tile && f.amx_int8 &&
             f.amx_bf16 && f.amx_granted;
    default:
      return false;
  }
}

// The widest level of 1..7 the host supports (8 is never automatic).
inline int best_level() {
  for (int l = 7; l >= 2; --l)
    if (level_supported(l)) return l;
  return 1;
}

inline const char* level_name(int level) {
  switch (level) {
    case 1:
      return "scalar";
    case 2:
      return "AVX2+FMA";
    case 3:
      return "AVX2+FMA+AVX-VNNI";
    case 4:
      return "AVX-512";
    case 5:
      return "AVX-512+VNNI";
    case 6:
      return "AVX-512+VNNI+BF16";
    case 7:
      return "AMX+AVX-512";
    case 8:
      return "AMX+AVX-512, card-class AMX-BF16 prefill";
    default:
      return "unknown";
  }
}

}  // namespace cpu_moe
