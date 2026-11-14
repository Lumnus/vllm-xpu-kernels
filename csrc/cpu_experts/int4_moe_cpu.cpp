// Host-side torch op over the int4 MoE experts of int4_moe_cpu_impl.h.

#include "cpu_experts/int4_moe_cpu.h"

#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "cpu_experts/int4_moe_cpu_impl.h"

namespace {

// The process's pools (one per thread count) and the settings every pool
// takes, now and when made later.
struct Pools {
  std::mutex mu;
  std::map<int64_t, std::unique_ptr<cpu_moe::Pool>> by_threads;
  std::vector<int> cores;  // empty: not pinned
  int64_t spin_us = cpu_moe::Pool::kDefaultSpinUs;
};

Pools& pools() {
  static Pools p;
  return p;
}

cpu_moe::Pool& pool_for(int64_t threads) {
  Pools& ps = pools();
  std::lock_guard<std::mutex> g(ps.mu);
  auto& p = ps.by_threads[threads];
  if (!p) {
    p = std::make_unique<cpu_moe::Pool>(static_cast<int>(threads));
    p->set_spin_us(ps.spin_us);
    if (!ps.cores.empty()) p->set_cores(ps.cores);
  }
  return *p;
}

void check_cpu(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(
      t.device().is_cpu(), "int4_moe_cpu: ", name, " must be on the CPU");
  TORCH_CHECK(t.is_contiguous(), "int4_moe_cpu: ", name, " must be contiguous");
}

}  // namespace

void int4_moe_cpu(
    torch::Tensor& out,
    const torch::Tensor& x,
    const torch::Tensor& units,
    const torch::Tensor& call_unit,
    const torch::Tensor& call_token,
    const torch::Tensor& call_weight,
    int64_t inter,
    int64_t group,
    c10::IntArrayRef offsets,
    int64_t num_threads,
    int64_t isa,
    int64_t arith) {
  check_cpu(out, "out");
  check_cpu(x, "x");
  check_cpu(units, "units");
  check_cpu(call_unit, "call_unit");
  check_cpu(call_token, "call_token");
  check_cpu(call_weight, "call_weight");
  TORCH_CHECK(
      x.dim() == 2 && x.scalar_type() == at::kBFloat16,
      "int4_moe_cpu: x must be [T, H] bfloat16");
  TORCH_CHECK(
      out.sizes() == x.sizes() && out.scalar_type() == at::kBFloat16,
      "int4_moe_cpu: out must be bfloat16 shaped like x");
  TORCH_CHECK(
      units.dim() == 2 && units.scalar_type() == at::kByte,
      "int4_moe_cpu: units must be [S, U] uint8");
  const int64_t C = call_unit.numel();
  TORCH_CHECK(
      call_unit.scalar_type() == at::kInt &&
          call_token.scalar_type() == at::kInt &&
          call_weight.scalar_type() == at::kFloat,
      "int4_moe_cpu: call_unit, call_token int32, call_weight float32");
  TORCH_CHECK(
      call_token.numel() == C && call_weight.numel() == C,
      "int4_moe_cpu: the call lists differ in length");
  TORCH_CHECK(
      offsets.size() == 4,
      "int4_moe_cpu: offsets are (w13, w2, w13 scales, w2 scales)");
  TORCH_CHECK(
      num_threads >= 1 && num_threads <= 1024,
      "int4_moe_cpu: num_threads out of range");
  TORCH_CHECK(isa >= 0 && isa <= 2, "int4_moe_cpu: isa is 0, 1 or 2");
  TORCH_CHECK(
      arith >= 0 && arith <= 3,
      "int4_moe_cpu: arith is 0 to 3 (bit 0: GEMM1, bit 1: GEMM2 take the "
      "grouped GEMM's bf16 weights instead of the GEMV's chunk-scaled sums)");
  cpu_moe::Layout L;
  L.hidden = static_cast<int>(x.size(1));
  L.inter = static_cast<int>(inter);
  L.group = static_cast<int>(group);
  L.unit_bytes = units.size(1);
  L.off_w13 = offsets[0];
  L.off_w2 = offsets[1];
  L.off_s13 = offsets[2];
  L.off_s2 = offsets[3];
  try {
    cpu_moe::run_calls(
        pool_for(num_threads),
        static_cast<cpu_moe::Isa>(isa),
        L,
        reinterpret_cast<uint16_t*>(out.data_ptr()),
        reinterpret_cast<const uint16_t*>(x.data_ptr()),
        x.size(0),
        units.data_ptr<uint8_t>(),
        units.size(0),
        call_unit.data_ptr<int32_t>(),
        call_token.data_ptr<int32_t>(),
        call_weight.data_ptr<float>(),
        C,
        static_cast<int>(arith));
  } catch (const std::runtime_error& e) {
    TORCH_CHECK(false, e.what());
  }
}

int64_t int4_moe_cpu_configure(c10::IntArrayRef cores, int64_t spin_us) {
  TORCH_CHECK(
      spin_us >= -1 && spin_us <= 1000000,
      "int4_moe_cpu_configure: spin_us out of range");
  std::vector<int> list;
  for (int64_t c : cores) {
    TORCH_CHECK(
        c >= 0 && c < 65536,
        "int4_moe_cpu_configure: core ",
        c,
        " out of range");
    list.push_back(static_cast<int>(c));
  }
  Pools& ps = pools();
  std::lock_guard<std::mutex> g(ps.mu);
  ps.cores = list;
  ps.spin_us = spin_us < 0 ? cpu_moe::Pool::kDefaultSpinUs : spin_us;
  bool ok = true;
  for (auto& [n, p] : ps.by_threads) {
    p->set_spin_us(ps.spin_us);
    ok = p->set_cores(ps.cores) && ok;
  }
  TORCH_CHECK(
      ok,
      "int4_moe_cpu_configure: a worker thread could not be pinned to "
      "the cores given (outside the process's CPU mask?)");
  return static_cast<int64_t>(list.size());
}

cpu_moe::Pool& int4_moe_cpu_pool(int64_t threads) { return pool_for(threads); }

int64_t int4_moe_cpu_isa(int64_t isa) {
  try {
    return static_cast<int64_t>(
        cpu_moe::resolve_isa(static_cast<cpu_moe::Isa>(isa)));
  } catch (const std::runtime_error& e) {
    TORCH_CHECK(false, e.what());
  }
  return 0;
}
