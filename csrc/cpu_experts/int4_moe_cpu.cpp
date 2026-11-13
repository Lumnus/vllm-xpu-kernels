// Host-side torch op over the int4 MoE experts of int4_moe_cpu_impl.h.

#include "cpu_experts/int4_moe_cpu.h"

#include <map>
#include <memory>
#include <mutex>

#include "cpu_experts/int4_moe_cpu_impl.h"

namespace {

cpu_moe::Pool& pool_for(int64_t threads) {
  static std::mutex mu;
  static std::map<int64_t, std::unique_ptr<cpu_moe::Pool>> pools;
  std::lock_guard<std::mutex> g(mu);
  auto& p = pools[threads];
  if (!p) p = std::make_unique<cpu_moe::Pool>(static_cast<int>(threads));
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
    int64_t isa) {
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
        C);
  } catch (const std::runtime_error& e) {
    TORCH_CHECK(false, e.what());
  }
}

int64_t int4_moe_cpu_isa(int64_t isa) {
  try {
    return static_cast<int64_t>(
        cpu_moe::resolve_isa(static_cast<cpu_moe::Isa>(isa)));
  } catch (const std::runtime_error& e) {
    TORCH_CHECK(false, e.what());
  }
  return 0;
}
