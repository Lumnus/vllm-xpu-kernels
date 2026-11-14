// Host-side torch op over the int4 MoE experts of int4_moe_cpu_impl.h.

#include "cpu_experts/int4_moe_cpu.h"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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
    int64_t arith,
    const std::optional<torch::Tensor>& call_x_row,
    const std::optional<torch::Tensor>& silu_table) {
  check_cpu(out, "out");
  check_cpu(x, "x");
  check_cpu(units, "units");
  check_cpu(call_unit, "call_unit");
  check_cpu(call_token, "call_token");
  check_cpu(call_weight, "call_weight");
  TORCH_CHECK(
      x.dim() == 2 && x.scalar_type() == at::kBFloat16,
      "int4_moe_cpu: x must be [T, H] bfloat16");
  const bool x_rows = call_x_row.has_value();
  TORCH_CHECK(
      out.scalar_type() == at::kBFloat16 &&
          (x_rows ? out.dim() == 2 && out.size(1) == x.size(1)
                  : out.sizes() == x.sizes()),
      "int4_moe_cpu: out must be bfloat16 shaped like x (with call_x_row: "
      "[rows, H])");
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
  if (x_rows) {
    check_cpu(*call_x_row, "call_x_row");
    TORCH_CHECK(
        call_x_row->scalar_type() == at::kInt && call_x_row->numel() == C,
        "int4_moe_cpu: call_x_row is int32, one per call");
  }
  TORCH_CHECK(
      offsets.size() == 4,
      "int4_moe_cpu: offsets are (w13, w2, w13 scales, w2 scales)");
  TORCH_CHECK(
      num_threads >= 1 && num_threads <= 1024,
      "int4_moe_cpu: num_threads out of range");
  TORCH_CHECK(
      isa >= 0 && isa <= cpu_moe::kIsaMax, "int4_moe_cpu: isa is 0 to 8");
  TORCH_CHECK(
      (arith >= 0 && arith <= 3) || arith == cpu_moe::kArithCardExact,
      "int4_moe_cpu: arith is 0 to 3 (bit 0: GEMM1, bit 1: GEMM2 take the "
      "grouped GEMM's bf16 weights instead of the GEMV's chunk-scaled sums), "
      "or 7 (card-exact: the card's grouped GEMM and silu exactly)");
  const uint16_t* table = nullptr;
  if (silu_table.has_value()) {
    check_cpu(*silu_table, "silu_table");
    TORCH_CHECK(
        silu_table->numel() == 65536 && silu_table->element_size() == 2,
        "int4_moe_cpu: silu_table is the card's bf16 silu of every bf16 gate: "
        "65,536 two-byte values (bfloat16 or int16)");
    table = reinterpret_cast<const uint16_t*>(silu_table->data_ptr());
  }
  TORCH_CHECK(
      arith != cpu_moe::kArithCardExact || table != nullptr,
      "int4_moe_cpu: arith 7 (card-exact) needs silu_table, the card's bf16 "
      "silu of every bf16 gate (65,536 entries, taken on the card)");
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
        out.size(0),
        units.data_ptr<uint8_t>(),
        units.size(0),
        call_unit.data_ptr<int32_t>(),
        call_token.data_ptr<int32_t>(),
        call_weight.data_ptr<float>(),
        C,
        static_cast<int>(arith),
        x_rows ? call_x_row->data_ptr<int32_t>() : nullptr,
        x.size(0),
        table);
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

// One pass over a layer call's routing: which calls the host computes, as the
// caller's Python would list them with numpy (see the header). No allocation
// beyond a per-thread table of seen experts.
std::vector<int64_t> int4_moe_cpu_select(
    const torch::Tensor& ids,
    const torch::Tensor& tier,
    const torch::Tensor& slot,
    const torch::Tensor& logical_of_slot,
    torch::Tensor& pair,
    torch::Tensor& logical,
    torch::Tensor& experts,
    torch::Tensor& row_map) {
  check_cpu(ids, "ids");
  check_cpu(tier, "tier");
  check_cpu(slot, "slot");
  check_cpu(logical_of_slot, "logical_of_slot");
  check_cpu(pair, "pair");
  check_cpu(logical, "logical");
  check_cpu(experts, "experts");
  TORCH_CHECK(
      ids.scalar_type() == torch::kInt32 || ids.scalar_type() == torch::kInt64,
      "int4_moe_cpu_select: ids must be int32 or int64");
  TORCH_CHECK(
      tier.scalar_type() == torch::kInt8, "int4_moe_cpu_select: tier is int8");
  TORCH_CHECK(
      slot.scalar_type() == torch::kInt64 &&
          logical_of_slot.scalar_type() == torch::kInt64 &&
          pair.scalar_type() == torch::kInt64 &&
          logical.scalar_type() == torch::kInt64 &&
          experts.scalar_type() == torch::kInt64,
      "int4_moe_cpu_select: slot, logical_of_slot, pair, logical and experts "
      "are int64");
  TORCH_CHECK(
      slot.numel() == tier.numel(),
      "int4_moe_cpu_select: tier and slot have one entry per physical id");
  const int64_t n = ids.numel();
  TORCH_CHECK(
      pair.numel() >= n && logical.numel() >= n && experts.numel() >= n,
      "int4_moe_cpu_select: pair, logical and experts need room for every id");
  const bool with_map = row_map.numel() > 0;
  if (with_map) {
    check_cpu(row_map, "row_map");
    TORCH_CHECK(
        row_map.scalar_type() == torch::kInt32 && row_map.numel() == n,
        "int4_moe_cpu_select: row_map is int32 with one entry per id");
  }
  const int64_t num_global = tier.numel();
  const int64_t num_slots = logical_of_slot.numel();
  const int8_t* tr = tier.data_ptr<int8_t>();
  const int64_t* sl = slot.data_ptr<int64_t>();
  const int64_t* lg = logical_of_slot.data_ptr<int64_t>();
  int64_t* po = pair.data_ptr<int64_t>();
  int64_t* lo = logical.data_ptr<int64_t>();
  int64_t* eo = experts.data_ptr<int64_t>();
  int32_t* mo = with_map ? row_map.data_ptr<int32_t>() : nullptr;
  // counts[t + 1] for tier t in -1 (another rank's), FAST, POOL, HOST
  int64_t counts[4] = {0, 0, 0, 0};
  int64_t calls = 0, distinct = 0, bad = -1;
  // Seen experts: a stamp per logical expert, this call's number when seen.
  thread_local std::vector<uint32_t> seen;
  thread_local uint32_t stamp = 0;
  if (++stamp == 0) {  // wrapped: clear once
    std::fill(seen.begin(), seen.end(), 0u);
    stamp = 1;
  }
  auto body = [&](auto* id) {
    for (int64_t i = 0; i < n; ++i) {
      const int64_t g = static_cast<int64_t>(id[i]);
      if (g < 0 || g >= num_global) {
        bad = i;
        return;
      }
      const int8_t t = tr[g];
      if (t >= -1 && t <= 2) ++counts[t + 1];
      if (mo) mo[i] = -1;
      if (t != 2) continue;  // HOST = 2
      const int64_t s = sl[g];
      TORCH_CHECK(
          s >= 0 && s < num_slots,
          "int4_moe_cpu_select: a host-tier physical id has no slot");
      const int64_t e = lg[s];
      TORCH_CHECK(e >= 0, "int4_moe_cpu_select: a negative logical expert");
      if (mo) mo[i] = static_cast<int32_t>(calls);
      po[calls] = i;
      lo[calls] = e;
      ++calls;
      if (static_cast<uint64_t>(e) >= seen.size()) seen.resize(e + 1, 0u);
      if (seen[e] != stamp) {
        seen[e] = stamp;
        eo[distinct++] = e;
      }
    }
  };
  if (ids.scalar_type() == torch::kInt32)
    body(ids.data_ptr<int32_t>());
  else
    body(ids.data_ptr<int64_t>());
  return {counts[1], counts[2], counts[3], calls, distinct, bad};
}

std::string int4_moe_cpu_isa_name(int64_t isa) {
  TORCH_CHECK(
      isa >= 1 && isa <= cpu_moe::kIsaMax, "int4_moe_cpu: isa is 1 to 8");
  return cpu_moe::level_name(static_cast<int>(isa));
}
