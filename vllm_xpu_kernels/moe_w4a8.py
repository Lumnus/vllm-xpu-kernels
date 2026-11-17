# SPDX-License-Identifier: Apache-2.0
"""W4A8 routed experts on the cutlass Xe2 grouped GEMM.

The int4 g128 expert weights stay the bytes ``XpuFusedMoe`` holds after
``implement_zp`` (signed nibbles, [E, N, K/2], bf16 scales [E, N, K/128]); no
copy, no repack. Activations are quantized to int8 per token (dynamic,
symmetric) once before each grouped GEMM: the token rows before the routing
remap (the remap moves int8 bytes and their scales together), the SiLU output
before gemm2. The GEMM is ``cutlass_grouped_gemm_w4a8``: the Xe2
mixed-precision DPAS s8 x s4 -> s32 reads the nibbles directly (no unpack),
the int32 partial of each 128-wide K group is scaled by w_scale[n, g] into
fp32, and the token scale is applied once at the end.

With ``fused_act=True`` (default off) gemm1 runs
``cutlass_grouped_gemm_w4a8_silu_quant`` instead: its epilogue computes
silu(gate) * up in fp32 and writes it as int8 with one fp32 scale per (row,
128 columns), which gemm2 consumes directly (A_scale [rows, inter/128]); the
bf16 gemm1 output, the silu_and_mul kernel and the second quant_rows go away.

Both ops are registered in ``_xpu_C``.

Around the GEMMs the path launches only what the result needs. Rows whose ids
are all negative (DP or graph padding) are not masked: the activation quant
is per row, ``remap_hidden_states`` never copies a slot whose id is negative,
out of range or not owned by this rank, and ``moe_gather`` writes zeros for a
row with no computed slot, so such a row comes out as zeros whatever its
activations hold (NaN included), as on ``XpuFusedMoe._apply_kernel``. The
remap clears ``rows_per_expert`` and counts every local expert's rows into
it, so the caller's buffer is handed to it directly: no zero fill and no
second count. ``VLLM_XPU_W4A8_STOCK_PREP=1`` restores the earlier sequence
(padding mask, out-of-place ``masked_fill``, own zeroed counts, and the
caller's counts derived again in torch). Read once at import.

``VLLM_XPU_W4A8_FUSED_MOE`` (default 0, read once at import) runs the layer
on the fused decode ops (csrc/xpu/grouped_gemm/xe_2_w4a8/w4a8_moe_fused.h)
when a call has at most ``FUSED_MAX_ROUTES`` routes and the unfused dispatch
would pick an 8, 16 or 32-row tile:
  1  map + quant/gather, gemm1 over the map's tiles, silu_and_mul + quant in
     one kernel, gemm2, moe_gather: 6 launches against 11.
  2  as 1, but silu_and_mul and the quant stay the two stock ops: 7.
The values per element are those of the unfused path; only row positions
and the GEMM scheduler change. Larger calls, ``fused_act`` and the stock
prep keep the unfused path.

Two optional inputs let a caller drop work it already did or still has to
do around the layer; neither changes a value:
  x_q     the token rows already quantized by the per-token int8 op
          (``_C.dynamic_per_token_scaled_int8_quant``, the op this path
          calls itself) at the MoE entry, shared with the other consumers of
          the row. The unfused path remaps them instead of quantizing again;
          the fused path gathers them (``moe_w4a8_route_gather_q``) instead
          of quantizing in its gather.
  shared  ``(out, gate)``: the shared expert's output and its sigmoid gate
          column. ``moe_gather`` adds ``gate * out`` to the routed sum with
          the rounding points of the separate multiply and add
          (``out = T(T(acc) + T(gate * out))``), so the caller drops both.
          ``apply`` returns True when it did; False leaves the add to the
          caller (a kernels build whose moe_gather has no shared term).
          ``(out, gate, True)`` asks for one fp32 rounding of
          ``acc + gate * out`` instead (different bits: no T rounding of
          the routed sum or of the gated product).
"""

import os
from typing import Optional

import torch

GROUP = 128

STOCK_PREP_ENV = "VLLM_XPU_W4A8_STOCK_PREP"
_STOCK_PREP = os.environ.get(STOCK_PREP_ENV, "0").strip().upper() in (
    "1", "TRUE", "YES", "ON")

# The grouped GEMM's M tile. The kernel's own rule divides the routed-row
# buffer (T * top_k: every route of the call, owned by this rank or not) by
# the LOCAL expert count, so under expert parallelism it sees ep_size x the
# rows an expert really gets and picks the 128 x 256 tile for prefill chunks
# whose experts hold 64-128 rows. "rows" (default) applies the same
# thresholds to the rows per expert this rank really gets in expectation,
# routes / total experts (n_moe / local / ep_size), and passes the tile to
# the op. The one-card sweep (both GEMMs, uniform and routed, 32-256 rows
# per local expert) measured 32 x 64 at 1.23-1.51x the 128 x 256 at 64-81
# rows and every tile bit-equal, so the tile changes speed, never bits.
# VLLM_XPU_W4A8_TILE_RULE=buffer restores the kernel's rule;
# VLLM_XPU_W4A8_TILE (a fixed tile for every call) still overrides both.
TILE_RULE_ENV = "VLLM_XPU_W4A8_TILE_RULE"
_TILE_RULE = os.environ.get(TILE_RULE_ENV, "rows").strip().lower() or "rows"
if _TILE_RULE not in ("rows", "buffer"):
    raise ValueError(f"{TILE_RULE_ENV}={_TILE_RULE!r}: rows or buffer")
_TILE_FIXED = os.environ.get("VLLM_XPU_W4A8_TILE", "").strip().lower() not in (
    "", "auto")
# avg rows per expert -> tile (the kernel's thresholds: 4 / 8 / 128).
TILE_TABLE = ((4, "8x64", 8), (8, "16x64", 16), (128, "32x64", 32))
TILE_BIG = "128x256"


def tile_rule() -> str:
    """VLLM_XPU_W4A8_TILE_RULE as read at import: rows or buffer."""
    return _TILE_RULE


def rows_per_expert_avg(n_moe: int, num_experts: int,
                        total_experts: Optional[int] = None) -> int:
    """The average the tile is chosen from: routes / total experts under
    the rows rule (this rank's real rows per local expert in expectation),
    routes / local experts under the buffer rule (the kernel's own)."""
    if _TILE_RULE == "rows" and total_experts:
        return n_moe // max(total_experts, 1)
    return n_moe // max(num_experts, 1)


def gemm_tile(n_moe: int, num_experts: int,
              total_experts: Optional[int] = None) -> Optional[str]:
    """The tile name to pass to cutlass_grouped_gemm_w4a8, or None to let
    the op choose (buffer rule, or a fixed VLLM_XPU_W4A8_TILE)."""
    if _TILE_RULE != "rows" or _TILE_FIXED:
        return None
    avg = rows_per_expert_avg(n_moe, num_experts, total_experts)
    for bound, name, _ in TILE_TABLE:
        if avg <= bound:
            return name
    return TILE_BIG


FUSED_ENV = "VLLM_XPU_W4A8_FUSED_MOE"
_FUSED_RAW = os.environ.get(FUSED_ENV, "0").strip()
_FUSED = int(_FUSED_RAW) if _FUSED_RAW in ("0", "1", "2") else 0
# The map kernel's bound (w4a8_moe_fused_prep.cpp kMaxRoutes, kMaxLocal).
FUSED_MAX_ROUTES = 2048
FUSED_MAX_LOCAL = 512

_OP = None
_GLU = None
_FOPS = None
_LOGGED_FUSED = False
_LOGGED_PREP = False
_LOGGED_FUSED_MOE = False


def stock_prep() -> bool:
    """True when VLLM_XPU_W4A8_STOCK_PREP=1 keeps the earlier padding mask
    and count re-derivation around the W4A8 GEMMs."""
    return _STOCK_PREP


def _log_prep_once() -> None:
    global _LOGGED_PREP
    if _LOGGED_PREP:
        return
    _LOGGED_PREP = True
    import logging
    if _STOCK_PREP:
        msg = ("W4A8 MoE prep: padding mask and torch re-count of "
               "rows_per_expert kept (%s=1)")
    else:
        msg = ("W4A8 MoE prep: no padding mask, no re-count; the remap fills "
               "the caller's rows_per_expert (%s=1 restores both)")
    logging.getLogger("vllm_xpu_kernels").warning(msg, STOCK_PREP_ENV)


def load_op():
    """The W4A8 grouped GEMM op from ``_xpu_C``."""
    global _OP
    if _OP is None:
        _OP = torch.ops._xpu_C.cutlass_grouped_gemm_w4a8
    return _OP


def load_glu_op():
    """The fused gemm1 op from ``_xpu_C``, or None if absent."""
    global _GLU
    if _GLU is None:
        try:
            _GLU = torch.ops._xpu_C.cutlass_grouped_gemm_w4a8_silu_quant
        except (AttributeError, RuntimeError):
            return None
    return _GLU


def fused_mode() -> int:
    """VLLM_XPU_W4A8_FUSED_MOE as read at import: 0 off, 1, 2."""
    return _FUSED


def load_fused_ops():
    """The fused decode ops from ``_xpu_C``, or None if this build lacks
    them."""
    global _FOPS
    if _FOPS is None:
        try:
            xo = torch.ops._xpu_C
            _FOPS = (xo.moe_w4a8_route_quant, xo.moe_w4a8_tile_gemm,
                     xo.moe_w4a8_silu_mul_quant)
        except (AttributeError, RuntimeError):
            _FOPS = False
    return _FOPS or None


def unfused_tile(n_moe: int, num_experts: int,
                 total_experts: Optional[int] = None) -> Optional[int]:
    """The M tile the unfused dispatch picks for a call of ``n_moe`` routed
    rows (the 4 / 8 / 128 thresholds on rows_per_expert_avg), when it is one
    of the fused path's 8 / 16 / 32; else None."""
    avg = rows_per_expert_avg(n_moe, num_experts, total_experts)
    for bound, _, rows in TILE_TABLE:
        if avg <= bound:
            return rows
    return None


def fused_umt(n_moe: int, tile_m: int, num_experts: int) -> int:
    """Tiles the map can need: each expert with c > 0 rows takes
    ceil(c / tile_m) <= c / tile_m + 1 tiles, and no tile is empty."""
    return min(n_moe, n_moe // tile_m + min(n_moe, num_experts))


def _log_fused_moe_once(mode: int, tile_m: int) -> None:
    global _LOGGED_FUSED_MOE
    if _LOGGED_FUSED_MOE:
        return
    _LOGGED_FUSED_MOE = True
    import logging
    logging.getLogger("vllm_xpu_kernels").warning(
        "W4A8 MoE fused decode on (%s=%d): map + quant/gather, gemm1 and "
        "gemm2 over the tile map%s, moe_gather; first call tile %d rows, "
        "up to %d routes per call", FUSED_ENV, mode,
        ", silu_and_mul + quant in one kernel" if mode == 1 else
        ", stock silu_and_mul and quant", tile_m, FUSED_MAX_ROUTES)


_GATHER_SHARED = None
_GATHER_Q = None


def gather_takes_shared() -> bool:
    """Whether ``_moe_C.moe_gather`` takes the shared expert's term."""
    global _GATHER_SHARED
    if _GATHER_SHARED is None:
        try:
            schema = torch.ops._moe_C.moe_gather.default._schema
            _GATHER_SHARED = any(a.name == "shared_gate"
                                 for a in schema.arguments)
        except (AttributeError, RuntimeError):
            _GATHER_SHARED = False
    return _GATHER_SHARED


def load_gather_q_op():
    """``_xpu_C.moe_w4a8_route_gather_q`` (the fused map with a gather of
    pre-quantized rows), or None if this build lacks it."""
    global _GATHER_Q
    if _GATHER_Q is None:
        try:
            _GATHER_Q = torch.ops._xpu_C.moe_w4a8_route_gather_q
        except (AttributeError, RuntimeError):
            _GATHER_Q = False
    return _GATHER_Q or None


# Markers the engine reads: apply takes x_q and shared.
TAKES_ENTRY_ROWS = True
TAKES_SHARED = True


def _entry_rows_ok(x_q, hs: torch.Tensor) -> bool:
    if x_q is None:
        return False
    q, sc = x_q
    return (q.dtype == torch.int8 and q.shape == hs.shape
            and q.device == hs.device and q.stride(-1) == 1
            and sc.dtype == torch.float32 and sc.numel() == hs.shape[0])


def _gather(output, g2, topk_weights, unp, num_experts, shared) -> bool:
    if shared is not None and gather_takes_shared():
        out, gate = shared[0], shared[1]
        if len(shared) > 2 and shared[2]:
            # Tier 2: one fp32 rounding of acc + gate * out.
            torch.ops._moe_C.moe_gather(output, g2, topk_weights, unp,
                                        num_experts, out, gate, True)
        else:
            torch.ops._moe_C.moe_gather(output, g2, topk_weights, unp,
                                        num_experts, out, gate)
        return True
    torch.ops._moe_C.moe_gather(output, g2, topk_weights, unp, num_experts)
    return False


_LOGGED_TILE = False


def _log_tile_once(n_moe: int, impl, tile: Optional[str]) -> None:
    global _LOGGED_TILE
    if _LOGGED_TILE:
        return
    _LOGGED_TILE = True
    import logging
    logging.getLogger("vllm_xpu_kernels").warning(
        "W4A8 MoE tile rule: %s (%s=%s); first call %d routes over %d of %d "
        "experts -> %s", "rows per expert on this rank (routes / total "
        "experts)" if _TILE_RULE == "rows" and not _TILE_FIXED else
        ("fixed VLLM_XPU_W4A8_TILE" if _TILE_FIXED else
         "kernel rule (routes / local experts)"), TILE_RULE_ENV, _TILE_RULE,
        n_moe, impl.num_experts, impl.total_experts_num, tile or "op's choice")


def quant_rows(x: torch.Tensor):
    """Symmetric dynamic int8 per row: (int8 [T, H], fp32 scale [T, 1])."""
    t = x.shape[0]
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    s = torch.empty((t, 1), dtype=torch.float32, device=x.device)
    if t == 0:
        return q, s
    try:
        torch.ops._C.dynamic_per_token_scaled_int8_quant(q, x.contiguous(), s)
    except (AttributeError, RuntimeError):
        xf = x.float()
        s = (xf.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-10)
        q = torch.round(xf / s).clamp_(-128, 127).to(torch.int8)
    return q, s


def eligible(impl) -> Optional[str]:
    """None if ``impl`` (an XpuFusedMoe) can run W4A8, else the reason."""
    if not getattr(impl, "is_int4", False):
        return "not an int4 MoE"
    if impl.activation != "silu" or impl.gemm1_clamp_limit:
        return f"activation {impl.activation} / clamp {impl.gemm1_clamp_limit}"
    if getattr(impl, "_astra_groups", None) is not None:
        return "astra group ownership re-laid out the weights"
    for w, s in ((impl.w13, impl.gemm1_wei_scales),
                 (impl.w2, impl.gemm2_wei_scales)):
        if w.element_size() != 1 or w.dim() != 3 or not w.is_contiguous():
            return f"weight {w.dtype} {tuple(w.shape)}"
        if (s is None or s.dim() != 3 or s.dtype != torch.bfloat16
                or s.shape[-1] * GROUP != w.shape[-1] * 2
                or not s.is_contiguous()):
            return "scales are not bf16 [E, N, K/128] contiguous"
        if w.shape[1] % 64 or (w.shape[2] * 2) % 64:
            return f"N/K of {tuple(w.shape)} not multiples of 64"
    return None


def apply(impl,
          output: torch.Tensor,
          hidden_states: torch.Tensor,
          topk_weights: torch.Tensor,
          topk_ids: torch.Tensor,
          expert_map: Optional[torch.Tensor] = None,
          fused_act: bool = False,
          rows_per_expert: Optional[torch.Tensor] = None,
          x_q=None,
          shared=None) -> bool:
    """Drop-in for XpuFusedMoe._apply_kernel with the W4A8 grouped GEMMs.

    ``rows_per_expert``: optional int32 ``[num_experts]`` buffer of the
    caller; the remap clears it and fills it with each local expert's rows.
    With VLLM_XPU_W4A8_STOCK_PREP=1 it is ignored here (the caller derives it
    again, see ``XpuFusedMoe.apply``).

    ``x_q`` / ``shared``: see the module docstring. Returns True when the
    shared expert's term was added in the gather."""
    op = load_op()
    _log_prep_once()
    num_rows, hidden = hidden_states.shape
    topk = impl.n_experts_per_token
    n_moe = num_rows * topk
    inter = impl.inter_size
    dev = hidden_states.device
    if expert_map is None and impl.ep_size > 1:
        expert_map = impl.expert_map
    if _STOCK_PREP:
        # Rows routed nowhere (all -1, DP padding) are zeroed.
        invalid = (topk_ids < 0).all(dim=1)
        hs = hidden_states.masked_fill(invalid.unsqueeze(1), 0)
        rows_per_expert = torch.zeros((impl.num_experts, ),
                                      dtype=torch.int32, device=dev)
        x_q = None
    else:
        # No padding mask: such a row is never remapped and is gathered as
        # zeros (module docstring). The remap clears rows_per_expert first.
        hs = hidden_states
        if rows_per_expert is None:
            rows_per_expert = torch.empty((impl.num_experts, ),
                                          dtype=torch.int32, device=dev)
        tile_m = (unfused_tile(n_moe, impl.num_experts,
                               impl.total_experts_num) if _FUSED else None)
        if (tile_m is not None and not fused_act and 0 < n_moe
                and n_moe <= FUSED_MAX_ROUTES
                and impl.num_experts <= FUSED_MAX_LOCAL and topk <= 16
                and hs.dtype == torch.bfloat16
                and output.dtype == torch.bfloat16 and hidden % 8 == 0
                and inter % 8 == 0 and load_fused_ops() is not None):
            return _apply_fused(impl, output, hs, topk_weights, topk_ids,
                                expert_map, rows_per_expert, tile_m, x_q,
                                shared)

    if _entry_rows_ok(x_q, hs):
        hq, hsc = x_q
        hsc = hsc.reshape(-1, 1)
    else:
        hq, hsc = quant_rows(hs)
    # The stock remap copies bytes (e4m3 = uint8 path) and routes the
    # per-token scale ([T, 1] -> block_k = hidden) with them.
    remapped = torch.empty((n_moe, hidden), dtype=torch.float8_e4m3fn,
                           device=dev)
    remapped_sc = torch.empty((n_moe, 1), dtype=torch.float32, device=dev)
    unp = torch.empty((num_rows, topk), dtype=torch.int32, device=dev)
    torch.ops._moe_C.remap_hidden_states(
        hidden_states=hq.view(torch.float8_e4m3fn),
        hidden_states_scales=hsc,
        remapped_hidden_states=remapped,
        remapped_hidden_states_scales=remapped_sc,
        expert_map=expert_map,
        rows_per_expert=rows_per_expert,
        unpermuted_row_to_permuted_row=unp,
        topk_ids=topk_ids,
        total_experts_num=impl.total_experts_num,
        local_experts_num=impl.local_experts_num)

    tile = gemm_tile(n_moe, impl.num_experts, impl.total_experts_num)
    _log_tile_once(n_moe, impl, tile)
    glu = load_glu_op() if fused_act and inter % 128 == 0 else None
    if glu is not None:
        global _LOGGED_FUSED
        if not _LOGGED_FUSED:
            _LOGGED_FUSED = True
            import logging
            logging.getLogger("vllm_xpu_kernels").warning(
                "W4A8 MoE fused gemm1 epilogue on (silu*up -> int8, "
                "per-row x 128-col scales; inter=%d)", inter)
        aq = torch.empty((n_moe, inter), dtype=torch.int8, device=dev)
        asc = torch.empty((n_moe, inter // 128), dtype=torch.float32,
                          device=dev)
        glu(remapped.view(torch.int8), remapped_sc, impl.w13,
            impl.gemm1_wei_scales, impl.w13_bias, aq, asc, rows_per_expert,
            inter, hidden, impl.num_experts)
    else:
        g1 = torch.empty((n_moe, 2 * inter), dtype=output.dtype, device=dev)
        op(remapped.view(torch.int8), remapped_sc, impl.w13,
           impl.gemm1_wei_scales, impl.w13_bias, g1, rows_per_expert,
           2 * inter, hidden, impl.num_experts, tile)
        act = torch.empty((n_moe, inter), dtype=output.dtype, device=dev)
        impl.act_func(act, g1)
        aq, asc = quant_rows(act)
    g2 = torch.empty((n_moe, hidden), dtype=output.dtype, device=dev)
    op(aq, asc, impl.w2, impl.gemm2_wei_scales, impl.w2_bias, g2,
       rows_per_expert, hidden, inter, impl.num_experts, tile)
    return _gather(output, g2, topk_weights, unp, impl.num_experts, shared)


def _apply_fused(impl, output, hs, topk_weights, topk_ids, expert_map,
                 rows_per_expert, tile_m: int, x_q=None,
                 shared=None) -> bool:
    """The layer on the fused decode ops (module docstring). Same contract as
    the unfused apply: output, the caller's rows_per_expert filled with each
    local expert's rows, padding / unrouted / unowned slots skipped."""
    route_quant, tile_gemm, silu_quant = load_fused_ops()
    num_rows, hidden = hs.shape
    topk = impl.n_experts_per_token
    n_moe = num_rows * topk
    inter = impl.inter_size
    dev = hs.device
    _log_fused_moe_once(_FUSED, tile_m)
    umt = fused_umt(n_moe, tile_m, impl.num_experts)
    ws = torch.empty((2 + 3 * umt, ), dtype=torch.int32, device=dev)
    aq = torch.empty((n_moe, hidden), dtype=torch.int8, device=dev)
    asc = torch.empty((n_moe, ), dtype=torch.float32, device=dev)
    unp = torch.empty((num_rows, topk), dtype=torch.int32, device=dev)
    # 1 + 2: map (rows_per_expert, route -> row, tiles), quant + gather; or
    # the gather of the rows quantized at the MoE entry.
    gather_q = load_gather_q_op() if _entry_rows_ok(x_q, hs) else None
    if gather_q is not None:
        gather_q(x_q[0], x_q[1], topk_ids, expert_map, aq, asc, unp,
                 rows_per_expert, ws, impl.total_experts_num,
                 impl.local_experts_num, tile_m)
    else:
        route_quant(hs, topk_ids, expert_map, aq, asc, unp, rows_per_expert,
                    ws, impl.total_experts_num, impl.local_experts_num,
                    tile_m)
    # 3: gemm1 over the tile map.
    g1 = torch.empty((n_moe, 2 * inter), dtype=output.dtype, device=dev)
    tile_gemm(aq, asc, impl.w13, impl.gemm1_wei_scales, impl.w13_bias, g1,
              ws, 2 * inter, hidden, tile_m)
    # 4: silu_and_mul + the per-token quant of its output.
    if _FUSED == 1:
        q2 = torch.empty((n_moe, inter), dtype=torch.int8, device=dev)
        s2 = torch.empty((n_moe, ), dtype=torch.float32, device=dev)
        silu_quant(g1, q2, s2, ws)
    else:
        act = torch.empty((n_moe, inter), dtype=output.dtype, device=dev)
        impl.act_func(act, g1)
        q2, s2 = quant_rows(act)
        s2 = s2.view(-1)
    # 5: gemm2 over the tile map; then the stock gather through the map.
    g2 = torch.empty((n_moe, hidden), dtype=output.dtype, device=dev)
    tile_gemm(q2, s2, impl.w2, impl.gemm2_wei_scales, impl.w2_bias, g2, ws,
              hidden, inter, tile_m)
    return _gather(output, g2, topk_weights, unp, impl.num_experts, shared)
