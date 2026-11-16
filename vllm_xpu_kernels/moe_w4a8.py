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
"""

import os
from typing import Optional

import torch

GROUP = 128

STOCK_PREP_ENV = "VLLM_XPU_W4A8_STOCK_PREP"
_STOCK_PREP = os.environ.get(STOCK_PREP_ENV, "0").strip().upper() in (
    "1", "TRUE", "YES", "ON")

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


def unfused_tile(n_moe: int, num_experts: int) -> Optional[int]:
    """The M tile the unfused dispatch picks for a call of ``n_moe`` routed
    rows (grouped_gemm_w4a8_xe2.cpp: avg = rows / experts against 4 / 8 /
    128), when it is one of the fused path's 8 / 16 / 32; else None."""
    avg = n_moe // max(num_experts, 1)
    if avg <= 4:
        return 8
    if avg <= 8:
        return 16
    if avg <= 128:
        return 32
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
          rows_per_expert: Optional[torch.Tensor] = None) -> None:
    """Drop-in for XpuFusedMoe._apply_kernel with the W4A8 grouped GEMMs.

    ``rows_per_expert``: optional int32 ``[num_experts]`` buffer of the
    caller; the remap clears it and fills it with each local expert's rows.
    With VLLM_XPU_W4A8_STOCK_PREP=1 it is ignored here (the caller derives it
    again, see ``XpuFusedMoe.apply``)."""
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
    else:
        # No padding mask: such a row is never remapped and is gathered as
        # zeros (module docstring). The remap clears rows_per_expert first.
        hs = hidden_states
        if rows_per_expert is None:
            rows_per_expert = torch.empty((impl.num_experts, ),
                                          dtype=torch.int32, device=dev)
        tile_m = unfused_tile(n_moe, impl.num_experts) if _FUSED else None
        if (tile_m is not None and not fused_act and 0 < n_moe
                and n_moe <= FUSED_MAX_ROUTES
                and impl.num_experts <= FUSED_MAX_LOCAL and topk <= 16
                and hs.dtype == torch.bfloat16
                and output.dtype == torch.bfloat16 and hidden % 8 == 0
                and inter % 8 == 0 and load_fused_ops() is not None):
            _apply_fused(impl, output, hs, topk_weights, topk_ids,
                         expert_map, rows_per_expert, tile_m)
            return

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
           2 * inter, hidden, impl.num_experts)
        act = torch.empty((n_moe, inter), dtype=output.dtype, device=dev)
        impl.act_func(act, g1)
        aq, asc = quant_rows(act)
    g2 = torch.empty((n_moe, hidden), dtype=output.dtype, device=dev)
    op(aq, asc, impl.w2, impl.gemm2_wei_scales, impl.w2_bias, g2,
       rows_per_expert, hidden, inter, impl.num_experts)
    torch.ops._moe_C.moe_gather(output, g2, topk_weights, unp,
                                impl.num_experts)


def _apply_fused(impl, output, hs, topk_weights, topk_ids, expert_map,
                 rows_per_expert, tile_m: int) -> None:
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
    # 1 + 2: map (rows_per_expert, route -> row, tiles), quant + gather.
    route_quant(hs, topk_ids, expert_map, aq, asc, unp, rows_per_expert, ws,
                impl.total_experts_num, impl.local_experts_num, tile_m)
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
    torch.ops._moe_C.moe_gather(output, g2, topk_weights, unp,
                                impl.num_experts)
