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

With ``B70_W4A8_FUSED_ACT=1`` (default off) gemm1 runs
``cutlass_grouped_gemm_w4a8_silu_quant`` instead: its epilogue computes
silu(gate) * up in fp32 and writes it as int8 with one fp32 scale per (row,
128 columns), which gemm2 consumes directly (A_scale [rows, inter/128]); the
bf16 gemm1 output, the silu_and_mul kernel and the second quant_rows go away.

Both ops are registered in ``_xpu_C``.
"""

import os
from typing import Optional

import torch

GROUP = 128

_OP = None
_GLU = None
_LOGGED_FUSED = False


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


def fused_act_enabled() -> bool:
    return os.environ.get("B70_W4A8_FUSED_ACT", "0") == "1"


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
          expert_map: Optional[torch.Tensor] = None) -> None:
    """Drop-in for XpuFusedMoe._apply_kernel with the W4A8 grouped GEMMs."""
    op = load_op()
    num_rows, hidden = hidden_states.shape
    topk = impl.n_experts_per_token
    n_moe = num_rows * topk
    inter = impl.inter_size
    dev = hidden_states.device
    if expert_map is None and impl.ep_size > 1:
        expert_map = impl.expert_map
    # Same as stock: rows routed nowhere (all -1, DP padding) are zeroed.
    invalid = (topk_ids < 0).all(dim=1)
    hs = hidden_states.masked_fill(invalid.unsqueeze(1), 0)

    hq, hsc = quant_rows(hs)
    # The stock remap copies bytes (e4m3 = uint8 path) and routes the
    # per-token scale ([T, 1] -> block_k = hidden) with them.
    remapped = torch.empty((n_moe, hidden), dtype=torch.float8_e4m3fn,
                           device=dev)
    remapped_sc = torch.empty((n_moe, 1), dtype=torch.float32, device=dev)
    rows_per_expert = torch.zeros((impl.num_experts, ), dtype=torch.int32,
                                  device=dev)
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

    glu = load_glu_op() if fused_act_enabled() and inter % 128 == 0 else None
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
