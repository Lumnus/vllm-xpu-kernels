# SPDX-License-Identifier: Apache-2.0
"""B70 K1: small-M int8 dense GEMMs over B70-IP-W1's int8 weights.

W1 (vLLM ``b70_dense_int8``) stores each converted linear as int8 ``[N, K]``
(symmetric per output row, RTN) plus an fp32 scale ``[N, 1]``. At decode
sizes its path (``scaled_int8_quant`` + Triton ``scaled_mm``) is dominated by
fixed cost. These ops consume the same weights unchanged:

  ``torch.ops._xpu_C.b70_w8a16_gemm(x, w, w_scale, bias)``
      weight-only: bf16 x, fp32 FMA, no activation quant pass. M <= ~16.
  ``torch.ops._xpu_C.b70_w8a8_fused_gemm(x, w, w_scale, bias)``
      per-token int8 quant (same rule as scaled_int8_quant) + the cutlass
      s8 x s8 streaming GEMM; weights read once. Any M (tuned for M <= 64), K % 32 == 0,
      N % 16 == 0, N >= 32, bf16 x.
  ``torch.ops._xpu_C.b70_w8a8_gemm(xq, xs, w, w_scale, bias)``  the GEMM alone.

``mm`` picks one by M and returns None when neither applies (the caller keeps
its own path). Thresholds come from the caller (vLLM env, see the patch).
"""

from typing import Optional

import torch


def available() -> bool:
    try:
        return (hasattr(torch.ops._xpu_C, "b70_w8a16_gemm")
                and hasattr(torch.ops._xpu_C, "b70_w8a8_fused_gemm"))
    except (AttributeError, RuntimeError):
        return False


def mm(x: torch.Tensor,
       w: torch.Tensor,
       w_scale: torch.Tensor,
       bias: Optional[torch.Tensor] = None,
       w8a16_max_m: int = 2,
       fused_max_m: int = 64) -> Optional[torch.Tensor]:
    """x [M, K] bf16/fp16 contiguous, w int8 [N, K], w_scale fp32 [N] / [N, 1]."""
    m, k = x.shape
    n = w.shape[0]
    if x.data_ptr() % 16 or w.data_ptr() % 16:
        return None  # the kernels use 16-byte vector loads
    if m <= w8a16_max_m and k % 16 == 0:
        return torch.ops._xpu_C.b70_w8a16_gemm(x, w, w_scale, bias)
    if (m <= fused_max_m and k % 32 == 0 and n % 16 == 0 and n >= 32
            and x.dtype == torch.bfloat16):
        return torch.ops._xpu_C.b70_w8a8_fused_gemm(x, w, w_scale, bias)
    return None


def quantize_weight_int8(w: torch.Tensor):
    """[N, K] float -> (int8 [N, K], fp32 scale [N, 1]): symmetric per row, RTN (W1's rule)."""
    n = w.shape[0]
    q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    s = torch.empty((n, 1), dtype=torch.float32, device=w.device)
    step = max(1, (64 << 20) // max(1, w.shape[1] * 4))
    for i in range(0, n, step):
        blk = w[i:i + step].float()
        sc = (blk.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
        q[i:i + step] = torch.round(blk / sc).clamp(-127, 127).to(torch.int8)
        s[i:i + step] = sc
    return q, s


def hc_down(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor) -> torch.Tensor:
    """HC down projection in int8 (B70 K1): per-token quant + the s8 x s8 GEMM.

    A row's output depends only on that row and the weights (per-row quant, exact int32
    accumulation for every K split, one fp32 epilogue): the same bits for any batch
    composition and run, which the ordered (DFlash2 fixed-order) HC path requires.
    x [R, K] bf16, w_q int8 [N, K], w_s fp32 [N, 1]."""
    return torch.ops._xpu_C.b70_w8a8_fused_gemm(x.contiguous(), w_q, w_s, None)
