# SPDX-License-Identifier: Apache-2.0
"""B70 K1: small-M int8 dense GEMMs over B70-IP-W1's int8 weights.

W1 (vLLM ``b70_dense_int8``) stores each converted linear as int8 ``[N, K]``
(symmetric per output row, RTN) plus an fp32 scale ``[N, 1]``. At decode
sizes its path (``scaled_int8_quant`` + Triton ``scaled_mm``) is dominated by
fixed cost. These ops consume the same weights unchanged:

  ``torch.ops._xpu_C.b70_w8a16_gemm(x, w, w_scale, bias)``
      weight-only: bf16 x, fp32 FMA, no activation quant pass. M <= ~16.
  ``torch.ops._xpu_C.b70_w8a8_fused_gemm(x, w, w_scale, bias)``
      per-token int8 quant of x inside the GEMM (same scale rule as
      scaled_int8_quant), s8 x s8 DPAS. M <= 64, K % 32 == 0, N % 8 == 0.

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
       w8a16_max_m: int = 8,
       fused_max_m: int = 64) -> Optional[torch.Tensor]:
    """x [M, K] bf16/fp16 contiguous, w int8 [N, K], w_scale fp32 [N] / [N, 1]."""
    m, k = x.shape
    n = w.shape[0]
    if x.data_ptr() % 16 or w.data_ptr() % 16:
        return None  # the kernels use 16-byte vector loads
    if m <= w8a16_max_m and k % 16 == 0:
        return torch.ops._xpu_C.b70_w8a16_gemm(x, w, w_scale, bias)
    if m <= min(fused_max_m, 64) and k % 32 == 0 and n % 8 == 0:
        return torch.ops._xpu_C.b70_w8a8_fused_gemm(x, w, w_scale, bias)
    return None
