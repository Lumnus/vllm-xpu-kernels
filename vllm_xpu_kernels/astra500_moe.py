# SPDX-License-Identifier: Apache-2.0
"""Measured BF16/W4 small-batch GEMM with a 64-element inner tile."""

from pathlib import Path

import torch

torch.ops.load_library(str(Path(__file__).with_name("libastra_moe.so")))


def astra_grouped_gemm(**kwargs):
    a, weight, scales, out = (
        kwargs["ptr_A"], kwargs["ptr_B"], kwargs.get("ptr_B_scale"), kwargs["ptr_D"]
    )
    if (
        a.dtype == torch.bfloat16
        and weight.dtype == torch.int8
        and a.shape[0] <= 320
        and scales is not None
        and scales.dtype == torch.bfloat16
        and scales.ndim == 3
        and scales.shape[2] * 128 == kwargs["K"]
        and kwargs.get("ptr_bias") is None
        and kwargs.get("ptr_A_scale") is None
    ):
        torch.ops.astra_moe_k.grouped(
            a, weight, scales, out, kwargs["rows_per_expert"],
            kwargs["N"], kwargs["K"], kwargs["num_experts"], 6,
        )
        return out
    return torch.ops._xpu_C.cutlass_grouped_gemm_interface(**kwargs)
