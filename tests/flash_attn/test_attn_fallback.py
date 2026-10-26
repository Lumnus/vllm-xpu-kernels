# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When the attention kernel for a config is not compiled,
flash_attn_varlen_func falls back to a PyTorch reference implementation.
These tests stub the compiled op, so they need no XPU kernel for the shape."""

import pytest
import torch
import torch.nn.functional as F

from vllm_xpu_kernels import flash_attn_interface

NOT_COMPILED = ("Paged decode kernel not compiled for this configuration.\n"
                "Add this line to your paged_decode config file: "
                "16,256,64,false,false,false")


def _not_compiled(*args, **kwargs):
    raise RuntimeError(NOT_COMPILED)


def _inputs(device):
    torch.manual_seed(0)
    seq_len, num_heads, head_size = 8, 2, 64
    q, k, v = (torch.randn(seq_len, num_heads, head_size, device=device)
               for _ in range(3))
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    return q, k, v, cu_seqlens, seq_len


def _call(q, k, v, cu_seqlens, seq_len):
    return flash_attn_interface.flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=seq_len,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_k=seq_len,
        cu_seqlens_k=cu_seqlens,
    )


@pytest.fixture
def missing_kernel(monkeypatch):
    monkeypatch.setattr(torch.ops._vllm_fa2_C,
                        "varlen_fwd",
                        _not_compiled,
                        raising=False)
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing",
                        lambda: False)
    monkeypatch.delenv("VLLM_XPU_DISABLE_ATTN_FALLBACK", raising=False)
    return monkeypatch


def test_missing_kernel_falls_back(missing_kernel):
    q, k, v, cu_seqlens, seq_len = _inputs("cpu")
    out = _call(q, k, v, cu_seqlens, seq_len)
    ref = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1),
                                         v.transpose(0, 1)).transpose(0, 1)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_missing_kernel_raises_during_graph_capture(missing_kernel):
    missing_kernel.setattr(torch.xpu, "is_current_stream_capturing",
                           lambda: True)
    with pytest.raises(RuntimeError, match="not compiled"):
        _call(*_inputs("cpu"))


def test_missing_kernel_raises_when_fallback_disabled(missing_kernel):
    missing_kernel.setenv("VLLM_XPU_DISABLE_ATTN_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="16,256,64,false,false,false"):
        _call(*_inputs("cpu"))
