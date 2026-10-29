# SPDX-License-Identifier: Apache-2.0
# B70 (Lumnus): native per-token int8 quant + oneDNN int8 GEMM vs torch refs.
import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

DEV = "xpu"


def quant_ref(x: torch.Tensor):
    """Eager copy of vllm._xpu_ops.xpu_ops.dynamic_per_token_int8_quant_ref."""
    x2 = x.reshape(-1, x.shape[-1])
    mn = torch.min(x2, dim=-1)[0].to(torch.float32).unsqueeze(-1)
    mx = torch.max(x2, dim=-1)[0].to(torch.float32).unsqueeze(-1)
    s = (torch.maximum(mn.abs(), mx.abs()) / 127).clamp(min=1e-5).to(x.dtype)
    q = torch.clamp(torch.round(x2 / s.to(torch.float32)), -128, 127)
    return q.to(torch.int8).view(x.shape), s.view(x.shape[:-1] + (1,))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,k", [(1, 5120), (7, 17408), (512, 5120),
                                 (33, 100), (3, 8)])
@pytest.mark.parametrize("scale_f32", [True, False])
def test_int8_quant_exact(dtype, m, k, scale_f32):
    torch.manual_seed(0)
    x = (torch.randn(m, k, device=DEV) * 3).to(dtype)
    x[0, :4] = 0
    if m > 2:
        x[2] = 0  # all-zero row -> clamp(1e-5)
    q_ref, s_ref = quant_ref(x)
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(m, 1, device=DEV,
                    dtype=torch.float32 if scale_f32 else dtype)
    torch.ops._C.dynamic_per_token_scaled_int8_quant(q, x, s)
    assert torch.equal(q, q_ref)
    assert torch.equal(s.to(torch.float32), s_ref.to(torch.float32))


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,n,k", [(1, 4096, 5120), (16, 1024, 5120),
                                   (600, 3072, 5120), (64, 5120, 17408)])
@pytest.mark.parametrize("per_token", [True, False])
@pytest.mark.parametrize("with_bias", [False, True])
def test_int8_scaled_mm(out_dtype, m, n, k, per_token, with_bias):
    torch.manual_seed(0)
    a = torch.randint(-128, 128, (m, k), dtype=torch.int8, device=DEV)
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device=DEV)
    sa = (torch.rand(m if per_token else 1, 1, device=DEV) + 0.1) * 1e-2
    sw = (torch.rand(n, 1, device=DEV) + 0.1) * 1e-2
    bias = torch.randn(n, device=DEV, dtype=out_dtype) if with_bias else None
    acc = (a.cpu().to(torch.int64) @ w.cpu().to(torch.int64).t()).to(
        torch.float64)
    ref = acc * sa.cpu().double() * sw.cpu().double().t()
    if bias is not None:
        ref = ref + bias.cpu().double()
    out = torch.ops._xpu_C.int8_scaled_mm(a, sa, w.t(), sw, out_dtype, bias)
    torch.testing.assert_close(out.cpu().double(), ref.to(out_dtype).double(),
                               rtol=1.6e-2, atol=1e-2)
