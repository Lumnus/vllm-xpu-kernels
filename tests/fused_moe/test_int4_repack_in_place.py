# SPDX-License-Identifier: Apache-2.0
"""XpuFusedMoe repacks int4 weights in their own storage, so weights that live
in host memory seen from the device stay there."""

import pytest
import torch

from vllm_xpu_kernels.fused_moe_interface import (XpuFusedMoe,
                                                  _repack_int4_in_place,
                                                  implement_zp)


def test_repack_in_place_matches_implement_zp():
    gen = torch.Generator().manual_seed(7)
    w = torch.nn.Parameter(torch.randint(0,
                                         256, (40, 24, 16),
                                         dtype=torch.uint8,
                                         generator=gen),
                           requires_grad=False)
    want = torch.stack([implement_zp(w.data[i]) for i in range(40)])
    ptr = w.data_ptr()
    _repack_int4_in_place(w)
    assert w.dtype == torch.int8
    assert w.data_ptr() == ptr
    assert torch.equal(w.data.view(torch.uint8), want)


def _host_view(t: torch.Tensor) -> torch.Tensor:
    import vllm_xpu_kernels._C  # noqa: F401
    return torch.ops._C.get_xpu_view_from_cpu_tensor(t)


@pytest.mark.skipif(
    not torch.xpu.is_available() or
    (not torch.ops._xpu_C.is_bmg(0) and not torch.ops._xpu_C.is_pvc(0)),
    reason="XE2 CUTLASS tests only run on BMG or PVC.")
def test_int4_weights_in_host_memory_stay_there():
    """Built over host-memory views, the int4 MoE allocates nothing for its
    weights on the device and gives the output of the device-resident one."""
    dev = torch.device("xpu:0")
    e, k, inter, group, topk, tokens = 32, 2560, 640, 128, 8, 16
    gen = torch.Generator().manual_seed(3)
    w13 = torch.randint(0,
                        256, (e, 2 * inter, k // 2),
                        dtype=torch.uint8,
                        generator=gen)
    w2 = torch.randint(0,
                       256, (e, k, inter // 2),
                       dtype=torch.uint8,
                       generator=gen)
    s13 = (torch.rand(e, 2 * inter, k // group, generator=gen) * 4e-3 +
           2e-3).to(torch.bfloat16)
    s2 = (torch.rand(e, k, inter // group, generator=gen) * 4e-3 +
          2e-3).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(e, generator=gen)[:topk] for _ in range(tokens)])
    weights = torch.softmax(torch.randn(tokens, topk, generator=gen), dim=-1)
    x = (torch.randn(tokens, k, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
    topk_w, topk_ids = weights.to(dev), ids.to(dev, torch.int32)

    def run(p13, p2):
        moe = XpuFusedMoe(w13=p13,
                          w13_scales=s13.to(dev),
                          w13_bias=None,
                          w2=p2,
                          w2_scales=s2.to(dev),
                          w2_bias=None,
                          n_experts_per_token=topk,
                          activation="silu",
                          num_experts=e)
        out = torch.empty_like(x)
        moe.apply(out, x, topk_w, topk_ids)
        return out

    ref = run(torch.nn.Parameter(w13.to(dev), requires_grad=False),
              torch.nn.Parameter(w2.to(dev), requires_grad=False))

    host13, host2 = w13.clone().pin_memory(), w2.clone().pin_memory()
    p13 = torch.nn.Parameter(_host_view(host13), requires_grad=False)
    p2 = torch.nn.Parameter(_host_view(host2), requires_grad=False)
    torch.xpu.synchronize()
    before = torch.xpu.memory_allocated(dev)
    out = run(p13, p2)
    torch.xpu.synchronize()
    grown = torch.xpu.memory_allocated(dev) - before
    assert p13.data_ptr() == host13.data_ptr()
    assert p2.data_ptr() == host2.data_ptr()
    # weights are 78 MiB; scales, activations and the output stay far below
    assert grown < (w13.numel() + w2.numel()) // 4
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
