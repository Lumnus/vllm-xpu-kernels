# SPDX-License-Identifier: Apache-2.0
"""Device bits of the MoE entry ops (one card; skipped without XPU):

1. moe_gather with the shared term against moe_gather followed by torch's
   bf16 multiply and add (the engine's sequence), bitwise; single_round
   bounded by an ulp of each operand.
2. moe_w4a8_route_gather_q against moe_w4a8_route_quant: the expert rows,
   their scales, the route map, rows_per_expert and the tile map, bitwise,
   with the token rows quantized by _C.dynamic_per_token_scaled_int8_quant.
3. topk_softmax_route against topk_softmax followed by a torch EPLB map
   (vLLM's rule): weights, ids and the expert-load record.

QFN decode shapes: hidden 2560, 512 experts (128 local at EP4), top-10,
1-33 tokens, NaN padding rows, moved placement."""

import pytest
import torch

xpu = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="XPU only")


def _gather_case(t, topk, hidden, n_local, seed, dev="xpu"):
    g = torch.Generator().manual_seed(seed)
    rows = t * topk
    g2 = (torch.randn((rows, hidden), generator=g) / 4).to(torch.bfloat16)
    w = torch.rand((t, topk), generator=g)
    unp = torch.randperm(rows, generator=g).to(torch.int32).view(t, topk)
    unp[::3, 1] = -1
    unp[-1] = -1  # a row with no computed slot
    sh = (torch.randn((t, hidden), generator=g) / 4).to(torch.bfloat16)
    sh[-1] = float("nan")
    gate = torch.sigmoid(torch.randn((t, 1), generator=g)).to(torch.bfloat16)
    return [x.to(dev) for x in (g2, w, unp, sh, gate)]


@xpu
@pytest.mark.parametrize("t", [1, 2, 4, 8, 13, 16, 33])
@pytest.mark.parametrize("topk", [8, 10])
def test_moe_gather_shared_bits(t, topk):
    import vllm_xpu_kernels._moe_C  # noqa: F401
    g2, w, unp, sh, gate = _gather_case(t, topk, 2560, 128, t * 31 + topk)
    plain = torch.empty((t, 2560), dtype=torch.bfloat16, device="xpu")
    torch.ops._moe_C.moe_gather(plain, g2, w, unp, 128)
    want = gate * sh + plain
    got = torch.empty_like(plain)
    torch.ops._moe_C.moe_gather(got, g2, w, unp, 128, sh, gate)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    one = torch.empty_like(plain)
    torch.ops._moe_C.moe_gather(one, g2, w, unp, 128, sh, gate, True)
    a, b = want.float(), one.float()
    fin = torch.isfinite(a)
    assert torch.equal(fin, torch.isfinite(b))
    eps = torch.finfo(torch.bfloat16).eps
    bound = eps * (plain.float().abs() + (gate * sh).float().abs() + a.abs())
    assert bool(((a - b).abs()[fin] <= bound[fin] + 1e-30).all())


@xpu
@pytest.mark.parametrize("t", [1, 2, 4, 8, 13, 16, 33])
@pytest.mark.parametrize("ep_rank", [None, 0, 3])
def test_route_gather_q_bits(t, ep_rank):
    import vllm_xpu_kernels._C  # noqa: F401
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    g = torch.Generator().manual_seed(t * 7 + (ep_rank or 0))
    hidden, topk, total = 2560, 10, 512
    local = total if ep_rank is None else 128
    x = (torch.randn((t + 2, hidden), generator=g) / 4).to(torch.bfloat16)
    x[t:] = float("nan")
    ids = torch.randint(0, total, (t + 2, topk), generator=g).to(torch.int32)
    ids[::3, 1] = -1
    ids[t:] = -1
    emap = None
    if ep_rank is not None:
        perm = torch.randperm(total, generator=g)
        emap = torch.full((total,), -1, dtype=torch.int32)
        emap[perm[:local]] = torch.arange(local, dtype=torch.int32)
        emap = emap.xpu()
    x, ids = x.xpu(), ids.xpu()
    T = x.shape[0]
    R = T * topk
    tile_m = 8
    umt = min(R, R // tile_m + min(R, local))
    q = torch.empty((T, hidden), dtype=torch.int8, device="xpu")
    s = torch.empty((T, 1), dtype=torch.float32, device="xpu")
    torch.ops._C.dynamic_per_token_scaled_int8_quant(q, x, s)
    outs = []
    for pre in (False, True):
        aq = torch.zeros((R, hidden), dtype=torch.int8, device="xpu")
        asc = torch.zeros((R,), dtype=torch.float32, device="xpu")
        pp = torch.empty((T, topk), dtype=torch.int32, device="xpu")
        rpe = torch.full((local,), 7, dtype=torch.int32, device="xpu")
        ws = torch.zeros((2 + 3 * umt,), dtype=torch.int32, device="xpu")
        if pre:
            torch.ops._xpu_C.moe_w4a8_route_gather_q(
                q, s, ids, emap, aq, asc, pp, rpe, ws, total, local, tile_m)
        else:
            torch.ops._xpu_C.moe_w4a8_route_quant(
                x, ids, emap, aq, asc, pp, rpe, ws, total, local, tile_m)
        outs.append((aq, asc, pp, rpe, ws))
    (aq0, as0, pp0, r0, w0), (aq1, as1, pp1, r1, w1) = outs
    assert torch.equal(r0, r1) and torch.equal(w0[:2], w1[:2])
    # Slots inside an expert come from atomics: compare per route.
    for a, b in ((pp0, pp1),):
        assert torch.equal(a >= 0, b >= 0)
    sel0, sel1 = pp0.view(-1), pp1.view(-1)
    ok = sel0 >= 0
    assert torch.equal(aq0[sel0[ok].long()], aq1[sel1[ok].long()])
    assert torch.equal(as0[sel0[ok].long()].view(torch.int32),
                       as1[sel1[ok].long()].view(torch.int32))


def _eplb_ref(ids, l2p, rc, load, record, unpadded, sentinel):
    out = ids.clone()
    for row in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            lg = int(ids[row, j])
            ph = -1
            if 0 <= lg < rc.shape[0]:
                h = (row * 2654435769) & 0xFFFFFFFF
                ph = int(l2p[lg, h % int(rc[lg])])
            if record and 0 <= ph < load.shape[0] and (
                    unpadded is None or row < unpadded):
                load[ph] += 1
            if sentinel is not None and ph < 0:
                ph = sentinel
            out[row, j] = ph
    return out


@xpu
@pytest.mark.parametrize("t", [1, 2, 8, 16, 33])
@pytest.mark.parametrize("sentinel", [None, 543, 0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_topk_softmax_route_bits(t, sentinel, dtype):
    import vllm_xpu_kernels._moe_C  # noqa: F401
    g = torch.Generator().manual_seed(t)
    n_log, n_phys, topk = 512, 544, 10
    p2l = torch.cat([torch.arange(n_log),
                     torch.randint(0, n_log, (n_phys - n_log,), generator=g)])
    p2l = p2l[torch.randperm(n_phys, generator=g)]
    l2p = torch.full((n_log, 1 + n_phys - n_log), -1, dtype=torch.long)
    rc = torch.zeros((n_log,), dtype=torch.long)
    for p, lg in enumerate(p2l.tolist()):
        l2p[lg, rc[lg]] = p
        rc[lg] += 1
    logits = torch.randn((t, n_log), generator=g).to(dtype)
    logits[0, 5] = logits[0, 4]
    pad = torch.zeros((t,), dtype=torch.bool)
    pad[-1] = t > 1
    unpadded = max(1, t - 1)
    dev = "xpu"
    w0 = torch.empty((t, topk), dtype=torch.float32, device=dev)
    i0 = torch.empty((t, topk), dtype=torch.int32, device=dev)
    e0 = torch.empty((t, topk), dtype=torch.int32, device=dev)
    torch.ops._moe_C.topk_softmax(w0, i0, e0, logits.to(dev), True, None,
                                  pad.to(dev))
    load_ref = torch.zeros((n_phys,), dtype=torch.int32)
    i_ref = _eplb_ref(i0.cpu(), l2p, rc, load_ref, True, unpadded, sentinel)
    w1 = torch.empty_like(w0)
    i1 = torch.empty_like(i0)
    e1 = torch.empty_like(e0)
    load = torch.zeros((n_phys,), dtype=torch.int32, device=dev)
    torch.ops._moe_C.topk_softmax_route(
        w1, i1, e1, logits.to(dev), True, pad.to(dev), l2p.to(dev),
        rc.to(dev), load, torch.tensor(True, device=dev),
        torch.tensor(unpadded, dtype=torch.int32, device=dev),
        0 if sentinel is None else sentinel, sentinel is not None)
    assert torch.equal(w0.view(torch.int32), w1.view(torch.int32))
    assert torch.equal(i1.cpu(), i_ref)
    assert torch.equal(load.cpu(), load_ref)
    assert torch.equal(e0, e1)
