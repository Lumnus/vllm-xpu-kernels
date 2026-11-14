# SPDX-License-Identifier: Apache-2.0
"""The host W4A8 MoE op against the card's W4A8 arithmetic, emulated in
torch bit for bit (int8 per-token activations as
dynamic_per_token_scaled_int8_quant, int32 per 128-K group, fp32 group sum
with fused multiply-adds, the activation scale at the end, bf16 rounding
points of moe_w4a8.apply, moe_gather's fused multiply-add). CPU only."""

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

OP = torch.ops._xpu_C.int4_moe_cpu_a8
ISA = torch.ops._xpu_C.int4_moe_cpu_isa

B70 = []  # the default scheme: the card's W4A8 path
GLU = [0, 0, 1, 128, 0, 1]  # the card's fused gemm1 epilogue (off in serving)


def _layout(hidden, inter, group):
    w13 = 2 * inter * hidden // 2
    w2 = hidden * inter // 2
    s13 = 2 * inter * (hidden // group) * 2
    s2 = hidden * (inter // group) * 2
    offsets = [0, w13, w13 + w2, w13 + w2 + s13]
    unit = -(-(w13 + w2 + s13 + s2) // 4096) * 4096
    return offsets, unit


def _units(n, hidden, inter, group, gen):
    offsets, unit = _layout(hidden, inter, group)
    units = torch.randint(0, 256, (n, unit), dtype=torch.uint8, generator=gen)
    for off, rows, cols in (
        (offsets[2], 2 * inter, hidden // group),
        (offsets[3], hidden, inter // group),
    ):
        scales = (torch.randn(n, rows * cols, generator=gen) * 0.01).to(
            torch.bfloat16
        )
        units[:, off : off + rows * cols * 2] = scales.view(torch.uint8)
    return units


def _bf16(t):
    return t.to(torch.bfloat16).float()


def _fma(a, b, c):
    """fp32 fused multiply-add: a * b is exact in float64 for these
    operands, and one rounding of the float64 sum to fp32 equals the single
    rounding of fma (no fp32 tie can be produced by the float64 step)."""
    return (a.double() * b.double() + c.double()).float()


def _quant_int8q(v, group):
    """dynamic_per_token_scaled_int8_quant on bf16 values (or rule 0 with
    per-group scales): q [T, K] int, scales [T, K / len]."""
    t, k = v.shape
    n = group or k
    g = v.view(t, k // n, n)
    amax = g.abs().amax(-1, keepdim=True)
    s = _bf16(torch.clamp(amax / 127.0, min=1e-5))
    x = g / s
    f = torch.floor(x)
    boundary = (f + 0.5) * s
    even = torch.floor(f * 0.5) == f * 0.5
    q = torch.where(
        g > boundary,
        f + 1,
        torch.where(g < boundary, f, torch.where(even, f, f + 1)),
    )
    q = q.clamp(-128, 127)
    return q.view(t, k).to(torch.int64), s.view(t, -1)


def _quant_amax127(v, group):
    t, k = v.shape
    g = v.view(t, k // group, group)
    mx = g.abs().amax(-1, keepdim=True)
    s = mx * (1.0 / 127.0)
    inv = torch.where(mx > 0, 127.0 / mx, torch.zeros_like(mx))
    q = torch.round(g * inv).clamp(-127, 127)  # round half to even
    return q.view(t, k).to(torch.int64), s.view(t, -1)


def _w4(w):
    b = w.to(torch.int16)
    q = torch.empty(w.shape[0], w.shape[1] * 2, dtype=torch.int64)
    for half, nib in ((0, b & 0xF), (1, (b >> 4) & 0xF)):
        q[:, half::2] = torch.where(nib >= 8, nib - 16, nib).to(torch.int64)
    return q


def _gemm(q, s, w, ws, group, ag):
    """q [T, K] int8 values, s activation scales ([T, 1] per row or [T, K/ag]),
    w [N, K/2] nibbles, ws [N, K/G] bf16: fp32 F per output, the card's way."""
    wq = _w4(w)
    t, k = q.shape
    fl = ag if ag and ag < group else group
    F = torch.zeros(t, wq.shape[0], dtype=torch.float32)
    wsf = ws.float()
    for k0 in range(0, k, fl):
        d = (q[:, k0 : k0 + fl] @ wq[:, k0 : k0 + fl].T).float()  # exact int
        m = wsf[:, k0 // group][None, :].expand(t, -1)
        if ag:
            m = m * s[:, k0 // ag : k0 // ag + 1]
        F = _fma(d, m, F)
    if not ag:
        F = F * s
    return F


def _silu(g):
    return g / (1.0 + torch.exp(-g.double()).float())


def _reference(x, units, cu, ct, cw, hidden, inter, group, scheme):
    offsets, _ = _layout(hidden, inter, group)
    glu = scheme == GLU
    out = torch.zeros(x.shape[0], hidden, dtype=torch.float32)
    for c in range(cu.numel()):
        u = units[int(cu[c])]
        w13 = u[offsets[0] : offsets[1]].view(2 * inter, hidden // 2)
        w2 = u[offsets[1] : offsets[2]].view(hidden, inter // 2)
        s13 = (
            u[offsets[2] : offsets[3]].view(torch.bfloat16).view(2 * inter, -1)
        )
        s2 = u[offsets[3] : offsets[3] + hidden * (inter // group) * 2]
        s2 = s2.view(torch.bfloat16).view(hidden, -1)
        xt = x[int(ct[c]) : int(ct[c]) + 1].float()
        q1, as1 = _quant_int8q(xt, 0)
        F1 = _gemm(q1, as1, w13, s13, group, 0)
        if not glu:
            h = _bf16(F1)
            a = _bf16(_bf16(_silu(h[:, :inter])) * h[:, inter:])
            q2, as2 = _quant_int8q(a, 0)
            y = _bf16(_gemm(q2, as2, w2, s2, group, 0))
        else:
            a = _silu(F1[:, :inter]) * F1[:, inter:]
            q2, as2 = _quant_amax127(a, 128)
            y = _bf16(_gemm(q2, as2, w2, s2, group, 128))
        t = int(ct[c])
        out[t] = _fma(y[0], torch.tensor(float(cw[c])), out[t])
    return out.to(torch.bfloat16)


def _calls(tokens, experts, topk, gen):
    cu, ct = [], []
    for t in range(tokens):
        for e in torch.randperm(experts, generator=gen)[:topk].tolist():
            cu.append(e)
            ct.append(t)
    cw = torch.rand(len(cu), generator=gen) * 0.9 + 0.05
    return (
        torch.tensor(cu, dtype=torch.int32),
        torch.tensor(ct, dtype=torch.int32),
        cw.float(),
    )


def _run(x, units, cu, ct, cw, inter, group, scheme=B70, threads=1, isa=0):
    offsets, _ = _layout(x.shape[1], inter, group)
    out = torch.empty_like(x)
    OP(out, x, units, cu, ct, cw, inter, group, offsets, scheme, threads, isa)
    return out


SHAPES = [(256, 128, 128, 5, 7, 3), (2560, 640, 128, 4, 3, 2)]


@pytest.mark.parametrize("scheme", [B70, GLU], ids=["b70", "glu"])
@pytest.mark.parametrize("shape", SHAPES, ids=["small", "qfn"])
def test_bit_identical_to_the_card_arithmetic(shape, scheme):
    hidden, inter, group, experts, tokens, topk = shape
    gen = torch.Generator().manual_seed(11)
    units = _units(experts, hidden, inter, group, gen)
    x = (torch.randn(tokens, hidden, generator=gen) * 2).to(torch.bfloat16)
    cu, ct, cw = _calls(tokens, experts, topk, gen)
    got = _run(x, units, cu, ct, cw, inter, group, scheme)
    want = _reference(x, units, cu, ct, cw, hidden, inter, group, scheme)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_threads_do_not_change_the_result():
    gen = torch.Generator().manual_seed(12)
    units = _units(6, 256, 128, 128, gen)
    x = torch.randn(9, 256, generator=gen).to(torch.bfloat16)
    cu, ct, cw = _calls(9, 6, 4, gen)
    one = _run(x, units, cu, ct, cw, 128, 128, threads=1)
    four = _run(x, units, cu, ct, cw, 128, 128, threads=4)
    assert torch.equal(one.view(torch.int16), four.view(torch.int16))


@pytest.mark.skipif(ISA(0) != 2, reason="no AVX2 + FMA on this host")
def test_avx2_and_scalar_agree_bit_for_bit():
    gen = torch.Generator().manual_seed(13)
    units = _units(5, 512, 256, 128, gen)
    x = (torch.randn(6, 512, generator=gen) * 4).to(torch.bfloat16)
    cu, ct, cw = _calls(6, 5, 3, gen)
    for scheme in (B70, GLU):
        a = _run(x, units, cu, ct, cw, 256, 128, scheme, isa=2)
        s = _run(x, units, cu, ct, cw, 256, 128, scheme, isa=1)
        assert torch.equal(a.view(torch.int16), s.view(torch.int16))


def test_a_single_call_with_weight_one_is_the_expert_output():
    """The per-pair rows the engine asks for: weight 1 rounds to y."""
    gen = torch.Generator().manual_seed(14)
    units = _units(2, 256, 128, 128, gen)
    x = torch.randn(2, 256, generator=gen).to(torch.bfloat16)
    cu = torch.tensor([1, 0], dtype=torch.int32)
    ct = torch.tensor([0, 1], dtype=torch.int32)
    cw = torch.ones(2)
    got = _run(x, units, cu, ct, cw, 128, 128)
    want = _reference(x, units, cu, ct, cw, 256, 128, 128, B70)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_no_calls_gives_zeros():
    gen = torch.Generator().manual_seed(15)
    units = _units(1, 256, 128, 128, gen)
    x = torch.randn(3, 256, generator=gen).to(torch.bfloat16)
    e = torch.empty(0, dtype=torch.int32)
    got = _run(x, units, e, e, torch.empty(0), 128, 128)
    assert torch.equal(got, torch.zeros_like(got))


@pytest.mark.parametrize(
    "case, match",
    [
        ("scheme", "scheme is"),
        ("group", "activation group"),
        ("unit", "names unit"),
        ("order", "token order"),
    ],
)
def test_refusals(case, match):
    gen = torch.Generator().manual_seed(16)
    units = _units(2, 256, 128, 128, gen)
    x = torch.randn(2, 256, generator=gen).to(torch.bfloat16)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    ct = torch.tensor([0, 1], dtype=torch.int32)
    cw = torch.ones(2)
    scheme = B70
    if case == "scheme":
        scheme = [0, 0, 0]
    elif case == "group":
        scheme = [0, 96, 0, 0, 0, 0]
    elif case == "unit":
        cu = torch.tensor([0, 2], dtype=torch.int32)
    elif case == "order":
        ct = torch.tensor([1, 0], dtype=torch.int32)
    with pytest.raises(RuntimeError, match=match):
        _run(x, units, cu, ct, cw, 128, 128, scheme)
