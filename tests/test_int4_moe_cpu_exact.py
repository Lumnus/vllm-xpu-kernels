# SPDX-License-Identifier: Apache-2.0
"""The host op's card-exact arithmetic (arith 7) against the card's measured
W4A16 grouped GEMM: weights truncated to bf16(q * s), products summed exactly
in blocks of 16, each block added to the fp32 accumulator with one rounding,
bf16 outputs, the activation from the card's own silu table.

The references: the card model in torch that reproduced the card's per-pair
rows from their inputs with 0 differing values (block sums in float64,
exact for these inputs), and an exact model in Python fractions for inputs
whose exponents span far beyond float64's reach. CPU only, no device."""

from fractions import Fraction

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

OP = torch.ops._xpu_C.int4_moe_cpu
ISA = torch.ops._xpu_C.int4_moe_cpu_isa
CARD_EXACT = 7


def _layout(hidden: int, inter: int, group: int) -> tuple[list[int], int]:
    w13 = 2 * inter * hidden // 2
    w2 = hidden * inter // 2
    s13 = 2 * inter * (hidden // group) * 2
    s2 = hidden * (inter // group) * 2
    offsets = [0, w13, w13 + w2, w13 + w2 + s13]
    unit = -(-(w13 + w2 + s13 + s2) // 4096) * 4096
    return offsets, unit


def _units(n, hidden, inter, group, gen, scale=0.01):
    offsets, unit = _layout(hidden, inter, group)
    units = torch.randint(0, 256, (n, unit), dtype=torch.uint8, generator=gen)
    for off, rows, cols in (
        (offsets[2], 2 * inter, hidden // group),
        (offsets[3], hidden, inter // group),
    ):
        scales = (torch.randn(n, rows * cols, generator=gen) * scale).to(
            torch.bfloat16
        )
        units[:, off : off + rows * cols * 2] = scales.view(torch.uint8)
    return units


def _tensors(units, u, hidden, inter, group):
    offsets, _ = _layout(hidden, inter, group)
    row = units[u]
    w13 = row[offsets[0] : offsets[1]].view(2 * inter, hidden // 2)
    w2 = row[offsets[1] : offsets[2]].view(hidden, inter // 2)
    s13 = row[offsets[2] : offsets[3]].view(torch.bfloat16).view(2 * inter, -1)
    s2 = (
        row[offsets[3] : offsets[3] + hidden * (inter // group) * 2]
        .view(torch.bfloat16)
        .view(hidden, -1)
    )
    return w13, w2, s13, s2


def _table(perturb: bool = True) -> torch.Tensor:
    """A stand-in for the card's table: the host's bf16 silu of every bf16
    gate, every 97th finite entry one bf16 step off (a wrong index shows)."""
    g = (
        torch.arange(65536, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    t = (g.float() / (1.0 + torch.exp(-g.float()))).to(torch.bfloat16)
    bits = t.view(torch.int16).clone()
    if perturb:
        idx = torch.arange(0, 65536, 97)
        finite = (bits[idx] & 0x7F80) != 0x7F80
        bits[idx[finite]] += 1
    return bits.view(torch.bfloat16)


def _q(w: torch.Tensor) -> torch.Tensor:
    b = w.to(torch.int16)
    q = torch.empty(w.shape[0], w.shape[1] * 2, dtype=torch.float32)
    for half, nib in ((0, b & 0xF), (1, (b >> 4) & 0xF)):
        q[:, half::2] = torch.where(nib >= 8, nib - 16, nib).float()
    return q


def _rtz(t: torch.Tensor) -> torch.Tensor:
    return (t.float().view(torch.int32) & ~0xFFFF).view(torch.float32)


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).float()


def _card_dot(w, s, a, group):
    """The card model: weights bf16_rtz(q * s), products in float64 summed
    per block of 16 (exact for these inputs), each block added to the fp32
    accumulator in float64 and rounded to fp32."""
    wq = _rtz(_q(w) * s.float().repeat_interleave(group, dim=1)).double()
    p = a.double()[:, None, :] * wq[None]  # [T, N, K]
    blocks = p.view(p.shape[0], p.shape[1], -1, 16).sum(-1)
    acc = torch.zeros(p.shape[0], p.shape[1], dtype=torch.float32)
    for j in range(blocks.shape[-1]):
        acc = (acc.double() + blocks[..., j]).float()
    return acc


def _card_model(x, units, cu, ct, cw, hidden, inter, group, table):
    tf = table.float()
    out = torch.zeros(x.shape[0], hidden, dtype=torch.float32)
    for c in range(cu.numel()):
        w13, w2, s13, s2 = _tensors(units, int(cu[c]), hidden, inter, group)
        t = int(ct[c])
        h = _bf16(_card_dot(w13, s13, x[t : t + 1].float(), group))
        gate = h[:, :inter].to(torch.bfloat16).view(torch.int16).long() & 0xFFFF
        act = _bf16(tf[gate] * h[:, inter:])
        y = _bf16(_card_dot(w2, s2, act, group))
        out[t] += y[0] * float(cw[c])
    return out.to(torch.bfloat16)


def _calls(tokens, num_units, topk, gen):
    cu, ct, cw = [], [], []
    for t in range(tokens):
        for u in torch.randperm(num_units, generator=gen)[:topk].tolist():
            cu.append(u)
            ct.append(t)
            cw.append(float(torch.rand(1, generator=gen)) * 0.3 + 0.01)
    return (
        torch.tensor(cu, dtype=torch.int32),
        torch.tensor(ct, dtype=torch.int32),
        torch.tensor(cw, dtype=torch.float32),
    )


def _run(x, units, cu, ct, cw, inter, group, table, threads=3, isa=0, arith=7):
    offsets, _ = _layout(x.shape[1], inter, group)
    out = torch.empty_like(x)
    OP(
        out,
        x,
        units,
        cu,
        ct,
        cw,
        inter,
        group,
        offsets,
        threads,
        isa,
        arith,
        None,
        table,
    )
    return out


def _levels() -> list[int]:
    out = []
    for level in range(2, 9):
        try:
            if ISA(level) == level:
                out.append(level)
        except RuntimeError:
            pass
    return out


@pytest.mark.parametrize(
    "hidden,inter,group,tokens,num_units,topk",
    [
        (2560, 640, 128, 1, 12, 10),
        (2560, 640, 128, 3, 12, 10),
        (2560, 640, 128, 9, 3, 2),
        (256, 128, 32, 5, 3, 2),
        (512, 256, 64, 7, 4, 3),
    ],
)
def test_matches_the_card_model_with_no_differing_value(
    hidden, inter, group, tokens, num_units, topk
):
    gen = torch.Generator().manual_seed(hidden + 7 * tokens)
    units = _units(num_units, hidden, inter, group, gen)
    x = torch.randn(tokens, hidden, generator=gen).to(torch.bfloat16)
    cu, ct, cw = _calls(tokens, num_units, topk, gen)
    table = _table()
    got = _run(x, units, cu, ct, cw, inter, group, table)
    want = _card_model(x, units, cu, ct, cw, hidden, inter, group, table)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    # the arithmetic of arith 3 (bf16 weights to nearest, 8 fp32 lanes, the
    # host's silu) is not the card's: most values differ
    old = _run(x, units, cu, ct, cw, inter, group, table, arith=3)
    assert (old != want).float().mean().item() > 0.3


def test_every_level_equals_the_scalar_code():
    levels = _levels()
    if not levels:
        pytest.skip("no vector level on this host")
    gen = torch.Generator().manual_seed(17)
    units = _units(6, 2560, 640, 128, gen)
    x = torch.randn(7, 2560, generator=gen).to(torch.bfloat16)
    cu, ct, cw = _calls(7, 6, 3, gen)
    table = _table()
    scalar = _run(x, units, cu, ct, cw, 640, 128, table, threads=1, isa=1)
    for level in levels:
        for threads in (1, 4):
            got = _run(x, units, cu, ct, cw, 640, 128, table, threads, level)
            assert torch.equal(
                got.view(torch.int16), scalar.view(torch.int16)
            ), (
                level,
                threads,
            )


# ---------------------------------------------------------------------------
# beyond float64: an exact model in fractions
# ---------------------------------------------------------------------------


def _round_f32(v: Fraction) -> float:
    """v rounded once to fp32, to nearest even (subnormals, overflow)."""
    if v == 0:
        return 0.0
    sign = -1.0 if v < 0 else 1.0
    m = abs(v)
    e = m.numerator.bit_length() - m.denominator.bit_length()
    if Fraction(2) ** e > m:
        e -= 1
    lsb = max(e - 23, -149)  # the kept grain
    scaled = m / Fraction(2) ** lsb
    n = scaled.numerator // scaled.denominator
    rem = scaled - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2 == 1):
        n += 1
    out = n * 2.0**lsb if lsb + n.bit_length() <= 128 else float("inf")
    return sign * out


def _exact_dot(w, s, a, group):
    """The card's chain in fractions: no float64 sum anywhere."""
    wq = _rtz(_q(w) * s.float().repeat_interleave(group, dim=1))
    wf = [[Fraction(float(v)) for v in row] for row in wq.tolist()]
    out = torch.empty(a.shape[0], wq.shape[0], dtype=torch.float32)
    for t, arow in enumerate(a.tolist()):
        af = [Fraction(float(v)) for v in arow]
        for n, wr in enumerate(wf):
            acc = 0.0
            for b in range(0, len(af), 16):
                block = sum(wr[k] * af[k] for k in range(b, b + 16))
                acc = _round_f32(Fraction(acc) + block)
            out[t, n] = acc
    return out


def _wide_inputs(n, k, gen, lo=-60, hi=10):
    """bf16 values 2^e (1 + m/128), e uniform in [lo, hi], random sign; a
    block of 16 then spans up to 70 exponents."""
    e = torch.randint(lo, hi + 1, (n, k), generator=gen).double()
    m = torch.randint(0, 128, (n, k), generator=gen).double() / 128 + 1
    sign = torch.randint(0, 2, (n, k), generator=gen).double() * 2 - 1
    return (sign * m * torch.pow(2.0, e)).to(torch.bfloat16)


def test_inputs_spanning_70_exponents_match_the_exact_model():
    """Blocks far beyond the 30-exponent window of exact float64 sums take
    the op's exact integer sum: GEMM1's rows equal the fraction model's."""
    hidden, inter, group = 256, 64, 32
    gen = torch.Generator().manual_seed(70)
    units = _units(1, hidden, inter, group, gen, scale=1.0)
    x = _wide_inputs(1, hidden, gen)
    w13, _, s13, _ = _tensors(units, 0, hidden, inter, group)
    h = _exact_dot(w13, s13, x.float(), group)
    # GEMM1's up rows read back through the op: a table of ones makes the
    # activation bf16(up), and GEMM2 the identity (q = 1 on its own column,
    # scale 1) passes it to the output unchanged
    table = torch.ones(65536, dtype=torch.bfloat16)
    _, w2b, _, s2b = _tensors(units, 0, hidden, inter, group)
    w2b.zero_()
    s2b.fill_(1.0)
    # GEMM2 row j reads activation j: q = 1 at its own column only
    for j in range(inter):
        w2b[j, j // 2] = 1 if j % 2 == 0 else 0x10
    cu = torch.zeros(1, dtype=torch.int32)
    cw = torch.ones(1, dtype=torch.float32)
    got = _run(x, units, cu, cu, cw, inter, group, table)
    up = _bf16(h[0, inter:])
    want = up.to(torch.bfloat16)  # y_j = bf16(1 * act_j) = bf16(up_j)
    assert torch.equal(got[0, :inter].view(torch.int16), want.view(torch.int16))


def test_refuses_card_exact_without_a_table():
    gen = torch.Generator().manual_seed(4)
    units = _units(1, 256, 128, 32, gen)
    x = torch.randn(1, 256, generator=gen).to(torch.bfloat16)
    one = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="silu_table"):
        _run(x, units, one, one, torch.ones(1), 128, 32, None)
    with pytest.raises(RuntimeError, match="silu_table"):
        _run(x, units, one, one, torch.ones(1), 128, 32, _table()[:1000])
    with pytest.raises(RuntimeError, match="arith"):
        _run(x, units, one, one, torch.ones(1), 128, 32, _table(), arith=4)


def test_the_table_is_read_only_by_card_exact():
    gen = torch.Generator().manual_seed(5)
    units = _units(3, 512, 256, 64, gen)
    x = torch.randn(2, 512, generator=gen).to(torch.bfloat16)
    cu, ct, cw = _calls(2, 3, 2, gen)
    for arith in (0, 3):
        plain = torch.empty_like(x)
        offsets, _ = _layout(512, 256, 64)
        OP(plain, x, units, cu, ct, cw, 256, 64, offsets, 2, 0, arith)
        with_table = _run(x, units, cu, ct, cw, 256, 64, _table(), arith=arith)
        assert torch.equal(
            plain.view(torch.int16), with_table.view(torch.int16)
        )
