# SPDX-License-Identifier: Apache-2.0
"""Every dispatch level of the host MoE ops against their scalar code: the
AVX2, AVX-VNNI, AVX-512, AVX-512 VNNI / BF16 and AMX bodies give the scalar
code's outputs bit for bit (W4A16 in all four arith modes, W4A8 in every
scheme); level 8's card-class AMX-BF16 prefill stays within a bf16 step.
A level the host lacks is skipped; under Intel SDE (sde64 -spr -- python -m
pytest ...) the emulated CPU's levels run. CPU only, no device."""

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

OP16 = torch.ops._xpu_C.int4_moe_cpu
OP8 = torch.ops._xpu_C.int4_moe_cpu_a8
ISA = torch.ops._xpu_C.int4_moe_cpu_isa
NAME = torch.ops._xpu_C.int4_moe_cpu_isa_name

LEVELS = list(range(2, 9))
EXACT = [level for level in LEVELS if level != 8]


def _have(level: int) -> bool:
    try:
        return ISA(level) == level
    except RuntimeError:
        return False


def _need(level: int):
    if not _have(level):
        pytest.skip(f"this host has no {NAME(level)}")


def _layout(hidden: int, inter: int, group: int) -> tuple[list[int], int]:
    w13 = 2 * inter * hidden // 2
    w2 = hidden * inter // 2
    s13 = 2 * inter * (hidden // group) * 2
    s2 = hidden * (inter // group) * 2
    offsets = [0, w13, w13 + w2, w13 + w2 + s13]
    unit = -(-(w13 + w2 + s13 + s2) // 4096) * 4096
    return offsets, unit


def _units(n: int, hidden: int, inter: int, group: int, gen) -> torch.Tensor:
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


def _calls(tokens: int, num_units: int, topk: int, gen, shared=False):
    """Calls in token order; shared: every token through the same experts
    (many rows per expert, the prefill shape the AMX code takes)."""
    first = torch.randperm(num_units, generator=gen)[:topk].tolist()
    cu, ct, cw = [], [], []
    for t in range(tokens):
        if shared:
            picks = first
        else:
            picks = torch.randperm(num_units, generator=gen)[:topk].tolist()
        for u in picks:
            cu.append(u)
            ct.append(t)
            cw.append(float(torch.rand(1, generator=gen)) * 0.3 + 0.01)
    return (
        torch.tensor(cu, dtype=torch.int32),
        torch.tensor(ct, dtype=torch.int32),
        torch.tensor(cw, dtype=torch.float32),
    )


def _problem(hidden, inter, group, experts, tokens, topk, seed, shared):
    gen = torch.Generator().manual_seed(seed)
    units = _units(experts, hidden, inter, group, gen)
    x = torch.randn(tokens, hidden, generator=gen).to(torch.bfloat16)
    return x, units, *_calls(tokens, experts, topk, gen, shared)


def _run16(x, units, cu, ct, cw, inter, group, isa, arith, threads=3):
    offsets, _ = _layout(x.shape[1], inter, group)
    out = torch.empty_like(x)
    OP16(out, x, units, cu, ct, cw, inter, group, offsets, threads, isa, arith)
    return out


def _run8(x, units, cu, ct, cw, inter, group, isa, scheme, threads=3):
    offsets, _ = _layout(x.shape[1], inter, group)
    out = torch.empty_like(x)
    OP8(out, x, units, cu, ct, cw, inter, group, offsets, scheme, threads, isa)
    return out


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


# (hidden, inter, group, experts, tokens, topk, shared)
SHAPES16 = [
    (2560, 640, 128, 8, 1, 8, False),  # decode at the model's shapes
    (256, 128, 32, 3, 5, 2, False),
    (512, 256, 128, 3, 20, 2, True),  # 20 rows per expert
]


def test_auto_is_a_level_and_levels_have_names():
    best = ISA(0)
    assert 1 <= best <= 7
    assert _have(best)
    assert [NAME(level) for level in (1, 2, 4, 7)] == [
        "scalar",
        "AVX2+FMA",
        "AVX-512",
        "AMX+AVX-512",
    ]
    with pytest.raises(RuntimeError):
        ISA(9)


@pytest.mark.parametrize("arith", [0, 1, 2, 3])
@pytest.mark.parametrize("shape", SHAPES16, ids=["decode", "small", "rows20"])
@pytest.mark.parametrize("level", EXACT)
def test_w4a16_level_is_bit_identical_to_scalar(level, shape, arith):
    _need(level)
    hidden, inter, group, experts, tokens, topk, shared = shape
    x, units, cu, ct, cw = _problem(
        hidden, inter, group, experts, tokens, topk, 31, shared
    )
    scalar = _run16(x, units, cu, ct, cw, inter, group, 1, arith, threads=1)
    got = _run16(x, units, cu, ct, cw, inter, group, level, arith)
    assert _same(got, scalar)


@pytest.mark.parametrize("level", [6, 7])
def test_bf16_code_steps_aside_for_subnormal_inputs(level):
    _need(level)
    x, units, cu, ct, cw = _problem(512, 256, 128, 3, 4, 2, 32, False)
    bits = x.view(torch.int16).clone()
    bits[:, ::7] = (bits[:, ::7] & -32768) | 0x0011  # subnormal bf16
    x = bits.view(torch.bfloat16)
    for arith in (1, 3):
        scalar = _run16(x, units, cu, ct, cw, 256, 128, 1, arith, threads=1)
        assert _same(
            _run16(x, units, cu, ct, cw, 256, 128, level, arith), scalar
        )


def test_card_class_amx_prefill_stays_within_a_bf16_step():
    _need(8)
    x, units, cu, ct, cw = _problem(512, 256, 128, 3, 40, 2, 33, True)
    for arith in (1, 2, 3):
        scalar = _run16(x, units, cu, ct, cw, 256, 128, 1, arith, threads=1)
        got = _run16(x, units, cu, ct, cw, 256, 128, 8, arith).float()
        step = torch.exp2(
            torch.floor(torch.log2(scalar.float().abs().max())) - 7
        )
        assert ((got - scalar.float()).abs() <= step).all()
    # the int4 GEMV's arithmetic and decode shapes stay exact at level 8
    scalar = _run16(x, units, cu, ct, cw, 256, 128, 1, 0, threads=1)
    assert _same(_run16(x, units, cu, ct, cw, 256, 128, 8, 0), scalar)


B70 = []
SCHEMES = {
    "b70": B70,
    "amax127_sep": [1, 0, 1, 0, 1, 0],
    "act_g64": [0, 64, 0, 64, 0, 0],
    "glu": [0, 0, 1, 128, 0, 1],
}
SHAPES8 = [
    (2560, 640, 128, 8, 1, 8, False),
    (512, 256, 128, 3, 33, 2, True),  # AMX tiles plus a tail row
]


@pytest.mark.parametrize("scheme", list(SCHEMES), ids=list(SCHEMES))
@pytest.mark.parametrize("shape", SHAPES8, ids=["decode", "rows33"])
@pytest.mark.parametrize("level", LEVELS)
def test_w4a8_level_is_bit_identical_to_scalar(level, shape, scheme):
    _need(level)
    hidden, inter, group, experts, tokens, topk, shared = shape
    x, units, cu, ct, cw = _problem(
        hidden, inter, group, experts, tokens, topk, 34, shared
    )
    sc = SCHEMES[scheme]
    scalar = _run8(x, units, cu, ct, cw, inter, group, 1, sc, threads=1)
    got = _run8(x, units, cu, ct, cw, inter, group, level, sc)
    assert _same(got, scalar)


@pytest.mark.parametrize("arith", [0, 3])
@pytest.mark.parametrize("shape", SHAPES16, ids=["decode", "small", "rows20"])
@pytest.mark.parametrize("level", LEVELS)
def test_inputs_read_in_place_at_every_level(level, shape, arith):
    """call_x_row (one output row per call, its input read from x in place)
    gives the gathered-input run's bits at every level, and every level but
    8 the scalar code's."""
    _need(level)
    hidden, inter, group, experts, tokens, topk, shared = shape
    x, units, cu, ct, cw = _problem(
        hidden, inter, group, experts, tokens, topk, 36, shared
    )
    offsets, _ = _layout(hidden, inter, group)
    rows = torch.arange(cu.numel(), dtype=torch.int32)
    gathered = x.index_select(0, ct.long())

    def run(isa, inplace, threads=3):
        out = torch.empty(cu.numel(), hidden, dtype=torch.bfloat16)
        src, xr = (x, ct) if inplace else (gathered, None)
        OP16(
            out,
            src,
            units,
            cu,
            rows,
            cw,
            inter,
            group,
            offsets,
            threads,
            isa,
            arith,
            xr,
        )
        return out

    got = run(level, False)
    assert _same(run(level, True), got)
    assert _same(run(level, True, threads=1), got)
    if level != 8:
        assert _same(got, run(1, False, threads=1))


def test_a_level_the_host_lacks_is_refused():
    missing = [level for level in LEVELS if not _have(level)]
    if not missing:
        pytest.skip("this host has every level")
    x, units, cu, ct, cw = _problem(256, 128, 128, 2, 1, 1, 35, False)
    with pytest.raises(RuntimeError, match="not available"):
        _run16(x, units, cu, ct, cw, 128, 128, missing[0], 0)
