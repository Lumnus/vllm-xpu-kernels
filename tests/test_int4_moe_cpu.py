# SPDX-License-Identifier: Apache-2.0
"""The host int4 MoE op against the card's W4A16 arithmetic, emulated in
torch at the same rounding points. Runs on the CPU only (no device)."""

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

OP = torch.ops._xpu_C.int4_moe_cpu
ISA = torch.ops._xpu_C.int4_moe_cpu_isa


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


def _dequant_dot(w: torch.Tensor, s: torch.Tensor, a: torch.Tensor, group: int):
    """a [T, K] fp32 times the int4 rows w [N, K/2] with scales s [N, K/G]:
    fp32 partial sums of 32-element chunks, each times its group's scale."""
    b = w.to(torch.int16)
    q = torch.empty(w.shape[0], w.shape[1] * 2, dtype=torch.float32)
    for half, nib in ((0, b & 0xF), (1, (b >> 4) & 0xF)):
        q[:, half::2] = torch.where(nib >= 8, nib - 16, nib).float()
    n, k = q.shape
    chunks = torch.einsum(
        "nck,tck->tnc", q.view(n, k // 32, 32), a.view(-1, k // 32, 32)
    )
    scale = s.float().repeat_interleave(group // 32, dim=1)  # [N, K/32]
    return (chunks * scale[None]).sum(-1)


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).float()


def _reference(x, units, cu, ct, cw, hidden, inter, group):
    offsets, _ = _layout(hidden, inter, group)
    out = torch.zeros(x.shape[0], hidden, dtype=torch.float32)
    for c in range(cu.numel()):
        u = units[int(cu[c])]
        w13 = u[offsets[0] : offsets[1]].view(2 * inter, hidden // 2)
        w2 = u[offsets[1] : offsets[2]].view(hidden, inter // 2)
        s13 = (
            u[offsets[2] : offsets[3]].view(torch.bfloat16).view(2 * inter, -1)
        )
        s2 = (
            u[offsets[3] : offsets[3] + hidden * (inter // group) * 2]
            .view(torch.bfloat16)
            .view(hidden, -1)
        )
        xt = x[int(ct[c]) : int(ct[c]) + 1].float()
        h = _bf16(_dequant_dot(w13, s13, xt, group))
        gate, up = h[:, :inter], h[:, inter:]
        act = _bf16(_bf16(gate / (1.0 + torch.exp(-gate))) * up)
        y = _bf16(_dequant_dot(w2, s2, act, group))
        out[int(ct[c])] += y[0] * float(cw[c])
    return out.to(torch.bfloat16)


def _calls(tokens: int, num_units: int, topk: int, gen):
    cu, ct, cw = [], [], []
    for t in range(tokens):
        picks = torch.randperm(num_units, generator=gen)[:topk]
        for u in picks.tolist():
            cu.append(u)
            ct.append(t)
            cw.append(float(torch.rand(1, generator=gen)) * 0.3 + 0.01)
    return (
        torch.tensor(cu, dtype=torch.int32),
        torch.tensor(ct, dtype=torch.int32),
        torch.tensor(cw, dtype=torch.float32),
    )


def _run(x, units, cu, ct, cw, inter, group, threads=4, isa=0):
    offsets, _ = _layout(x.shape[1], inter, group)
    out = torch.empty_like(x)
    OP(out, x, units, cu, ct, cw, inter, group, offsets, threads, isa)
    return out


def _steps(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest difference in bf16 steps at the larger output's scale."""
    ref = max(float(b.float().abs().max()), 1e-30)
    spacing = 2.0 ** (torch.tensor(ref).log2().floor().item() - 7)
    return float((a.float() - b.float()).abs().max()) / spacing


@pytest.mark.parametrize(
    "hidden,inter,group,tokens,num_units,topk",
    [
        (2560, 640, 128, 1, 4, 1),
        (2560, 640, 128, 4, 8, 3),
        (2560, 640, 128, 9, 12, 10),
        (256, 128, 32, 5, 3, 2),
        (512, 256, 64, 3, 4, 4),
    ],
)
def test_matches_card_arithmetic(hidden, inter, group, tokens, num_units, topk):
    gen = torch.Generator().manual_seed(hidden + tokens)
    units = _units(num_units, hidden, inter, group, gen)
    x = torch.randn(tokens, hidden, generator=gen).to(torch.bfloat16)
    cu, ct, cw = _calls(tokens, num_units, topk, gen)
    got = _run(x, units, cu, ct, cw, inter, group)
    want = _reference(x, units, cu, ct, cw, hidden, inter, group)
    assert _steps(got, want) <= 1.0


def test_scalar_and_avx2_agree_bit_for_bit():
    if ISA(0) != 2:
        pytest.skip("no AVX2 + FMA on this host")
    gen = torch.Generator().manual_seed(7)
    units = _units(6, 2560, 640, 128, gen)
    x = torch.randn(8, 2560, generator=gen).to(torch.bfloat16)
    cu, ct, cw = _calls(8, 6, 3, gen)
    scalar = _run(x, units, cu, ct, cw, 640, 128, threads=1, isa=1)
    avx2 = _run(x, units, cu, ct, cw, 640, 128, threads=4, isa=2)
    assert torch.equal(scalar.view(torch.int16), avx2.view(torch.int16))


def test_threads_do_not_change_the_result():
    gen = torch.Generator().manual_seed(3)
    units = _units(5, 256, 128, 32, gen)
    x = torch.randn(6, 256, generator=gen).to(torch.bfloat16)
    cu, ct, cw = _calls(6, 5, 2, gen)
    one = _run(x, units, cu, ct, cw, 128, 32, threads=1)
    many = _run(x, units, cu, ct, cw, 128, 32, threads=7)
    assert torch.equal(one.view(torch.int16), many.view(torch.int16))


def test_tokens_without_calls_are_zero_and_no_calls_is_all_zero():
    gen = torch.Generator().manual_seed(5)
    units = _units(2, 256, 128, 32, gen)
    x = torch.randn(4, 256, generator=gen).to(torch.bfloat16)
    cu = torch.tensor([1], dtype=torch.int32)
    ct = torch.tensor([2], dtype=torch.int32)
    cw = torch.tensor([0.5], dtype=torch.float32)
    out = _run(x, units, cu, ct, cw, 128, 32)
    assert out[[0, 1, 3]].abs().sum() == 0 and out[2].abs().sum() > 0
    empty = torch.empty(0, dtype=torch.int32)
    out = _run(x, units, empty, empty, empty.float(), 128, 32)
    assert out.abs().sum() == 0


def test_refuses_bad_calls():
    gen = torch.Generator().manual_seed(9)
    units = _units(2, 256, 128, 32, gen)
    x = torch.randn(3, 256, generator=gen).to(torch.bfloat16)
    w = torch.tensor([0.5, 0.5], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="unit 2 of 2"):
        _run(
            x,
            units,
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            w,
            128,
            32,
        )
    with pytest.raises(RuntimeError, match="token order"):
        _run(
            x,
            units,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([2, 1], dtype=torch.int32),
            w,
            128,
            32,
        )
    with pytest.raises(RuntimeError, match="token 3 of 3"):
        _run(
            x,
            units,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 3], dtype=torch.int32),
            w,
            128,
            32,
        )


def test_many_tokens_per_expert_and_avx2_blocks():
    # 13 and 40 tokens per expert: several passes of 8 tokens and a remainder
    # (the AVX2 code's blocks), against the scalar code and the reference.
    gen = torch.Generator().manual_seed(11)
    units = _units(3, 256, 128, 128, gen)
    x = torch.randn(40, 256, generator=gen).to(torch.bfloat16)
    cu, ct, cw = _calls(40, 3, 3, gen)
    got = _run(x, units, cu, ct, cw, 128, 128, threads=5)
    scalar = _run(x, units, cu, ct, cw, 128, 128, threads=2, isa=1)
    assert torch.equal(got.view(torch.int16), scalar.view(torch.int16))
    want = _reference(x, units, cu, ct, cw, 256, 128, 128)
    assert _steps(got, want) <= 1.0


def test_configure_pins_and_keeps_the_result():
    import os

    configure = torch.ops._xpu_C.int4_moe_cpu_configure
    gen = torch.Generator().manual_seed(13)
    units = _units(4, 256, 128, 32, gen)
    x = torch.randn(5, 256, generator=gen).to(torch.bfloat16)
    cu, ct, cw = _calls(5, 4, 2, gen)
    before = _run(x, units, cu, ct, cw, 128, 32, threads=3)
    cores = sorted(os.sched_getaffinity(0))[:3]
    try:
        assert configure(cores, 0) == len(cores)
        pinned = _run(x, units, cu, ct, cw, 128, 32, threads=3)
        fresh = _run(x, units, cu, ct, cw, 128, 32, threads=6)  # a new pool
        assert torch.equal(before.view(torch.int16), pinned.view(torch.int16))
        assert torch.equal(before.view(torch.int16), fresh.view(torch.int16))
        with pytest.raises(RuntimeError, match="spin_us out of range"):
            configure(cores, -5)
    finally:
        assert configure([], -1) == 0
    after = _run(x, units, cu, ct, cw, 128, 32, threads=3)
    assert torch.equal(before.view(torch.int16), after.view(torch.int16))
