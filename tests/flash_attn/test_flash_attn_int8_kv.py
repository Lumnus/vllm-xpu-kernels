# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 per-token-head KV cache (vLLM `int8_per_token_head`) for paged FA2.

The cache uses the block-local split layout: per block
[int8 K|V data for all (slot, head) | K scales fp32 | V scales fp32], i.e. the
same bytes per block as vLLM's inline per-token-head layout (a block may be
padded at its tail, as hybrid models' unified pages are). INT8 KV is
supported by the paged decode kernel, including small causal multi-query
batches (speculative-decode verification, uniform or ragged), which the
interface runs through the decode kernel; batches routed to chunk prefill are
rejected (callers dequantize the referenced blocks and use the fp16/bf16
path) unless ``int8_prefill`` is set, in which case the chunk-prefill kernel
reads the cache in place (head size 256, bfloat16 query, causal). The kernel
output is compared with an fp32 reference computed on the same quantized
cache.
"""
import pytest
import torch

from tests.utils import format_tc
from vllm_xpu_kernels.flash_attn_interface import (_SPEC_DECODE_MAX_QLEN,
                                                   flash_attn_varlen_func)

NUM_HEADS = [(8, 8), (16, 4), (32, 4), (24, 2)]
HEAD_SIZES = [64, 128, 192, 256, 512]
BLOCK_SIZES = [16, 32, 64, 128]
DTYPES = [torch.half, torch.bfloat16]
SLIDING_WINDOWS = [None, 128]
SINK = [False, True]
MODES = ["decode", "spec_decode", "spec_decode_ragged"]

MINI_PYTEST_PARAMS = {
    "default": {
        "num_heads": [(16, 4)],
        "head_size": [128, 256],
        "block_size": [64],
        "window": [None],
    },
    "test_varlen_int8_kv_padded_page": {
        "num_heads": [(16, 2)],
        "block_size": [64],
    },
    "test_varlen_int8_kv_int8_dot": {
        "num_heads": [(6, 1)],
        "head_size": [256],
        "block_size": [64, 1600],
        "window": [None],
    },
    "test_varlen_int8_kv_prefill": {
        "num_heads": [(6, 1)],
        "block_size": [64, 1600],
    },
}


def quantize_per_token_head(x: torch.Tensor):
    """vLLM int8_per_token_head write semantics (fp32): scale = absmax / 127,
    round half away from zero, clamp to [-128, 127]."""
    xf = x.float()
    scale = torch.clamp(xf.abs().amax(-1) / 127.0, min=1e-6)
    y = xf * (1.0 / scale)[..., None]
    y = torch.where(y >= 0, y + 0.5, y - 0.5).clamp(-128.0, 127.0).trunc()
    return y.to(torch.int8), scale


def split_layout_views(raw, block_size, num_kv_heads, head_size):
    """K/V int8 views [nb, B, H, D] and scale views [nb, B, H] over raw
    [num_blocks, page_bytes] uint8 (the page may be padded at its tail)."""
    nb, page = raw.shape
    data = block_size * num_kv_heads * 2 * head_size
    sb = block_size * num_kv_heads * 4
    assert page >= data + 2 * sb and page % 64 == 0
    d = torch.as_strided(raw.view(torch.int8),
                         (nb, block_size, num_kv_heads, 2 * head_size),
                         (page, num_kv_heads * 2 * head_size, 2 * head_size,
                          1))
    f = raw.view(torch.float32)

    def scales(off):
        return torch.as_strided(f, (nb, block_size, num_kv_heads),
                                (page // 4, num_kv_heads, 1),
                                f.storage_offset() + off // 4)

    return (d[..., :head_size], d[..., head_size:], scales(data),
            scales(data + sb))


def ref_attn(q, k, v, scale, q_pos, window, sinks):
    g = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    s = torch.einsum("qhd,khd->hqk", q, k) * scale
    kpos = torch.arange(k.shape[0], device=q.device)
    ok = kpos[None, :] <= q_pos[:, None]
    if window:
        ok &= (q_pos[:, None] - kpos[None, :]) < window
    s = s.masked_fill(~ok[None], float("-inf"))
    if sinks is not None:
        m = torch.maximum(s.amax(-1), sinks[:, None].float())
        p = torch.exp(s - m[..., None])
        p = p / (p.sum(-1) + torch.exp(sinks[:, None].float() - m))[..., None]
    else:
        p = torch.softmax(s, -1)
    return torch.einsum("hqk,khd->qhd", p, v)


def build_int8_cache(seq_lens, block_size, hkv, head_size, dtype, dev,
                     pad_bytes=0, fill=None, codes=False):
    """Quantize random K/V into a split-layout cache with random block
    placement. Returns the views, the block table and the dequantized
    per-sequence K/V for the reference."""
    nblk = [(L + block_size - 1) // block_size for L in seq_lens]
    num_blocks = sum(nblk) + 3
    page = 2 * (head_size + 4) * hkv * block_size + pad_bytes
    if fill is None:
        # Garbage in unused slots, scales and page padding.
        raw = torch.randint(0, 256, (num_blocks, page),
                            dtype=torch.uint8).to(dev)
    else:
        raw = torch.full((num_blocks, page), fill, dtype=torch.uint8,
                         device=dev)
    kc, vc, ks, vs = split_layout_views(raw, block_size, hkv, head_size)
    perm = (torch.randperm(num_blocks - 1, device=dev) + 1).to(torch.int32)
    block_table = torch.zeros(len(seq_lens), max(nblk), dtype=torch.int32,
                              device=dev)
    keys, values, c = [], [], 0
    for i, (L, n) in enumerate(zip(seq_lens, nblk)):
        block_table[i, :n] = perm[c:c + n]
        c += n
        key = (torch.randn(L, hkv, head_size, device=dev) * 2).to(dtype)
        key[:, :, 7] *= 20  # outlier channel, as in real K
        val = torch.randn(L, hkv, head_size, device=dev).to(dtype)
        pos = torch.arange(L, device=dev)
        b = block_table[i, pos // block_size].long()
        o = pos % block_size
        kq, kscale = quantize_per_token_head(key)
        vq, vscale = quantize_per_token_head(val)
        kc[b, o], vc[b, o], ks[b, o], vs[b, o] = kq, vq, kscale, vscale
        keys.append(kq.float() * kscale[..., None])
        values.append(
            (vq, vscale) if codes else vq.float() * vscale[..., None])
    return kc, vc, ks, vs, block_table, keys, values


def quantize_query_rule(q: torch.Tensor):
    """The decode kernel's int8_dot query rule (int8_attn_query_scale /
    int8_attn_query_code), exactly: scale = max(absmax / 127, 1e-10) and
    1 / scale as IEEE fp32 divisions (a float64 quotient of fp32 operands
    rounds to the same fp32), y = q * (1 / scale), codes half away from zero:
    sign(y) * min((trunc(2|y|) + 1) >> 1, 127)."""
    x = q.float()
    scale = torch.clamp((x.abs().amax(-1).double() / 127.0).float(), min=1e-10)
    inv = (1.0 / scale.double()).float()
    y = x * inv[..., None]
    c = (torch.clamp(y.abs() * 2.0, max=255.0).to(torch.int32) + 1) >> 1
    c = torch.clamp(c, max=127)
    return torch.where(y < 0, -c, c).to(torch.int8), scale


def quantize_p_rule(p: torch.Tensor):
    """The chunk-prefill kernel's int8 P rule per row of a 32-slot tile
    (int8_attn_p_scale / int8_attn_query_code), P >= 0: scale = max(row max /
    127, 1e-30) and 1 / scale as IEEE fp32 divisions, codes half up."""
    scale = torch.clamp((p.amax(-1).double() / 127.0).float(), min=1e-30)
    inv = (1.0 / scale.double()).float()
    y = p * inv[..., None]
    c = (torch.clamp(y * 2.0, max=255.0).to(torch.int32) + 1) >> 1
    return torch.clamp(c, max=127).float(), scale


def ref_attn_pv8(q, k, kv_codes, kv_scales, scale, q_pos):
    """fp32 reference of int8_prefill=3: online softmax over 32-slot tiles,
    P * v_scale quantized per (row, tile) by the kernel's rule, P*V on the
    codes, times the tile's P scale. q [q, hq, D] (codes x scales), k the
    dequantized keys [L, hkv, D], V as (codes [L, hkv, D], scales [L, hkv])."""
    vq, vs = kv_codes, kv_scales
    g = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(g, 1)
    vq, vs = vq.repeat_interleave(g, 1).float(), vs.repeat_interleave(g, 1)
    s = torch.einsum("qhd,khd->hqk", q, k) * scale
    kpos = torch.arange(k.shape[0], device=q.device)
    s = s.masked_fill(~(kpos[None, :] <= q_pos[:, None])[None], float("-inf"))
    m = s.amax(-1, keepdim=True)
    p = torch.exp(s - m)
    den = p.sum(-1)
    out = torch.zeros(q.shape[1], q.shape[0], q.shape[2], device=q.device)
    for t0 in range(0, k.shape[0], 32):
        pv = p[..., t0:t0 + 32] * vs[t0:t0 + 32].T[:, None, :]
        codes, ps = quantize_p_rule(pv)
        out += torch.einsum("hqk,khd->hqd", codes,
                            vq[t0:t0 + 32]) * ps[..., None]
    return (out / den[..., None]).transpose(0, 1)


def run_and_check(q_lens, seq_lens, hq, hkv, head_size, block_size, window,
                  sinks, dtype, dev, pad_bytes=0, fill=None, int8_dot=False,
                  int8_prefill=0, tol=1.5e-2):
    kc, vc, ks, vs, block_table, keys, values = build_int8_cache(
        seq_lens, block_size, hkv, head_size, dtype, dev, pad_bytes, fill,
        codes=int8_prefill == 3)
    q = torch.randn(sum(q_lens), hq, head_size, device=dev).to(dtype)
    cu_q = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(),
                        dtype=torch.int32, device=dev)
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    scale = head_size**-0.5
    out = torch.empty_like(q)
    flash_attn_varlen_func(q,
                           kc,
                           vc,
                           max(q_lens),
                           cu_q,
                           max(seq_lens),
                           seqused_k=seqused_k,
                           softmax_scale=scale,
                           causal=True,
                           window_size=[window - 1, 0] if window else None,
                           block_table=block_table,
                           out=out,
                           k_descale=ks,
                           v_descale=vs,
                           s_aux=sinks,
                           int8_dot=int8_dot,
                           int8_prefill=int8_prefill)
    # Prefill rows take int8 Q*K with int8_prefill 2 and 3; decode rows of the
    # batch take it with int8_dot.
    int8_q = (int8_prefill >= 2) if int8_prefill else int8_dot
    if int8_q:
        # The int8 query (codes x scales) the kernel multiplies with.
        qc, qs = quantize_query_rule(q)
        q_ref = qc.float() * qs[..., None]
    else:
        q_ref = q.float()
    ref = []
    for i, (L, ql) in enumerate(zip(seq_lens, q_lens)):
        q0 = int(cu_q[i])
        q_pos = torch.arange(L - ql, L, device=dev)
        if int8_prefill == 3 and ql > 1:
            vq, vsc = values[i]
            ref.append(
                ref_attn_pv8(q_ref[q0:q0 + ql], keys[i], vq, vsc, scale, q_pos))
            continue
        if int8_prefill == 3:
            vq, vsc = values[i]
            v_deq = vq.float() * vsc[..., None]
        else:
            v_deq = values[i]
        # Decode rows of a mixed batch: the decode kernel's query.
        q_rows = q_ref if (ql > 1 or not int8_prefill) else (
            _int8_query(q) if int8_dot else q.float())
        ref.append(
            ref_attn(q_rows[q0:q0 + ql], keys[i], v_deq, scale, q_pos, window,
                     sinks))
    torch.testing.assert_close(out.float(),
                               torch.cat(ref),
                               atol=tol,
                               rtol=tol)
    return out


def _int8_query(q):
    qc, qs = quantize_query_rule(q)
    return qc.float() * qs[..., None]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("window", SLIDING_WINDOWS)
@pytest.mark.parametrize("is_sink", SINK)
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@torch.inference_mode()
def test_varlen_int8_kv(mode, num_heads, head_size, block_size, window,
                        is_sink, dtype):
    if mode != "decode" and _SPEC_DECODE_MAX_QLEN < 5:
        pytest.skip("speculative-decode route disabled "
                    "(VLLM_XPU_SPEC_DECODE_MAX_QLEN)")
    torch.manual_seed(0)
    dev = "xpu"
    hq, hkv = num_heads
    seq_lens = [17, 64, 200, 1 if mode == "decode" else 3, 129]
    q_lens = {
        "decode": [1] * 5,
        # Every sequence has the same q_len (<= seq_len).
        "spec_decode": [3] * 5,
        # Adaptive verification: per-sequence q_len.
        "spec_decode_ragged": [3, 1, 5, 2, 4],
    }[mode]
    sinks = (torch.randn(hq, device=dev) * 2).to(dtype) if is_sink else None
    run_and_check(q_lens, seq_lens, hq, hkv, head_size, block_size, window,
                  sinks, dtype, dev, fill=0x7F)


@pytest.mark.parametrize("num_heads", [(16, 2), (24, 4)])
@pytest.mark.parametrize("block_size", [64, 832])
@pytest.mark.parametrize("pad_bytes", [0, 1024])
@torch.inference_mode()
def test_varlen_int8_kv_padded_page(num_heads, block_size, pad_bytes):
    """Large and padded pages (hybrid models unify the page size across
    attention and state layers), garbage in every unused byte."""
    torch.manual_seed(0)
    hq, hkv = num_heads
    run_and_check([1] * 4, [1, 700, 1500, 3000], hq, hkv, 256, block_size,
                  None, None, torch.bfloat16, "xpu", pad_bytes)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("num_heads", [(6, 1), (16, 4), (24, 2)])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("block_size", [64, 128, 1600])
@pytest.mark.parametrize("window", SLIDING_WINDOWS)
@pytest.mark.parametrize("is_sink", SINK)
@torch.inference_mode()
def test_varlen_int8_kv_int8_dot(mode, num_heads, head_size, block_size,
                                 window, is_sink):
    """int8_dot: Q quantized per (token, query head) in the kernel, Q*K on
    s8 x s8 -> s32 DPAS. Checked against the fp32 reference over the int8
    query of the shared rule (codes x scales) and the same cache; and, more
    loosely, against the BF16-query kernel."""
    if mode != "decode" and _SPEC_DECODE_MAX_QLEN < 5:
        pytest.skip("speculative-decode route disabled "
                    "(VLLM_XPU_SPEC_DECODE_MAX_QLEN)")
    torch.manual_seed(0)
    dev = "xpu"
    hq, hkv = num_heads
    seq_lens = [17, 64, 200, 1 if mode == "decode" else 3, 2100]
    q_lens = {
        "decode": [1] * 5,
        "spec_decode": [3] * 5,
        "spec_decode_ragged": [3, 1, 5, 2, 16],
    }[mode]
    sinks = (torch.randn(hq, device=dev) * 2).to(
        torch.bfloat16) if is_sink else None
    args = (q_lens, seq_lens, hq, hkv, head_size, block_size, window, sinks,
            torch.bfloat16, dev)
    torch.manual_seed(1)
    out8 = run_and_check(*args, fill=0x7F, int8_dot=True)
    torch.manual_seed(1)
    out16 = run_and_check(*args, fill=0x7F)
    assert torch.isfinite(out8).all()
    torch.testing.assert_close(out8.float(), out16.float(), atol=4e-2,
                               rtol=4e-2)


def test_varlen_int8_kv_int8_dot_refusals():
    dev = "xpu"
    raw = torch.zeros((4, 2 * (512 + 4) * 1 * 64),
                      dtype=torch.uint8,
                      device=dev)
    for head_size, dtype in ((128, torch.half), (512, torch.bfloat16)):
        kc, vc, ks, vs = split_layout_views(raw, 64, 1, head_size)
        q = torch.randn(1, 6, head_size, device=dev, dtype=dtype)
        with pytest.raises((RuntimeError, ValueError), match="int8_dot"):
            flash_attn_varlen_func(q,
                                   kc,
                                   vc,
                                   1,
                                   torch.tensor([0, 1],
                                                dtype=torch.int32,
                                                device=dev),
                                   64,
                                   seqused_k=torch.full((1, ),
                                                        64,
                                                        dtype=torch.int32,
                                                        device=dev),
                                   causal=True,
                                   block_table=torch.ones((1, 1),
                                                          dtype=torch.int32,
                                                          device=dev),
                                   k_descale=ks,
                                   v_descale=vs,
                                   int8_dot=True)


# (q_lens, seq_lens): a chunk after cached context, a fresh prompt crossing
# blocks, a mixed batch with a decode row and a short verify row, a chunk whose
# context ends inside a block.
PREFILL_CASES = [
    ([300], [2100]),
    ([257], [257]),
    ([1, 40, 5, 64], [90, 40, 700, 1700]),
    ([96], [1631]),
]


@pytest.mark.parametrize("int8_prefill", [1, 2, 3])
@pytest.mark.parametrize("num_heads", [(6, 1), (16, 4), (24, 2)])
@pytest.mark.parametrize("block_size", [64, 128, 1600])
@pytest.mark.parametrize("case", range(len(PREFILL_CASES)))
@pytest.mark.parametrize("int8_dot", [False, True])
@torch.inference_mode()
def test_varlen_int8_kv_prefill(int8_prefill, num_heads, block_size, case,
                                int8_dot):
    """int8_prefill: the chunk-prefill kernel reads the int8 cache in place
    (garbage in every unused byte). 1: BF16 dots over the widened codes,
    against the BF16-query reference; 2: int8 Q*K (the decode kernel's query
    rule), against the int8-query reference; 3: also int8 P*V, against the
    reference with the kernel's P rounding. Decode rows of a mixed batch take
    the paged decode kernel (int8_dot as passed)."""
    torch.manual_seed(case)
    hq, hkv = num_heads
    q_lens, seq_lens = PREFILL_CASES[case]
    run_and_check(q_lens, seq_lens, hq, hkv, 256, block_size, None, None,
                  torch.bfloat16, "xpu", int8_dot=int8_dot,
                  int8_prefill=int8_prefill,
                  tol=3e-2 if int8_prefill == 3 else 1.5e-2)


def test_varlen_int8_kv_prefill_refusals():
    dev = "xpu"
    for head_size, dtype, window, sink in ((128, torch.bfloat16, None, False),
                                           (256, torch.half, None, False),
                                           (256, torch.bfloat16, 128, False),
                                           (256, torch.bfloat16, None, True)):
        page = 2 * (head_size + 4) * 64
        raw = torch.zeros((4, page), dtype=torch.uint8, device=dev)
        kc, vc, ks, vs = split_layout_views(raw, 64, 1, head_size)
        q = torch.randn(40, 6, head_size, device=dev, dtype=dtype)
        with pytest.raises((RuntimeError, ValueError), match="int8_prefill"):
            flash_attn_varlen_func(
                q, kc, vc, 40,
                torch.tensor([0, 40], dtype=torch.int32, device=dev), 64,
                seqused_k=torch.full((1, ), 64, dtype=torch.int32, device=dev),
                causal=True,
                window_size=[window - 1, 0] if window else None,
                block_table=torch.ones((1, 1), dtype=torch.int32, device=dev),
                k_descale=ks, v_descale=vs,
                s_aux=torch.zeros(6, device=dev, dtype=dtype) if sink else None,
                int8_prefill=2)


@pytest.mark.parametrize("q_lens", [[33], [1, 33]])
def test_varlen_int8_kv_rejects_prefill(q_lens):
    """Batches routed to chunk prefill (q_len above the speculative-decode
    route) are rejected."""
    if max(q_lens) <= _SPEC_DECODE_MAX_QLEN:
        pytest.skip("q_len within the speculative-decode route")
    dev = "xpu"
    hq, hkv, head_size, block_size = 16, 4, 128, 64
    page = 2 * (head_size + 4) * hkv * block_size
    raw = torch.zeros((4, page), dtype=torch.uint8, device=dev)
    kc, vc, ks, vs = split_layout_views(raw, block_size, hkv, head_size)
    n = len(q_lens)
    q = torch.randn(sum(q_lens), hq, head_size, device=dev, dtype=torch.half)
    cu_q = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(),
                        dtype=torch.int32,
                        device=dev)
    with pytest.raises(RuntimeError, match="decode"):
        flash_attn_varlen_func(q,
                               kc,
                               vc,
                               max(q_lens),
                               cu_q,
                               64,
                               seqused_k=torch.full((n, ),
                                                    64,
                                                    dtype=torch.int32,
                                                    device=dev),
                               causal=True,
                               block_table=torch.ones((n, 1),
                                                      dtype=torch.int32,
                                                      device=dev),
                               k_descale=ks,
                               v_descale=vs)


def test_varlen_int8_kv_requires_scales():
    dev = "xpu"
    raw = torch.zeros((4, 2 * (128 + 4) * 4 * 64),
                      dtype=torch.uint8,
                      device=dev)
    kc, vc, _, _ = split_layout_views(raw, 64, 4, 128)
    q = torch.randn(2, 16, 128, device=dev, dtype=torch.half)
    with pytest.raises(ValueError, match="scale"):
        flash_attn_varlen_func(q,
                               kc,
                               vc,
                               1,
                               torch.tensor([0, 1, 2],
                                            dtype=torch.int32,
                                            device=dev),
                               64,
                               seqused_k=torch.full((2, ),
                                                    64,
                                                    dtype=torch.int32,
                                                    device=dev),
                               causal=True,
                               block_table=torch.ones((2, 1),
                                                      dtype=torch.int32,
                                                      device=dev))
