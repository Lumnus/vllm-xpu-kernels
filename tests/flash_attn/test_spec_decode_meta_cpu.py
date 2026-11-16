# SPDX-License-Identifier: Apache-2.0
"""The spec-decode fast path's metadata built once per forward
(``spec_decode_meta_cache``) gives every layer the values it builds itself,
bit for bit, and launches the 14 metadata kernels once. CPU only: the FA op is
replaced by a stand-in that records what it is given and writes an output
that depends on every metadata value; it runs in a subprocess so the stand-in
never reaches this session.

Cases: 1-16 query rows, requests of 1-8 query tokens, empty requests,
padding rows past the last query, short KV (the clamp to 1), 12 layers per
forward; a second forward with a new cache, and a call with another
seqused_k tensor in the same forward, rebuild."""

import os
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HEADS, DIM = 2, 4


def _reference_metadata(torch, cu_seqlens_q, seqused_k, block_table, total):
    """The metadata as _spec_decode_varlen_fwd built it before the cache."""
    batch = cu_seqlens_q.numel() - 1
    cu_seqlens_q_spec = torch.arange(total + 1, dtype=torch.int32)
    dummy_cu_seqlens_k = torch.zeros_like(cu_seqlens_q_spec)
    positions = cu_seqlens_q_spec[:-1]
    request_ids = torch.searchsorted(cu_seqlens_q[1:], positions,
                                     right=True).clamp_(max=batch - 1)
    query_ends = cu_seqlens_q[1:].index_select(0, request_ids)
    is_padding = positions >= cu_seqlens_q[-1]
    seqused_k_spec = (seqused_k.index_select(0, request_ids) -
                      (query_ends - positions - 1)).clamp_(min=1).to(
                          torch.int32)
    seqused_k_spec.masked_fill_(is_padding, 1)
    block_table_spec = block_table.index_select(0, request_ids)
    block_table_spec.masked_fill_(is_padding[:, None], 0)
    return (cu_seqlens_q_spec, dummy_cu_seqlens_k, seqused_k_spec,
            block_table_spec, is_padding)


def _check():
    import torch
    from torch.utils._python_dispatch import (TorchDispatchMode,
                                              _disable_current_modes)
    try:
        import vllm_xpu_kernels._vllm_fa2_C  # noqa: F401
    except ImportError:
        sys.modules["vllm_xpu_kernels._vllm_fa2_C"] = types.ModuleType("x")
    from vllm_xpu_kernels import flash_attn_interface as fai

    seen = []

    def varlen_fwd(q, k, v, out, cu_q, cu_k, seqused, _a, bt, *rest):
        with _disable_current_modes():
            seen.append(
                (cu_q.clone(), cu_k.clone(), seqused.clone(), bt.clone()))
            val = (seqused.double() * 7 + bt.double().sum(1) +
                   cu_q[:-1].double() + cu_k[:-1].double())
            out.copy_(val[:, None, None].to(out.dtype).expand_as(out))
        return out, None

    fai.torch.ops._vllm_fa2_C.varlen_fwd = varlen_fwd

    class Count(TorchDispatchMode):

        def __init__(self):
            super().__init__()
            self.ops = []

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            name = func.overloadpacket.__name__
            if name not in ("slice", "select", "unsqueeze", "view", "expand",
                            "detach", "clone", "copy_", "_to_copy",
                            "alias", "lift_fresh"):
                self.ops.append(name)
            return func(*args, **(kwargs or {}))

    gen = torch.Generator().manual_seed(0)
    cases = 0
    for trial in range(60):
        batch = int(torch.randint(1, 6, (1, ), generator=gen))
        lens = torch.randint(0, 9, (batch, ), generator=gen)
        if trial % 7 == 0:
            lens[0] = 0  # an empty request
        cu = torch.zeros(batch + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(lens, 0)
        real = int(cu[-1])
        pad = int(torch.randint(0, 4, (1, ), generator=gen))
        total = max(1, min(16, real + pad))
        if total < real:
            continue
        seqused = (torch.randint(1, 40, (batch, ), generator=gen) +
                   lens).to(torch.int32)
        if trial % 5 == 0:
            seqused[-1] = 1  # KV shorter than the query: the clamp to 1
        bt = torch.randint(0, 1000, (batch, 6), generator=gen,
                           dtype=torch.int32)
        q = torch.randn(total, HEADS, DIM)
        k = v = torch.zeros(1)
        args = (q, k, v)

        ref = _reference_metadata(torch, cu, seqused, bt, total)
        for a, b in zip(ref, fai.spec_decode_metadata(cu, seqused, bt, total)):
            assert torch.equal(a, b)

        def layers(cache,
                   n=12,
                   args=args,
                   cu=cu,
                   seqused=seqused,
                   bt=bt,
                   total=total):
            outs, counts = [], []
            for _ in range(n):
                out = torch.empty(total, HEADS, DIM)
                with Count() as c:
                    fai._spec_decode_varlen_fwd(*args, out, cu, seqused, bt,
                                                64, None, None, 1.0, None,
                                                (-1, -1), 0.0, cache)
                outs.append(out)
                counts.append(len(c.ops))
            return outs, counts

        seen.clear()
        base, base_counts = layers(None)
        base_seen = list(seen)
        seen.clear()
        cache = {}
        once, once_counts = layers(cache)
        for a, b in zip(base, once):
            assert torch.equal(a, b)
        for sa, sb in zip(base_seen, seen):
            for a, b in zip(sa, sb):
                assert torch.equal(a, b)
        # the metadata kernels run in the first layer only (the output
        # masked_fill stays in every layer)
        assert base_counts == [base_counts[0]] * 12
        assert once_counts[0] == base_counts[0]
        assert once_counts[1:] == [1] * 11, once_counts
        assert len(cache["spec_decode_meta"]) == 1
        # a new forward starts with a new cache and builds again
        again, again_counts = layers({}, 2)
        assert again_counts[0] == base_counts[0]
        assert torch.equal(again[0], base[0])
        # another seqused_k tensor in the same forward is its own entry
        other = seqused.clone()
        out = torch.empty(total, HEADS, DIM)
        fai._spec_decode_varlen_fwd(*args, out, cu, other, bt, 64, None, None,
                                    1.0, None, (-1, -1), 0.0, cache)
        assert len(cache["spec_decode_meta"]) == 2
        assert torch.equal(out, base[0])
        cases += 1
    print(f"cases {cases}, metadata ops per layer {base_counts[0] - 1}, "
          f"once: first layer only; outputs and metadata bitwise")
    assert cases >= 40


@pytest.mark.skipif(sys.platform != "linux", reason="CPU check")
def test_spec_decode_metadata_once_per_forward():
    r = subprocess.run([sys.executable, os.path.abspath(__file__)],
                       cwd="/",
                       env={
                           **os.environ, "PYTHONPATH": ROOT
                       },
                       capture_output=True,
                       text=True,
                       timeout=600)
    print(r.stdout)
    assert r.returncode == 0, r.stderr[-4000:] + r.stdout


if __name__ == "__main__":
    _check()
