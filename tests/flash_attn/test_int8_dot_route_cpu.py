# SPDX-License-Identifier: Apache-2.0
"""flash_attn_varlen_func(..., int8_dot=...) reaches the FA op: as the op's
int8_dot argument when true (plain decode rows and the spec-decode route),
not at all when false (so the call is the one without the argument), and is
refused for a non-INT8 cache, a non-bfloat16 query or rows above the
spec-decode bound. CPU only: the op is a stand-in that records its keyword
arguments; it runs in a subprocess so the stand-in never reaches this
session."""

import os
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def _check():
    import torch
    try:
        import vllm_xpu_kernels._vllm_fa2_C  # noqa: F401
    except ImportError:
        sys.modules["vllm_xpu_kernels._vllm_fa2_C"] = types.ModuleType("x")
    from vllm_xpu_kernels import flash_attn_interface as fai

    assert fai.INT8_DOT_DECODE
    seen = []

    def varlen_fwd(q, k, v, out, *rest, **kwargs):
        seen.append(dict(kwargs))
        return out, None

    fai.torch.ops._vllm_fa2_C.varlen_fwd = varlen_fwd
    hq, hkv, d, blk = 6, 1, 256, 64
    k = torch.zeros(4, blk, hkv, d, dtype=torch.int8)
    ks = torch.ones(4, blk, hkv)
    table = torch.ones(1, 2, dtype=torch.int32)
    seqused = torch.tensor([100], dtype=torch.int32)

    def call(rows, dot, q_dtype=torch.bfloat16, kv=k):
        q = torch.zeros(rows, hq, d, dtype=q_dtype)
        cu = torch.tensor([0, rows], dtype=torch.int32)
        seen.clear()
        sc = ks if kv.dtype == torch.int8 else None
        fai.flash_attn_varlen_func(q, kv, kv, rows, cu, 100, seqused_k=seqused,
                                   causal=True, block_table=table,
                                   out=torch.empty_like(q), k_descale=sc,
                                   v_descale=sc, spec_decode_meta_cache={},
                                   int8_dot=dot)
        return list(seen)

    assert call(1, True) == [{"int8_dot": True}]
    assert call(4, True) == [{"int8_dot": True}]  # spec-decode route
    assert call(16, True) == [{"int8_dot": True}]
    assert call(1, False) == [{}]
    assert call(4, False) == [{}]
    for bad in (dict(q_dtype=torch.half), dict(kv=k.to(torch.bfloat16))):
        try:
            call(1, True, **bad)
        except ValueError as e:
            assert "int8_dot" in str(e)
        else:
            raise AssertionError(f"accepted {bad}")
    try:
        call(fai._SPEC_DECODE_MAX_QLEN + 1, True)
    except ValueError as e:
        assert "int8_dot" in str(e)
    else:
        raise AssertionError("accepted rows above the spec-decode bound")
    print("int8_dot route: ok")


@pytest.mark.skipif(sys.platform != "linux", reason="CPU check")
def test_int8_dot_reaches_the_op():
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
