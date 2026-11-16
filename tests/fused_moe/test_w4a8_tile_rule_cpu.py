# SPDX-License-Identifier: Apache-2.0
"""The W4A8 grouped GEMM's tile rule (moe_w4a8.gemm_tile / unfused_tile),
CPU only. The default rule takes the rows an expert of this rank really gets
(routes / total experts) through the kernel's 4 / 8 / 128 thresholds; the
buffer rule (VLLM_XPU_W4A8_TILE_RULE=buffer) is the kernel's own (routes /
local experts, which under expert parallelism counts every rank's routes);
a fixed VLLM_XPU_W4A8_TILE leaves the choice to the op. Every tile gives
the same bits (one-card sweep, both GEMMs, 32-256 rows per local expert);
the table below is the rule the sweep's timings support: 32 x 64 up to 128
real rows per expert (1.23-1.51x the 128 x 256 at 64-81), 128 x 256 above."""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# real rows per local expert -> tile, and the fused path's M tile
TABLE = [(4, "8x64", 8), (8, "16x64", 16), (128, "32x64", 32),
         (10**9, "128x256", None)]

_CHECK = r"""
import json, sys, types
try:
    import vllm_xpu_kernels._C  # noqa: F401
except ImportError:
    for name in ("_C", "_moe_C", "_xpu_C"):
        sys.modules[f"vllm_xpu_kernels.{name}"] = types.ModuleType(name)
from vllm_xpu_kernels import moe_w4a8 as m
out = []
for ep in (1, 2, 4):
    total = 512
    local = total // ep
    for r in range(1, 257):
        n_moe = r * total  # every expert gets r rows on its rank
        out.append([ep, r, m.gemm_tile(n_moe, local, total),
                    m.unfused_tile(n_moe, local, total)])
print("TILES " + json.dumps(out))
"""


def _run(env_extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", **env_extra)
    if "VLLM_XPU_W4A8_TILE" not in env_extra:
        env.pop("VLLM_XPU_W4A8_TILE", None)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    run = subprocess.run([sys.executable, "-P", "-c", _CHECK], env=env,
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    import json
    line = [l for l in run.stdout.splitlines() if l.startswith("TILES ")][0]
    return json.loads(line[6:])


def _want(avg):
    for bound, name, rows in TABLE:
        if avg <= bound:
            return name, rows


def test_tile_rule_rows_default():
    pytest.importorskip("torch")
    rows = _run({"VLLM_XPU_W4A8_TILE_RULE": "rows"})
    assert len(rows) == 3 * 256
    for ep, r, tile, tm in rows:
        name, trows = _want(r)
        assert (tile, tm) == (name, trows), (ep, r, tile, tm)


def test_tile_rule_buffer_restores_kernel_rule():
    pytest.importorskip("torch")
    rows = _run({"VLLM_XPU_W4A8_TILE_RULE": "buffer"})
    for ep, r, tile, tm in rows:
        # The op chooses (None passed); the fused path's tile follows the
        # kernel's average over local experts: ep x the real rows.
        assert tile is None, (ep, r)
        assert tm == _want(r * ep)[1], (ep, r, tm)


def test_fixed_tile_leaves_choice_to_op():
    pytest.importorskip("torch")
    rows = _run({"VLLM_XPU_W4A8_TILE": "16x64"})
    assert all(tile is None for _, _, tile, _ in rows)
