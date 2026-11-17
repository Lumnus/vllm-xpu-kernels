# SPDX-License-Identifier: Apache-2.0
"""The W4A8 MoE path's Python around its ops (moe_w4a8.apply through
XpuFusedMoe.apply): without the padding mask and the torch re-count of
rows_per_expert it gives the same output and the same rows_per_expert as
with them (VLLM_XPU_W4A8_STOCK_PREP=1), bit for bit, with expert parallelism
off and on, unrouted (-1) ids, sentinel-mapped ids, out-of-range ids, NaN
padding rows and a stale caller buffer. CPU only.

The device ops are replaced by torch references of their contracts as
csrc/moe/remap_hidden_states.cpp and csrc/moe/moe_gather.cpp implement them
(the remap clears rows_per_expert, then counts and places every slot whose id
is in [0, total_experts) and maps to a local expert; the gather skips -1
slots and writes zeros for a row with none), and an int4 x int8 grouped GEMM
reference that reads only the rows rows_per_expert assigns. The check runs in
a subprocess so stand-in extension modules, when the compiled ones are
absent, never reach this session.

Run as a script, ``--save <file>`` stores the outputs, rows_per_expert and
the aten ops each call launched (outside the op references), so trees can be
compared."""

import os
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def _import_kernels():
    """vllm_xpu_kernels' Python modules, with empty stand-ins for the
    compiled extensions when they are not importable here."""
    try:
        import vllm_xpu_kernels._C  # noqa: F401
        import vllm_xpu_kernels._moe_C  # noqa: F401
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except ImportError:
        for name in ("_C", "_moe_C", "_xpu_C"):
            sys.modules[f"vllm_xpu_kernels.{name}"] = types.ModuleType(name)
    from vllm_xpu_kernels import fused_moe_interface, moe_w4a8
    return fused_moe_interface, moe_w4a8


def _run_cases():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    in_ref = [False]
    launched = []

    class Count(TorchDispatchMode):

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            if not in_ref[0]:
                launched.append(str(func.overloadpacket.__name__))
            return func(*args, **(kwargs or {}))

    def ref(fn):

        def wrapped(*args, **kwargs):
            in_ref[0] = True
            try:
                return fn(*args, **kwargs)
            finally:
                in_ref[0] = False

        return wrapped

    def local_id(g, expert_map, total, n_local):
        if g < 0 or g >= total:
            return -1
        loc = g if expert_map is None else int(expert_map[g])
        return loc if 0 <= loc < n_local else -1

    @ref
    def remap(hidden_states, hidden_states_scales, remapped_hidden_states,
              remapped_hidden_states_scales, expert_map, rows_per_expert,
              unpermuted_row_to_permuted_row, topk_ids, total_experts_num,
              local_experts_num, expert_scale_desc=None):
        topk = topk_ids.shape[1]
        ids = topk_ids.reshape(-1).tolist()
        loc = [local_id(g, expert_map, total_experts_num, local_experts_num)
               for g in ids]
        rows_per_expert.zero_()
        counts = [0] * local_experts_num
        for x in loc:
            if x >= 0:
                counts[x] += 1
        rows_per_expert.add_(torch.tensor(counts, dtype=torch.int32))
        base, run = [], 0
        for c in counts:
            base.append(run)
            run += c
        seen = [0] * local_experts_num
        src = hidden_states.view(torch.uint8)
        dst = remapped_hidden_states.view(torch.uint8)
        unp = unpermuted_row_to_permuted_row.view(-1)
        for slot, x in enumerate(loc):
            if x < 0:
                unp[slot] = -1
                continue
            row = base[x] + seen[x]
            seen[x] += 1
            unp[slot] = row
            dst[row] = src[slot // topk]
            remapped_hidden_states_scales[row] = hidden_states_scales[slot //
                                                                      topk]

    def dequant_int4(w, s):
        b = w.view(torch.uint8).to(torch.int32)
        lo, hi = b & 0xF, (b >> 4) & 0xF
        v = torch.stack([lo, hi], dim=-1).reshape(w.shape[0], -1)
        v = torch.where(v >= 8, v - 16, v).float()
        return v * s.float().repeat_interleave(128, dim=-1)

    @ref
    def grouped_gemm_w4a8(a, a_scale, w, w_scale, bias, d, rows_per_expert,
                          n, k, num_experts, tile=None):
        start = 0
        for e, rows in enumerate(rows_per_expert.tolist()):
            if rows == 0:
                continue
            x = a[start:start + rows].float()
            sc = a_scale[start:start + rows].float()
            x = x * (sc if sc.shape[1] == 1 else sc.repeat_interleave(
                k // sc.shape[1], dim=1))
            y = x @ dequant_int4(w[e], w_scale[e]).t()
            if bias is not None:
                y = y + bias[e].float()
            d[start:start + rows] = y.to(d.dtype)
            start += rows

    @ref
    def act_func(out, g1):
        inter = out.shape[1]
        gate, up = g1[:, :inter].float(), g1[:, inter:].float()
        out.copy_((torch.nn.functional.silu(gate) * up).to(out.dtype))

    @ref
    def moe_gather(output, g2, topk_weights, unp, num_experts):
        for t in range(output.shape[0]):
            acc = torch.zeros(output.shape[1], dtype=torch.float32)
            for k, row in enumerate(unp[t].tolist()):
                if row == -1:
                    continue
                acc += g2[row].float() * topk_weights[t, k].float()
            output[t] = acc.to(output.dtype)

    moe_ns = types.SimpleNamespace(remap_hidden_states=remap,
                                   moe_gather=moe_gather)
    xpu_ns = types.SimpleNamespace(
        cutlass_grouped_gemm_w4a8=grouped_gemm_w4a8)

    class COps:
        """``torch.ops._C``: placeholders for the ops moe_utils binds at
        import; no per-token int8 quant op, so quant_rows takes its torch
        path (per row, as the op)."""

        def __getattr__(self, name):
            if name == "dynamic_per_token_scaled_int8_quant":
                raise AttributeError(name)

            def unused(*args, **kwargs):
                raise AssertionError(f"_C.{name} is not on the W4A8 path")

            return unused

    c_ns = COps()
    real_getattr = type(torch.ops).__getattr__

    class Ops(type(torch.ops)):

        def __getattr__(self, name):
            if name == "_moe_C":
                return moe_ns
            if name == "_xpu_C":
                return xpu_ns
            if name == "_C":
                return c_ns
            return real_getattr(self, name)

    for name in ("_C", "_moe_C", "_xpu_C"):
        torch.ops.__dict__.pop(name, None)  # namespaces cached on first use
    torch.ops.__class__ = Ops
    fmi, moe_w4a8 = _import_kernels()
    moe_w4a8._OP = None
    moe_w4a8._GLU = None

    hidden, inter, topk = 256, 128, 4
    gen = torch.Generator().manual_seed(11)

    def make_impl(n_local, ep_rank, ep_size, expert_map):
        impl = object.__new__(fmi.XpuFusedMoe)
        impl.n_experts_per_token = topk
        impl.inter_size = inter
        impl.num_experts = n_local
        impl.ep_rank, impl.ep_size = ep_rank, ep_size
        impl.expert_map = expert_map
        impl.total_experts_num = (expert_map.shape[0] if expert_map
                                  is not None else n_local * ep_size)
        impl.local_experts_num = n_local
        impl.w13 = torch.randint(-128, 128, (n_local, 2 * inter, hidden // 2),
                                 dtype=torch.int8, generator=gen)
        impl.w2 = torch.randint(-128, 128, (n_local, hidden, inter // 2),
                                dtype=torch.int8, generator=gen)
        impl.gemm1_wei_scales = (torch.rand(
            (n_local, 2 * inter, hidden // 128), generator=gen) /
                                 64).to(torch.bfloat16)
        impl.gemm2_wei_scales = (torch.rand((n_local, hidden, inter // 128),
                                            generator=gen) /
                                 64).to(torch.bfloat16)
        impl.w13_bias = impl.w2_bias = None
        impl.act_func = act_func
        impl.activation = "silu"
        impl.gemm1_clamp_limit = None
        impl.is_int4 = True
        impl._use_ref = False
        impl._w4a8_why = None
        return impl

    def routed(m, n_global, pad):
        a = (torch.randn((m + pad, hidden), generator=gen) / 8).to(
            torch.bfloat16)
        a[m:] = float("nan")
        logits = torch.randn((m + pad, n_global), generator=gen)
        w, ids = torch.topk(logits.softmax(-1), topk, dim=-1)
        ids = ids.to(torch.int32)
        ids[::3, 1] = -1  # unrouted slots
        ids[m:] = -1  # padding rows
        return a, w, ids

    results = {}
    for m in (1, 7, 33):
        cases = []
        # Expert parallelism off: no expert map, the routed ids as they are.
        impl = make_impl(8, 0, 1, None)
        a, w, ids = routed(m, 8, 3)
        cases.append(("ep1", impl, a, w, ids, {}))
        for ep_rank in (0, 3):
            n_global, n_local, ep_size = 16, 4, 4
            emap = torch.full((n_global, ), -1, dtype=torch.int32)
            emap[ep_rank * n_local:(ep_rank + 1) * n_local] = torch.arange(
                n_local, dtype=torch.int32)
            impl = make_impl(n_local, ep_rank, ep_size, emap)
            a, w, ids = routed(m, n_global, 3)
            sentinel = n_global - 1 if ep_rank == 0 else 0
            sids = torch.where(ids < 0, sentinel, ids)
            # The routed ids as they are (the op filters them).
            cases.append((f"ep4r{ep_rank}-raw", impl, a, w, ids, {}))
            # The layer's sentinel and a stale counts buffer (ids_valid).
            cases.append((f"ep4r{ep_rank}-sentinel", impl, a, w, sids, {
                "ids_valid": True,
                "rows_per_expert": "stale"
            }))
            oob = ids.clone()
            oob[::4, 0] = n_global + 5
            cases.append((f"ep4r{ep_rank}-oob", impl, a, w, oob, {
                "ids_valid": True,
                "rows_per_expert": "zero"
            }))
            # A moved placement (EPLB): this rank owns scattered experts.
            perm = torch.randperm(n_global, generator=gen)
            pmap = torch.full((n_global, ), -1, dtype=torch.int32)
            pmap[perm[:n_local]] = torch.arange(n_local, dtype=torch.int32)
            pimpl = make_impl(n_local, ep_rank, ep_size, pmap)
            unowned = int(perm[n_local])
            cases.append((f"ep4r{ep_rank}-moved", pimpl, a, w,
                          torch.where(ids < 0, unowned, ids), {
                              "ids_valid": True,
                              "rows_per_expert": "stale"
                          }))
        for name, impl, a, w, ids, kw in cases:
            kw = dict(kw)
            rpe = None
            if kw.get("rows_per_expert") is not None:
                rpe = torch.full((impl.num_experts, ),
                                 7 if kw["rows_per_expert"] == "stale" else 0,
                                 dtype=torch.int32)
                kw["rows_per_expert"] = rpe
            out = torch.full_like(a, 3.0)
            launched.clear()
            with Count():
                fmi.XpuFusedMoe.apply(impl, output=out, hidden_states=a,
                                      topk_weights=w, topk_ids=ids,
                                      w4a8=True, **kw)
            results[f"{name}-m{m}"] = {
                "out": out.clone(),
                "rows_per_expert": None if rpe is None else rpe.clone(),
                "ops": list(launched),
            }
    return results


def _main(argv):
    import torch

    stock = os.environ.get("VLLM_XPU_W4A8_STOCK_PREP", "0") == "1"
    res = _run_cases()
    if "--save" in argv:
        torch.save(res, argv[argv.index("--save") + 1])
    for key, r in res.items():
        if r["out"].isnan().any():
            raise SystemExit(f"{key}: NaN in the output")
    print("W4A8-PREP-RUN-OK", "stock" if stock else "lean", len(res))


def _run(stock: bool, save: str):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               VLLM_XPU_W4A8_STOCK_PREP="1" if stock else "0")
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    run = subprocess.run(
        [sys.executable, "-P", os.path.abspath(__file__), "--save", save],
        env=env, capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0 and "W4A8-PREP-RUN-OK" in run.stdout, (
        run.stdout[-2000:] + run.stderr[-4000:])


def test_w4a8_prep_bit_identical(tmp_path):
    torch = pytest.importorskip("torch")
    lean, stock = str(tmp_path / "lean.pt"), str(tmp_path / "stock.pt")
    _run(False, lean)
    _run(True, stock)
    a, b = torch.load(lean), torch.load(stock)
    assert a.keys() == b.keys() and len(a) == 27
    for key in a:
        assert torch.equal(a[key]["out"], b[key]["out"]), key
        ra, rb = a[key]["rows_per_expert"], b[key]["rows_per_expert"]
        assert (ra is None) == (rb is None), key
        if ra is not None:
            assert torch.equal(ra, rb), key
        # Padding rows (all ids negative, NaN activations) come out as zeros.
        assert not a[key]["out"][-3:].any(), key
        # Without the mask and the re-count the path launches no masked_fill,
        # no index_add_ and no bool reduction.
        for op in ("masked_fill", "index_add", "all", "bitwise_and"):
            assert not any(x.startswith(op) for x in a[key]["ops"]), (key, op)
        assert len(a[key]["ops"]) < len(b[key]["ops"]), key


if __name__ == "__main__":
    _main(sys.argv)
