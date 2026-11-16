# SPDX-License-Identifier: Apache-2.0
"""The fused W4A8 MoE decode path (moe_w4a8 with VLLM_XPU_W4A8_FUSED_MOE=1
or 2) gives the output and the rows_per_expert of the unfused path, bit for
bit, CPU only: expert parallelism off and on (ranks 0 and 3), unrouted (-1),
sentinel-mapped (unowned and owned), out-of-range ids, a moved placement,
NaN padding rows, a stale caller buffer, 1 to 16 rows and 33, top-4 and
top-10, and routings that put many rows on one expert (several tiles per
expert at tile 8, 16 and 32).

The device ops are torch references:
1. the ops both paths share, of their contracts as the csrc implements them:
   the int8 quant op (int8_quant_math.h, exact tie rule, bf16-rounded scale),
   silu_and_mul (bf16 silu, then the bf16 product), remap_hidden_states,
   moe_gather, and the W4A8 GEMM as its mainloop computes an element (exact
   int32 per 128-wide K group, fp32 sum of group partials times the weight
   scale in group order, times the row's activation scale, bf16 out), which
   makes an element independent of its row's position, as on the device;
2. the three fused ops, mirroring their kernels step by step: the map's
   lane ranges, scan, tile map, its bound (umt) and its slot order (the
   atomics are stood in for by a shuffled order, so the test also shows
   that the order inside an expert reaches no value); the quant + gather;
   the tile-map GEMM grid with surplus tiles; silu + quant up to the
   map's row count.

Each case also runs with the token rows quantized once outside the layer
(``x_q``, as the engine's MoE entry passes them) and with the shared
expert's term added in the gather (``shared``): the output must equal the
plain run's, and the plain output followed by torch's bf16 multiply and add
(``sigmoid(g) * shared``, then ``shared + routed``, the engine's unfused
sequence), bit for bit, NaN padding rows included.

What it proves is the composition: the map, the tile map and the gather
through it reproduce the unfused remap, grouped GEMM and gather. That the
device kernels compute each element as the references do is the card test
(test_fused_moe_xe2.py::test_fused_moe_int4_w4a8_fused_decode_bits).

Run as a script, ``--save <file>`` stores per case the output, the
rows_per_expert and the device launches by code."""

import os
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# Device launches per op call, by the csrc: remap_hidden_states clears,
# counts, places (3); cutlass_grouped_gemm_w4a8 zero-fills its atomic
# counter and runs (2); route_quant runs map and quant+gather (2).
LAUNCHES = {
    "quant": 1, "remap": 3, "gemm": 2, "act": 1, "gather": 1,
    "route_quant": 2, "tile_gemm": 1, "silu_quant": 1, "route_gather_q": 2,
}


def _import_kernels():
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

    calls = []
    gen = torch.Generator().manual_seed(23)
    shuffle_gen = torch.Generator().manual_seed(5)

    # ---------------------------------------------------------- shared ops
    def quant_ref(x):
        """int8_quant_math.h for bf16 rows: (int8 [T, H], fp32 scale [T])."""
        xf = x.float()
        absmax = xf.abs().amax(dim=1)
        s = absmax / 127.0
        s = torch.where(s < 1e-5, torch.full_like(s, 1e-5), s)
        s = s.to(torch.bfloat16).float()
        sc = s[:, None]
        t = xf / sc
        f = torch.floor(t)
        b = (f + 0.5) * sc
        half = f * 0.5
        even = torch.where(torch.floor(half) == half, f, f + 1.0)
        q = torch.where(xf > b, f + 1.0, torch.where(xf < b, f, even))
        q = q.clamp(-128.0, 127.0)
        return q.to(torch.int8), s

    def int8_quant(q, x, s):
        calls.append("quant")
        qq, ss = quant_ref(x)
        q.copy_(qq)
        s.view(-1).copy_(ss)

    def silu_mul_ref(g1):
        inter = g1.shape[1] // 2
        g = g1[:, :inter].float()
        u = g1[:, inter:].float()
        silu = (g / (1.0 + torch.exp(-g))).to(torch.bfloat16).float()
        return (silu * u).to(torch.bfloat16)

    def act_func(out, g1):
        calls.append("act")
        out.copy_(silu_mul_ref(g1))

    def local_id(g, expert_map, total, n_local):
        if g < 0 or g >= total:
            return -1
        loc = g if expert_map is None else int(expert_map[g])
        return loc if 0 <= loc < n_local else -1

    def unpack_int4(w):
        b = w.view(torch.uint8).to(torch.int64)
        lo, hi = b & 0xF, (b >> 4) & 0xF
        v = torch.stack([lo, hi], dim=-1).reshape(w.shape[0], -1)
        return torch.where(v >= 8, v - 16, v)

    def gemm_rows(a, a_scale, w, w_scale, bias):
        """One expert's rows as the W4A8 mainloop computes each element."""
        wi = unpack_int4(w)
        k = wi.shape[1]
        acc = torch.zeros((a.shape[0], wi.shape[0]), dtype=torch.float32)
        ai = a.to(torch.int64)
        for g in range(k // 128):
            sl = slice(g * 128, (g + 1) * 128)
            part = (ai[:, sl] @ wi[:, sl].t()).to(torch.float32)
            acc = acc + part * w_scale[:, g].float()[None, :]
        acc = acc * a_scale.reshape(-1).float()[:, None]
        if bias is not None:
            acc = acc + bias.float()[None, :]
        return acc.to(torch.bfloat16)

    def grouped_gemm_w4a8(a, a_scale, w, w_scale, bias, d, rows_per_expert,
                          n, k, num_experts, tile=None):
        calls.append("gemm")
        start = 0
        for e, rows in enumerate(rows_per_expert.tolist()):
            if rows:
                d[start:start + rows] = gemm_rows(
                    a[start:start + rows], a_scale.reshape(-1)[start:start +
                                                             rows], w[e],
                    w_scale[e], None if bias is None else bias[e])
            start += rows

    def remap(hidden_states, hidden_states_scales, remapped_hidden_states,
              remapped_hidden_states_scales, expert_map, rows_per_expert,
              unpermuted_row_to_permuted_row, topk_ids, total_experts_num,
              local_experts_num, expert_scale_desc=None):
        calls.append("remap")
        topk = topk_ids.shape[1]
        loc = [local_id(int(g), expert_map, total_experts_num,
                        local_experts_num)
               for g in topk_ids.reshape(-1).tolist()]
        counts = [0] * local_experts_num
        for x in loc:
            if x >= 0:
                counts[x] += 1
        rows_per_expert.copy_(torch.tensor(counts, dtype=torch.int32))
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

    def moe_gather(output, g2, topk_weights, unp, num_experts, shared=None,
                   shared_gate=None, single_round=False):
        """moe_gather.cpp; with the shared term, its SHARED instantiation:
        T(float(T(acc)) + float(T(gate * shared))) per element."""
        calls.append("gather")
        for t in range(output.shape[0]):
            acc = torch.zeros(output.shape[1], dtype=torch.float32)
            for k, row in enumerate(unp.view(output.shape[0], -1)[t].tolist()):
                if row == -1:
                    continue
                acc += g2[row].float() * topk_weights[t, k].float()
            if shared is None:
                output[t] = acc.to(output.dtype)
                continue
            gate = shared_gate.reshape(-1)[t].float()
            if single_round:
                output[t] = (acc + gate * shared[t].float()).to(output.dtype)
                continue
            routed = acc.to(output.dtype)
            gated = (gate * shared[t].float()).to(output.dtype)
            output[t] = (routed.float() + gated.float()).to(output.dtype)

    moe_gather.__dict__["_schema_shared"] = True

    # ---------------------------------------------------------- fused ops
    MAP_WG, MAX_ROUTES, MAX_LOCAL = 256, 2048, 512

    def route_quant(x, topk_ids, expert_map, a_q, a_scale, perm_pos,
                    rows_per_expert, tile_ws, total, local, tile_m):
        """w4a8_moe_fused_prep.cpp launch_map + launch_quant_gather."""
        calls.append("route_quant")
        ids = topk_ids.reshape(-1).tolist()
        R = len(ids)
        assert R <= MAX_ROUTES and local <= MAX_LOCAL
        assert tile_m in (8, 16, 32)
        shift = {8: 3, 16: 4, 32: 5}[tile_m]
        umt = (tile_ws.numel() - 2) // 3
        loc = [local_id(int(g), expert_map, total, local) for g in ids]
        # Atomic slots: any order; a shuffled one here.
        cnt = [0] * local
        slot = [0] * R
        for r in torch.randperm(R, generator=shuffle_gen).tolist():
            if loc[r] >= 0:
                slot[r] = cnt[loc[r]]
                cnt[loc[r]] += 1
        # Lane ranges of experts, exclusive scans over lanes.
        epl = (local + MAP_WG - 1) // MAP_WG
        rows_l, tiles_l = [], []
        for lane in range(MAP_WG):
            lo = min(lane * epl, local)
            hi = min(lo + epl, local)
            rows_l.append(sum(cnt[lo:hi]))
            tiles_l.append(sum((c + tile_m - 1) >> shift for c in cnt[lo:hi]))
        ws = tile_ws.view(-1)
        ws[0] = sum(tiles_l)
        ws[1] = sum(rows_l)
        offs = [0] * local
        row_off = tile_off = 0
        for lane in range(MAP_WG):
            lo = min(lane * epl, local)
            hi = min(lo + epl, local)
            for e in range(lo, hi):
                c = cnt[e]
                offs[e] = row_off
                rows_per_expert[e] = c
                nt = (c + tile_m - 1) >> shift
                for t in range(nt):
                    s = tile_off + t
                    # The kernel guards the write; the bound must hold.
                    assert s < umt, ("tile map bound", s, umt)
                    rem = c - (t << shift)
                    ws[2 + s] = e
                    ws[2 + umt + s] = row_off + (t << shift)
                    ws[2 + 2 * umt + s] = min(rem, tile_m)
                row_off += c
                tile_off += nt
        pp = perm_pos.view(-1)
        for r in range(R):
            pp[r] = offs[loc[r]] + slot[r] if loc[r] >= 0 else -1
        # Quant + gather.
        topk = topk_ids.shape[1]
        q, s = quant_ref(x)
        for r in range(R):
            if pp[r] >= 0:
                a_q[pp[r]] = q[r // topk]
                a_scale[pp[r]] = s[r // topk]

    def route_gather_q(x_q, x_scale, topk_ids, expert_map, a_q, a_scale,
                       perm_pos, rows_per_expert, tile_ws, total, local,
                       tile_m):
        """launch_map + launch_gather_q: the map of route_quant, then the
        rows quantized at the entry copied to the expert rows."""
        n = len(calls)
        route_quant(torch.zeros((x_q.shape[0], x_q.shape[1]),
                                dtype=torch.bfloat16), topk_ids, expert_map,
                    a_q, a_scale, perm_pos, rows_per_expert, tile_ws, total,
                    local, tile_m)
        del calls[n:]
        calls.append("route_gather_q")
        topk = topk_ids.shape[1]
        pp = perm_pos.view(-1)
        for r in range(pp.numel()):
            if pp[r] >= 0:
                a_q[pp[r]] = x_q[r // topk]
                a_scale[pp[r]] = x_scale.reshape(-1)[r // topk]

    def tile_gemm(a, a_scale, b, b_scale, bias, d, tile_ws, n, k, tile_m):
        """w4a8_moe_fused_gemm.cpp: grid umt x N tiles, surplus tiles
        leave, each tile one expert's rows at row0."""
        calls.append("tile_gemm")
        ws = tile_ws.view(-1)
        umt = (ws.numel() - 2) // 3
        for m_tile in range(umt):
            if m_tile >= int(ws[0]):
                continue
            e = int(ws[2 + m_tile])
            row0 = int(ws[2 + umt + m_tile])
            rows = int(ws[2 + 2 * umt + m_tile])
            assert 1 <= rows <= tile_m
            d[row0:row0 + rows] = gemm_rows(
                a[row0:row0 + rows], a_scale.reshape(-1)[row0:row0 + rows],
                b[e], b_scale[e], None if bias is None else bias[e])

    def silu_quant(g1, q_out, q_scale, tile_ws):
        """launch_silu_quant: rows below tile_ws[1]."""
        calls.append("silu_quant")
        n = int(tile_ws.view(-1)[1])
        if n:
            qq, ss = quant_ref(silu_mul_ref(g1[:n]))
            q_out[:n] = qq
            q_scale[:n] = ss

    moe_ns = types.SimpleNamespace(remap_hidden_states=remap,
                                   moe_gather=moe_gather)
    xpu_ns = types.SimpleNamespace(
        cutlass_grouped_gemm_w4a8=grouped_gemm_w4a8,
        moe_w4a8_route_quant=route_quant,
        moe_w4a8_route_gather_q=route_gather_q,
        moe_w4a8_tile_gemm=tile_gemm,
        moe_w4a8_silu_mul_quant=silu_quant)

    class COps:

        def __getattr__(self, name):
            if name == "dynamic_per_token_scaled_int8_quant":
                return int8_quant

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
        torch.ops.__dict__.pop(name, None)
    torch.ops.__class__ = Ops
    fmi, moe_w4a8 = _import_kernels()
    moe_w4a8._OP = None
    moe_w4a8._GLU = None
    moe_w4a8._FOPS = None
    moe_w4a8._GATHER_Q = None
    moe_w4a8._GATHER_SHARED = True  # the stand-in takes the shared term

    hidden, inter = 256, 128

    def make_impl(n_local, ep_rank, ep_size, expert_map, topk):
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

    def routed(m, n_global, pad, topk, skew=None):
        a = (torch.randn((m + pad, hidden), generator=gen) / 8).to(
            torch.bfloat16)
        a[m:] = float("nan")
        logits = torch.randn((m + pad, n_global), generator=gen)
        if skew is not None:
            logits[:, skew] += 50.0  # every token's top choice: one expert
        w, ids = torch.topk(logits.softmax(-1), topk, dim=-1)
        ids = ids.to(torch.int32)
        ids[::3, 1] = -1
        ids[m:] = -1
        return a, w, ids

    cases = []
    for topk, rows in ((4, (1, 2, 4, 7, 8, 16, 33)), (10, (1, 5, 8, 16))):
        for m in rows:
            impl = make_impl(8 if topk == 4 else 16, 0, 1, None, topk)
            a, w, ids = routed(m, impl.num_experts, 3, topk)
            cases.append((f"k{topk}-ep1-m{m}", impl, a, w, ids, {}))
            n_global, n_local, ep_size = (16, 4, 4) if topk == 4 else (64,
                                                                       16, 4)
            for ep_rank in (0, 3):
                emap = torch.full((n_global, ), -1, dtype=torch.int32)
                emap[ep_rank * n_local:(ep_rank + 1) * n_local] = torch.arange(
                    n_local, dtype=torch.int32)
                impl = make_impl(n_local, ep_rank, ep_size, emap, topk)
                a, w, ids = routed(m, n_global, 3, topk)
                tag = f"k{topk}-ep4r{ep_rank}-m{m}"
                sentinel = n_global - 1 if ep_rank == 0 else 0
                cases.append((f"{tag}-raw", impl, a, w, ids, {}))
                cases.append((f"{tag}-sentinel", impl, a, w,
                              torch.where(ids < 0, sentinel, ids), {
                                  "ids_valid": True,
                                  "rows_per_expert": "stale"
                              }))
                # A sentinel this rank owns: padding rows (NaN) are computed
                # as a real expert on both paths (AX-E9 2.3).
                owned = ep_rank * n_local
                cases.append((f"{tag}-owned", impl, a, w,
                              torch.where(ids < 0, owned, ids), {
                                  "ids_valid": True,
                                  "rows_per_expert": "stale"
                              }))
                oob = ids.clone()
                oob[::4, 0] = n_global + 5
                cases.append((f"{tag}-oob", impl, a, w, oob, {
                    "ids_valid": True,
                    "rows_per_expert": "zero"
                }))
                perm = torch.randperm(n_global, generator=gen)
                pmap = torch.full((n_global, ), -1, dtype=torch.int32)
                pmap[perm[:n_local]] = torch.arange(n_local, dtype=torch.int32)
                pimpl = make_impl(n_local, ep_rank, ep_size, pmap, topk)
                cases.append((f"{tag}-moved", pimpl, a, w,
                              torch.where(ids < 0, int(perm[n_local]), ids), {
                                  "ids_valid": True,
                                  "rows_per_expert": "stale"
                              }))
                # Skew: one owned expert takes a row of every token.
                sa, sw, sids = routed(m, n_global, 3, topk,
                                      skew=ep_rank * n_local + 1)
                cases.append((f"{tag}-skew", impl, sa, sw, sids, {}))

    results = {}
    for name, impl, a, w, ids, kw in cases:
        kw = dict(kw)
        rpe = None
        fill = 0
        if kw.get("rows_per_expert") is not None:
            fill = 7 if kw["rows_per_expert"] == "stale" else 0
            rpe = torch.full((impl.num_experts, ), fill, dtype=torch.int32)
            kw["rows_per_expert"] = rpe
        n_moe = a.shape[0] * impl.n_experts_per_token
        # The shared expert's output and gate column; NaN on padding rows,
        # as the shared expert computes them from NaN rows.
        m_real = a.shape[0] - 3
        sh = (torch.randn(a.shape, generator=gen) / 4).to(torch.bfloat16)
        sh[m_real:] = float("nan")
        gate = torch.sigmoid(torch.randn((a.shape[0], 1), generator=gen)).to(
            torch.bfloat16)
        xq, xs = quant_ref(a)
        for variant, extra in (("", {}), ("|xq", {"x_q": (xq, xs.view(-1, 1))}),
                               ("|xq+sh", {"x_q": (xq, xs.view(-1, 1)),
                                           "shared": (sh, gate)}),
                               ("|xq+sh1", {"x_q": (xq, xs.view(-1, 1)),
                                            "shared": (sh, gate, True)})):
            if rpe is not None:
                rpe.fill_(fill)
            out = torch.full_like(a, 3.0)
            calls.clear()
            folded = fmi.XpuFusedMoe.apply(impl, output=out, hidden_states=a,
                                           topk_weights=w, topk_ids=ids,
                                           w4a8=True, **kw, **extra)
            results[name + variant] = {
                "out": out.clone(),
                "rows_per_expert": None if rpe is None else rpe.clone(),
                "calls": list(calls),
                "launches": sum(LAUNCHES[c] for c in calls),
                "tile_m": moe_w4a8.unfused_tile(n_moe, impl.num_experts),
                "folded": folded,
                "shared": (sh, gate),
            }
    return results


def _main(argv):
    import torch

    from vllm_xpu_kernels import moe_w4a8  # noqa: F401  (env read at import)
    res = _run_cases()
    if "--save" in argv:
        torch.save(res, argv[argv.index("--save") + 1])
    mode = os.environ.get("VLLM_XPU_W4A8_FUSED_MOE", "0")
    print("W4A8-FUSED-RUN-OK", mode, len(res))


def _run(mode: str, save: str):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               VLLM_XPU_W4A8_STOCK_PREP="0", VLLM_XPU_W4A8_FUSED_MOE=mode)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    run = subprocess.run(
        [sys.executable, "-P", os.path.abspath(__file__), "--save", save],
        env=env, capture_output=True, text=True, timeout=3600)
    assert run.returncode == 0 and "W4A8-FUSED-RUN-OK" in run.stdout, (
        run.stdout[-2000:] + run.stderr[-4000:])


def _bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


torch = None


def test_w4a8_fused_decode_bit_identical(tmp_path):
    global torch
    torch = pytest.importorskip("torch")
    files = {m: str(tmp_path / f"m{m}.pt") for m in ("0", "1", "2")}
    for m, f in files.items():
        _run(m, f)
    base = torch.load(files["0"])
    # 13 case kinds (EP off; ranks 0 and 3 x raw, sentinel, owned sentinel,
    # out of range, moved, skew) x 11 (top-4 at 7 row counts, top-10 at 4),
    # each plain, with entry rows, and with entry rows + the shared term.
    assert len(base) == 4 * 143
    plain = [k for k in base if "|" not in k]
    t2_stats = []
    assert len(plain) == 143
    for m in ("0", "1", "2"):
        got = torch.load(files[m])
        for key in plain:
            ref = base[key]["out"]
            xq = got[key + "|xq"]
            # Entry rows: same output and counts as the layer's own quant.
            assert torch.equal(_bits(xq["out"]), _bits(ref)), (m, key)
            assert xq["folded"] is False, (m, key)
            # No quant of the token rows (the act quant before gemm2 stays).
            assert xq["calls"][0] != "quant", (m, key, xq["calls"])
            if m == "0":
                assert xq["launches"] == 10, (m, key, xq)
            else:
                assert xq["calls"][0] == "route_gather_q", (m, key)
                assert xq["launches"] == (6 if m == "1" else 7), (m, key)
            sh = got[key + "|xq+sh"]
            out_s, gate = sh["shared"]
            # The engine's unfused sequence: sigmoid(g) * shared (bf16), then
            # shared + routed (bf16), on the plain output.
            want = (gate * out_s) + ref
            assert sh["folded"] is True, (m, key)
            assert torch.equal(_bits(sh["out"]), _bits(want)), (m, key)
            assert sh["launches"] == xq["launches"], (m, key)
            # Tier 2 (single fp32 rounding): bounded against tier 1.
            t2 = got[key + "|xq+sh1"]
            assert t2["folded"] is True, (m, key)
            a1, a2 = want.float(), t2["out"].float()
            fin = torch.isfinite(a1)
            assert torch.equal(fin, torch.isfinite(a2)), (m, key)
            d = (a2[fin] - a1[fin]).abs()
            rel = d / a1[fin].abs().clamp_min(1e-3)
            t2_stats.append((rel.max().item(), rel.mean().item(),
                             (d > 0).float().mean().item()))
            # Tier 1 rounds the routed sum, the gated product and the sum to
            # bf16 (half an ulp each); tier 2 rounds once. So the two differ
            # by at most an ulp of each operand plus one of the result.
            eps = torch.finfo(torch.bfloat16).eps
            bound = eps * (ref.float()[fin].abs()
                           + (gate * out_s).float()[fin].abs()
                           + a1[fin].abs()) + torch.finfo(torch.bfloat16).tiny
            assert bool((d <= bound).all()), (m, key, d.max().item())
    mx = max(r[0] for r in t2_stats)
    mean = sum(r[1] for r in t2_stats) / len(t2_stats)
    frac = sum(r[2] for r in t2_stats) / len(t2_stats)
    print(f"TIER2-SINGLE-ROUND cases={len(t2_stats)} max_rel={mx:.3e} "
          f"mean_rel={mean:.3e} frac_elems_differ={frac:.4f}")
    for m in ("1", "2"):
        got = torch.load(files[m])
        assert got.keys() == base.keys()
        for key in plain:
            a, b = got[key], base[key]
            # Bit for bit, NaN rows (owned sentinel) included.
            assert torch.equal(_bits(a["out"]), _bits(b["out"])), (m, key)
            ra, rb = a["rows_per_expert"], b["rows_per_expert"]
            assert (ra is None) == (rb is None), (m, key)
            if ra is not None:
                assert torch.equal(ra, rb), (m, key)
            # The fused path ran, with its launch count by code.
            assert b["calls"][0] == "quant" and b["launches"] == 11, key
            assert a["calls"][0] == "route_quant", (m, key)
            assert a["launches"] == (6 if m == "1" else 7), (m, key, a)
        # Padding rows with unowned / negative ids are zeros (plain runs;
        # with the shared term they carry the shared expert's NaN rows).
        for key in plain:
            if key.endswith(("-raw", "-sentinel", "-oob", "-moved",
                             "-skew")) or (
                    "-ep1-" in key):
                assert not got[key]["out"][-3:].any(), (m, key)


if __name__ == "__main__":
    import torch  # noqa: F811
    _main(sys.argv)
