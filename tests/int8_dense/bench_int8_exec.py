#!/usr/bin/env python3
"""GPU test + microbench of the int8 dense GEMMs and the W4A8 MoE grouped GEMMs, one card.

Needs a vllm_xpu_kernels build whose _xpu_C registers the int8_dense_* and
cutlass_grouped_gemm_w4a8* ops.

Modes (comma list, default all):
  dense    exactness of int8_dense_w8a16_gemm and int8_dense_w8a8_fused_gemm per rank shape, M 1..64:
           vs an fp64 reference over the same int8 weights (W8A16: bf16 x; W8A8: x quantized per token
           exactly as scaled_int8_quant does) and vs BF16 F.linear on the unquantized weights.
  dbench   us per call per shape and M, eager and XPU-graph, rotating R weight copies (defeats L2):
           bf16 F.linear | oneDNN int8_scaled_mm + quant (if present) | w8a16 | w8a8_fused | w8a8 GEMM only.
  moe      W4A8 MoE (512 experts / EP4 = 128 local, top-10, hidden 2560, inter 640):
           gemm-level check of the fused gemm1 (silu*up -> int8 + per-128-col scales) and of gemm2 with
           those scales vs fp64; then the full MoE: fused-act vs unfused W4A8 vs stock W4A16 vs fp32.
  mbench   ms of the full MoE (graph replay and eager) for stock W4A16, W4A8 unfused, W4A8 fused-act;
           plus GEMM-only gemm1 / gemm2 ms and TOPS at 4096 tokens (prefill).
  hcdet    determinism + batch invariance of int8_dense_w8a8_fused_gemm (the int8 HC down path): for each
           shape, a 256-row pool; every row's output must be bit-identical across 2 runs and across batch
           compositions (sub-batches of 1..256 rows in shuffled order, so M crosses the 8/16/32/64 tiles
           and the split-K choice).

Prints one JSON line per case (DENSE / DBENCH / GLU / MOE / MBENCH / HCDET) and writes --out.
"""
import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

DEV = "xpu"
DENSE_SHAPES = {  # name: (K, N) per rank, TP4
    "shared_gate_up": (2560, 320),
    "shared_down": (160, 2560),
    "attn_o": (1536, 2560),
    "gdn_in": (2560, 4096),
    "hc_down": (10240, 336),
    "in_proj_ba": (2560, 24),
}
H, INTER, E_GLOBAL, TOPK, G = 2560, 640, 512, 10, 128
AGC = 128  # activation-scale group of the fused gemm1 output


def sync():
    torch.xpu.synchronize()


def ebench(fn, iters=50, reps=5, warm=3):
    """us per call, eager (host launch cost included)."""
    for _ in range(warm):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        ts.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(ts)


def gbench(fn, calls=1, iters=100, reps=5):
    """us per call under XPU graph replay; fn() makes `calls` calls (rotating weights)."""
    s = torch.xpu.Stream()
    with torch.xpu.stream(s):
        for _ in range(3):
            fn()
    sync()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g, stream=s):
        fn()
    for _ in range(3):
        g.replay()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        sync()
        ts.append((time.perf_counter() - t0) / iters / calls * 1e6)
    return statistics.median(ts)


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def q_weight(w):
    """W1's quantize_weight_int8: symmetric per row, RTN, fp32 [N, 1]."""
    wf = w.float()
    s = (wf.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
    return torch.round(wf / s).clamp(-127, 127).to(torch.int8), s


def q_act(x):
    """scaled_int8_quant symmetric dynamic per token: scale = amax / 127, q = rint(x / scale)."""
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True)
    inv = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
    return torch.round(xf * inv).clamp(-127, 127).to(torch.int8), amax / 127.0


# --------------------------------------------------------------------------- dense
def run_dense(ms):
    res = []
    g = torch.Generator(device="cpu").manual_seed(7)
    for name, (K, N) in DENSE_SHAPES.items():
        w = (torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16)
        wq, ws = q_weight(w)
        wqd, wsd, wd = wq.to(DEV), ws.to(DEV), w.to(DEV)
        bias = (torch.randn(N, generator=g) * 0.1).to(torch.bfloat16)
        for M in ms:
            for kind in ("gauss", "outlier"):
                x = torch.randn(M, K, generator=g).to(torch.bfloat16)
                if kind == "outlier":
                    x[:, : min(16, K)] *= 20
                xd = x.to(DEV)
                wdq = wq.double() * ws.double()
                ref16 = x.double() @ wdq.t() + bias.double()  # W8A16 target
                xq, xs = q_act(x)
                ref88 = (xq.double() @ wq.double().t()) * xs.double() * ws.double().t() + bias.double()
                refbf = x.double() @ w.double().t() + bias.double()  # unquantized
                r = {"shape": name, "K": K, "N": N, "M": M, "input": kind}
                o16 = torch.ops._xpu_C.int8_dense_w8a16_gemm(xd, wqd, wsd, bias.to(DEV))
                o88 = torch.ops._xpu_C.int8_dense_w8a8_fused_gemm(xd, wqd, wsd, bias.to(DEV)) if (M <= 64 and N % 16 == 0 and N >= 32) else None
                sync()
                r["w8a16_rel_vs_ref"] = rel(o16.cpu(), ref16)
                r["w8a16_rel_vs_bf16w"] = rel(o16.cpu(), refbf)
                r["w8a16_nan"] = bool(torch.isnan(o16).any())
                if o88 is not None:
                    r["w8a8f_rel_vs_ref"] = rel(o88.cpu(), ref88)
                    r["w8a8f_rel_vs_bf16w"] = rel(o88.cpu(), refbf)
                    r["w8a8f_nan"] = bool(torch.isnan(o88).any())
                # bf16 output rounding is ~2e-3 relative; integer math is exact.
                r["ok"] = (not r["w8a16_nan"]) and r["w8a16_rel_vs_ref"] < 5e-3 and (
                    o88 is None or ((not r["w8a8f_nan"]) and r["w8a8f_rel_vs_ref"] < 5e-3))
                print("DENSE", json.dumps(r), flush=True)
                res.append(r)
    return res


def run_dbench(ms, rot):
    res = []
    for name, (K, N) in DENSE_SHAPES.items():
        ws_ = [(torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16) for _ in range(rot)]
        qs = [q_weight(w) for w in ws_]
        for M in ms:
            x = torch.randn(M, K, device=DEV).to(torch.bfloat16)
            r = {"shape": name, "K": K, "N": N, "M": M, "rot": rot}
            arms = {
                "bf16": lambda i: F.linear(x, ws_[i]),
                "w8a16": lambda i: torch.ops._xpu_C.int8_dense_w8a16_gemm(x, qs[i][0], qs[i][1], None),
            }
            if M <= 64 and N % 16 == 0 and N >= 32:
                arms["w8a8f"] = lambda i: torch.ops._xpu_C.int8_dense_w8a8_fused_gemm(x, qs[i][0], qs[i][1], None)
                xq8, xs8 = q_act(x)
                xq8, xs8 = xq8.contiguous(), xs8.reshape(-1).contiguous()
                arms["w8a8_gemm_only"] = lambda i: torch.ops._xpu_C.int8_dense_w8a8_gemm(xq8, xs8, qs[i][0], qs[i][1], None)
            if hasattr(torch.ops._xpu_C, "int8_scaled_mm") and hasattr(torch.ops._C, "dynamic_per_token_scaled_int8_quant"):
                def onednn(i):
                    xq = torch.empty((M, K), dtype=torch.int8, device=DEV)
                    xs = torch.empty((M, 1), dtype=torch.float32, device=DEV)
                    torch.ops._C.dynamic_per_token_scaled_int8_quant(xq, x, xs)
                    return torch.ops._xpu_C.int8_scaled_mm(xq, xs.reshape(-1), qs[i][0].t(), qs[i][1].reshape(-1),
                                                           x.dtype, None)
                arms["onednn_i8"] = onednn
            for a, f in arms.items():
                try:
                    r[f"{a}_graph_us"] = round(gbench(lambda: [f(i) for i in range(rot)], calls=rot), 2)
                    r[f"{a}_eager_us"] = round(ebench(lambda: f(0)), 2)
                except Exception as e:  # noqa
                    r[f"{a}_err"] = repr(e)[:200]
            if "bf16_graph_us" in r:
                for a in arms:
                    if a != "bf16" and f"{a}_graph_us" in r:
                        r[f"{a}_over_bf16"] = round(r[f"{a}_graph_us"] / r["bf16_graph_us"], 3)
            print("DBENCH", json.dumps(r), flush=True)
            res.append(r)
    return res


# --------------------------------------------------------------------------- moe
def unpack_s4(b):
    u = b.to(torch.int16) & 0xFF
    lo, hi = u & 0xF, (u >> 4) & 0xF
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack((lo, hi), dim=-1).flatten(-2).to(torch.int8)


def dequant(b, s):
    w = unpack_s4(b).float()
    return (w.view(*w.shape[:-1], -1, G) * s.float().unsqueeze(-1)).flatten(-2)


def run_glu_gemm(seed):
    """gemm-level: fused gemm1 vs fp64, and gemm2 consuming its AG scales vs fp64."""
    glu = torch.ops._xpu_C.cutlass_grouped_gemm_w4a8_silu_quant
    op = torch.ops._xpu_C.cutlass_grouped_gemm_w4a8
    res = []
    cases = [[1, 2, 3, 4, 0, 6], [5, 7, 9, 0, 11, 6], [33, 0, 64, 127, 17, 130], [300, 0, 129, 513, 7, 257]]
    for tk in (64,):
        for pf in (4,):
            for i, rows in enumerate(cases):
                g = torch.Generator(device="cpu").manual_seed(seed + i)
                E, M, K = len(rows), sum(rows), H
                a = torch.randint(-127, 128, (M, K), dtype=torch.int8, generator=g)
                a_s = torch.rand(M, 1, generator=g) * 0.02 + 0.001
                wb = torch.randint(-128, 128, (E, 2 * INTER, K // 2), dtype=torch.int8, generator=g)
                ws = (torch.rand(E, 2 * INTER, K // G, generator=g) * 0.004 + 0.001).to(torch.bfloat16)
                w2 = torch.randint(-128, 128, (E, H, INTER // 2), dtype=torch.int8, generator=g)
                s2 = (torch.rand(E, H, INTER // G, generator=g) * 0.004 + 0.001).to(torch.bfloat16)
                rpe = torch.tensor(rows, dtype=torch.int32)
                # fp64 reference of silu(gate) * up
                act = torch.empty(M, INTER, dtype=torch.float64)
                r0 = 0
                for e, m in enumerate(rows):
                    if m:
                        h = (a[r0:r0 + m].double() * a_s[r0:r0 + m].double()) @ dequant(wb[e], ws[e]).double().t()
                        act[r0:r0 + m] = F.silu(h[:, :INTER]) * h[:, INTER:]
                    r0 += m
                q = torch.empty(M, INTER, dtype=torch.int8, device=DEV)
                qs = torch.empty(M, INTER // AGC, dtype=torch.float32, device=DEV)
                glu(a.to(DEV), a_s.to(DEV), wb.to(DEV), ws.to(DEV), None, q, qs, rpe.to(DEV), INTER, K, E)
                sync()
                deq = (q.cpu().double().view(M, INTER // AGC, AGC) * qs.cpu().double().unsqueeze(-1)).view(M, INTER)
                # gemm2 with the AG scales vs fp64 over the same int8 + scales
                d2 = torch.empty(M, H, dtype=torch.bfloat16, device=DEV)
                op(q, qs, w2.to(DEV), s2.to(DEV), None, d2, rpe.to(DEV), H, INTER, E)
                sync()
                ref2 = torch.empty(M, H, dtype=torch.float64)
                r0 = 0
                for e, m in enumerate(rows):
                    if m:
                        ref2[r0:r0 + m] = deq[r0:r0 + m] @ dequant(w2[e], s2[e]).double().t()
                    r0 += m
                r = {"rows": rows, "tile_k": tk, "prefetch": pf,
                     "glu_rel_vs_f64": rel(deq, act),  # includes the int8 quant of the act (~0.5 %)
                     "glu_nan": bool(torch.isnan(qs).any()),
                     "g2_ag_rel_vs_f64": rel(d2.cpu(), ref2),
                     "g2_nan": bool(torch.isnan(d2).any())}
                r["ok"] = (not r["glu_nan"]) and (not r["g2_nan"]) and r["glu_rel_vs_f64"] < 1.5e-2 \
                    and r["g2_ag_rel_vs_f64"] < 5e-3
                print("GLU", json.dumps(r), flush=True)
                res.append(r)
    return res


def make_moe(seed, n_local, ep):
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe
    g = torch.Generator(device="cpu").manual_seed(seed)
    w13 = torch.randint(0, 256, (n_local, 2 * INTER, H // 2), dtype=torch.uint8, generator=g)
    w2 = torch.randint(0, 256, (n_local, H, INTER // 2), dtype=torch.uint8, generator=g)
    s13 = (torch.rand(n_local, 2 * INTER, H // G, generator=g) * 0.004 + 0.001).to(torch.bfloat16)
    s2 = (torch.rand(n_local, H, INTER // G, generator=g) * 0.004 + 0.001).to(torch.bfloat16)
    return XpuFusedMoe(w13.to(DEV), s13.to(DEV), None, w2.to(DEV), s2.to(DEV), None,
                       n_experts_per_token=TOPK, activation="silu", num_experts=n_local,
                       ep_rank=0, ep_size=ep)


def routing(T, seed, skew=0.8):
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(E_GLOBAL, generator=g)
    p = torch.empty(E_GLOBAL)
    p[perm] = 1.0 / torch.arange(1, E_GLOBAL + 1).float() ** skew
    ids = torch.multinomial(p.expand(T, E_GLOBAL), TOPK, replacement=False, generator=g)
    w = torch.softmax(torch.randn(T, TOPK, generator=g), dim=-1)
    return ids.to(torch.int32).to(DEV), w.to(DEV)


def moe_ref(fm, x, tw, ids):
    out = torch.zeros(x.shape, dtype=torch.float32, device=DEV)
    loc = fm.expert_map[ids.long()]
    xf = x.float()
    for e in range(fm.num_experts):
        tok, slot = (loc == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h1 = xf[tok] @ dequant(fm.w13[e], fm.gemm1_wei_scales[e]).t()
        act = F.silu(h1[:, :INTER]) * h1[:, INTER:]
        out.index_add_(0, tok, (act @ dequant(fm.w2[e], fm.gemm2_wei_scales[e]).t()) * tw[tok, slot].unsqueeze(1))
    return out


def w4a8_apply(w4, fm, o, x, tw, ids, fused):
    w4.apply(fm, o, x, tw, ids, fused_act=fused)


def run_moe(fm, w4, tokens, seed):
    res = []
    for T in tokens:
        ids, tw = routing(T, seed + T)
        g = torch.Generator(device="cpu").manual_seed(seed + 7 * T)
        x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
        for kind in ("gauss", "outlier"):
            xx = x.clone()
            if kind == "outlier":
                xx[:, :16] *= 20
            o_s, o_u, o_f = (torch.zeros(T, H, dtype=torch.bfloat16, device=DEV) for _ in range(3))
            fm.apply(o_s, xx, tw, ids)
            w4a8_apply(w4, fm, o_u, xx, tw, ids, False)
            w4a8_apply(w4, fm, o_f, xx, tw, ids, True)
            ref = moe_ref(fm, xx, tw, ids)
            sync()
            r = {"T": T, "input": kind, "w4a16_rel": rel(o_s, ref), "w4a8_rel": rel(o_u, ref),
                 "w4a8_fused_rel": rel(o_f, ref), "fused_vs_unfused_rel": rel(o_f, o_u),
                 "nan": bool(torch.isnan(o_f).any())}
            print("MOE", json.dumps(r), flush=True)
            res.append(r)
    return res


def run_mbench(fm, w4, tokens, seed, grid):
    res = []
    for T in tokens:
        ids, tw = routing(T, seed + T)
        x = (torch.randn(T, H, device=DEV) * 0.5).to(torch.bfloat16)
        o = torch.empty(T, H, dtype=torch.bfloat16, device=DEV)
        r = {"T": T}
        bench = gbench if T <= 1024 else (lambda f, **k: ebench(f, iters=10))
        r["w4a16_us"] = round(bench(lambda: fm.apply(o, x, tw, ids)), 1)
        for pf, st, tk in grid:
            tag = f"pf{pf}_st{st}_tk{tk}"
            for fused in (False, True):
                key = f"w4a8{'_fused' if fused else ''}_{tag}_us"
                try:
                    r[key] = round(bench(lambda: w4a8_apply(w4, fm, o, x, tw, ids, fused)), 1)
                except Exception as e:  # noqa
                    r[key] = "ERR " + repr(e)[:160]
        if T <= 64:
            r["w4a16_eager_us"] = round(ebench(lambda: fm.apply(o, x, tw, ids)), 1)
            r["w4a8_fused_eager_us"] = round(ebench(lambda: w4a8_apply(w4, fm, o, x, tw, ids, True)), 1)
        print("MBENCH", json.dumps(r), flush=True)
        res.append(r)
    return res


def run_gemm_prefill(fm, w4, T, seed, grid):
    """GEMM-only gemm1 / gemm2 at a T-token prefill (rows routed as in the MoE): ms and TOPS."""
    ids, _ = routing(T, seed + T)
    rpe = torch.zeros(fm.num_experts, dtype=torch.int32, device=DEV)
    loc = fm.expert_map[ids.long()]
    sel = loc[loc >= 0].long()
    rpe.index_add_(0, sel, torch.ones_like(sel, dtype=torch.int32))
    rows = int(rpe.sum().item())
    op = torch.ops._xpu_C.cutlass_grouped_gemm_w4a8
    glu = torch.ops._xpu_C.cutlass_grouped_gemm_w4a8_silu_quant
    a8, as8 = w4.quant_rows((torch.randn(rows, H, device=DEV) * 0.5).to(torch.bfloat16))
    i8, is8 = w4.quant_rows((torch.randn(rows, INTER, device=DEV) * 0.5).to(torch.bfloat16))
    q = torch.empty(rows, INTER, dtype=torch.int8, device=DEV)
    qs = torch.empty(rows, INTER // AGC, dtype=torch.float32, device=DEV)
    d1 = torch.empty(rows, 2 * INTER, dtype=torch.bfloat16, device=DEV)
    d2 = torch.empty(rows, H, dtype=torch.bfloat16, device=DEV)
    E = fm.num_experts
    f1, f2 = 2.0 * rows * 2 * INTER * H, 2.0 * rows * H * INTER
    r = {"T": T, "rows": rows}
    a16 = (torch.randn(rows, H, device=DEV) * 0.5).to(torch.bfloat16)
    i16 = (torch.randn(rows, INTER, device=DEV) * 0.5).to(torch.bfloat16)
    s1 = ebench(lambda: torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=a16, ptr_A_scale=None, ptr_B=fm.w13, ptr_B_scale=fm.gemm1_wei_scales, ptr_bias=None,
        ptr_D=d1, rows_per_expert=rpe, N=2 * INTER, K=H, num_experts=E), iters=10)
    s2 = ebench(lambda: torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=i16, ptr_A_scale=None, ptr_B=fm.w2, ptr_B_scale=fm.gemm2_wei_scales, ptr_bias=None,
        ptr_D=d2, rows_per_expert=rpe, N=H, K=INTER, num_experts=E), iters=10)
    r["g1_w4a16_tflops"], r["g2_w4a16_tflops"] = round(f1 / s1 / 1e6, 1), round(f2 / s2 / 1e6, 1)
    seen = set()
    for pf, _st, tk in grid:
        if (pf, tk) in seen:
            continue
        seen.add((pf, tk))
        tag = f"pf{pf}_tk{tk}"
        t1 = ebench(lambda: op(a8, as8, fm.w13, fm.gemm1_wei_scales, None, d1, rpe, 2 * INTER, H, E), iters=10)
        tg = ebench(lambda: glu(a8, as8, fm.w13, fm.gemm1_wei_scales, None, q, qs, rpe, INTER, H, E), iters=10)
        t2 = ebench(lambda: op(i8, is8, fm.w2, fm.gemm2_wei_scales, None, d2, rpe, H, INTER, E), iters=10)
        t2g = ebench(lambda: op(q, qs, fm.w2, fm.gemm2_wei_scales, None, d2, rpe, H, INTER, E), iters=10)
        r[f"g1_{tag}_tops"] = round(f1 / t1 / 1e6, 1)
        r[f"g1glu_{tag}_tops"] = round(f1 / tg / 1e6, 1)
        r[f"g2_{tag}_tops"] = round(f2 / t2 / 1e6, 1)
        r[f"g2ag_{tag}_tops"] = round(f2 / t2g / 1e6, 1)
    print("PREFILL", json.dumps(r), flush=True)
    return r


# --------------------------------------------------------------------------- hcdet
def run_hcdet(seed):
    """Bitwise: same input -> same output for every run, batch composition, split and tile."""
    res = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for name in ("hc_down", "gdn_in", "attn_o", "shared_gate_up"):
        K, N = DENSE_SHAPES[name]
        w = (torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16)
        wq, ws = (t.to(DEV) for t in q_weight(w))
        pool = torch.randn(256, K, generator=g).to(torch.bfloat16)
        pool[:, :16] *= 20  # outlier channels, as in HC streams
        pool = pool.to(DEV)
        op = lambda x: torch.ops._xpu_C.int8_dense_w8a8_fused_gemm(x.contiguous(), wq, ws, None)  # noqa: E731
        ref = op(pool)
        ref2 = op(pool)
        sync()
        r = {"shape": name, "K": K, "N": N, "run_to_run_equal": bool(torch.equal(ref, ref2))}
        bad = []
        perm = torch.randperm(256, generator=g)
        for m in (1, 2, 3, 7, 8, 9, 13, 16, 17, 31, 33, 63, 64, 65, 96, 128, 200, 256):
            idx = perm[:m].to(DEV)
            out = op(pool.index_select(0, idx))
            sync()
            if not torch.equal(out, ref.index_select(0, idx)):
                bad.append(f"M{m}")
        r["mismatches"] = bad
        r["ok"] = r["run_to_run_equal"] and not bad
        print("HCDET", json.dumps(r), flush=True)
        res.append(r)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hcdet,dense,dbench,moe,mbench")
    ap.add_argument("--dense-m", default="1,2,4,8,16,32,64")
    ap.add_argument("--rot", type=int, default=8, help="weight copies rotated per graph (defeats L2)")
    ap.add_argument("--experts", type=int, default=128, help="local experts (D8: 512 / EP4)")
    ap.add_argument("--ep", type=int, default=4)
    ap.add_argument("--moe-tokens", default="1,8,16,64,256,1024")
    ap.add_argument("--bench-tokens", default="1,2,4,8,16,32,64,96,256,1024")
    ap.add_argument("--prefill-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="result-int8-exec.json")
    args = ap.parse_args()

    import vllm_xpu_kernels  # noqa
    import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C)
    from vllm_xpu_kernels import moe_w4a8 as w4
    print("DEVICE", torch.xpu.get_device_name(0), "kernels", getattr(vllm_xpu_kernels, "__version__", "?"),
          flush=True)
    for o in ("int8_dense_w8a16_gemm", "int8_dense_w8a8_fused_gemm", "int8_dense_w8a8_gemm", "cutlass_grouped_gemm_w4a8_silu_quant"):
        assert hasattr(torch.ops._xpu_C, o), f"_xpu_C has no {o}: wrong wheel?"
    grid = [(4, 0, 64)]  # the fixed tuning: prefetch 4, 64-wide small tiles, tile K 64
    modes = args.mode.split(",")
    ms = [int(m) for m in args.dense_m.split(",")]
    out = {"args": vars(args)}
    if "hcdet" in modes:
        out["hcdet"] = run_hcdet(args.seed)
        print("HCDET_ALL_OK", all(r["ok"] for r in out["hcdet"]), flush=True)
    if "dense" in modes:
        out["dense"] = run_dense(ms)
        print("DENSE_ALL_OK", all(r["ok"] for r in out["dense"]), flush=True)
    if "dbench" in modes:
        out["dbench"] = run_dbench(ms, args.rot)
    if "moe" in modes or "mbench" in modes:
        if "moe" in modes:
            out["glu"] = run_glu_gemm(args.seed)
            print("GLU_ALL_OK", all(r["ok"] for r in out["glu"]), flush=True)
        fm = make_moe(args.seed, args.experts, args.ep)
        sync()
        if "moe" in modes:
            out["moe"] = run_moe(fm, w4, [int(t) for t in args.moe_tokens.split(",")], args.seed)
        if "mbench" in modes:
            out["mbench"] = run_mbench(fm, w4, [int(t) for t in args.bench_tokens.split(",")], args.seed, grid)
            out["prefill"] = run_gemm_prefill(fm, w4, args.prefill_tokens, args.seed, grid)
    out["peak_gib"] = torch.xpu.max_memory_allocated() / 2**30
    print("PEAK_GIB", round(out["peak_gib"], 3), flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
