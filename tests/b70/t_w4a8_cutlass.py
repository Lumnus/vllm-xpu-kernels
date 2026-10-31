#!/usr/bin/env python3
"""GPU test + microbench: W4A8 cutlass grouped GEMM (s8 x s4 DPAS) vs stock W4A16.

Loads the standalone _b70_w4a8_C module from --lib-dir (torch.ops.load_library, the
installed vllm_xpu_kernels wheel is untouched) and b70_moe_w4a8.py from this directory.

  gemm   GEMM-level exactness vs a float64 reference over the same int4 bytes and int8 rows
         (every M policy, both tile K), plus agreement with the stock W4A16 op.
  moe    MoE-level (per-rank shape of a 512-expert top-10 MoE: hidden 2560, inter 640, 512 experts / EP2 = 256 local,
         top-10): W4A8 vs stock XpuFusedMoe.apply vs an fp32 reference.
  bench  ms of the full MoE (stock apply vs W4A8) at decode 1..256 and prefill 512..4096,
         and GEMM-only ms / TOPS of gemm1 and gemm2 (stock cutlass bf16 vs W4A8) at the
         routed rows of each token count.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:  # the staged copy next to this file (installed wheel untouched)
    import b70_moe_w4a8 as w4  # noqa: E402
except ImportError:  # in-tree: tests/b70/ of vllm-xpu-kernels
    from vllm_xpu_kernels import b70_moe_w4a8 as w4  # noqa: E402
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe  # noqa: E402

DEV = "xpu"
H, INTER, E_GLOBAL, EP, TOPK, G = 2560, 640, 512, 2, 10, 128


def sync():
    torch.xpu.synchronize()


def bench(fn, iters=20, reps=5, warm=3):
    for _ in range(warm):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        ts.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(ts)


def unpack_s4(b):
    """[.., N, K/2] int8 (implement_zp bytes) -> [.., N, K] int8 values in [-8, 7]."""
    u = b.to(torch.int16) & 0xFF
    lo = u & 0xF
    hi = (u >> 4) & 0xF
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack((lo, hi), dim=-1).flatten(-2).to(torch.int8)


def dequant(b, s):
    """W[n, k] = s4 * scale[n, k // 128] (fp32)."""
    w = unpack_s4(b).float()
    return (w.view(*w.shape[:-1], -1, G) * s.float().unsqueeze(-1)).flatten(-2)


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def cos_rows(a, b):
    c = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
    return c.mean().item(), c.min().item()


# --------------------------------------------------------------------------- gemm
def gemm_case(op, rows, N, K, tile_k, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    E = len(rows)
    M = sum(rows)
    a = torch.randint(-127, 128, (M, K), dtype=torch.int8, generator=g)
    a_s = torch.rand(M, 1, generator=g) * 0.02 + 0.001
    wb = torch.randint(-128, 128, (E, N, K // 2), dtype=torch.int8, generator=g)
    ws = (torch.rand(E, N, K // G, generator=g) * 0.02 + 0.002).to(torch.bfloat16)
    rpe = torch.tensor(rows, dtype=torch.int32)
    # float64 reference: as[m] * sum_g ws[n,g] * sum_{k in g} a[m,k] * w[n,k]
    ref = torch.empty(M, N, dtype=torch.float64)
    wv = unpack_s4(wb).double()
    r0 = 0
    for e, m in enumerate(rows):
        if m:
            ae = a[r0:r0 + m].double().view(m, K // G, G)
            we = wv[e].view(N, K // G, G)
            part = torch.einsum("mgk,ngk->mng", ae, we)
            ref[r0:r0 + m] = (part * ws[e].double().unsqueeze(0)).sum(-1) * a_s[r0:r0 + m].double()
        r0 += m
    os.environ["B70_W4A8_TILE_K"] = str(tile_k)
    d = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    op(a.to(DEV), a_s.to(DEV), wb.to(DEV), ws.to(DEV), None, d, rpe.to(DEV), N, K, E)
    sync()
    out = d.cpu().double()
    err = (out - ref).abs()
    # stock W4A16 on the dequantized activations (bf16 A), same weights
    ab = (a.float() * a_s).to(torch.bfloat16).to(DEV)
    ds = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=ab, ptr_A_scale=None, ptr_B=wb.to(DEV), ptr_B_scale=ws.to(DEV), ptr_bias=None,
        ptr_D=ds, rows_per_expert=rpe.to(DEV), N=N, K=K, num_experts=E)
    sync()
    return {
        "rows": rows, "N": N, "K": K, "tile_k": tile_k, "avg_m": M / E,
        "max_abs_err": err.max().item(), "ref_max": ref.abs().max().item(),
        "rel_vs_f64": rel(out, ref), "max_rel_elem": (err / ref.abs().clamp_min(1e-3 * ref.abs().max())).max().item(),
        "rel_stock_vs_f64": rel(ds.cpu().double(), ref),
        "rel_w4a8_vs_stock": rel(out, ds.cpu().double()),
        "nan": bool(torch.isnan(out).any()),
    }


def run_gemm(op):
    cases = [
        [1, 2, 3, 4, 0, 6],            # avg <= 4   -> m_8 tile
        [5, 7, 9, 0, 11, 6],           # avg <= 8   -> m_16 tile
        [33, 0, 64, 127, 17, 130],     # avg <= 128 -> m_32 tile
        [300, 0, 129, 513, 7, 257],    # avg > 128  -> 128 x 256 tile
    ]
    res = []
    for tk in (64, 32):
        for i, rows in enumerate(cases):
            for N, K in ((1280, 2560), (2560, 640)):
                r = gemm_case(op, rows, N, K, tk, seed=100 + i)
                r["ok"] = (not r["nan"]) and r["rel_vs_f64"] < 5e-3
                print("GEMM", json.dumps(r), flush=True)
                res.append(r)
    return res


# --------------------------------------------------------------------------- moe
def make_moe(seed, n_local):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w13 = torch.randint(0, 256, (n_local, 2 * INTER, H // 2), dtype=torch.uint8, generator=g)
    w2 = torch.randint(0, 256, (n_local, H, INTER // 2), dtype=torch.uint8, generator=g)
    s13 = (torch.rand(n_local, 2 * INTER, H // G, generator=g) * 0.004 + 0.001).to(torch.bfloat16)
    s2 = (torch.rand(n_local, H, INTER // G, generator=g) * 0.004 + 0.001).to(torch.bfloat16)
    fm = XpuFusedMoe(w13.to(DEV), s13.to(DEV), None, w2.to(DEV), s2.to(DEV), None,
                     n_experts_per_token=TOPK, activation="silu", num_experts=n_local,
                     ep_rank=0, ep_size=EP)
    return fm


def routing(T, seed, skew=0.8):
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(E_GLOBAL, generator=g)
    p = torch.empty(E_GLOBAL)
    p[perm] = 1.0 / torch.arange(1, E_GLOBAL + 1).float() ** skew
    ids = torch.multinomial(p.expand(T, E_GLOBAL), TOPK, replacement=False, generator=g)
    w = torch.softmax(torch.randn(T, TOPK, generator=g), dim=-1)
    return ids.to(torch.int32).to(DEV), w.to(DEV)


def moe_ref(fm, x, tw, ids):
    """fp32 over the dequantized int4 weights, one local expert at a time."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=DEV)
    emap = fm.expert_map
    loc = emap[ids.long()]  # [T, K] local id or -1
    xf = x.float()
    for e in range(fm.num_experts):
        tok, slot = (loc == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        w13 = dequant(fm.w13[e], fm.gemm1_wei_scales[e])
        w2 = dequant(fm.w2[e], fm.gemm2_wei_scales[e])
        h1 = xf[tok] @ w13.t()
        act = torch.nn.functional.silu(h1[:, :INTER]) * h1[:, INTER:]
        out.index_add_(0, tok, (act @ w2.t()) * tw[tok, slot].unsqueeze(1))
    return out


def run_moe(fm, tokens, seed):
    res = []
    for T in tokens:
        ids, tw = routing(T, seed + T)
        g = torch.Generator(device="cpu").manual_seed(seed + 7 * T)
        x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
        for kind in ("gauss", "outlier"):
            xx = x.clone()
            if kind == "outlier":
                xx[:, :16] *= 20
            o_s = torch.zeros(T, H, dtype=torch.bfloat16, device=DEV)
            o_q = torch.zeros_like(o_s)
            fm.apply(o_s, xx, tw, ids)
            w4.apply(fm, o_q, xx, tw, ids)
            ref = moe_ref(fm, xx, tw, ids)
            sync()
            r = {"T": T, "input": kind,
                 "w4a16_rel": rel(o_s, ref), "w4a16_cos": cos_rows(o_s, ref),
                 "w4a8_rel": rel(o_q, ref), "w4a8_cos": cos_rows(o_q, ref),
                 "w4a8_vs_w4a16_rel": rel(o_q, o_s),
                 "nan": bool(torch.isnan(o_q).any())}
            print("MOE", json.dumps(r), flush=True)
            res.append(r)
    return res


# --------------------------------------------------------------------------- bench
def routed_rows(fm, T, seed):
    ids, tw = routing(T, seed + T)
    x = torch.randn(T, H, device=DEV).to(torch.bfloat16)
    rpe = torch.zeros(fm.num_experts, dtype=torch.int32, device=DEV)
    rem = torch.empty(T * TOPK, H, dtype=torch.bfloat16, device=DEV)
    unp = torch.empty(T, TOPK, dtype=torch.int32, device=DEV)
    torch.ops._moe_C.remap_hidden_states(
        hidden_states=x, hidden_states_scales=None, remapped_hidden_states=rem,
        remapped_hidden_states_scales=None, expert_map=fm.expert_map, rows_per_expert=rpe,
        unpermuted_row_to_permuted_row=unp, topk_ids=ids, total_experts_num=fm.total_experts_num,
        local_experts_num=fm.local_experts_num)
    return x, ids, tw, rpe


def run_bench(fm, op, tokens, seed, tile_ks):
    res = []
    for T in tokens:
        x, ids, tw, rpe = routed_rows(fm, T, seed)
        rows = int(rpe.sum().item())
        o = torch.empty(T, H, dtype=torch.bfloat16, device=DEV)
        r = {"T": T, "local_rows": rows, "max_rows_per_expert": int(rpe.max().item())}
        r["moe_w4a16_ms"] = bench(lambda: fm.apply(o, x, tw, ids))
        for tk in tile_ks:
            os.environ["B70_W4A8_TILE_K"] = str(tk)
            r[f"moe_w4a8_tk{tk}_ms"] = bench(lambda: w4.apply(fm, o, x, tw, ids))
        if rows:
            # GEMM-only at the routed grouping (rows sorted by expert)
            a16 = (torch.randn(rows, H, device=DEV) * 0.5).to(torch.bfloat16)
            a8, as8 = w4.quant_rows(a16)
            i16 = (torch.randn(rows, INTER, device=DEV) * 0.5).to(torch.bfloat16)
            i8, is8 = w4.quant_rows(i16)
            d1 = torch.empty(rows, 2 * INTER, dtype=torch.bfloat16, device=DEV)
            d2 = torch.empty(rows, H, dtype=torch.bfloat16, device=DEV)
            E = fm.num_experts

            def s1():
                torch.ops._xpu_C.cutlass_grouped_gemm_interface(
                    ptr_A=a16, ptr_A_scale=None, ptr_B=fm.w13, ptr_B_scale=fm.gemm1_wei_scales,
                    ptr_bias=None, ptr_D=d1, rows_per_expert=rpe, N=2 * INTER, K=H, num_experts=E)

            def s2():
                torch.ops._xpu_C.cutlass_grouped_gemm_interface(
                    ptr_A=i16, ptr_A_scale=None, ptr_B=fm.w2, ptr_B_scale=fm.gemm2_wei_scales,
                    ptr_bias=None, ptr_D=d2, rows_per_expert=rpe, N=H, K=INTER, num_experts=E)

            f1 = 2.0 * rows * 2 * INTER * H
            f2 = 2.0 * rows * H * INTER
            r["g1_w4a16_ms"] = bench(s1)
            r["g2_w4a16_ms"] = bench(s2)
            r["g1_w4a16_tflops"] = f1 / r["g1_w4a16_ms"] / 1e9
            r["g2_w4a16_tflops"] = f2 / r["g2_w4a16_ms"] / 1e9
            for tk in tile_ks:
                os.environ["B70_W4A8_TILE_K"] = str(tk)
                r[f"g1_w4a8_tk{tk}_ms"] = bench(lambda: op(a8, as8, fm.w13, fm.gemm1_wei_scales, None, d1,
                                                          rpe, 2 * INTER, H, E))
                r[f"g2_w4a8_tk{tk}_ms"] = bench(lambda: op(i8, is8, fm.w2, fm.gemm2_wei_scales, None, d2,
                                                          rpe, H, INTER, E))
                r[f"g1_w4a8_tk{tk}_tops"] = f1 / r[f"g1_w4a8_tk{tk}_ms"] / 1e9
                r[f"g2_w4a8_tk{tk}_tops"] = f2 / r[f"g2_w4a8_tk{tk}_ms"] / 1e9
            r["quant_rows_h_ms"] = bench(lambda: w4.quant_rows(a16))
        print("BENCH", json.dumps(r), flush=True)
        res.append(r)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib-dir", default=os.path.join(HERE, "lib"))
    ap.add_argument("--mode", default="gemm,moe,bench")
    ap.add_argument("--experts", type=int, default=E_GLOBAL // EP, help="local experts (256 = 512 experts / EP2)")
    ap.add_argument("--moe-tokens", default="1,16,256,1024")
    ap.add_argument("--bench-tokens", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096")
    ap.add_argument("--tile-k", default="64,32")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "result.json"))
    args = ap.parse_args()

    op = w4.load_op(args.lib_dir)
    print("OP", op, "device", torch.xpu.get_device_name(0), flush=True)
    out = {"lib_dir": args.lib_dir}
    modes = args.mode.split(",")
    if "gemm" in modes:
        out["gemm"] = run_gemm(op)
        print("GEMM_ALL_OK", all(r["ok"] for r in out["gemm"]), flush=True)
    if "moe" in modes or "bench" in modes:
        fm = make_moe(args.seed, args.experts)
        sync()
        if "moe" in modes:
            out["moe"] = run_moe(fm, [int(t) for t in args.moe_tokens.split(",")], args.seed)
        if "bench" in modes:
            out["bench"] = run_bench(fm, op, [int(t) for t in args.bench_tokens.split(",")], args.seed,
                                     [int(t) for t in args.tile_k.split(",")])
    out["peak_gib"] = torch.xpu.max_memory_allocated() / 2**30
    print("PEAK_GIB", round(out["peak_gib"], 3), flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
