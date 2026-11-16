# SPDX-License-Identifier: Apache-2.0
"""Card harness for the mailbox all-reduce (needs 2 or 4 XPU cards).

python tests/mailbox_ar/mailbox_ar_cards.py <mode> [--world N] [...]

modes:
  check     every size x dtype x partial mode, eager and graph replay with
            fresh data each replay, against the CPU contract bit for bit and
            across ranks (sha1); plus a skewed arm (rank r runs r*k GEMMs
            before each call)
  bench     microseconds per call, per rank: mailbox vs dist.all_reduce
            (XCCL; set CCL_SYCL_ALLREDUCE_ARC=1 for the ARC arm), 96 calls
            in a graph, chained and interleaved with a GEMM like the engine;
            JSON lines on stdout
  soak      N minutes of graph replays with random skew, counting mismatches
            and FAILED words
  failstop  the last rank skips one call: every other rank must report
            FAILED with the stage and tag, write NaN, and return (no hang)

Run from the kernels tree root with the venv's python; ZE_AFFINITY_MASK picks
the cards (e.g. 0,3 for the two-card pair, 0,1,2,3 for four).
"""

import argparse
import hashlib
import json
import os
import socket
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

EPW = 6
CLOCK_HZ = 2.8e9


def partners(rank, world, order):
    if world == 2:
        return [rank ^ 1]
    return [rank ^ order[0], rank ^ order[1]]


def reference(inputs, order, fp32p):
    world, dt = len(inputs), inputs[0].dtype
    vals = [x.float() for x in inputs]
    for s in range(len(partners(0, world, order))):
        if s > 0 and not fp32p:
            vals = [v.to(dt).float() for v in vals]
        vals = [
            vals[r] + vals[partners(r, world, order)[s]] for r in range(world)
        ]
    return [v.to(dt) for v in vals]


def swap_fd(name, server, fd):
    addr = "\0" + name
    if server:
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(addr)
        srv.listen(1)
        conn, _ = srv.accept()
        srv.close()
    else:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        for _ in range(3000):
            try:
                conn.connect(addr)
                break
            except OSError:
                time.sleep(0.01)
    with conn:
        socket.send_fds(conn, [b"f"], [fd])
        _, fds, _, _ = socket.recv_fds(conn, 1, 1)
    return fds[0]


class Mailbox:
    def __init__(self, rank, world, order, max_numel, token):
        import vllm_xpu_kernels._xpu_C  # noqa: F401

        self.ops = ops = torch.ops._xpu_C
        self.rank, self.world, self.order = rank, world, order
        self.parts = partners(rank, world, order)
        slot = max(
            ops.mailbox_ar_slot_bytes(max_numel, True),
            ops.mailbox_ar_slot_bytes(max_numel, False),
        )
        self.slot = (slot + 63) // 64 * 64
        hs = [ops.mono_ipc_create(2 * self.slot) for _ in self.parts]
        infos = [list(ops.mono_ipc_export(h)) for h in hs]
        allinfo = [None] * world
        dist.all_gather_object(allinfo, infos)
        fds = []
        for s, p in enumerate(self.parts):
            lo, hi = min(rank, p), max(rank, p)
            fds.append(
                swap_fd(
                    f"mbox-cards-{token}-{s}-{lo}-{hi}", rank == lo, infos[s][1]
                )
            )
        self.modes = []
        for s, p in enumerate(self.parts):
            self.modes.append(ops.mono_ipc_open(hs[s], allinfo[p][s], fds[s]))
        self.inbox = [ops.mono_ipc_ptr(h, False) for h in hs]
        self.outbox = [ops.mono_ipc_ptr(h, True) for h in hs]
        self.host = ops.mailbox_ar_host_alloc()
        self.state = torch.zeros(4, dtype=torch.int32, device="xpu")
        torch.xpu.synchronize()

    def __call__(self, t, fp32p=True, timeout_s=20.0):
        self.ops.mailbox_all_reduce(
            t,
            self.inbox,
            self.outbox,
            self.state.data_ptr(),
            self.host,
            self.slot,
            fp32p,
            int(timeout_s * CLOCK_HZ),
        )
        return t

    def health(self):
        return list(self.ops.mailbox_ar_host_read(self.host))


def sha(t):
    return hashlib.sha1(t.cpu().view(torch.int16).numpy().tobytes()).hexdigest()


def gen(rank, numel, dt, seed):
    g = torch.Generator().manual_seed(seed * 131 + rank)
    x = torch.randn(numel, generator=g) * torch.exp2(
        torch.randint(-8, 8, (numel,), generator=g).float()
    )
    return x.to(dt)


def all_inputs(world, numel, dt, seed):
    return [gen(r, numel, dt, seed) for r in range(world)]


def mode_check(rank, world, mb, args, log):
    sizes = [2560, 8 * 2560, 16 * 2560, 25 * 2560 + 3]
    bad = 0
    seed = 0  # the same sequence on every rank
    gemm_a = torch.randn(512, 2560, device="xpu", dtype=torch.bfloat16)
    gemm_b = torch.randn(2560, 2560, device="xpu", dtype=torch.bfloat16)
    for dt in (torch.bfloat16, torch.float16):
        for fp32p in (True, False):
            for n in sizes:
                for arm in ("eager", "skew"):
                    for _ in range(5):
                        seed += 1
                        xs = all_inputs(world, n, dt, seed)
                        x = xs[rank].to("xpu")
                        if arm == "skew":
                            for _ in range(rank * 4):
                                gemm_a @ gemm_b
                        mb(x, fp32p)
                        torch.xpu.synchronize()
                        ref = reference(xs, mb.order, fp32p)[rank]
                        ok = torch.equal(
                            x.cpu().view(torch.int16), ref.view(torch.int16)
                        )
                        hs = [None] * world
                        dist.all_gather_object(hs, sha(x))
                        bad += (not ok) + (len(set(hs)) != 1)
                # graph replay, fresh data per replay
                xs0 = all_inputs(world, n, dt, 1)
                static = xs0[rank].to("xpu")
                g = torch.xpu.XPUGraph()
                torch.xpu.synchronize()
                dist.barrier()
                with torch.xpu.graph(g):
                    mb(static, fp32p)
                for rep in range(20):
                    xs = all_inputs(world, n, dt, 1000 + rep)
                    static.copy_(xs[rank].to("xpu"))
                    torch.xpu.synchronize()
                    dist.barrier()
                    g.replay()
                    torch.xpu.synchronize()
                    ref = reference(xs, mb.order, fp32p)[rank]
                    bad += not torch.equal(
                        static.cpu().view(torch.int16), ref.view(torch.int16)
                    )
                log(
                    {
                        "mode": "check",
                        "dtype": str(dt),
                        "fp32p": fp32p,
                        "numel": n,
                        "bad_so_far": bad,
                    }
                )
    log(
        {
            "mode": "check",
            "result": "PASS" if bad == 0 else "FAIL",
            "bad": bad,
            "health": mb.health(),
            "ipc_modes": mb.modes,
        }
    )


def time_graph(fn, x, calls=96, reps=50, gemm=None):
    g = torch.xpu.XPUGraph()
    torch.xpu.synchronize()
    dist.barrier()
    with torch.xpu.graph(g):
        for _ in range(calls):
            if gemm is not None:
                gemm()
            fn(x)
    torch.xpu.synchronize()
    for _ in range(3):
        g.replay()
    torch.xpu.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / (reps * calls) * 1e6


def mode_bench(rank, world, mb, args, log):
    a = torch.randn(8, 2560, device="xpu", dtype=torch.bfloat16)
    w = torch.randn(2560, 2560, device="xpu", dtype=torch.bfloat16)
    out = torch.empty(8, 2560, device="xpu", dtype=torch.bfloat16)
    gemm = lambda: torch.matmul(a, w, out=out)  # noqa: E731
    g_only = time_graph(lambda x: None, a, gemm=gemm)
    for rows in (1, 2, 4, 8, 16, 25):
        x = torch.randn(rows, 2560, device="xpu", dtype=torch.bfloat16)
        row = {
            "mode": "bench",
            "rows": rows,
            "bytes": x.numel() * 2,
            "rank": rank,
        }
        for name, fn in (
            ("mailbox_fp32", lambda t: mb(t, True)),
            ("mailbox_input", lambda t: mb(t, False)),
            ("xccl", lambda t: dist.all_reduce(t)),
        ):
            row[name + "_chain_us"] = round(time_graph(fn, x), 2)
            row[name + "_inter_us"] = round(
                time_graph(fn, x, gemm=gemm) - g_only, 2
            )
        row["arc"] = os.environ.get("CCL_SYCL_ALLREDUCE_ARC", "")
        log(row)


def mode_soak(rank, world, mb, args, log):
    n = 8 * 2560
    xs0 = all_inputs(world, n, torch.bfloat16, 3)
    static = xs0[rank].to("xpu")
    gemm_a = torch.randn(256, 2560, device="xpu", dtype=torch.bfloat16)
    gemm_b = torch.randn(2560, 2560, device="xpu", dtype=torch.bfloat16)
    g = torch.xpu.XPUGraph()
    torch.xpu.synchronize()
    dist.barrier()
    with torch.xpu.graph(g):
        for _ in range(rank * 3):
            gemm_a @ gemm_b
        mb(static, True)
    end = time.time() + args.minutes * 60
    reps = bad = 0
    while time.time() < end:
        xs = all_inputs(world, n, torch.bfloat16, 5000 + reps % 64)
        static.copy_(xs[rank].to("xpu"))
        torch.xpu.synchronize()
        g.replay()
        torch.xpu.synchronize()
        if reps % 16 == 0:
            ref = reference(xs, mb.order, True)[rank]
            bad += not torch.equal(
                static.cpu().view(torch.int16), ref.view(torch.int16)
            )
        reps += 1
        dist.barrier()
    log(
        {
            "mode": "soak",
            "replays": reps,
            "checked": (reps + 15) // 16,
            "mismatch": bad,
            "health": mb.health(),
        }
    )


def mode_failstop(rank, world, mb, args, log):
    n = 8 * 2560
    x = torch.ones(n, device="xpu", dtype=torch.bfloat16)
    mb(x, True, timeout_s=2.0)
    torch.xpu.synchronize()
    dist.barrier()
    t0 = time.time()
    if rank != world - 1:  # the last rank skips this call
        mb(x, True, timeout_s=2.0)
        torch.xpu.synchronize()
    took = time.time() - t0
    h = mb.health()
    nan = bool(torch.isnan(x.float()).any()) if rank != world - 1 else None
    log(
        {
            "mode": "failstop",
            "rank": rank,
            "returned_after_s": round(took, 2),
            "health": h,
            "nan_written": nan,
            "result": (
                "PASS" if rank == world - 1 or (h[0] == 1 and nan) else "FAIL"
            ),
        }
    )


def worker(rank, world, args, init, q):
    torch.xpu.set_device(rank)
    dist.init_process_group(
        "xccl",
        init_method=f"file://{init}",
        rank=rank,
        world_size=world,
        device_id=torch.device("xpu", rank),
    )
    gloo = dist.new_group(backend="gloo")
    dist.barrier(group=gloo)
    tok = [os.getpid() if rank == 0 else 0]
    dist.broadcast_object_list(tok, 0, group=gloo)
    order = tuple(int(v) for v in args.order.split(","))
    # the gloo group carries the object exchange of the setup
    dist.all_gather_object = lambda o, v, _f=dist.all_gather_object: _f(
        o, v, group=gloo
    )
    mb = Mailbox(rank, world, order, 32 * 2560, tok[0])
    log = lambda d: q.put(json.dumps(d))  # noqa: E731
    {
        "check": mode_check,
        "bench": mode_bench,
        "soak": mode_soak,
        "failstop": mode_failstop,
    }[args.mode](rank, world, mb, args, log)
    q.put(None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "bench", "soak", "failstop"])
    p.add_argument("--world", type=int, default=0)
    p.add_argument("--order", default="3,1")
    p.add_argument("--minutes", type=float, default=10.0)
    args = p.parse_args()
    world = args.world or torch.xpu.device_count()
    assert world in (2, 4), world
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        procs = [
            ctx.Process(
                target=worker, args=(r, world, args, os.path.join(d, "init"), q)
            )
            for r in range(world)
        ]
        for pr in procs:
            pr.start()
        done = 0
        while done < world:
            m = q.get()
            if m is None:
                done += 1
            else:
                print(m, flush=True)
        for pr in procs:
            pr.join()
    sys.exit(0)


if __name__ == "__main__":
    main()
