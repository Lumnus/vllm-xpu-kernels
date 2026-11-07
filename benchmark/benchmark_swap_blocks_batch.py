# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""H2D swap_blocks_batch: staged vs direct copy from pinned host memory.

Sources, one host pool per run:
  registered  shared mmap page-locked with xpu_host_register (the vLLM XPU
              KV-offload pool on /dev/shm); SYCL reports it as "unknown"
  pinned      torch pin_memory tensor (USM host)
  pageable    the same mmap, not registered: always staged, in both
              columns (baseline; checks the fallback)

Providers: snapshot_src=True (staged: host memcpy into a pinned staging
buffer, then DMA) and snapshot_src=False (direct DMA from the source).

Per (source, block size, blocks per batch) it reports the median over
--iters of:
  call_us   host time spent inside swap_blocks_batch (the caller's thread)
  e2e_us    call + stream synchronize
  GB/s      batch bytes / e2e

Example:
  python benchmark/benchmark_swap_blocks_batch.py \
      --source registered pinned pageable \
      --block-kib 16 64 256 --blocks 64 512 2048
"""

import argparse
import mmap
import statistics
import time
from typing import Optional

import numpy as np
import torch
from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

from tests import register_ops as ops


class HostPool:
    """A host pool of n_bytes in one of the three source classes."""

    def __init__(self, source: str, n_bytes: int):
        self.source = source
        self.region: Optional[mmap.mmap] = None
        self.registered = False
        if source == "pinned":
            self.tensor = torch.empty(n_bytes,
                                      dtype=torch.uint8,
                                      pin_memory=True)
            return
        self.region = mmap.mmap(-1, n_bytes)
        self.tensor = torch.frombuffer(memoryview(self.region),
                                       dtype=torch.uint8)
        if source == "registered":
            self.registered = ops.xpu_host_register(self.tensor.data_ptr(),
                                                    n_bytes)
            if not self.registered:
                raise RuntimeError("xpu_host_register failed")

    def close(self) -> None:
        torch.xpu.synchronize()
        if self.registered:
            ops.xpu_host_unregister(self.tensor.data_ptr())
        del self.tensor
        if self.region is not None:
            self.region.close()


def batch_args(pool: HostPool, dst: torch.Tensor, block: int, n: int,
               seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """n random distinct source blocks -> dst blocks 0..n-1."""
    n_src = pool.tensor.numel() // block
    rng = np.random.default_rng(seed)
    src_idx = rng.choice(n_src, size=n, replace=False).astype(np.uint64)
    src = np.uint64(pool.tensor.data_ptr()) + src_idx * np.uint64(block)
    dsts = np.uint64(dst.data_ptr()) + np.arange(
        n, dtype=np.uint64) * np.uint64(block)
    sizes = np.full(n, block, dtype=np.uint64)
    return (torch.from_numpy(src), torch.from_numpy(dsts),
            torch.from_numpy(sizes))


def run_one(args3, snapshot_src: bool, iters: int,
            warmup: int) -> tuple[float, float]:
    call, e2e = [], []
    for i in range(warmup + iters):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        ops.swap_blocks_batch(*args3, snapshot_src=snapshot_src)
        t1 = time.perf_counter()
        torch.xpu.synchronize()
        t2 = time.perf_counter()
        if i >= warmup:
            call.append((t1 - t0) * 1e6)
            e2e.append((t2 - t0) * 1e6)
    return statistics.median(call), statistics.median(e2e)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source",
                   nargs="+",
                   default=["registered", "pinned", "pageable"],
                   choices=["registered", "pinned", "pageable"])
    p.add_argument("--block-kib", nargs="+", type=int, default=[16, 64, 256])
    p.add_argument("--blocks", nargs="+", type=int, default=[64, 512, 2048])
    p.add_argument("--pool-gib",
                   type=float,
                   default=4.0,
                   help="host pool size; must hold the largest batch")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="xpu:0")
    args = p.parse_args()

    torch.xpu.set_device(args.device)
    pool_bytes = int(args.pool_gib * (1 << 30))
    max_batch = max(args.block_kib) * 1024 * max(args.blocks)
    if max_batch > pool_bytes:
        raise SystemExit(f"--pool-gib too small for a {max_batch} B batch")
    dst = torch.empty(max_batch, dtype=torch.uint8, device=args.device)

    print(f"device {torch.xpu.get_device_name(args.device)}, "
          f"torch {torch.__version__}, pool {args.pool_gib} GiB")
    hdr = (f"{'source':<11}{'block':>8}{'n':>6}{'MiB':>9}  "
           f"{'staged call/e2e us':>22}{'GB/s':>7}  "
           f"{'direct call/e2e us':>22}{'GB/s':>7}  {'e2e x':>6}")
    print(hdr)
    print("-" * len(hdr))
    with torch.inference_mode():
        for source in args.source:
            pool = HostPool(source, pool_bytes)
            try:
                pool.tensor[::4096].fill_(1)  # fault the pages in
                for kib in args.block_kib:
                    block = kib * 1024
                    for n in args.blocks:
                        a3 = batch_args(pool, dst, block, n, seed=n + kib)
                        nbytes = block * n
                        s_call, s_e2e = run_one(a3, True, args.iters,
                                                args.warmup)
                        d_call, d_e2e = run_one(a3, False, args.iters,
                                                args.warmup)
                        print(f"{source:<11}{kib:>6}Ki{n:>6}"
                              f"{nbytes / (1 << 20):>9.1f}  "
                              f"{s_call:>10.0f} /{s_e2e:>10.0f}"
                              f"{nbytes / s_e2e / 1e3:>7.2f}  "
                              f"{d_call:>10.0f} /{d_e2e:>10.0f}"
                              f"{nbytes / d_e2e / 1e3:>7.2f}  "
                              f"{s_e2e / d_e2e:>6.2f}")
            finally:
                pool.close()


if __name__ == "__main__":
    main()
