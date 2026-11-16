# SPDX-License-Identifier: Apache-2.0
"""CPU model of the mailbox all-reduce (csrc/xpu/mailbox_ar), no device.

The model runs the kernel's per-item algorithm line for line: 16-byte lines
(three payload words + tag), two slots per stage mailbox selected by the
tag's low bit, the epoch read at work-group start and published by the last
work-group, the wait with its poll budget and the sticky FAILED word. A
random scheduler interleaves the work-items of all ranks one memory access
at a time, with launch skew between ranks, slow ranks and dead ranks.

Checks:
1. every rank's output equals the summation contract bit for bit
   (reference below), for 2 and 4 ranks, every partner order, both partial
   modes, tails (numel not a multiple of 6) and grid-stride loops;
2. a slow rank (within the timeout) changes nothing;
3. a dead rank makes every live rank fail (FAILED set, NaN written) within a
   bounded number of steps: no hang;
4. negative controls that show the design's two load-bearing assumptions:
   one slot instead of two hangs (a fast rank overwrites a line its partner
   has not read), and a line that lands in 4-byte pieces with the tag first
   gives wrong sums (the 16-byte store/load unit the kernel relies on).
"""

import random

import pytest
import torch

EPW = 6


def partners(rank, world, order):
    if world == 2:
        return [rank ^ 1]
    return [rank ^ order[0], rank ^ order[1]]


def reference(inputs, order, fp32p):
    world = len(inputs)
    dt = inputs[0].dtype
    vals = [x.float() for x in inputs]
    for s in range(len(partners(0, world, order))):
        if s > 0 and not fp32p:
            vals = [v.to(dt).float() for v in vals]
        vals = [
            vals[r] + vals[partners(r, world, order)[s]] for r in range(world)
        ]
    return [v.to(dt) for v in vals]


def u16(t):
    return [int(v) & 0xFFFF for v in t.view(torch.int16).tolist()]


def from_u16(words, dt):
    return torch.tensor(
        [w - 65536 if w >= 32768 else w for w in words], dtype=torch.int16
    ).view(dt)


def f32_bits(t):
    return [int(v) & 0xFFFFFFFF for v in t.view(torch.int32).tolist()]


def from_f32_bits(words):
    return torch.tensor(
        [w - (1 << 32) if w >= (1 << 31) else w for w in words],
        dtype=torch.int32,
    ).view(torch.float32)


class Model:
    def __init__(
        self,
        world,
        n,
        nwi,
        wg,
        order,
        fp32p,
        slots=2,
        torn=False,
        timeout_polls=4000,
        seed=0,
        dt=torch.bfloat16,
    ):
        self.world, self.n, self.nwi, self.wg = world, n, nwi, wg
        self.order, self.fp32p, self.slots, self.torn = (
            order,
            fp32p,
            slots,
            torn,
        )
        self.timeout = timeout_polls
        self.rng = random.Random(seed)
        self.dt = dt
        self.items = (n + EPW - 1) // EPW
        self.nst = len(partners(0, world, order))
        lines = self.items * (2 if fp32p else 1)
        # mbox[r][s][slot][line] = (w0, w1, w2, tag)
        self.mbox = [
            [
                [[(0, 0, 0, 0)] * lines for _ in range(2)]
                for _ in range(self.nst)
            ]
            for _ in range(world)
        ]
        self.state = [[0, 0] for _ in range(world)]  # epoch, done
        self.failed = [0] * world

    # -- memory accesses (each a scheduler step)
    def store(self, r, s, slot, line, words):
        tgt = self.mbox[partners(r, self.world, self.order)[s]][s][slot]
        if not self.torn:
            tgt[line] = tuple(words)
            yield
            return
        cur = list(tgt[line])
        idx = [3, 0, 1, 2]  # the tag word lands first
        for k in idx:
            cur[k] = words[k]
            tgt[line] = tuple(cur)
            yield

    def wait(self, r, s, slot, line, tag):
        polls = 0
        while True:
            w = self.mbox[r][s][slot][line]
            if w[3] == tag:
                return w
            polls += 1
            if polls % 256 == 0 and (self.failed[r] or polls > self.timeout):
                self.failed[r] = 1
                return None
            yield

    def item(self, r, x, out, i, tag):
        slot = (tag & 1) % self.slots
        e0 = i * EPW
        idx = [e0 + j for j in range(EPW)]
        t = torch.stack(
            [
                x[k] if k < self.n else torch.zeros((), dtype=self.dt)
                for k in idx
            ]
        )
        b = u16(t)
        v = t.float()
        ok = True
        for s in range(self.nst):
            if s == 0 or not self.fp32p:
                if s > 0:
                    rr = v.to(self.dt)
                    b, v = u16(rr), rr.float()
                words = [
                    b[0] | b[1] << 16,
                    b[2] | b[3] << 16,
                    b[4] | b[5] << 16,
                    tag,
                ]
                yield from self.store(r, s, slot, i, words)
                w = yield from self.wait(r, s, slot, i, tag)
                if w is None:
                    ok = False
                    break
                peer = []
                for j in range(EPW):
                    peer.append((w[j >> 1] >> ((j & 1) * 16)) & 0xFFFF)
                v = v + from_u16(peer, self.dt).float()
            else:
                fb = f32_bits(v)
                yield from self.store(r, s, slot, 2 * i, fb[0:3] + [tag])
                yield from self.store(r, s, slot, 2 * i + 1, fb[3:6] + [tag])
                w0 = yield from self.wait(r, s, slot, 2 * i, tag)
                w1 = (
                    (yield from self.wait(r, s, slot, 2 * i + 1, tag))
                    if w0
                    else None
                )
                if w0 is None or w1 is None:
                    ok = False
                    break
                v = v + from_f32_bits(list(w0[0:3]) + list(w1[0:3]))
        res = (
            v.to(self.dt)
            if ok
            else torch.full((EPW,), float("nan"), dtype=self.dt)
        )
        for j, k in enumerate(idx):
            if k < self.n:
                out[k] = res[j]

    def workgroup(self, r, x, out, g, nwg, box):
        yield  # the work-group starts later than the launch
        epoch = self.state[r][0]
        tag = (epoch + 1) & 0xFFFFFFFF or 1
        box["tags"].add(tag)
        gens = []
        for lid in range(self.wg):
            wi = g * self.wg + lid
            gens.append(self.wi_loop(r, x, out, wi, tag))
        while gens:  # work-items of a group interleave too
            k = self.rng.randrange(len(gens))
            try:
                next(gens[k])
            except StopIteration:
                gens.pop(k)
            yield
        old = self.state[r][1]
        self.state[r][1] = old + 1
        if old == nwg - 1:
            self.state[r][1] = 0
            self.state[r][0] = tag

    def wi_loop(self, r, x, out, wi, tag):
        for i in range(wi, self.items, self.nwi):
            yield from self.item(r, x, out, i, tag)

    def run(
        self,
        calls,
        inputs,
        skew=0,
        slow=None,
        dead=None,
        dead_after=0,
        max_steps=3_000_000,
    ):
        """inputs[c][r]: rank r's tensor at call c. Returns outputs[c][r]."""
        nwg = self.nwi // self.wg
        outs = [[None] * self.world for _ in range(calls)]
        nxt = [0] * self.world  # next call per rank
        delay = [self.rng.randrange(skew + 1) for _ in range(self.world)]
        active = {}  # rank -> list of work-group generators
        steps = 0
        while True:
            for r in range(self.world):
                if r in active or nxt[r] >= calls:
                    continue
                if dead is not None and r == dead and nxt[r] >= dead_after:
                    continue
                if delay[r] > 0:
                    delay[r] -= 1
                    continue
                c = nxt[r]
                out = inputs[c][r].clone()
                outs[c][r] = out
                box = {"tags": set()}
                active[r] = [
                    self.workgroup(r, inputs[c][r], out, g, nwg, box)
                    for g in range(nwg)
                ]
            if not active:
                if all(
                    nxt[r] >= calls or (r == dead and nxt[r] >= dead_after)
                    for r in range(self.world)
                ):
                    return outs
                steps += 1  # only launch delays are pending
                continue
            ranks = list(active)
            weights = [
                0.05 if (slow is not None and r == slow) else 1.0 for r in ranks
            ]
            r = self.rng.choices(ranks, weights)[0]
            gl = active[r]
            k = self.rng.randrange(len(gl))
            try:
                next(gl[k])
            except StopIteration:
                gl.pop(k)
                if not gl:
                    del active[r]
                    nxt[r] += 1
                    delay[r] = self.rng.randrange(skew + 1)
            steps += 1
            if steps > max_steps:
                raise TimeoutError("model did not finish: hang")


def make_inputs(calls, world, n, seed, dt=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    res = []
    for _ in range(calls):
        row = []
        for _ in range(world):
            t = torch.randn(n, generator=g) * torch.exp2(
                torch.randint(-12, 12, (n,), generator=g).float()
            )
            row.append(t.to(dt))
        res.append(row)
    return res


CASES = [
    # world, n, nwi, wg, order
    (2, 31, 4, 2, (1, 2)),
    (4, 31, 4, 2, (3, 1)),
    (4, 40, 4, 4, (1, 2)),
    (4, 13, 8, 4, (2, 3)),
    (4, 66, 4, 2, (2, 1)),
]


@pytest.mark.parametrize("fp32p", [True, False])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("seed", range(6))
def test_contract_bitwise(case, fp32p, seed):
    world, n, nwi, wg, order = case
    calls = 7
    ins = make_inputs(calls, world, n, 1000 + seed)
    m = Model(world, n, nwi, wg, order, fp32p, seed=seed)
    outs = m.run(calls, ins, skew=40)
    assert m.failed == [0] * world
    for c in range(calls):
        ref = reference(ins[c], order, fp32p)
        for r in range(world):
            assert torch.equal(
                outs[c][r].view(torch.int16), ref[r].view(torch.int16)
            ), (c, r)
            assert torch.equal(
                outs[c][r].view(torch.int16), outs[c][0].view(torch.int16)
            )
    assert all(st[0] == calls for st in m.state)  # epoch advanced once per call


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
def test_contract_fp16_and_bf16(dt):
    world, n, nwi, wg, order = 4, 25, 4, 2, (3, 1)
    ins = make_inputs(4, world, n, 7, dt=dt)
    m = Model(world, n, nwi, wg, order, True, seed=3, dt=dt)
    outs = m.run(4, ins, skew=10)
    for c in range(4):
        ref = reference(ins[c], order, True)
        for r in range(world):
            assert torch.equal(
                outs[c][r].view(torch.int16), ref[r].view(torch.int16)
            )


@pytest.mark.parametrize("seed", range(4))
def test_slow_rank_within_timeout(seed):
    world, n, order = 4, 30, (3, 1)
    ins = make_inputs(5, world, n, 50 + seed)
    m = Model(world, n, 4, 2, order, True, seed=seed, timeout_polls=10**9)
    outs = m.run(5, ins, skew=200, slow=seed % world)
    assert m.failed == [0] * world
    for c in range(5):
        ref = reference(ins[c], order, True)
        for r in range(world):
            assert torch.equal(
                outs[c][r].view(torch.int16), ref[r].view(torch.int16)
            )


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("dead_after", [0, 2])
def test_dead_rank_fails_every_live_rank(world, dead_after):
    n, order, calls, dead = 24, (3, 1), 5, world - 1
    ins = make_inputs(calls, world, n, 9)
    m = Model(world, n, 4, 2, order, True, seed=1, timeout_polls=2000)
    outs = m.run(
        calls,
        ins,
        skew=5,
        dead=dead,
        dead_after=dead_after,
        max_steps=2_000_000,
    )
    for r in range(world):
        if r == dead:
            continue
        assert m.failed[r] == 1, r
        # calls before the death are exact, the call that waited on the dead
        # rank (and every later one) is NaN, never a silent wrong sum
        for c in range(dead_after):
            ref = reference(ins[c], order, True)
            assert torch.equal(
                outs[c][r].view(torch.int16), ref[r].view(torch.int16)
            )
        assert torch.isnan(outs[dead_after][r].float()).any()
        for c in range(dead_after, calls):
            o = outs[c][r].float()
            ref = reference(ins[c], order, True)[r].float()
            # any element is either NaN or the exact sum (lines that completed
            # before the partner chain broke); none is a wrong number
            good = torch.isnan(o) | (o == ref)
            assert bool(good.all()), (c, r)


def test_negative_control_one_slot_hangs_or_fails():
    """One slot per stage: a fast rank overwrites a line its partner has not
    read yet. Some schedule must fail (that is why the kernel has two)."""
    world, n, order = 2, 12, (1, 2)
    bad = 0
    for seed in range(40):
        ins = make_inputs(6, world, n, seed)
        m = Model(
            world, n, 2, 2, order, True, slots=1, seed=seed, timeout_polls=3000
        )
        try:
            outs = m.run(6, ins, skew=0, slow=seed % 2, max_steps=400_000)
        except TimeoutError:
            bad += 1
            continue
        if any(m.failed):
            bad += 1
            continue
        for c in range(6):
            ref = reference(ins[c], order, True)
            if not all(
                torch.equal(
                    outs[c][r].view(torch.int16), ref[r].view(torch.int16)
                )
                for r in range(world)
            ):
                bad += 1
                break
    assert bad > 0


def test_negative_control_torn_lines_give_wrong_sums():
    """Lines landing in 4-byte pieces, tag first: a reader can take a stale
    payload. Some schedule must produce a wrong value (the kernel relies on
    a 16-byte store landing, and a 16-byte load reading, as a unit)."""
    world, n, order = 4, 24, (3, 1)
    wrong = 0
    for seed in range(30):
        ins = make_inputs(4, world, n, 300 + seed)
        m = Model(world, n, 4, 2, order, True, torn=True, seed=seed)
        outs = m.run(4, ins, skew=3)
        for c in range(4):
            ref = reference(ins[c], order, True)
            for r in range(world):
                if not torch.equal(
                    outs[c][r].view(torch.int16), ref[r].view(torch.int16)
                ):
                    wrong += 1
    assert wrong > 0
