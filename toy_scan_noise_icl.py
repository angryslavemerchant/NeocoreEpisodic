"""SCAN-lite v6: NOISE + CONNECTORS, frozen kernel, still gradient-free.

Two axes at once, separably instrumented:
- NOISE (inference organ): every teaching is corrupted with prob p
  (atom actions flipped; demo output tokens flipped). Exact hypothesis
  elimination must become ACCUMULATED EVIDENCE: counts/votes across
  repeated teachings (the schedule now repeats R rounds). Consolidation
  returns as counting — and with it the within-lifetime LEARNING CURVE,
  exactly where the DINO campaign diagnosed it must live: where one
  observation stops sufficing.
- CONNECTORS (kernel discipline): two-clause commands "C1 conn C2";
  conn forms (4, per-lifetime) mean AND (emit in order) or AFTER (swap
  clause order). The ordering op enters as a 2-way learned PROGRAM over
  an order-blind concatenation kernel — brittleness bout #2: no new
  task-shaped operation, only a swap bit inferred from one/few demos.

Zero trained parameters, runs locally in seconds. Metrics: exact-match
per frame (single + multi-clause) as a function of teaching round, for
p in {0, 0.15, 0.3}. Pre-registered: clean = 100 flat at round 1;
noisy = rising curves converging toward 100 as votes accumulate;
connector frames track single-clause frames (the swap-bit is learned
as easily as any other program) — if instead connectors lag
structurally, the kernel needs redesign and we report that.
"""

import argparse
import time

import torch

from toy_scan_icl import (N_C, N_D, N_S, N_V, NA, NF, OUTM, PAD_CLS, S0,
                          C0, exec_cmd)

N_N = 4                      # connector forms
OUT2 = 2 * OUTM


def make_lifetime(E, device):
    return {"v": torch.randint(0, NA, (E, N_V), device=device),
            "d": torch.randint(0, NA, (E, N_D), device=device),
            "s": torch.randint(0, 2, (E, N_S), device=device),
            "c": torch.randint(0, 2, (E, N_C), device=device),
            "n": torch.randint(0, 2, (E, N_N), device=device)}  # 0 and 1 aft


def corrupt(out, p, device):
    if p <= 0:
        return out
    flip = (torch.rand_like(out.float()) < p) & (out != PAD_CLS)
    rnd = torch.randint(0, NA, out.shape, device=device)
    return torch.where(flip, rnd, out)


def concat_swap(c1, c2, swap):
    """c1,c2 (E,OUTM) contiguous-prefix outputs; swap (E,) bool ->
    (E,OUT2). Order-blind concatenation kernel + a swap bit."""
    a = torch.where(swap.unsqueeze(1), c2, c1)
    b = torch.where(swap.unsqueeze(1), c1, c2)
    la = (a != PAD_CLS).sum(1)
    j = torch.arange(OUT2, device=c1.device).unsqueeze(0)
    from_a = j < la.unsqueeze(1)
    ja = j.expand(c1.shape[0], -1).clamp(max=OUTM - 1)
    jb = (j - la.unsqueeze(1)).clamp(min=0, max=OUTM - 1)
    return torch.where(from_a, a.gather(1, ja),
                       torch.where(j - la.unsqueeze(1)
                                   < (b != PAD_CLS).sum(1, keepdim=True),
                                   b.gather(1, jb),
                                   torch.full_like(j, PAD_CLS)))


class VoteKernel:
    def __init__(self, E, device):
        self.av = torch.zeros(E, NF, NA, device=device)      # atom votes
        self.sv = torch.zeros(E, N_S, 2, device=device)
        self.cv = torch.zeros(E, N_C, 2, device=device)
        self.nv = torch.zeros(E, N_N, 2, device=device)
        self.ar = torch.arange(E, device=device)

    def atoms(self):
        return self.av.argmax(-1)                            # (E, NF)

    def teach_atom(self, form, act):
        self.av[self.ar, form, act] += 1

    def _vote2(self, store, local, preds, demo):
        d0 = (preds[0] != demo).sum(1)
        d1 = (preds[1] != demo).sum(1)
        store[self.ar, local, 0] += (d0 < d1).float()
        store[self.ar, local, 1] += (d1 < d0).float()

    def teach_s(self, local, demo, vi, di):
        A = self.atoms()
        v, d = A[self.ar, vi], A[self.ar, di + N_V]
        preds = [exec_cmd(v, d, torch.full_like(v, m + 2),
                          torch.ones_like(v)) for m in (0, 1)]
        self._vote2(self.sv, local, preds, demo)

    def teach_c(self, local, demo, vi):
        A = self.atoms()
        v = A[self.ar, vi]
        z = torch.zeros_like(v)
        preds = [exec_cmd(v, z, z, torch.full_like(v, k)) for k in (2, 3)]
        self._vote2(self.cv, local, preds, demo)

    def teach_n(self, local, demo, v1, v2):
        A = self.atoms()
        a1 = exec_cmd(A[self.ar, v1], torch.zeros_like(v1),
                      torch.zeros_like(v1), torch.ones_like(v1))
        a2 = exec_cmd(A[self.ar, v2], torch.zeros_like(v2),
                      torch.zeros_like(v2), torch.ones_like(v2))
        f = torch.zeros(v1.shape[0], dtype=torch.bool, device=v1.device)
        t = ~f
        preds = [concat_swap(a1, a2, f)[:, :OUT2],
                 concat_swap(a1, a2, t)[:, :OUT2]]
        self._vote2(self.nv, local, preds, demo[:, :OUT2])

    def clause(self, vi, di, si, ci, hd, hs, hc):
        A = self.atoms()
        v, d = A[self.ar, vi], A[self.ar, di + N_V]
        mode = self.sv[self.ar, si].argmax(-1)
        cnt = self.cv[self.ar, ci].argmax(-1) + 2
        mc = torch.where(hs, mode + 2,
                         torch.where(hd, torch.ones_like(mode),
                                     torch.zeros_like(mode)))
        return exec_cmd(v, d, mc, torch.where(hc, cnt,
                                              torch.ones_like(cnt)))


def true_clause(maps, vi, di, si, ci, hd, hs, hc):
    ar = torch.arange(vi.shape[0], device=vi.device)
    v = maps["v"][ar, vi]
    d = maps["d"][ar, di]
    mode = maps["s"][ar, si]
    cnt = maps["c"][ar, ci] + 2
    mc = torch.where(hs, mode + 2, torch.where(hd, torch.ones_like(mode),
                                               torch.zeros_like(mode)))
    return exec_cmd(v, d, mc, torch.where(hc, cnt, torch.ones_like(cnt)))


def rand_clause(E, device):
    return (torch.randint(0, N_V, (E,), device=device),
            torch.randint(0, N_D, (E,), device=device),
            torch.randint(0, N_S, (E,), device=device),
            torch.randint(0, N_C, (E,), device=device))


def run(E, device, rounds, p, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    maps = make_lifetime(E, device)
    ker = VoteKernel(E, device)
    ar = torch.arange(E, device=device)
    frames = {"V": (0, 0, 0), "VD": (1, 0, 0), "VDC": (1, 0, 1),
              "VSD": (1, 1, 0), "VSDC": (1, 1, 1)}
    multi = [("V+V", (0, 0, 0), (0, 0, 0)),
             ("VD+VC", (1, 0, 0), (0, 0, 1)),
             ("VSD+VDC", (1, 1, 0), (1, 0, 1))]
    curves = {k: [] for k in list(frames) + [m[0] for m in multi]}
    for r in range(rounds):
        # teach every word once per round, corrupted
        for f in range(N_V):
            form = torch.full((E,), f, dtype=torch.long, device=device)
            ker.teach_atom(form, corrupt(
                maps["v"][:, f].unsqueeze(1), p, device).squeeze(1))
        for f in range(N_D):
            form = torch.full((E,), f + N_V, dtype=torch.long,
                              device=device)
            ker.teach_atom(form, corrupt(
                maps["d"][:, f].unsqueeze(1), p, device).squeeze(1))
        for f in range(N_S):
            local = torch.full((E,), f, dtype=torch.long, device=device)
            vi = torch.randint(0, N_V, (E,), device=device)
            di = torch.randint(0, N_D, (E,), device=device)
            demo = exec_cmd(maps["v"][ar, vi], maps["d"][ar, di],
                            maps["s"][ar, local] + 2, torch.ones_like(vi))
            ker.teach_s(local, corrupt(demo, p, device), vi, di)
        for f in range(N_C):
            local = torch.full((E,), f, dtype=torch.long, device=device)
            vi = torch.randint(0, N_V, (E,), device=device)
            z = torch.zeros_like(vi)
            demo = exec_cmd(maps["v"][ar, vi], z, z,
                            maps["c"][ar, local] + 2)
            ker.teach_c(local, corrupt(demo, p, device), vi)
        for f in range(N_N):
            local = torch.full((E,), f, dtype=torch.long, device=device)
            v1 = torch.randint(0, N_V, (E,), device=device)
            v2 = torch.randint(0, N_V, (E,), device=device)
            o = torch.ones_like(v1)
            z = torch.zeros_like(v1)
            a1 = exec_cmd(maps["v"][ar, v1], z, z, o)
            a2 = exec_cmd(maps["v"][ar, v2], z, z, o)
            demo = concat_swap(a1, a2, maps["n"][ar, local].bool())
            ker.teach_n(local, corrupt(demo, p, device), v1, v2)
        # eval
        for name, (hd, hs, hc) in frames.items():
            vi, di, si, ci = rand_clause(E, device)
            hdt = torch.full((E,), bool(hd), dtype=torch.bool,
                             device=device)
            hst = torch.full((E,), bool(hs), dtype=torch.bool,
                             device=device)
            hct = torch.full((E,), bool(hc), dtype=torch.bool,
                             device=device)
            tgt = true_clause(maps, vi, di, si, ci, hdt, hst, hct)
            out = ker.clause(vi, di, si, ci, hdt, hst, hct)
            curves[name].append(float((out == tgt).all(1).float().mean()))
        for name, f1, f2 in multi:
            args1 = rand_clause(E, device)
            args2 = rand_clause(E, device)
            ni = torch.randint(0, N_N, (E,), device=device)
            hs1 = [torch.full((E,), bool(x), dtype=torch.bool,
                              device=device) for x in f1]
            hs2 = [torch.full((E,), bool(x), dtype=torch.bool,
                              device=device) for x in f2]
            t1 = true_clause(maps, *args1, *hs1)
            t2 = true_clause(maps, *args2, *hs2)
            tgt = concat_swap(t1, t2, maps["n"][ar, ni].bool())
            k1 = ker.clause(*args1, *hs1)
            k2 = ker.clause(*args2, *hs2)
            swap = ker.nv[ar, ni].argmax(-1).bool()
            out = concat_swap(k1, k2, swap)
            curves[name].append(float((out == tgt).all(1).float().mean()))
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--eval-batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    t0 = time.time()
    for p in (0.0, 0.15, 0.3):
        c = run(args.eval_batch, device, args.rounds, p, args.seed)
        print(f"\np={p} — exact-match by teaching round "
              f"(rows=frames, cols=rounds 1..{args.rounds}):")
        for k, v in c.items():
            print(f"  {k:>8s}  " + "  ".join(f"{x * 100:5.1f}" for x in v))
    print(f"\ntotal {time.time() - t0:.0f}s, zero gradients", flush=True)


if __name__ == "__main__":
    main()
