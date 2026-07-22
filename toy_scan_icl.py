"""Toy: SCAN-lite under the lifetime meta-protocol (frozen kernel).

The generality witness. Single-clause SCAN-family commands where EVERY
content word's meaning is per-lifetime (unsmearable):

  frames:  V | V D | V C | V D C | V S D | V S D C
  verbs    (12 forms) -> one of 8 abstract actions
  dirs     (4 forms)  -> one of 8 actions (the "turn")
  struct   (4 forms)  -> mode OPPOSITE (d d v) or AROUND ((d v) x4)
  count    (4 forms)  -> 2 or 3 (multiplies the clause block)

Study protocol per lifetime (6 episodes x 4 words, fixed class
schedule, forms shuffled): verbs/dirs taught as ATOM pairs (form ->
action) and filed in the key-value book; struct/count words taught by
one DEMO command each — the walker learns them by EXACT HYPOTHESIS
ELIMINATION: enumerate the word's 2 candidate meanings, execute both
with book-decoded atoms, keep the one consistent with the demo. All
inference is hardcoded computation (posterior over an enumerable
hypothesis space); the kernel (templates = nested repeat/emit walks)
is frozen; THE WALKER HAS ZERO TRAINED PARAMETERS — fixed random form
keys and action prototypes, exact decoding. Pre-registered: walker at
~100 exact on all frames incl. combinations never demonstrated
(struct/count applied to verbs seen only as atoms — add-jump at full
strength).

Baseline: AR transformer (3L d96), ALL study material (atoms + full
demos) in context, teacher-forced, greedy decode — the
learned-in-context path, meta-trained over lifetimes.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

N_V, N_D, N_S, N_C = 12, 4, 4, 4
NF = N_V + N_D + N_S + N_C          # 24 forms; ids: V 0-11 D 12-15
S0, C0 = 16, 20                     # struct 16-19, count 20-23
NA = 8                              # abstract actions 0-7
PAD_CLS = NA                        # output pad class (classes 0..8)
OUTM = 24                           # (d v)x4 x cnt3 = 24
EPS = 6
QN = 8                              # eval queries per lifetime
# class schedule per episode: (n_verbs, dir?, struct?, count?)
SCHED = [("V", "V", "D", "C"), ("V", "V", "D", "S"), ("V", "V", "D", "S"),
         ("V", "V", "D", "C"), ("V", "V", "S", "C"), ("V", "V", "S", "C")]
# AR token space: forms 0-23, actions 24-31, PAD 32, BOS 33, SEP 34
T_ACT0, T_PAD, T_BOS, T_SEP = 24, 32, 33, 34
VOCAB = 35

# templates: mode_case {0 none,1 d-pre,2 opp,3 around} x cnt {1,2,3}
# cell values: 0 = verb slot, 1 = dir slot, 2 = pad
def build_templates():
    base = {0: [0], 1: [1, 0], 2: [1, 1, 0], 3: [1, 0] * 4}
    t = torch.full((4, 3, OUTM), 2, dtype=torch.long)
    for mc, blk in base.items():
        for c in (1, 2, 3):
            seq = blk * c
            t[mc, c - 1, :len(seq)] = torch.tensor(seq)
    return t


TMPL = build_templates()


def exec_cmd(v_act, d_act, mode_case, cnt):
    """All args (B,) long -> outputs (B, OUTM). Frozen kernel."""
    t = TMPL.to(v_act.device)[mode_case, cnt - 1]           # (B, OUTM)
    out = torch.where(t == 0, v_act.unsqueeze(1),
                      torch.where(t == 1, d_act.unsqueeze(1),
                                  torch.full_like(t, PAD_CLS)))
    return out


def make_lifetime(E, device):
    maps = {"v": torch.randint(0, NA, (E, N_V), device=device),
            "d": torch.randint(0, NA, (E, N_D), device=device),
            "s": torch.randint(0, 2, (E, N_S), device=device),   # 0 opp 1 arnd
            "c": torch.randint(0, 2, (E, N_C), device=device)}   # 0->2 1->3
    perm = {k: torch.rand(E, n, device=device).argsort(1)
            for k, n in (("v", N_V), ("d", N_D), ("s", N_S), ("c", N_C))}
    return maps, perm


def sample_cmd(E, device, maps, vi, di, si, ci, has_d, has_s, has_c):
    """Ground-truth execution for sampled component indices."""
    v_act = maps["v"].gather(1, vi.unsqueeze(1)).squeeze(1)
    d_act = maps["d"].gather(1, di.unsqueeze(1)).squeeze(1)
    mode = maps["s"].gather(1, si.unsqueeze(1)).squeeze(1)
    cnt2 = maps["c"].gather(1, ci.unsqueeze(1)).squeeze(1) + 2
    mode_case = torch.where(has_s, mode + 2,
                            torch.where(has_d, torch.ones_like(mode),
                                        torch.zeros_like(mode)))
    cnt = torch.where(has_c, cnt2, torch.ones_like(cnt2))
    return exec_cmd(v_act, d_act, mode_case, cnt)


class Kernel:
    """The gradient-free walker: fixed random keys/prototypes, exact
    book, exact hypothesis elimination for struct/count words."""

    def __init__(self, E, device, d=64, seed=7):
        g = torch.Generator().manual_seed(seed)
        self.F = F.normalize(torch.randn(NF, d, generator=g), dim=-1) \
            .to(device)
        self.A = F.normalize(torch.randn(NA, d, generator=g), dim=-1) \
            .to(device)
        self.atom = torch.full((E, NF), -1, dtype=torch.long,
                               device=device)   # decoded action per form
        self.post_s = torch.full((E, N_S), -1, dtype=torch.long,
                                 device=device)
        self.post_c = torch.full((E, N_C), -1, dtype=torch.long,
                                 device=device)

    def teach_atom(self, form_idx, act):
        ar = torch.arange(form_idx.shape[0], device=form_idx.device)
        self.atom[ar, form_idx] = act        # book write + exact decode

    def teach_demo_s(self, s_local, demo_out, vi, di):
        """Eliminate: execute both modes with decoded atoms, keep match."""
        E = s_local.shape[0]
        ar = torch.arange(E, device=s_local.device)
        v = self.atom[ar, vi]
        d = self.atom[ar, di + N_V]
        ok = []
        for m in (0, 1):
            pred = exec_cmd(v, d, torch.full_like(v, m + 2),
                            torch.ones_like(v))
            ok.append((pred == demo_out).all(1))
        choice = torch.where(ok[1] & ~ok[0], torch.ones_like(s_local),
                             torch.zeros_like(s_local))
        self.post_s[ar, s_local] = choice

    def teach_demo_c(self, c_local, demo_out, vi):
        E = c_local.shape[0]
        ar = torch.arange(E, device=c_local.device)
        v = self.atom[ar, vi]
        ok = []
        for cv in (2, 3):
            pred = exec_cmd(v, torch.zeros_like(v), torch.zeros_like(v),
                            torch.full_like(v, cv))
            ok.append((pred == demo_out).all(1))
        choice = torch.where(ok[1] & ~ok[0], torch.ones_like(c_local),
                             torch.zeros_like(c_local))
        self.post_c[ar, c_local] = choice

    def answer(self, vi, di, si, ci, has_d, has_s, has_c):
        E = vi.shape[0]
        ar = torch.arange(E, device=vi.device)
        v = self.atom[ar, vi]
        d = self.atom[ar, di + N_V]
        mode = self.post_s[ar, si]
        cnt = self.post_c[ar, ci] + 2
        mode_case = torch.where(has_s, mode.clamp(min=0) + 2,
                                torch.where(has_d, torch.ones_like(mode),
                                            torch.zeros_like(mode)))
        cnt = torch.where(has_c, cnt, torch.ones_like(cnt))
        return exec_cmd(v.clamp(min=0), d.clamp(min=0), mode_case, cnt)


# ---------------------------------------------------------------------------
# Lifetime script (shared by kernel and AR): study stream + eval queries
# ---------------------------------------------------------------------------

def run_lifetime_kernel(E, device, eval_only_last=True):
    maps, perm = make_lifetime(E, device)
    ker = Kernel(E, device)
    taught_v = []
    taught_d = []
    ar = torch.arange(E, device=device)
    vp, dp, sp, cp = 0, 0, 0, 0
    for e in range(EPS):
        for cls in SCHED[e]:
            if cls == "V":
                local = perm["v"][:, vp]
                vp += 1
                ker.teach_atom(local, maps["v"][ar, local])
                taught_v.append(local)
            elif cls == "D":
                local = perm["d"][:, dp]
                dp += 1
                ker.teach_atom(local + N_V, maps["d"][ar, local])
                taught_d.append(local)
            elif cls == "S":
                local = perm["s"][:, sp]
                sp += 1
                vi = taught_v[torch.randint(len(taught_v), (1,)).item()]
                di = taught_d[torch.randint(len(taught_d), (1,)).item()]
                demo = sample_cmd(E, device, maps, vi, di, local,
                                  torch.zeros_like(local),
                                  torch.ones_like(local).bool(),
                                  torch.ones_like(local).bool(),
                                  torch.zeros_like(local).bool())
                ker.teach_demo_s(local, demo, vi, di)
            else:
                local = perm["c"][:, cp]
                cp += 1
                vi = taught_v[torch.randint(len(taught_v), (1,)).item()]
                demo = sample_cmd(E, device, maps, vi,
                                  torch.zeros_like(local), local, local,
                                  torch.zeros_like(local).bool(),
                                  torch.zeros_like(local).bool(),
                                  torch.ones_like(local).bool())
                ker.teach_demo_c(local, demo, vi)
    # eval: all frames, uniform over ALL forms (most verbs never demoed)
    res = {}
    frames = {"V": (0, 0, 0), "VD": (1, 0, 0), "VC": (0, 0, 1),
              "VDC": (1, 0, 1), "VSD": (1, 1, 0), "VSDC": (1, 1, 1)}
    for name, (hd, hs, hc) in frames.items():
        vi = torch.randint(0, N_V, (E,), device=device)
        di = torch.randint(0, N_D, (E,), device=device)
        si = torch.randint(0, N_S, (E,), device=device)
        ci = torch.randint(0, N_C, (E,), device=device)
        hdt = torch.full((E,), bool(hd), dtype=torch.bool, device=device)
        hst = torch.full((E,), bool(hs), dtype=torch.bool, device=device)
        hct = torch.full((E,), bool(hc), dtype=torch.bool, device=device)
        tgt = sample_cmd(E, device, maps, vi, di, si, ci, hdt, hst, hct)
        out = ker.answer(vi, di, si, ci, hdt, hst, hct)
        res[name] = float((out == tgt).all(1).float().mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    agg = {}
    t0 = time.time()
    for _ in range(args.eval_batches):
        r = run_lifetime_kernel(args.eval_batch, device)
        for k, v in r.items():
            agg[k] = agg.get(k, 0.0) + v / args.eval_batches
    print(f"SCAN-lite, frozen kernel, ZERO trained parameters "
          f"({time.time() - t0:.0f}s):")
    print("  exact-match by frame: "
          + "  ".join(f"{k}={v * 100:.1f}" for k, v in agg.items()))


if __name__ == "__main__":
    main()
