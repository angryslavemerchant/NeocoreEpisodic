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


# ---------------------------------------------------------------------------
# AR contrast arm: full study stream in context, teacher-forced
# ---------------------------------------------------------------------------

def gen_study_and_tokens(E, device):
    """One lifetime's full study protocol, both as kernel teachings and
    as a flat token stream for the AR model. Returns maps + tokens."""
    maps, perm = make_lifetime(E, device)
    ker = Kernel(E, device)
    ar = torch.arange(E, device=device)
    toks = []
    taught_v, taught_d = [], []
    vp, dp, sp, cp = 0, 0, 0, 0
    for e in range(EPS):
        for cls in SCHED[e]:
            if cls == "V" or cls == "D":
                if cls == "V":
                    local = perm["v"][:, vp]; vp += 1
                    form = local
                    act = maps["v"][ar, local]
                    taught_v.append(local)
                else:
                    local = perm["d"][:, dp]; dp += 1
                    form = local + N_V
                    act = maps["d"][ar, local]
                    taught_d.append(local)
                ker.teach_atom(form, act)
                toks += [form, act + T_ACT0,
                         torch.full_like(form, T_SEP)]
            elif cls == "S":
                local = perm["s"][:, sp]; sp += 1
                vi = taught_v[torch.randint(len(taught_v), (1,)).item()]
                di = taught_d[torch.randint(len(taught_d), (1,)).item()]
                demo = sample_cmd(E, device, maps, vi, di, local,
                                  torch.zeros_like(local),
                                  torch.ones_like(local).bool(),
                                  torch.ones_like(local).bool(),
                                  torch.zeros_like(local).bool())
                ker.teach_demo_s(local, demo, vi, di)
                out8 = demo[:, :8]
                toks += [vi, local + S0, di + N_V]
                toks += [torch.where(out8[:, t] < NA, out8[:, t] + T_ACT0,
                                     torch.full_like(vi, T_PAD))
                         for t in range(8)]
                toks += [torch.full_like(vi, T_SEP)]
            else:
                local = perm["c"][:, cp]; cp += 1
                vi = taught_v[torch.randint(len(taught_v), (1,)).item()]
                demo = sample_cmd(E, device, maps, vi,
                                  torch.zeros_like(local), local, local,
                                  torch.zeros_like(local).bool(),
                                  torch.zeros_like(local).bool(),
                                  torch.ones_like(local).bool())
                ker.teach_demo_c(local, demo, vi)
                out3 = demo[:, :3]
                toks += [vi, local + C0]
                toks += [torch.where(out3[:, t] < NA, out3[:, t] + T_ACT0,
                                     torch.full_like(vi, T_PAD))
                         for t in range(3)]
                toks += [torch.full_like(vi, T_SEP)]
    study = torch.stack(toks, dim=1)                        # (E, ~120)
    return maps, ker, study


def gen_eval_queries(E, device, maps):
    frames = {"V": (0, 0, 0), "VD": (1, 0, 0), "VC": (0, 0, 1),
              "VDC": (1, 0, 1), "VSD": (1, 1, 0), "VSDC": (1, 1, 1)}
    out = {}
    for name, (hd, hs, hc) in frames.items():
        vi = torch.randint(0, N_V, (E,), device=device)
        di = torch.randint(0, N_D, (E,), device=device)
        si = torch.randint(0, N_S, (E,), device=device)
        ci = torch.randint(0, N_C, (E,), device=device)
        hdt = torch.full((E,), bool(hd), dtype=torch.bool, device=device)
        hst = torch.full((E,), bool(hs), dtype=torch.bool, device=device)
        hct = torch.full((E,), bool(hc), dtype=torch.bool, device=device)
        tgt = sample_cmd(E, device, maps, vi, di, si, ci, hdt, hst, hct)
        pad = torch.full_like(vi, T_PAD)
        q = torch.stack([vi,
                         torch.where(hst, si + S0, pad),
                         torch.where(hdt, di + N_V, pad),
                         torch.where(hct, ci + C0, pad)], dim=1)
        out[name] = (vi, di, si, ci, hdt, hst, hct, q, tgt)
    return out


class ARScan(nn.Module):
    def __init__(self, base_len, d=96, layers=3, heads=4):
        super().__init__()
        import math
        self.d = d
        self.base = base_len + 4                # study + query
        seq = self.base + 1 + OUTM              # BOS + answers
        pos = torch.zeros(seq, d)
        t = torch.arange(seq).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float()
                        * (-math.log(10000.0) / d))
        pos[:, 0::2] = torch.sin(t * div)
        pos[:, 1::2] = torch.cos(t * div)
        self.register_buffer("pos", pos * 0.3)
        allowed = torch.zeros(seq, seq, dtype=torch.bool)
        allowed[:, :self.base] = True
        for i in range(self.base, seq):
            allowed[i, self.base:i + 1] = True
        self.register_buffer("mask", ~allowed)
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList(
            nn.ModuleDict({}) for _ in range(0))  # placeholder unused
        from toy_composer_icl import MBlock
        self.blks = nn.ModuleList(MBlock(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, PAD_CLS + 1)

    def _run(self, study, q, ans_in):
        x = self.emb(torch.cat([study, q, ans_in], dim=1))
        x = x + self.pos[: x.shape[1]]
        m = self.mask[: x.shape[1], : x.shape[1]]
        for b in self.blks:
            x = b(x, mask=m)
        return self.head(self.norm(x[:, self.base:]))

    def loss(self, study, q, tgt):
        bos = torch.full((tgt.shape[0], 1), T_BOS, dtype=torch.long,
                         device=tgt.device)
        prev = torch.where(tgt[:, :-1] < NA, tgt[:, :-1] + T_ACT0,
                           torch.full_like(tgt[:, :-1], T_PAD))
        logits = self._run(study, q, torch.cat([bos, prev], dim=1))
        return F.cross_entropy(logits.reshape(-1, PAD_CLS + 1),
                               tgt.reshape(-1))

    @torch.no_grad()
    def generate(self, study, q):
        E = study.shape[0]
        dev = study.device
        ans = torch.full((E, OUTM), T_PAD, dtype=torch.long, device=dev)
        out = torch.full((E, OUTM), PAD_CLS, dtype=torch.long, device=dev)
        cur = torch.full((E, 1), T_BOS, dtype=torch.long, device=dev)
        for t in range(OUTM):
            ans_in = torch.cat([cur, torch.where(
                out[:, :OUTM - 1] < NA, out[:, :OUTM - 1] + T_ACT0,
                torch.full_like(out[:, :OUTM - 1], T_PAD))], dim=1)
            lg = self._run(study, q, ans_in)
            out[:, t] = lg[:, t].argmax(-1)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=str, default="kernel",
                    choices=["kernel", "ar", "both"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-prefix", type=str, default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="neocore-lex")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"scanlite-{args.arm}-s{args.steps}",
                         config=vars(args))
    results = {}

    if args.arm in ("kernel", "both"):
        agg = {}
        t0 = time.time()
        for _ in range(args.eval_batches):
            r = run_lifetime_kernel(args.eval_batch, device)
            for k, v in r.items():
                agg[k] = agg.get(k, 0.0) + v / args.eval_batches
        print(f"kernel ({time.time() - t0:.0f}s): "
              + "  ".join(f"{k}={v * 100:.1f}" for k, v in agg.items()),
              flush=True)
        results["kernel"] = agg

    if args.arm in ("ar", "both"):
        base = gen_study_and_tokens(2, device)[2].shape[1]
        model = ARScan(base).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=0.01)
        t0 = time.time()
        for step in range(1, args.steps + 1):
            maps, _, study = gen_study_and_tokens(args.batch, device)
            qs = gen_eval_queries(args.batch, device, maps)
            loss = 0.0
            for name, (_, _, _, _, _, _, _, q, tgt) in qs.items():
                loss = loss + model.loss(study, q, tgt)
            loss = loss / len(qs)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step == 1 or step % 200 == 0:
                print(f"[ar] step {step:5d}  loss {loss.item():.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
                if run:
                    run.log({"ar/loss": loss.item(), "ar/step": step})
        agg = {}
        with torch.no_grad():
            for _ in range(args.eval_batches):
                maps, _, study = gen_study_and_tokens(args.eval_batch // 2,
                                                      device)
                qs = gen_eval_queries(args.eval_batch // 2, device, maps)
                for name, (_, _, _, _, _, _, _, q, tgt) in qs.items():
                    out = model.generate(study, q)
                    agg[name] = agg.get(name, 0.0) + float(
                        (out == tgt).all(1).float().mean()) \
                        / args.eval_batches
        print("ar:      "
              + "  ".join(f"{k}={v * 100:.1f}" for k, v in agg.items()),
              flush=True)
        results["ar"] = agg
        if args.save_prefix:
            torch.save(model.state_dict(), args.save_prefix + "_ar.pt")

    if run:
        import wandb
        for arm, r in results.items():
            for k, v in r.items():
                run.summary[f"{arm}_{k}"] = v
        with open("results.json", "w") as f:
            json.dump(results, f, indent=1)
        art = wandb.Artifact(f"scanlite-{run.id}", type="results")
        art.add_file("results.json")
        if args.save_prefix and os.path.exists(args.save_prefix + "_ar.pt"):
            art.add_file(args.save_prefix + "_ar.pt")
        run.log_artifact(art)
        art.wait()
        print("ARTIFACT_VERIFIED", flush=True)
        run.finish()


if __name__ == "__main__":
    main()
