"""SimpleCore — base implementation of RCORE_SPEC v1 (2026-07-23).

One dense decoder-only LM, cut once: F (lower dense layers) -> CORE
-> G (upper dense layers). The core is a small dense transformer
over a working set of SELECTED F-states, with self-directed
selection: dedicated GAZE TOKENS ride the core's own attention
layers and their outputs are the next loop's queries (they can
express "what is missing," not just "what resembles what I hold").
Gaze tokens and working set both persist across sessions (chunk
cadence); S loops per session.

Delivery (spec section 3 compile): each session's K member states
are placed as SLOT TOKENS at their session's boundary inside ONE
interleaved causal sequence [ ... chunk u-1 || slots_u || chunk u
... ] — single parallel G pass, honest causal semantics, no
cross-attention module. Slots carry their boundary's position
embedding + a learned slot-type embedding. Loss on real tokens only.

IDENTITY PROPERTY (the smoke test): with slot columns masked out of
G's attention, SimpleCore's real-token logits equal the same
weights run with no core at all — bit-exact dense. Every deviation
from dense is attributable to the core.

BASE = NO restrictions (F and G fully dense causal — the core is a
redundant-but-present channel; restrictions that make it
load-bearing come later per spec section 5), NO scaffolding (no
answer boost, no gaze aux, no curricula — plain mix; pressure knobs
are diagnosis tools now, not defaults).

Twins: simplecore / dense (same trunk, no core, no slots).
Reduced vocab: union of TinyStories-pool tokens + the bank world's
vocab (~15k) — ~11M params, ~3x cheaper loss head than full GPT-2.

Telemetry (passive, eval-only): bufhit@question, gaze fact-frac,
admission recency. Metrics: story ppl, far/near h1/h2.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_stream_icl import Block
import rcore_lm as R
from rcore_lm import (build_story_pool, setup_world, build_real_batch,
                      grade, GKEYS, buffer_hit, gaze_stats, EOT)


# ---------------------------------------------------------------------------
# Reduced vocabulary
# ---------------------------------------------------------------------------

def build_reduced_vocab(pool, SW, cache="vocab_simplecore.json"):
    """LUT (50257,) mapping GPT-2 ids -> reduced ids. Coverage =
    story-pool uniques + the bank world's own vocab file (bank
    templates, attribute words, nonce syllables — stream_text_v2
    build_vocab's coverage set)."""
    if os.path.exists(cache):
        with open(cache) as f:
            ids = json.load(f)
    else:
        pool_ids = set(int(x) for x in np.unique(np.asarray(pool)))
        import stream_text_v2 as SWmod
        # bank coverage: reuse its vocab builder's id set
        if os.path.exists("vocab_text_v2.json"):
            with open("vocab_text_v2.json") as f:
                bank_ids = set(json.load(f))
        else:
            saved_remap, saved_nv = SWmod._REMAP, SWmod._NVOCAB
            SWmod.build_vocab("vocab_text_v2.json")
            with open("vocab_text_v2.json") as f:
                bank_ids = set(json.load(f))
            SWmod._REMAP, SWmod._NVOCAB = saved_remap, saved_nv
        ids = sorted(pool_ids | bank_ids | {EOT, R.NL})
        with open(cache, "w") as f:
            json.dump(ids, f)
    lut = np.zeros(50257, dtype=np.int64)   # unk -> 0 (rare)
    for i, g in enumerate(ids):
        lut[g] = i + 1
    return torch.from_numpy(lut), len(ids) + 1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Core(nn.Module):
    """Small dense transformer over [working set || gaze tokens].
    Gaze-token outputs are the next loop's selection queries."""

    def __init__(self, d, heads, layers=2, k_set=32, n_gaze=4,
                 s_loops=2, mode="accumulate", policy="learned",
                 explore=0.125):
        super().__init__()
        assert mode in ("accumulate", "replace")
        assert policy in ("learned", "random")
        self.d, self.K, self.Q = d, k_set, n_gaze
        self.S, self.mode, self.policy = s_loops, mode, policy
        self.explore = explore
        self.blocks = nn.ModuleList(Block(d, heads)
                                    for _ in range(layers))
        self.key_ln = nn.LayerNorm(d)
        self.key_proj = nn.Linear(d, d)     # per-token, no mixing
        self.q_proj = nn.Linear(d, d)
        self.gaze0 = nn.Parameter(torch.randn(1, n_gaze, d) * 0.02)
        self.members0 = nn.Parameter(torch.randn(1, k_set, d) * 0.02)
        self.scale = 1.0 / math.sqrt(d)

    def init_state(self, B):
        return (self.members0.expand(B, -1, -1),
                self.gaze0.expand(B, -1, -1))

    def session(self, cand, keys, members, gaze, policy=None,
                training=False):
        """cand/keys: (B, n, d) F-states so far + their score keys.
        Returns updated (members, gaze), plus admitted indices of
        the LAST loop (telemetry)."""
        pol = policy or self.policy
        B, n, _ = cand.shape
        k_admit = max(1, self.K // self.S) \
            if self.mode == "accumulate" else self.K
        idx = None
        for _ in range(self.S):
            q = self.q_proj(gaze)
            s = torch.einsum("bqd,bnd->bqn", q, keys) * self.scale
            s = s.max(1).values
            if pol == "random":
                key = torch.rand_like(s)
            else:
                key = s.detach().float()
            k = min(k_admit, n)
            idx = key.topk(k, dim=1).indices
            if pol == "learned" and training and self.explore > 0:
                n_exp = max(1, int(k * self.explore))
                rnd = torch.rand_like(key)
                rnd.scatter_(1, idx, -1.0)
                idx = torch.cat(
                    [idx[:, :k - n_exp],
                     rnd.topk(n_exp, dim=1).indices], dim=1)
            sel = cand.gather(
                1, idx.unsqueeze(-1).expand(-1, -1, self.d))
            if pol == "learned":
                gate = torch.sigmoid(s.gather(1, idx)).unsqueeze(-1)
                sel = sel * (1 + gate)
            if self.mode == "accumulate":
                members = torch.cat([members, sel], dim=1)[:, -self.K:]
            else:
                members = sel
            x = torch.cat([members, gaze], dim=1)
            for blk in self.blocks:
                x = blk(x, None)
            members, gaze = x[:, :members.shape[1]], \
                x[:, members.shape[1]:]
        if members.shape[1] < self.K:       # pad early sessions
            members = torch.cat(
                [self.members0.expand(B, -1, -1)
                 [:, :self.K - members.shape[1]], members], dim=1)
        return members, gaze, idx


class SimpleCore(nn.Module):
    def __init__(self, vocab, d=256, f_layers=3, g_layers=5,
                 heads=8, max_t=2048, chunk=64, use_core=True,
                 k_set=32, n_gaze=4, s_loops=2, core_layers=2,
                 mode="accumulate", policy="learned"):
        super().__init__()
        assert max_t % chunk == 0
        self.d, self.C, self.use_core = d, chunk, use_core
        self.K = k_set
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos = nn.Parameter(torch.randn(1, max_t, d) * 0.02)
        self.f = nn.ModuleList(Block(d, heads)
                               for _ in range(f_layers))
        self.g = nn.ModuleList(Block(d, heads)
                               for _ in range(g_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        if use_core:
            self.core = Core(d, heads, core_layers, k_set, n_gaze,
                             s_loops, mode, policy)
            self.slot_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self._masks = {}

    def _causal(self, L, device):
        key = (L, device)
        if key not in self._masks:
            i = torch.arange(L, device=device)
            self._masks[key] = ~(i.unsqueeze(0) <= i.unsqueeze(1))
        return self._masks[key]

    def _interleave_ix(self, T, device):
        """Static index maps for the interleaved G sequence:
        [c0 tokens | slots_1 | c1 tokens | slots_2 | ... ]."""
        key = ("ix", T, device)
        if key not in self._masks:
            nch = T // self.C
            real_ix, slot_at = [], []
            L = 0
            for u in range(nch):
                if u > 0:
                    slot_at.append(L)
                    L += self.K
                real_ix += list(range(L, L + self.C))
                L += self.C
            self._masks[key] = (
                torch.tensor(real_ix, device=device), slot_at, L)
        return self._masks[key]

    def forward(self, toks, policy=None, collect=False,
                mask_slots=False):
        B, T = toks.shape
        dev = toks.device
        x = self.emb(toks) + self.pos[:, :T]
        cm = self._causal(T, dev)
        for blk in self.f:
            x = blk(x, cm)
        if not self.use_core:
            y = x
            for blk in self.g:
                y = blk(y, cm)
            return self.head(self.norm(y)), []
        nch = T // self.C
        keys = self.core.key_proj(self.core.key_ln(x))
        members, gaze = self.core.init_state(B)
        slots, admits = [], []
        for u in range(1, nch):
            n = u * self.C
            members, gaze, idx = self.core.session(
                x[:, :n], keys[:, :n], members, gaze,
                policy=policy, training=self.training)
            slots.append(members + self.slot_type
                         + self.pos[:, n:n + 1])
            if collect:
                admits.append(idx.detach().cpu())
        real_ix, slot_at, L = self._interleave_ix(T, dev)
        y = torch.zeros(B, L, self.d, device=dev, dtype=x.dtype)
        y[:, real_ix] = x
        for s_i, pos0 in enumerate(slot_at):
            y[:, pos0:pos0 + self.K] = slots[s_i]
        gm = self._causal(L, dev)
        if mask_slots:                      # identity test: slots
            gm = gm.clone()                 # invisible to attention
            for pos0 in slot_at:
                gm[:, pos0:pos0 + self.K] = True
            gm.fill_diagonal_(False)
        for blk in self.g:
            y = blk(y, gm)
        logits = self.head(self.norm(y[:, real_ix]))
        return logits, admits


# ---------------------------------------------------------------------------
# Identity smoke test
# ---------------------------------------------------------------------------

def identity_test(device="cpu"):
    """Slots masked out of G's attention == no core at all,
    bit-exact on real-token logits (spec's anchor property)."""
    torch.manual_seed(0)
    m = SimpleCore(500, d=64, f_layers=2, g_layers=2, heads=4,
                   max_t=256, chunk=32, k_set=8, n_gaze=2,
                   s_loops=2, core_layers=1).to(device).eval()
    toks = torch.randint(0, 500, (2, 256), device=device)
    with torch.no_grad():
        la, _ = m(toks, mask_slots=True)
        m.use_core = False
        lb, _ = m(toks)
        m.use_core = True
    diff = (la - lb).abs().max().item()
    ok = diff < 1e-4
    print(f"[identity] masked-slots vs no-core max-diff {diff:.2e} "
          f"{'PASS' if ok else 'FAIL'}")
    # grad-flow: scorer + gaze receive gradient
    m.train()
    logits, _ = m(toks)
    logits.mean().backward()
    gq = m.core.q_proj.weight.grad.abs().sum().item()
    gg = m.core.gaze0.grad.abs().sum().item()
    ok2 = gq > 0 and gg > 0
    print(f"[identity] grad-flow q {gq:.2e} gaze {gg:.2e} "
          f"{'PASS' if ok2 else 'FAIL'}")
    return ok and ok2


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def make_model(arm, args, vocab, device):
    torch.manual_seed(args.seed)
    return SimpleCore(
        vocab, d=args.d, f_layers=args.f_layers,
        g_layers=args.g_layers, heads=args.heads, max_t=args.t,
        chunk=args.c, use_core=(arm != "dense"), k_set=args.k,
        n_gaze=args.n_gaze, s_loops=args.s, core_layers=args.core_layers,
        mode=args.core_mode,
        policy=("random" if arm == "random" else "learned")
    ).to(device)


def train_arm(model, SW, pool, lut, args, arm, device, log_fn):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)))
    model.train()
    rng = random.Random(1234 + args.seed)
    # base is dense (no windows) so far/near has no leak meaning yet;
    # the flag still buckets questions by distance for later stages
    far_gap = args.far_gap
    t0 = time.time()
    for step in range(1, args.steps + 1):
        toks, fmask, ans_tgt, qs = build_real_batch(
            SW, args.batch, args.t, rng, pool, args.fact_frac,
            device, n_stream_q=args.n_stream_q, n_quiz=args.n_quiz,
            stmts=args.stmts, filler_frac=args.filler_frac,
            far_gap=far_gap)
        toks = lut.to(device)[toks]
        inp, tgt = toks[:, :-1], toks[:, 1:]
        collect = (step % args.log_every == 0)
        with torch.autocast(device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            logits, admits = model(inp, collect=collect)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if collect or step == 1:
            g = grade(logits, toks, qs)
            n_fr = int(round(args.batch * args.fact_frac))
            gz, rec = gaze_stats(admits, fmask, n_fr, args.c)
            h1 = g["h1_far"][0] / max(g["h1_far"][1], 1)
            h2 = g["h2_far"][0] / max(g["h2_far"][1], 1)
            print(f"[{arm:10s}] step {step:5d} loss {loss:.3f} "
                  f"h1f {h1:.2f} h2f {h2:.2f} gaze {gz:.3f} "
                  f"rec {rec:.2f} ({time.time()-t0:.0f}s)",
                  flush=True)
            if log_fn:
                log_fn({f"{arm}/loss": float(loss), f"{arm}/h1": h1,
                        f"{arm}/h2": h2, f"{arm}/gaze": gz,
                        f"{arm}/step": step})


@torch.no_grad()
def evaluate(model, SW, pool_val, pool, lut, args, device,
             policy=None):
    model.eval()
    out = {}
    lutd = lut.to(device)
    rng = random.Random(4242)
    ce = n = 0.0
    for _ in range(args.eval_story_batches):
        toks, _, _, _ = build_real_batch(
            SW, args.batch, args.t, rng, pool_val, 0.0, device)
        toks = lutd[toks]
        inp, tgt = toks[:, :-1], toks[:, 1:]
        with torch.autocast(device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            logits, _ = model(inp, policy=policy)
        ce += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            tgt.reshape(-1)))
        n += 1
    out["lm_loss"] = ce / max(n, 1)
    out["ppl"] = math.exp(out["lm_loss"])
    for part in ("train", "hold"):
        rng = random.Random(999)
        agg = {k: [0, 0] for k in GKEYS}
        bh_s = bh_n = gz_s = gz_c = 0.0
        for _ in range(args.eval_fact_batches):
            toks, fmask, _, qs = build_real_batch(
                SW, args.batch, args.t, rng, pool, 1.0, device,
                bank_part=part, n_stream_q=args.n_stream_q,
                n_quiz=args.n_quiz, stmts=args.stmts,
                filler_frac=args.filler_frac, all_fact=True,
                far_gap=args.far_gap)
            toks = lutd[toks]
            inp = toks[:, :-1]
            with torch.autocast(device_type="cuda",
                                dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                logits, admits = model(inp, policy=policy,
                                       collect=True)
            g = grade(logits, toks, qs)
            for k in agg:
                agg[k][0] += g[k][0]
                agg[k][1] += g[k][1]
            bh, bn = buffer_hit(admits, qs, args.c)
            bh_s += bh * bn
            bh_n += bn
            gz, _ = gaze_stats(admits, fmask, args.batch, args.c)
            gz_s += gz
            gz_c += 1
        sfx = "" if part == "train" else "_hold"
        for k in GKEYS:
            out[f"{k}{sfx}"] = agg[k][0] / max(agg[k][1], 1)
        out[f"bufhit{sfx}"] = bh_s / max(bh_n, 1)
        if part == "train":
            out["gaze_fact"] = gz_s / max(gz_c, 1)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--t", type=int, default=2048)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--s", type=int, default=2)
    ap.add_argument("--n-gaze", type=int, default=4)
    ap.add_argument("--core-layers", type=int, default=2)
    ap.add_argument("--core-mode", type=str, default="accumulate")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--f-layers", type=int, default=3)
    ap.add_argument("--g-layers", type=int, default=5)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--fact-frac", type=float, default=0.2)
    ap.add_argument("--n-stream-q", type=int, default=8)
    ap.add_argument("--n-quiz", type=int, default=10)
    ap.add_argument("--stmts", type=int, default=2)
    ap.add_argument("--filler-frac", type=float, default=0.3)
    ap.add_argument("--n-c", type=int, default=3)
    ap.add_argument("--n-p", type=int, default=6)
    ap.add_argument("--min-gap", type=int, default=45)
    ap.add_argument("--far-gap", type=int, default=512)
    ap.add_argument("--pool-tokens", type=int, default=120_000_000)
    ap.add_argument("--val-pool-tokens", type=int, default=2_000_000)
    ap.add_argument("--arms", type=str, default="simplecore,dense")
    ap.add_argument("--eval-story-batches", type=int, default=8)
    ap.add_argument("--eval-fact-batches", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-prefix", type=str, default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str,
                    default="neocore-simplecore")
    args = ap.parse_args()

    if args.identity:
        ok = identity_test("cpu")
        raise SystemExit(0 if ok else 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    SW = setup_world(args)
    pool = build_story_pool(args.pool_tokens, "train",
                            "stories_train.npy")
    pool_val = build_story_pool(args.val_pool_tokens, "validation",
                                "stories_val.npy")
    lut, vocab = build_reduced_vocab(pool, SW)
    print(f"device={device} reduced vocab={vocab}", flush=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"sc-{args.arms.replace(',', '-')}"
                              f"-s{args.steps}",
                         config=vars(args))
    log_fn = (lambda x: run.log(x)) if run else None

    models, results = {}, {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        model = make_model(arm, args, vocab, device)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"\n=== arm {arm}: {n_par/1e6:.1f}M params", flush=True)
        train_arm(model, SW, pool, lut, args, arm, device, log_fn)
        models[arm] = model
        if args.save_prefix:
            torch.save(model.state_dict(),
                       f"{args.save_prefix}_{arm}.pt")
        results[arm] = evaluate(model, SW, pool_val, pool, lut,
                                args, device)
        if arm == "simplecore":
            results["simplecore-randgaze"] = evaluate(
                models[arm], SW, pool_val, pool, lut, args, device,
                policy="random")

    print("\n=== SIMPLECORE RESULTS")
    names = list(results)
    keys = ("ppl", "h1_far", "h2_far", "h1_near", "h2_near",
            "h1_far_hold", "bufhit", "gaze_fact")
    print("  metric        " + "  ".join(f"{n:>20s}" for n in names))
    for k in keys:
        row = "  ".join(f"{results[n].get(k, float('nan')):20.3f}"
                        for n in names)
        print(f"  {k:>12s}  {row}")

    if run:
        for a, st in results.items():
            for k, v in st.items():
                run.summary[f"{a}_{k}"] = v
        with open("simplecore_summary.json", "w") as f:
            json.dump(results, f, indent=1)
        import wandb as wb
        art = wb.Artifact(f"simplecore-{run.id}", type="results")
        art.add_file("simplecore_summary.json")
        if args.save_prefix:
            for arm in models:
                art.add_file(f"{args.save_prefix}_{arm}.pt")
        run.log_artifact(art).wait()
        run.finish()
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
