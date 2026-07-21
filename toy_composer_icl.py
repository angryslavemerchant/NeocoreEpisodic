"""Toy: the ITERATIVE COMPOSER (2026 — successor to toy_lexicon_icl v3).

v3 measured the productivity failure: a parallel-slot template composer
trained on 2-word queries scores 0.0 exact on 3-word ones (oracle ==
live, so the book/lexicon is exonerated — the COMPOSER is the wall).
This toy tests the repair, as a contrast of two ownerships of
iteration:

  AR      iteration as HOPE: same tower + book, answers emitted one
          token at a time (teacher-forced causal training, fed-back
          sampling at test). Does sequential emission alone let SGD
          discover the counting rule?
  WALKER  iteration as CIRCUIT: a hardcoded walk — cursor over query
          primitives, repeat counter, advance-on-zero, stop-at-end.
          Nothing about length is learnable. Learned pieces only:
          payload -> output-token head, and the modifier interpreter
          (TWICE -> counter=2) trained by EXACT soft-alignment
          (enumerate the <=2^n repeat combinations, expected CE).
          The walker needs no transformer at all at query time.

Both train ONLY on 2-primitive queries; eval poses 2, 3 AND 4-
primitive ones (outputs to 8 tokens). Pre-registered: walker 100 at
every length by construction; AR above the template's floor, below
ceiling, degrading with length. If so: the same law that governed
memory reads governs composition — built circuits generalize, hoped
circuits template-ize. Note what the walker's modifier interpreter
is: the first function word whose semantics is an OPERATION on a
machine (counter := 2), not a vector — rung 2's smallest specimen.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_lexicon_icl import (EPS, P, PAD_CLS, QN, SPE, TOK_NOOP, TOK_PAD,
                             TOK_TWICE, VOCAB, LexBook, ctx_episode,
                             make_lifetime)

OUTM = 8                # up to 4 primitives x 2 reps
QMAX = 8                # query region: 4 x (prim, mod)
TOK_BOS = VOCAB         # AR answer-region start token
VOCAB2 = VOCAB + 1


def sinpos(seq, d):
    pos = torch.zeros(seq, d)
    t = torch.arange(seq).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float()
                    * (-math.log(10000.0) / d))
    pos[:, 0::2] = torch.sin(t * div)
    pos[:, 1::2] = torch.cos(t * div)
    return pos * 0.3


def gen_queries(lex, sched, e, device, n):
    """n-primitive queries over everything studied so far. Returns
    prims (E,QN,n), mods (E,QN,n), q_tokens (E,QN,QMAX), tgt
    (E,QN,OUTM)."""
    E = lex.shape[0]
    pool = sched[:, :e + 1].reshape(E, (e + 1) * SPE)
    idx = torch.randint(0, pool.shape[1], (E, QN * n), device=device)
    prims = pool.gather(1, idx).reshape(E, QN, n)
    mods = torch.randint(0, 2, (E, QN, n), device=device)
    parts = []
    for k in range(n):
        parts += [prims[:, :, k],
                  torch.where(mods[:, :, k] == 1, TOK_TWICE, TOK_NOOP)]
    q = torch.stack(parts, dim=2)
    if q.shape[2] < QMAX:
        pad = torch.full((E, QN, QMAX - q.shape[2]), TOK_PAD,
                         dtype=torch.long, device=device)
        q = torch.cat([q, pad], dim=2)
    reps = 1 + mods
    m = lex.gather(1, prims.reshape(E, -1)).reshape(E, QN, n)
    csum = torch.cumsum(reps, dim=2)
    start = csum - reps
    pos = torch.arange(OUTM, device=device).view(1, 1, OUTM)
    tgt = torch.full((E, QN, OUTM), PAD_CLS, dtype=torch.long,
                     device=device)
    for k in range(n):
        mk = (pos >= start[:, :, k:k + 1]) & (pos < csum[:, :, k:k + 1])
        tgt = torch.where(mk, m[:, :, k:k + 1], tgt)
    return prims, mods, q, tgt


def write_episode(model, book, now, lex):
    with torch.no_grad():
        for s in range(SPE):
            f = now[:, s]
            book.write(F.normalize(model.emb(f), dim=-1),
                       model.emb(lex.gather(1, f.unsqueeze(1))
                                 .squeeze(1) + 32))


# ---------------------------------------------------------------------------
# Arm A: autoregressive composer (iteration as hope)
# ---------------------------------------------------------------------------

class MBlock(nn.Module):
    def __init__(self, d, heads=4, mlp_ratio=3.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x, mask=None):
        y = self.n1(x)
        a, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class ARComposer(nn.Module):
    def __init__(self, d=96, layers=3, heads=4):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(VOCAB2, d)
        self.base = 2 * SPE + QMAX          # ctx + query length
        seq = self.base + OUTM
        self.register_buffer("pos", sinpos(seq, d))
        allowed = torch.zeros(seq, seq, dtype=torch.bool)
        allowed[:, :self.base] = True                       # all see ctx+q
        for i in range(self.base, seq):
            allowed[i, self.base:i + 1] = True              # causal answers
        self.register_buffer("mask", ~allowed)
        self.blocks = nn.ModuleList(MBlock(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, PAD_CLS + 1)

    def _embed(self, ctx, q_tokens, ans_in, book):
        E, Q = q_tokens.shape[:2]
        toks = torch.cat([ctx, q_tokens, ans_in], dim=2)
        x = self.emb(toks.reshape(E * Q, -1))
        prims = q_tokens[:, :, 0::2]                        # (E,Q,4)
        valid = (prims < P).unsqueeze(-1).float()
        qv = self.emb(prims.clamp(max=P - 1))
        pay = book.read(qv.reshape(E, prims.shape[2] * Q, -1)) \
            .reshape(E, Q, prims.shape[2], -1) * valid
        x = x.reshape(E, Q, -1, self.d)
        for k in range(prims.shape[2]):
            p = 2 * SPE + 2 * k
            x[:, :, p] = x[:, :, p] + pay[:, :, k]
        return x.reshape(E * Q, -1, self.d)

    def forward_tf(self, ctx, q_tokens, tgt, book):
        """Teacher-forced training loss."""
        E, Q = q_tokens.shape[:2]
        prev = torch.cat([torch.full_like(tgt[:, :, :1], TOK_BOS),
                          torch.where(tgt[:, :, :-1] < 32,
                                      tgt[:, :, :-1] + 32,
                                      torch.full_like(tgt[:, :, :-1],
                                                      TOK_PAD))], dim=2)
        x = self._embed(ctx, q_tokens, prev, book)
        x = x + self.pos[: x.shape[1]]
        for b in self.blocks:
            x = b(x, mask=self.mask[: x.shape[1], : x.shape[1]])
        logits = self.head(self.norm(x[:, self.base:]))
        return F.cross_entropy(logits.reshape(-1, PAD_CLS + 1),
                               tgt.reshape(-1))

    @torch.no_grad()
    def generate(self, ctx, q_tokens, book):
        E, Q = q_tokens.shape[:2]
        dev = q_tokens.device
        ans = torch.full((E, Q, OUTM), TOK_PAD, dtype=torch.long,
                         device=dev)
        out = torch.full((E, Q, OUTM), PAD_CLS, dtype=torch.long,
                         device=dev)
        ans[:, :, 0] = TOK_BOS
        for t in range(OUTM):
            x = self._embed(ctx, q_tokens, ans, book)
            x = x + self.pos[: x.shape[1]]
            for b in self.blocks:
                x = b(x, mask=self.mask[: x.shape[1], : x.shape[1]])
            lg = self.head(self.norm(x[:, self.base + t]))
            pred = lg.argmax(-1).reshape(E, Q)
            out[:, :, t] = pred
            if t + 1 < OUTM:
                ans[:, :, t + 1] = torch.where(pred < 32, pred + 32,
                                               torch.full_like(pred,
                                                               TOK_PAD))
        return out


# ---------------------------------------------------------------------------
# Arm B: the walker (iteration as circuit)
# ---------------------------------------------------------------------------

class Walker(nn.Module):
    """Hardcoded walk: cursor over primitives, repeat counter,
    advance-on-zero, stop-at-end. Learned: emission head, modifier
    interpreter, pad logits. No transformer at query time."""

    def __init__(self, d=96):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(VOCAB2, d)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, PAD_CLS + 1)
        self.mod_lin = nn.Linear(d, 1)
        self.pad_logit = nn.Parameter(torch.zeros(PAD_CLS + 1))

    def per_prim(self, prims, mods, book):
        E, Q, n = prims.shape
        pv = book.read(self.emb(prims).reshape(E, Q * n, -1)) \
            .reshape(E, Q, n, -1)
        Lj = self.head(self.norm(pv))                       # (E,Q,n,C)
        mt = torch.where(mods == 1, TOK_TWICE, TOK_NOOP)
        q2 = torch.sigmoid(self.mod_lin(self.emb(mt)).squeeze(-1))
        return Lj, q2                                       # P(rep==2)

    def loss(self, prims, mods, tgt, book):
        """Exact expected CE over the <=2^n repeat combinations."""
        E, Q, n = prims.shape
        Lj, q2 = self.per_prim(prims, mods, book)
        total = torch.zeros(E, Q, device=prims.device)
        for combo in range(2 ** n):
            bits = [(combo >> k) & 1 for k in range(n)]
            pc = torch.ones(E, Q, device=prims.device)
            for k in range(n):
                pc = pc * (q2[:, :, k] if bits[k] else 1 - q2[:, :, k])
            seq = []
            for k in range(n):
                seq += [Lj[:, :, k]] * (bits[k] + 1)
            while len(seq) < OUTM:
                seq.append(self.pad_logit.view(1, 1, -1).expand(E, Q, -1))
            Lc = torch.stack(seq[:OUTM], dim=2)
            ce = F.cross_entropy(
                Lc.reshape(-1, PAD_CLS + 1), tgt.reshape(-1),
                reduction="none").reshape(E, Q, OUTM).mean(-1)
            total = total + pc * ce
        return total.mean()

    @torch.no_grad()
    def generate(self, prims, mods, book):
        E, Q, n = prims.shape
        Lj, q2 = self.per_prim(prims, mods, book)
        reps = 1 + (q2 > 0.5).long()                        # (E,Q,n)
        tok = Lj.argmax(-1)                                 # (E,Q,n)
        csum = torch.cumsum(reps, dim=2)
        start = csum - reps
        pos = torch.arange(OUTM, device=prims.device).view(1, 1, OUTM)
        out = torch.full((E, Q, OUTM), PAD_CLS, dtype=torch.long,
                         device=prims.device)
        for k in range(n):
            mk = (pos >= start[:, :, k:k + 1]) & (pos < csum[:, :, k:k + 1])
            out = torch.where(mk, tok[:, :, k:k + 1], out)
        return out


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def run_lifetime(model, arm, E, device, K, training):
    lex, sched = make_lifetime(E, device)
    book = LexBook(E, K, model.d, device)
    loss = torch.zeros((), device=device)
    stats = {}
    for e in range(EPS):
        now = sched[:, e]
        write_episode(model, book, now, lex)
        prims, mods, q, tgt = gen_queries(lex, sched, e, device, 2)
        ctx = ctx_episode(now, lex, QN)
        if arm == "ar":
            loss = loss + model.forward_tf(ctx, q, tgt, book)
        else:
            loss = loss + model.loss(prims, mods, tgt, book)
        if not training and e == EPS - 1:
            for n in (2, 3, 4):
                pr, md, qn_, tg = gen_queries(lex, sched, e, device, n)
                cx = ctx_episode(now, lex, QN)
                out = (model.generate(cx, qn_, book) if arm == "ar"
                       else model.generate(pr, md, book))
                ok = (out == tg).float()
                stats[f"len{n}"] = float(ok.mean())
                stats[f"len{n}_exact"] = float(
                    (out == tg).all(-1).float().mean())
    return loss / EPS, stats


def train(model, arm, steps, E, lr, device, K, tag, log_fn=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        loss, _ = run_lifetime(model, arm, E, device, K, True)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 200 == 0:
            print(f"[{tag}] step {step:5d}  loss {loss.item():.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if log_fn:
                log_fn({f"{tag}/loss": loss.item(), f"{tag}/step": step})


@torch.no_grad()
def eval_arm(model, arm, E, device, K, batches):
    model.eval()
    keys = [f"len{n}{s}" for n in (2, 3, 4) for s in ("", "_exact")]
    acc = {k: 0.0 for k in keys}
    for _ in range(batches):
        _, st = run_lifetime(model, arm, E, device, K, False)
        for k in keys:
            acc[k] += st[k] / batches
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--eval-batches", type=int, default=4)
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
                         name=f"composer-s{args.steps}",
                         config=vars(args))
    log_fn = (lambda d: run.log(d)) if run else None

    models = {"walker": Walker().to(device),
              "ar": ARComposer().to(device)}
    for arm, m in models.items():
        train(m, arm, args.steps, args.batch, args.lr, device, args.k,
              arm, log_fn=log_fn)
    if args.save_prefix:
        for arm, m in models.items():
            torch.save(m.state_dict(), f"{args.save_prefix}_{arm}.pt")

    results = {}
    for arm, m in models.items():
        results[arm] = eval_arm(m, arm, args.eval_batch, device, args.k,
                                args.eval_batches)

    print("\nLENGTH GENERALIZATION (trained on 2-primitive queries only; "
          "per-position / exact):")
    print(f"  {'arm':>8s}   len2         len3         len4")
    for arm, r in results.items():
        print(f"  {arm:>8s}   "
              + "   ".join(f"{r[f'len{n}'] * 100:5.1f}/"
                           f"{r[f'len{n}_exact'] * 100:5.1f}"
                           for n in (2, 3, 4)))

    if run:
        import wandb
        for arm, r in results.items():
            for k, v in r.items():
                run.summary[f"{arm}_{k}"] = v
        with open("results.json", "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1)
        art = wandb.Artifact(f"composer-{run.id}", type="results")
        art.add_file("results.json")
        for arm in models:
            p = f"{args.save_prefix}_{arm}.pt"
            if args.save_prefix and os.path.exists(p):
                art.add_file(p)
        run.log_artifact(art)
        art.wait()
        print("ARTIFACT_VERIFIED", flush=True)
        run.finish()


if __name__ == "__main__":
    main()
