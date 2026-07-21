"""Toy: lifetime lexicon acquisition with a codebook LAYER (2026-07-21).

Rung 1 of the proto-symbol program (POINTS_OF_INTEREST, the pinned
reader/bridge thread): codes whose payloads must be RETRIEVED and
COMPOSED. Successor to toy_codebook_icl.py with the two lessons of the
DINO campaign built in: (1) single-lookup probes never grade
accumulation -> every answer here needs 2 retrievals + composition;
(2) the book's intelligence ceiling is its encoder -> the book is a
LAYER inside the reasoner (its consumer is the layers above), not a
module after a stem.

World (grammar permanent, lexicon per-lifetime):
- P=32 primitive FORM tokens; per lifetime a random bijection maps
  forms -> 32 MEANING symbols (unsmearable: the binding exists only
  in the lifetime — the label-permutation trick promoted to language).
- Grammar, fixed forever (weights may learn it): a query is
  [p1, mod1, p2, mod2], mod in {NOOP, TWICE}; the answer is p1's
  meaning repeated 1 or 2 times, then p2's, padded to 4 with PAD.
- Lifetime = 8 episodes; each episode STUDIES 4 fresh primitives
  (form+meaning pairs shown in-context) — each primitive is studied
  in exactly ONE episode per lifetime.
- Queries per episode, by construction: 2 WITHIN (both primitives
  studied this episode — answerable by attention over the episode's
  study pairs, the meta-seq2seq/MLC regime), 2 CROSS (both studied in
  EARLIER episodes, absent from context — answerable ONLY through the
  persistent book), 2 MIXED. Cross-episode information reaches the
  query pass through the book alone (no side channel — constants law).

Model: 3-layer transformer (d=96). Sequence = [this episode's 4 study
pairs][query][4 answer slots]; logits read from the slots. The book is
a KEY-VALUE store (the pointer-array factorization arrives here
naturally: key = normalized form embedding, payload = meaning
embedding) written by the validated v6b economy — hard-assign +
count-capped running means on BOTH tensors, novelty-gated birth
(theta), lazy bar-merge — and read as a similarity-softmax mixture
over used keys, the payload mixture ADDED to the query primitive's
token embedding (rejoins the residual stream). Writes are no_grad;
embeddings train through the in-context/output pathways, so the book
stores vectors whose geometry is shaped by the rest of the tower
(healthier bootstrap than 20-way: the context channel gives dense
non-book gradient from step one).

Arms:
  live      book on (writes at study, reads at query)
  frozen    same weights, writes off                  -> cross floor
  oracle    book injected with the true lexicon      -> cross ceiling
  ctx-all   separate model, ALL pairs studied so far in context
            (the MLC regime at lifetime scale — in-context upper
            baseline while the lexicon fits the window)
  ctx-cap   ctx-all model, context capped to the last 2 episodes'
            pairs -> the context-boundary collapse, simulated
  episodic  separate model, current episode's pairs only, no book
            (the existing literature's regime) -> cross ~ chance

Smoke criteria: (a) WITHIN high for all trained arms (grammar +
in-context binding learned); (b) the dissociation on CROSS:
live ~ oracle ~ ctx-all >> ctx-cap > episodic ~ frozen ~ chance;
(c) book diagnostics sane (used ~ #studied, ~no collisions, merges
rare). Chance per output position ~ 1/33.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

P = 32                  # primitives (= meanings)
EPS = 8                 # episodes per lifetime
SPE = 4                 # primitives studied per episode
QN = 6                  # queries per episode (2 within / 2 cross / 2 mixed)
OUT = 4                 # answer slots
PAD_CLS = 32            # output class for padding (classes 0..32)
TOK_TWICE, TOK_NOOP, TOK_SLOT, TOK_PAD = 64, 65, 66, 67
VOCAB = 68              # forms 0-31, meanings 32-63, specials


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d, heads=4, mlp_ratio=3.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x):
        y = self.n1(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class LexBook:
    """Per-lifetime key-value codebook, v6b economy. Keys and payloads
    are separate tensors — the pointer-array factorization. All updates
    are hardcoded rules under no_grad; nothing here is a Parameter."""

    def __init__(self, E, K, d, device, theta=0.45, theta_merge=0.6,
                 cap=32, code_temp=10.0):
        self.K, self.theta, self.theta_merge = K, theta, theta_merge
        self.cap, self.code_temp = cap, code_temp
        g = torch.Generator(device="cpu").manual_seed(7)
        init = F.normalize(torch.randn(K, d, generator=g), dim=-1)
        self.keys = init.to(device).unsqueeze(0).repeat(E, 1, 1)
        self.pays = torch.zeros(E, K, d, device=device)
        self.counts = torch.zeros(E, K, device=device)
        self.merges = torch.zeros(E, device=device)

    @torch.no_grad()
    def write(self, kv, pv):
        """kv,pv: (E,d). Assign among used keys; birth below theta;
        drag key AND payload running means; lazy bar-merge."""
        kv, pv = kv.detach(), pv.detach()
        E = kv.shape[0]
        ar = torch.arange(E, device=kv.device)
        used = self.counts > 0
        sim = torch.einsum("ed,ekd->ek", F.normalize(kv, dim=-1),
                           F.normalize(self.keys, dim=-1))
        neg = torch.finfo(sim.dtype).min
        sim_u = sim.masked_fill(~used, neg)
        best, bestv = sim_u.argmax(1), sim_u.max(1).values
        free = ~used
        birth = (bestv < self.theta) & free.any(1)
        i = torch.where(birth, free.float().argmax(1), best)
        n = self.counts[ar, i] + 1
        lr = (1.0 / n.clamp(max=self.cap)).unsqueeze(-1)
        self.keys[ar, i] += lr * (kv - self.keys[ar, i])
        self.pays[ar, i] += lr * (pv - self.pays[ar, i])
        self.counts[ar, i] = n
        # lazy bar-merge on keys; count-weighted collapse of both
        ki = self.keys[ar, i]
        msim = torch.einsum("ed,ekd->ek", F.normalize(ki, dim=-1),
                            F.normalize(self.keys, dim=-1))
        mask = self.counts > 0
        mask[ar, i] = False
        msim = msim.masked_fill(~mask, neg)
        j = msim.argmax(1)
        do = (msim.max(1).values > self.theta_merge) & mask.any(1)
        if do.any():
            ni = self.counts[ar, i].unsqueeze(-1)
            nj = self.counts[ar, j].unsqueeze(-1)
            mk = (self.keys[ar, i] * ni + self.keys[ar, j] * nj) / (ni + nj)
            mp = (self.pays[ar, i] * ni + self.pays[ar, j] * nj) / (ni + nj)
            keep = torch.where(self.counts[ar, i]
                               >= self.counts[ar, j], i, j)
            drop = torch.where(keep == i, j, i)
            ard = ar[do]
            self.keys[ard, keep[do]] = mk[do]
            self.pays[ard, keep[do]] = mp[do]
            self.counts[ard, keep[do]] = (ni + nj).squeeze(-1)[do]
            self.counts[ard, drop[do]] = 0.0
            self.merges += do.float()

    @torch.no_grad()
    def read(self, q):
        """q: (E,Q,d) -> payload mixtures (E,Q,d)."""
        sim = torch.einsum("eqd,ekd->eqk", F.normalize(q, dim=-1),
                           F.normalize(self.keys, dim=-1))
        used = (self.counts > 0).unsqueeze(1)
        neg = torch.finfo(sim.dtype).min
        sim_u = sim.masked_fill(~used, neg)
        sim_eff = torch.where(used.any(-1, keepdim=True), sim_u, sim)
        w = torch.softmax(sim_eff * self.code_temp, dim=-1)
        return torch.einsum("eqk,ekd->eqd", w, self.pays)


class LexModel(nn.Module):
    def __init__(self, d=96, layers=3, heads=4, use_book=True,
                 ctx_mode="episode"):
        super().__init__()
        self.use_book, self.ctx_mode, self.d = use_book, ctx_mode, d
        self.emb = nn.Embedding(VOCAB, d)
        seq = (2 * SPE if ctx_mode == "episode" else 2 * P) + 4 + OUT
        self.pos = nn.Parameter(torch.randn(seq, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, PAD_CLS + 1)

    def forward(self, ctx_tokens, q_tokens, book=None):
        """ctx_tokens: (E,Q,C) query-repeated context; q_tokens: (E,Q,4).
        Returns logits (E,Q,OUT,33)."""
        E, Q, C = ctx_tokens.shape
        slots = torch.full((E, Q, OUT), TOK_SLOT, dtype=torch.long,
                           device=ctx_tokens.device)
        toks = torch.cat([ctx_tokens, q_tokens, slots], dim=2)
        x = self.emb(toks.reshape(E * Q, -1))
        if self.use_book and book is not None:
            # book read for the two query-primitive positions
            prims = q_tokens[:, :, [0, 2]]                    # (E,Q,2)
            qv = self.emb(prims)                              # (E,Q,2,d)
            pay = book.read(qv.reshape(E, Q * 2, -1)).reshape(E, Q, 2, -1)
            x = x.reshape(E, Q, -1, self.d)
            x[:, :, C] = x[:, :, C] + pay[:, :, 0]
            x[:, :, C + 2] = x[:, :, C + 2] + pay[:, :, 1]
            x = x.reshape(E * Q, -1, self.d)
        x = x + self.pos[: x.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x[:, -OUT:])).reshape(E, Q, OUT, -1)


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

def make_lifetime(E, device):
    lex = torch.rand(E, P, device=device).argsort(1)        # form -> meaning
    sched = torch.rand(E, P, device=device).argsort(1).reshape(E, EPS, SPE)
    return lex, sched


HOLDOUT_TWICE = 8       # forms 0..7: never modified by TWICE in training


def make_episode(lex, sched, e, device, holdout=False):
    """Returns study pairs, query tokens, targets (E,QN,OUT), and a
    per-output-position origin map: 0=within, 1=cross, 2=pad.
    holdout=True (training only): forms < HOLDOUT_TWICE never receive
    TWICE — the add-jump split. At eval, unconstrained queries measure
    whether the factorization yields systematicity for free."""
    E = lex.shape[0]
    now = sched[:, e]                                       # (E,SPE)
    ridx = torch.randint(0, SPE, (E, 6), device=device)
    w_pool = now.gather(1, ridx)                            # within picks
    if e > 0:
        past = sched[:, :e].reshape(E, e * SPE)
        cidx = torch.randint(0, e * SPE, (E, 6), device=device)
        c_pool = past.gather(1, cidx)
        is_cross_possible = True
    else:
        c_pool = w_pool
        is_cross_possible = False
    # queries: [w,w] [w,w] [c,c] [c,c] [w,c] [c,w]
    p1 = torch.stack([w_pool[:, 0], w_pool[:, 1], c_pool[:, 0],
                      c_pool[:, 1], w_pool[:, 2], c_pool[:, 4]], dim=1)
    p2 = torch.stack([w_pool[:, 3], w_pool[:, 4], c_pool[:, 2],
                      c_pool[:, 3], c_pool[:, 5], w_pool[:, 5]], dim=1)
    o1 = torch.tensor([0, 0, 1, 1, 0, 1], device=device)
    o2 = torch.tensor([0, 0, 1, 1, 1, 0], device=device)
    if not is_cross_possible:
        o1 = torch.zeros_like(o1)
        o2 = torch.zeros_like(o2)
    mods = torch.randint(0, 2, (E, QN, 2), device=device)   # 0 NOOP 1 TWICE
    if holdout:
        mods[:, :, 0] = torch.where(p1 < HOLDOUT_TWICE,
                                    torch.zeros_like(mods[:, :, 0]),
                                    mods[:, :, 0])
        mods[:, :, 1] = torch.where(p2 < HOLDOUT_TWICE,
                                    torch.zeros_like(mods[:, :, 1]),
                                    mods[:, :, 1])
    q_tokens = torch.stack(
        [p1, torch.where(mods[:, :, 0] == 1, TOK_TWICE, TOK_NOOP),
         p2, torch.where(mods[:, :, 1] == 1, TOK_TWICE, TOK_NOOP)],
        dim=2)                                              # (E,QN,4)
    m1 = lex.gather(1, p1)
    m2 = lex.gather(1, p2)
    r1 = 1 + mods[:, :, 0]
    r2 = 1 + mods[:, :, 1]
    pos = torch.arange(OUT, device=device).view(1, 1, OUT)
    tgt = torch.full((E, QN, OUT), PAD_CLS, dtype=torch.long, device=device)
    o_map = torch.full((E, QN, OUT), 2, dtype=torch.long, device=device)
    in1 = pos < r1.unsqueeze(-1)
    in2 = (pos >= r1.unsqueeze(-1)) & (pos < (r1 + r2).unsqueeze(-1))
    tgt = torch.where(in1, m1.unsqueeze(-1), tgt)
    tgt = torch.where(in2, m2.unsqueeze(-1), tgt)
    o_map = torch.where(in1, o1.view(1, QN, 1).expand(E, QN, OUT), o_map)
    o_map = torch.where(in2, o2.view(1, QN, 1).expand(E, QN, OUT), o_map)
    nov1 = ((p1 < HOLDOUT_TWICE) & (mods[:, :, 0] == 1)).unsqueeze(-1)
    nov2 = ((p2 < HOLDOUT_TWICE) & (mods[:, :, 1] == 1)).unsqueeze(-1)
    novel = (in1 & nov1) | (in2 & nov2)          # held-out combinations
    return now, q_tokens, tgt, o_map, novel


def ctx_episode(now, lex, Q):
    """(E,SPE) studied-now -> (E,Q,2*SPE) [form, meaning-token] pairs."""
    m = lex.gather(1, now) + 32
    pairs = torch.stack([now, m], dim=2).reshape(now.shape[0], -1)
    return pairs.unsqueeze(1).expand(-1, Q, -1)


def ctx_all(sched, lex, e, Q, cap_eps=None):
    """All pairs studied in episodes <= e (others PAD). Optionally only
    the last cap_eps episodes' pairs stay visible."""
    E = sched.shape[0]
    forms = sched.reshape(E, EPS, SPE)
    m = lex.gather(1, forms.reshape(E, -1)).reshape(E, EPS, SPE) + 32
    pairs = torch.stack([forms, m], dim=3).reshape(E, EPS, 2 * SPE)
    lo = 0 if cap_eps is None else max(0, e + 1 - cap_eps)
    vis = torch.zeros(E, EPS, 1, dtype=torch.bool, device=sched.device)
    vis[:, lo:e + 1] = True
    pairs = torch.where(vis, pairs, torch.full_like(pairs, TOK_PAD))
    return pairs.reshape(E, -1).unsqueeze(1).expand(-1, Q, -1)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def run_lifetime(model, E, device, K, training, arm="live",
                 cap_eps=None, lex=None, sched=None, holdout=False):
    if lex is None:
        lex, sched = make_lifetime(E, device)
    book = None
    if model.use_book:
        book = LexBook(E, K, model.d, device)
        if arm == "oracle":
            with torch.no_grad():
                forms = torch.arange(P, device=device)
                kv = F.normalize(model.emb(forms), dim=-1)
                pv = model.emb(lex + 32)
                book.keys[:, :P] = kv.unsqueeze(0)
                book.pays[:, :P] = pv
                book.counts[:, :P] = book.cap
    loss = torch.zeros((), device=device)
    stats = []
    for e in range(EPS):
        now, q_tokens, tgt, o_map, novel = make_episode(
            lex, sched, e, device, holdout=holdout)
        if model.use_book and arm == "live":
            with torch.no_grad():
                for s in range(SPE):
                    f = now[:, s]
                    book.write(F.normalize(model.emb(f), dim=-1),
                               model.emb(lex.gather(1, f.unsqueeze(1))
                                         .squeeze(1) + 32))
        if model.ctx_mode == "episode":
            ctx = ctx_episode(now, lex, QN)
        else:
            ctx = ctx_all(sched, lex, e, QN, cap_eps=cap_eps)
        logits = model(ctx, q_tokens, book=book)
        loss = loss + F.cross_entropy(
            logits.reshape(-1, PAD_CLS + 1), tgt.reshape(-1))
        with torch.no_grad():
            pred = logits.argmax(-1)
            ok = (pred == tgt).float()
            st = {"e": e}
            for name, oid in (("within", 0), ("cross", 1)):
                m = o_map == oid
                st[name] = float(ok[m].mean()) if m.any() else float("nan")
            st["novel"] = (float(ok[novel].mean()) if novel.any()
                           else float("nan"))
            st["exact"] = float((pred == tgt).all(-1).float().mean())
            if book is not None:
                st["used"] = float((book.counts > 0).float().sum(1).mean())
                st["merges"] = float(book.merges.mean())
            stats.append(st)
    return loss / EPS, stats


def train(model, steps, E, lr, device, K, tag, arm="live", log_every=200,
          log_fn=None, holdout=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        loss, st = run_lifetime(model, E, device, K, True, arm=arm,
                                holdout=holdout)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % log_every == 0:
            last = st[-1]
            print(f"[{tag}] step {step:5d}  loss {loss.item():.3f}  "
                  f"within {last['within']:.3f}  cross {last['cross']:.3f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
            if log_fn is not None:
                log_fn({f"{tag.strip()}/loss": loss.item(),
                        f"{tag.strip()}/within": last["within"],
                        f"{tag.strip()}/cross": last["cross"],
                        f"{tag.strip()}/step": step})


@torch.no_grad()
def eval_arm(model, E, device, K, arm, batches, cap_eps=None):
    model.eval()
    acc = {k: torch.zeros(EPS) for k in ("within", "cross", "novel",
                                         "exact", "used", "merges")}
    cnt = {k: torch.zeros(EPS) for k in acc}
    for _ in range(batches):
        _, stats = run_lifetime(model, E, device, K, False, arm=arm,
                                cap_eps=cap_eps)
        for st in stats:
            for k in acc:
                v = st.get(k)
                if v is not None and v == v:      # skip NaN
                    acc[k][st["e"]] += v
                    cnt[k][st["e"]] += 1
    return {k: (acc[k] / cnt[k].clamp(min=1)) for k in acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--baseline-steps", type=int, default=0,
                    help="extra budget for the attention baselines "
                         "(0 = same as --steps); v0 showed they fail to "
                         "IGNITE at 2k steps while the book model is "
                         "perfect by 200 — don't let the win be a "
                         "strawman")
    ap.add_argument("--baseline-lr", type=float, default=0.0,
                    help="0 = same as --lr")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--save-prefix", type=str, default="")
    ap.add_argument("--load-prefix", type=str, default="")
    ap.add_argument("--wandb", action="store_true",
                    help="stream + VERIFIED artifact (required on cloud: "
                         "the instance self-destroys on exit 0)")
    ap.add_argument("--wandb_project", type=str, default="neocore-lex")
    ap.add_argument("--holdout", action="store_true",
                    help="add-jump analog: forms 0..7 never meet TWICE "
                         "during training; eval is unconstrained and "
                         "reports the held-out combinations separately")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"lex-s{args.steps}-b"
                              f"{args.baseline_steps or args.steps}",
                         config=vars(args))
    log_fn = (lambda d: run.log(d)) if run else None

    m_book = LexModel(use_book=True, ctx_mode="episode").to(device)
    m_all = LexModel(use_book=False, ctx_mode="all").to(device)
    m_epi = LexModel(use_book=False, ctx_mode="episode").to(device)

    if args.load_prefix:
        for m, n in ((m_book, "book"), (m_all, "all"), (m_epi, "epi")):
            m.load_state_dict(torch.load(f"{args.load_prefix}_{n}.pt"))
        print(f"models loaded from {args.load_prefix}_*", flush=True)
    else:
        bsteps = args.baseline_steps or args.steps
        blr = args.baseline_lr or args.lr
        train(m_book, args.steps, args.batch, args.lr, device, args.k,
              "book    ", log_fn=log_fn, holdout=args.holdout)
        train(m_all, bsteps, args.batch, blr, device, args.k,
              "ctx-all ", log_fn=log_fn, holdout=args.holdout)
        train(m_epi, bsteps, args.batch, blr, device, args.k,
              "episodic", log_fn=log_fn, holdout=args.holdout)
    if args.save_prefix:
        for m, n in ((m_book, "book"), (m_all, "all"), (m_epi, "epi")):
            torch.save(m.state_dict(), f"{args.save_prefix}_{n}.pt")

    arms = {
        "live": eval_arm(m_book, args.eval_batch, device, args.k, "live",
                         args.eval_batches),
        "frozen": eval_arm(m_book, args.eval_batch, device, args.k,
                           "frozen", args.eval_batches),
        "oracle": eval_arm(m_book, args.eval_batch, device, args.k,
                           "oracle", args.eval_batches),
        "ctx-all": eval_arm(m_all, args.eval_batch, device, args.k,
                            "live", args.eval_batches),
        "ctx-cap": eval_arm(m_all, args.eval_batch, device, args.k,
                            "live", args.eval_batches, cap_eps=2),
        "episodic": eval_arm(m_epi, args.eval_batch, device, args.k,
                             "live", args.eval_batches),
    }

    names = list(arms)
    print("\nCROSS-episode per-position accuracy (chance ~3.0) by "
          "episode index — the lexicon-accumulation curve:")
    print("  ep-idx  " + "  ".join(f"{n:>8s}" for n in names)
          + "    used  merges")
    for e in range(1, EPS):
        row = "  ".join(f"{arms[n]['cross'][e] * 100:8.2f}" for n in names)
        print(f"  {e:6d}  {row}  {arms['live']['used'][e]:6.1f}"
              f"  {arms['live']['merges'][e]:6.2f}")
    print("\nWITHIN-episode accuracy, same layout:")
    for e in range(EPS):
        row = "  ".join(f"{arms[n]['within'][e] * 100:8.2f}" for n in names)
        print(f"  {e:6d}  {row}")
    for n in names:
        c = arms[n]["cross"][1:] * 100
        w = arms[n]["within"] * 100
        print(f"  {n:>8s}: cross {c.mean().item():5.1f}  "
              f"within {w.mean().item():5.1f}  "
              f"novel-combo {arms[n]['novel'].mean().item() * 100:5.1f}  "
              f"exact {arms[n]['exact'].mean().item() * 100:5.1f}")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for n in names:
            ax.plot(range(1, EPS), (arms[n]["cross"][1:] * 100).numpy(),
                    marker="o", label=n)
        ax.axhline(100 / 33, color="gray", ls=":", label="chance")
        ax.set_xlabel("episode index (cross-episode queries)")
        ax.set_ylabel("per-position accuracy (%)")
        ax.set_title("lifetime lexicon: composition across the "
                     "context boundary")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"curve -> {args.out}", flush=True)

    if run:
        import wandb
        for n in names:
            run.summary[f"{n}_cross"] = arms[n]["cross"][1:].mean().item()
            run.summary[f"{n}_within"] = arms[n]["within"].mean().item()
        results = {n: {k: v.tolist() for k, v in arms[n].items()}
                   for n in names}
        with open("results.json", "w") as f:
            json.dump({"args": vars(args), "arms": results}, f, indent=1)
        if args.out:
            run.log({"cross_curve": wandb.Image(args.out)})
        art = wandb.Artifact(f"lex-icl-{run.id}", type="results")
        art.add_file("results.json")
        if args.out:
            art.add_file(args.out)
        for n in ("book", "all", "epi"):
            p = f"{args.save_prefix}_{n}.pt"
            if args.save_prefix and os.path.exists(p):
                art.add_file(p)
        run.log_artifact(art)
        art.wait()      # VERIFIED before the instance self-destroys
        print("ARTIFACT_VERIFIED", flush=True)
        run.finish()


if __name__ == "__main__":
    main()
