"""Stream world v0: a small LM with a rule-written codebook MEMORY LAYER
(2026-07-21 overnight). The program's pivot to its actual goal: the
MISSING MIDDLE TIMESCALE — weights learn over training then freeze;
context holds anything then forgets; nothing in the standard recipe
acquires knowledge permanently at deployment. This toy is the smallest
LM-shaped test of the one mechanism we validated all campaign: a
key-value book written by the v6b economy (hard-assign running means,
novelty-gated birth, lazy bar-merge), read as a soft mixture into the
residual stream, meta-trained so the book is the ONLY path to the
answers.

World (per lifetime; everything entity-specific re-randomized -> the
weights cannot smear it):
- N_E entities, names drawn fresh from a pool each lifetime. Each
  entity has one RELATION (rel(A)=B, "A partner B .") and one
  ATTRIBUTE (attr(A)=v, "A color v ."), each stated twice.
- The lifetime is a STREAM of tiny documents processed in ROUNDS:
  round r introduces group-r entities' facts and asks questions about
  entities from EARLIER rounds only; a final quiz round asks about
  everyone. Attention is DOCUMENT-LOCAL (the context window is the
  sentence) — cross-document information has no path except the book.
- Questions (answer graded at the answer token):
    lookup-rel   [QREL  A ANS ?] -> rel(A)          (a name token)
    lookup-attr  [QATTR A ANS ?] -> attr(A)         (a value token)
    composition  [QCOMP A ANS ?] -> attr(rel(A))    two facts, filed
                 in different documents, possibly different rounds —
                 the reader-thesis question type that made the book's
                 value total instead of marginal in the lexicon toy.

Model: 4-layer causal transformer LM (d=96 smoke), document-local.
The book is read at TWO depths (after blocks 1 and 3, same book):
read 1 can fetch rel(A), the mixed residual lets read 2 form a query
for attr(B) — two hops inside one forward pass, composition as
re-reading. Read output projections are ZERO-INIT (the book starts as
a no-op; the aux loss ignites the pathway — bootstrap-whisper law).

v0 training wheels, declared: (a) writes fire on fact documents (the
harness knows which they are); (b) write key = normalized
emb(subject)+emb(typeword) (typed so A's two facts do not collide),
payload = mean of the fact's token embeddings — the tower trains this
geometry through its other pathways (the lexicon toy's trick); (c)
auxiliary retrieval CE on both reads' attention (bare gate-whisper
never bootstraps — 4th domain), annealed to zero mid-training.
v1 relaxations, in order: learned writer head on hidden states;
novelty-gated write timing (no harness signal); interleaved noise.

Arms: live (economy writes) / exemplar (append-only, every statement
a new slot — Ibanis's control) / frozen (random book) / oracle (true
facts pre-filled) / dense (separate model, no book).

Pre-registered gate (local smoke, before any launch): dense ~ chance
on ALL question types (no channel by construction); oracle >> dense;
live -> oracle as meta-training proceeds. Chance: rel ~ 1/N_E..1/20
(name tokens), attr/comp ~ 1/6. If oracle fails, the harness is
broken — fix before renting anything.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- world
N_NAMES = 20
N_V = 6
L_DOC = 6
TOK_VAL0 = N_NAMES                       # 20..25 value tokens
TOK_REL, TOK_ATTR = 26, 27
TOK_QREL, TOK_QATTR, TOK_QCOMP = 28, 29, 30
TOK_ANS, TOK_DOT, TOK_PAD = 31, 32, 33
VOCAB = 34
QTYPES = ("rel", "attr", "comp")


class World:
    """Per-lifetime entities/relations/attributes, batched (B,...)."""

    def __init__(self, B, n_e, device):
        self.B, self.n_e, self.device = B, n_e, device
        self.names = torch.rand(B, N_NAMES, device=device) \
            .argsort(1)[:, :n_e]                       # (B,n_e) tokens
        # rel(i) = j with group(j) <= group(i), j != i (2 per group)
        rel = torch.zeros(B, n_e, dtype=torch.long, device=device)
        for i in range(n_e):
            hi = 2 * (i // 2 + 1)                      # entities 0..hi-1
            j0 = torch.randint(0, max(hi - 1, 1), (B,), device=device)
            rel[:, i] = torch.where(torch.tensor(hi > 1, device=device),
                                    j0 + (j0 >= i).long(),
                                    torch.zeros_like(j0))
        self.rel = rel
        self.attr = torch.randint(0, N_V, (B, n_e), device=device) \
            + TOK_VAL0

    def name(self, ent):
        return self.names.gather(1, ent)

    def fact_doc(self, ent, kind):
        """ent: (B,) entity index; kind 0=rel 1=attr -> (B,L_DOC)."""
        B = self.B
        subj = self.name(ent.unsqueeze(1)).squeeze(1)
        if kind == 0:
            typ = torch.full((B,), TOK_REL, device=self.device)
            obj = self.name(self.rel.gather(
                1, ent.unsqueeze(1))).squeeze(1)
        else:
            typ = torch.full((B,), TOK_ATTR, device=self.device)
            obj = self.attr.gather(1, ent.unsqueeze(1)).squeeze(1)
        dot = torch.full((B,), TOK_DOT, device=self.device)
        pad = torch.full((B,), TOK_PAD, device=self.device)
        return torch.stack([subj, typ, obj, dot, pad, pad], 1)

    def q_doc(self, ent, qt):
        """qt in 0..2 (rel/attr/comp) -> tokens (B,L_DOC), answer (B,),
        and the two aux fact ids (B,) for read-1 / read-2."""
        B = self.B
        qtok = torch.full((B,), (TOK_QREL, TOK_QATTR, TOK_QCOMP)[qt],
                          device=self.device)
        subj = self.name(ent.unsqueeze(1)).squeeze(1)
        ans_slot = torch.full((B,), TOK_ANS, device=self.device)
        relt = self.rel.gather(1, ent.unsqueeze(1)).squeeze(1)
        if qt == 0:
            ans = self.name(relt.unsqueeze(1)).squeeze(1)
            f1 = f2 = ent * 2                          # rel fact of A
        elif qt == 1:
            ans = self.attr.gather(1, ent.unsqueeze(1)).squeeze(1)
            f1 = f2 = ent * 2 + 1                      # attr fact of A
        else:
            ans = self.attr.gather(1, relt.unsqueeze(1)).squeeze(1)
            f1 = ent * 2                               # rel fact of A
            f2 = relt * 2 + 1                          # attr fact of B
        dot = torch.full((B,), TOK_DOT, device=self.device)
        pad = torch.full((B,), TOK_PAD, device=self.device)
        toks = torch.stack([qtok, subj, ans_slot, ans, dot, pad], 1)
        return toks, ans, f1, f2


# ---------------------------------------------------------------- book
class Book:
    """v6b economy, batched over lifetimes. fact id -> slot tracked by
    the harness for aux supervision."""

    def __init__(self, B, K, d, device, theta=0.75, theta_merge=0.9,
                 cap=32, code_temp=12.0):
        self.K, self.theta, self.theta_merge = K, theta, theta_merge
        self.cap, self.code_temp = cap, code_temp
        g = torch.Generator(device="cpu").manual_seed(7)
        init = F.normalize(torch.randn(K, d, generator=g), dim=-1)
        self.keys = init.to(device).unsqueeze(0).repeat(B, 1, 1)
        self.pays = torch.zeros(B, K, d, device=device)
        self.counts = torch.zeros(B, K, device=device)
        self.merges = torch.zeros(B, device=device)
        self.ptr = torch.zeros(B, dtype=torch.long, device=device)

    @torch.no_grad()
    def write(self, kv, pv, exemplar=False):
        """kv,pv: (B,d). Returns chosen slot (B,)."""
        kv, pv = kv.detach(), pv.detach()
        B = kv.shape[0]
        ar = torch.arange(B, device=kv.device)
        used = self.counts > 0
        neg = torch.finfo(kv.dtype).min
        if exemplar:
            # append-only; FIFO ring under overflow (recency memory)
            i = self.ptr % self.K
            self.ptr += 1
            self.keys[ar, i] = kv
            self.pays[ar, i] = pv
            self.counts[ar, i] = 1
            return i
        sim = torch.einsum("bd,bkd->bk", F.normalize(kv, dim=-1),
                           F.normalize(self.keys, dim=-1))
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
        ki = self.keys[ar, i]
        msim = torch.einsum("bd,bkd->bk", F.normalize(ki, dim=-1),
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
        return i

    def read(self, q):
        """q: (B,T,d) -> payload mixture (B,T,d), attn (B,T,K).
        Differentiable in q (keys/pays are constants) — the gradient
        into the query pathway is the aux loss's channel."""
        sim = torch.einsum("btd,bkd->btk", F.normalize(q, dim=-1),
                           F.normalize(self.keys.detach(), dim=-1))
        used = (self.counts > 0).unsqueeze(1)
        neg = torch.finfo(sim.dtype).min
        sim_u = sim.masked_fill(~used, neg)
        sim_eff = torch.where(used.any(-1, keepdim=True), sim_u, sim)
        w = torch.softmax(sim_eff * self.code_temp, dim=-1)
        # clone: later in-place writes must not touch autograd's saved copy
        return torch.einsum("btk,bkd->btd", w,
                            self.pays.detach().clone()), w


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, heads=4, mlp_ratio=3.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(),
                                 nn.Linear(h, d))

    def forward(self, x, mask):
        y = self.n1(x)
        a, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class ReadHead(nn.Module):
    """Query proj into key space + ZERO-INIT output proj (book starts
    as a no-op; aux ignites it)."""

    def __init__(self, d):
        super().__init__()
        self.wq = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        nn.init.zeros_(self.wo.weight)
        nn.init.zeros_(self.wo.bias)

    def forward(self, x, book):
        mix, w = book.read(self.wq(x))
        return x + self.wo(mix), w


class StreamLM(nn.Module):
    def __init__(self, d=96, layers=4, heads=4, use_book=True):
        super().__init__()
        self.d, self.use_book = d, use_book
        self.emb = nn.Embedding(VOCAB, d)
        pos = torch.zeros(L_DOC, d)
        t = torch.arange(L_DOC).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / d))
        pos[:, 0::2] = torch.sin(t * div)
        pos[:, 1::2] = torch.cos(t * div)
        self.register_buffer("pos", pos * 0.3)
        m = torch.full((L_DOC, L_DOC), float("-inf"))
        self.register_buffer("mask", torch.triu(m, diagonal=1))
        self.blocks = nn.ModuleList(Block(d, heads)
                                    for _ in range(layers))
        self.read1 = ReadHead(d) if use_book else None
        self.read2 = ReadHead(d) if use_book else None
        self.ri = (1, layers - 1)               # read after these blocks
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, toks, book=None):
        """toks: (B,N,L_DOC) a round's documents (B lifetimes x N docs).
        Returns logits (B,N,L_DOC,VOCAB) and read attns
        (w1,w2): (B,N,L_DOC,K) or (None,None)."""
        B, N, L = toks.shape
        x = self.emb(toks.reshape(B * N, L)) + self.pos[:L]
        w1 = w2 = None
        for li, blk in enumerate(self.blocks):
            x = blk(x, self.mask[:L, :L])
            if self.use_book and book is not None:
                if li == self.ri[0]:
                    xb = x.reshape(B, N * L, self.d)
                    xb, w1 = self.read1(xb, book)
                    x = xb.reshape(B * N, L, self.d)
                elif li == self.ri[1]:
                    xb = x.reshape(B, N * L, self.d)
                    xb, w2 = self.read2(xb, book)
                    x = xb.reshape(B * N, L, self.d)
        logits = self.head(self.norm(x)).reshape(B, N, L, VOCAB)
        shp = (B, N, L, -1)
        return logits, \
            (w1.reshape(shp) if w1 is not None else None), \
            (w2.reshape(shp) if w2 is not None else None)


# ------------------------------------------------------------- lifetime
def run_lifetime(model, B, n_e, K, device, arm, q_per_round, quiz_q,
                 aux_w, stmts=2, noise=0.0):
    """One batched lifetime. Returns (loss, stats dict). noise: each
    fact STATEMENT's object token is corrupted with this prob (the LM
    sees the corrupted doc; the write payload inherits it; questions
    grade against the TRUE fact) — one statement stops sufficing,
    consolidation becomes gradeable."""
    world = World(B, n_e, device)
    groups = n_e // 2
    book = None
    slots = torch.zeros(B, n_e * 2, dtype=torch.long, device=device)
    if model.use_book:
        book = Book(B, K, model.d, device)
        if arm == "frozen":
            book.counts[:] = 1.0
            book.pays = torch.randn_like(book.pays)
        elif arm == "oracle":
            with torch.no_grad():
                for i in range(n_e):
                    ent = torch.full((B,), i, dtype=torch.long,
                                     device=device)
                    for kind in (0, 1):
                        doc = world.fact_doc(ent, kind)
                        kv = F.normalize(
                            model.emb(doc[:, 0])
                            + model.emb(doc[:, 1]), dim=-1)
                        pv = model.emb(doc[:, :3]).mean(1)
                        fid = i * 2 + kind
                        book.keys[:, fid] = kv
                        book.pays[:, fid] = pv
                        book.counts[:, fid] = book.cap
                        slots[:, fid] = fid
    loss = torch.zeros((), device=device)
    n_loss = 0
    ok = {t: [0.0, 0] for t in QTYPES}       # overall correct/count
    quiz_ok = {t: [0.0, 0] for t in QTYPES}
    curve = []
    for r in range(groups + 1):
        docs, metas = [], []
        if r < groups:
            ents = [2 * r, 2 * r + 1]
            fdocs = [(e, kind) for e in ents for kind in (0, 1)] * stmts
            fdocs = [fdocs[i] for i in torch.randperm(len(fdocs))]
            for e, kind in fdocs:
                ent = torch.full((B,), e, dtype=torch.long,
                                 device=device)
                doc = world.fact_doc(ent, kind)
                if noise > 0:
                    flip = torch.rand(B, device=device) < noise
                    if kind == 0:
                        rnd = world.name(torch.randint(
                            0, n_e, (B, 1), device=device)).squeeze(1)
                    else:
                        rnd = torch.randint(0, N_V, (B,),
                                            device=device) + TOK_VAL0
                    doc[:, 2] = torch.where(flip, rnd, doc[:, 2])
                if model.use_book and arm in ("live", "exemplar"):
                    with torch.no_grad():
                        kv = F.normalize(model.emb(doc[:, 0])
                                         + model.emb(doc[:, 1]), dim=-1)
                        pv = model.emb(doc[:, :3]).mean(1)
                    i = book.write(kv, pv, exemplar=(arm == "exemplar"))
                    slots[:, e * 2 + kind] = i
                docs.append(doc)
                metas.append(None)
        nq = quiz_q if r == groups else (q_per_round if r > 0 else 0)
        hi = 2 * r if r < groups else n_e
        for qi in range(nq):
            qt = qi % 3
            ent = torch.randint(0, hi, (B,), device=device)
            toks, ans, f1, f2 = world.q_doc(ent, qt)
            docs.append(toks)
            metas.append((qt, ans, f1, f2, r == groups))
        toks = torch.stack(docs, 1)                    # (B,N,L)
        logits, w1, w2 = model(toks, book=book)
        # LM loss (small): predict every next token, PAD excluded
        tgt_lm = toks[:, :, 1:]
        lm_mask = tgt_lm != TOK_PAD
        lm = F.cross_entropy(
            logits[:, :, :-1].reshape(-1, VOCAB)[lm_mask.reshape(-1)],
            tgt_lm.reshape(-1)[lm_mask.reshape(-1)])
        loss = loss + 0.1 * lm
        for di, meta in enumerate(metas):
            if meta is None:
                continue
            qt, ans, f1, f2, is_quiz = meta
            lg = logits[:, di, 2]                      # predicts idx 3
            loss = loss + F.cross_entropy(lg, ans)
            if model.use_book and arm != "frozen" and aux_w > 0:
                s1 = slots.gather(1, f1.unsqueeze(1)).squeeze(1)
                s2 = slots.gather(1, f2.unsqueeze(1)).squeeze(1)
                ar = torch.arange(B, device=device)
                p1 = w1[:, di, 1, :][ar, s1]           # read1 @ subject
                p2 = w2[:, di, 2, :][ar, s2]           # read2 @ ANS
                loss = loss - aux_w * (torch.log(p1 + 1e-9)
                                       + torch.log(p2 + 1e-9)).mean()
            n_loss += 1
            with torch.no_grad():
                hit = (lg.argmax(-1) == ans).float().mean().item()
                t = QTYPES[qt]
                ok[t][0] += hit
                ok[t][1] += 1
                if is_quiz:
                    quiz_ok[t][0] += hit
                    quiz_ok[t][1] += 1
                else:
                    curve.append((r, qt, hit))
    stats = {f"{t}": ok[t][0] / max(ok[t][1], 1) for t in QTYPES}
    stats.update({f"quiz_{t}": quiz_ok[t][0] / max(quiz_ok[t][1], 1)
                  for t in QTYPES})
    stats["curve"] = curve
    if book is not None:
        stats["used"] = float((book.counts > 0).float().sum(1).mean())
        stats["merges"] = float(book.merges.mean())
    return loss / max(n_loss, 1), stats


# ----------------------------------------------------------- train/eval
def train(model, steps, B, n_e, K, lr, device, tag, arm="live",
          q_per_round=6, quiz_q=12, aux_anneal=0.6, log_every=100,
          log_fn=None, stmts=2, noise=0.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        aux_w = max(0.0, 1.0 - step / max(steps * aux_anneal, 1)) \
            if model.use_book else 0.0
        loss, st = run_lifetime(model, B, n_e, K, device, arm,
                                q_per_round, quiz_q, aux_w,
                                stmts=stmts, noise=noise)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % log_every == 0:
            print(f"[{tag}] step {step:5d}  loss {loss.item():.3f}  "
                  f"rel {st['rel']:.3f}  attr {st['attr']:.3f}  "
                  f"comp {st['comp']:.3f}  aux_w {aux_w:.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if log_fn is not None:
                log_fn({f"{tag}/loss": loss.item(),
                        f"{tag}/rel": st["rel"],
                        f"{tag}/attr": st["attr"],
                        f"{tag}/comp": st["comp"],
                        f"{tag}/step": step})


@torch.no_grad()
def eval_arm(model, B, n_e, K, device, arm, batches, q_per_round=6,
             quiz_q=24, stmts=2, noise=0.0):
    model.eval()
    agg = {}
    curve_agg = {}
    for _ in range(batches):
        _, st = run_lifetime(model, B, n_e, K, device, arm,
                             q_per_round, quiz_q, aux_w=0.0,
                             stmts=stmts, noise=noise)
        for k, v in st.items():
            if k == "curve":
                for r, qt, hit in v:
                    curve_agg.setdefault((r, qt), []).append(hit)
            else:
                agg.setdefault(k, []).append(v)
    out = {k: sum(v) / len(v) for k, v in agg.items()}
    out["curve"] = {f"r{r}_{QTYPES[qt]}": sum(v) / len(v)
                    for (r, qt), v in sorted(curve_agg.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-e", type=int, default=8)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux-anneal", type=float, default=0.6,
                    help="aux weight hits 0 at this fraction of steps")
    ap.add_argument("--stmts", type=int, default=2,
                    help="times each fact is stated in the stream")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="per-statement object-token corruption prob")
    ap.add_argument("--k-sweep", type=str, default="",
                    help="comma-separated K values for a live-arm "
                         "capacity sweep at eval (economy under "
                         "forced-join pressure)")
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-prefix", type=str, default="")
    ap.add_argument("--load-prefix", type=str, default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="neocore-stream")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"device={device}", flush=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"stream-s{args.steps}-d{args.d}"
                              f"-ne{args.n_e}",
                         config=vars(args))
    log_fn = (lambda d: run.log(d)) if run else None

    kw = dict(d=args.d, layers=args.layers)
    m_book = StreamLM(use_book=True, **kw).to(device)
    m_dense = StreamLM(use_book=False, **kw).to(device)

    if args.load_prefix:
        m_book.load_state_dict(
            torch.load(f"{args.load_prefix}_book.pt"))
        m_dense.load_state_dict(
            torch.load(f"{args.load_prefix}_dense.pt"))
        print(f"loaded {args.load_prefix}_*", flush=True)
    else:
        train(m_book, args.steps, args.batch, args.n_e, args.k, args.lr,
              device, "book ", arm="live", aux_anneal=args.aux_anneal,
              log_fn=log_fn, stmts=args.stmts, noise=args.noise)
        train(m_dense, args.steps, args.batch, args.n_e, args.k,
              args.lr, device, "dense", log_fn=log_fn,
              stmts=args.stmts, noise=args.noise)
    if args.save_prefix:
        torch.save(m_book.state_dict(), f"{args.save_prefix}_book.pt")
        torch.save(m_dense.state_dict(), f"{args.save_prefix}_dense.pt")

    arms = {}
    for arm, model in (("live", m_book), ("exemplar", m_book),
                       ("frozen", m_book), ("oracle", m_book),
                       ("dense", m_dense)):
        arms[arm] = eval_arm(model, args.eval_batch, args.n_e, args.k,
                             device, arm, args.eval_batches,
                             stmts=args.stmts, noise=args.noise)

    names = list(arms)
    print(f"\nQUIZ accuracy (all entities, final round; chance: "
          f"rel ~{100 / args.n_e:.0f} floor {100 / N_NAMES:.0f}, "
          f"attr/comp ~{100 / N_V:.0f}):")
    print("  qtype  " + "  ".join(f"{n:>9s}" for n in names))
    for t in QTYPES:
        row = "  ".join(f"{arms[n][f'quiz_{t}'] * 100:9.1f}"
                        for n in names)
        print(f"  {t:>5s}  {row}")
    print("\nSTREAM (within-lifetime) accuracy by round, live arm:")
    for k, v in arms["live"]["curve"].items():
        print(f"  {k}: {v * 100:.1f}")
    print(f"\nbook: used {arms['live'].get('used', 0):.1f} "
          f"(facts={2 * args.n_e}), merges "
          f"{arms['live'].get('merges', 0):.2f}; exemplar used "
          f"{arms['exemplar'].get('used', 0):.1f}", flush=True)

    sweep = {}
    if args.k_sweep:
        print("\nCAPACITY sweep, live economy vs exemplar-FIFO "
              f"(facts={2 * args.n_e}, statements="
              f"{2 * args.n_e * args.stmts}):")
        for K_ in (int(x) for x in args.k_sweep.split(",")):
            row = {}
            for arm in ("live", "exemplar"):
                st = eval_arm(m_book, args.eval_batch, args.n_e, K_,
                              device, arm, args.eval_batches,
                              stmts=args.stmts, noise=args.noise)
                row[arm] = {t: st[f"quiz_{t}"] for t in QTYPES}
            sweep[K_] = row
            print(f"  K={K_:3d}  live " + " ".join(
                f"{t} {row['live'][t] * 100:5.1f}" for t in QTYPES)
                + "   exemplar " + " ".join(
                f"{t} {row['exemplar'][t] * 100:5.1f}" for t in QTYPES),
                flush=True)

    if run is not None:
        summary = {a: {k: v for k, v in st.items() if k != "curve"}
                   for a, st in arms.items()}
        if sweep:
            summary["k_sweep"] = {str(k): v for k, v in sweep.items()}
        for a, st in arms.items():
            for k, v in st.items():
                if k != "curve":
                    run.summary[f"{a}_{k}"] = v
        with open("stream_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        art = wandb.Artifact(f"stream-icl-{run.id}", type="results")
        art.add_file("stream_summary.json")
        if args.save_prefix:
            art.add_file(f"{args.save_prefix}_book.pt")
            art.add_file(f"{args.save_prefix}_dense.pt")
        logged = run.log_artifact(art)
        logged.wait()                    # VERIFIED before exit 0
        run.finish()
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
