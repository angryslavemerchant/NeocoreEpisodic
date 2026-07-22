"""Stream world v1: REAL TEXT (GPT-2 BPE), the dataset pipeline smoke
(2026-07-22). Stage 1 of the real-text campaign agreed with Ibanis:
build fact-graph -> template -> stream assembly at subword level, prove
the book-LM answers through it, THEN grow the template bank via LLM
generation and graft into a pretrained model.

Layers (the architecture from the design discussion):
1. FACT GRAPH (pure code, per lifetime): persons and companies with
   NONCE names (syllable-generated, multi-token under BPE — the honest
   hard mode); closed real-word answer sets for industries/professions/
   cities. Relations: founded(P,C), industry(C,I), based_in(C,city),
   works_as(P,prof), partner(P,P2). Two-hop chains by construction:
   P -> founded C -> industry/city; P -> partner P2 -> profession.
2. TEMPLATE BANK (hand-written starter, ~6 paraphrases per relation,
   slot-based; the full LLM-generated bank is stage 2). A held-out
   template per relation is EVAL-ONLY (paraphrase generalization).
3. STREAM ASSEMBLY (pure code, infinite): per lifetime fill slots with
   fresh names; each fact stated `stmts` times through different
   templates; facts phase then quiz phase (round interleaving returns
   at scale); attention is DOCUMENT-LOCAL (one sentence), so the book
   is the only cross-document channel.
4. PROBES: single-hop and two-hop questions, teacher-forced exact
   match over the (multi-token) answer span.

v1 smoke training wheels, declared: writes fire on fact sentences with
HARNESS-ASSIGNED slots (one per fact — filing by identity); the write
CONTENT is the doc's mean token embedding (no learned writer yet).
This isolates the question this smoke must answer — can a from-scratch
BPE-level LM learn to answer multi-token questions THROUGH the book —
from the filing question (learned writer head = stage 3). A live-theta
arm (novelty-gate filing over bag-of-embedding keys) is evaluated as
an instrument to measure how far embedding-bag filing gets at BPE
level; its failure is expected and informative, not gating.

Gate (pre-registered): oracle and live >> dense == floor on all
question types; dense floor: ~0 exact on name answers (multi-token),
~1/8 on closed-set answers. If oracle fails, the harness is broken.
"""

import argparse
import json
import random
import time

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

ENC = tiktoken.get_encoding("gpt2")
L_DOC = 28
PAD = 0                          # compact-vocab pad
UNK = 1                          # unseen BPE piece (rare; rate logged)
_REMAP = None                    # gpt2 id -> compact id
_NVOCAB = None

INDUSTRIES = ["shipping", "textiles", "robotics", "farming",
              "publishing", "insurance", "mining", "tourism"]
PROFESSIONS = ["baker", "lawyer", "pilot", "surgeon", "carpenter",
               "chemist", "fisherman", "architect"]
CITIES = ["Bristol", "Osaka", "Denver", "Turin", "Cairo", "Quebec",
          "Lagos", "Oslo"]

SYL_A = ["Vor", "Kre", "Tav", "Quen", "Mar", "Dor", "Bel", "Sar",
         "Fen", "Gal", "Hul", "Jor", "Lim", "Nav", "Pel", "Rud"]
SYL_B = ["lath", "ssin", "rix", "mar", "dan", "vek", "lor", "nis",
         "bram", "dell", "fort", "gen", "hart", "kell", "mond", "toll"]
CO_SUFFIX = ["Industries", "Group", "Logistics", "Works", "Holdings",
             "Systems"]

# statement templates per relation; LAST one is EVAL-ONLY (held out)
T_FOUNDED = [
    "{P} founded {C}.",
    "{C} was started by {P}.",
    "Years ago, {P} launched {C}.",
    "{P} is the founder of {C}.",
    "It was {P} who built {C} from nothing.",
    "{C} owes its existence to {P}.",
]
T_INDUSTRY = [
    "{C} operates in the {I} business.",
    "{C} is a {I} company.",
    "The core trade of {C} is {I}.",
    "{C} makes its money in {I}.",
    "At heart, {C} has always been a {I} firm.",
    "{C} built its name on {I}.",
]
T_BASED = [
    "{C} is headquartered in {X}.",
    "{C} runs its operations out of {X}.",
    "The main office of {C} is in {X}.",
    "{C} calls {X} home.",
    "You will find {C} based in {X}.",
    "{C} set up shop in {X}.",
]
T_WORKS = [
    "{P} works as a {J}.",
    "{P} makes a living as a {J}.",
    "By trade, {P} is a {J}.",
    "{P} earns his keep as a {J}.",
    "Professionally, {P} is a {J}.",
    "{P} spends his days working as a {J}.",
]
T_PARTNER = [
    "{P} is married to {Q}.",
    "{Q} and {P} are married.",
    "{P} tied the knot with {Q}.",
    "The spouse of {P} is {Q}.",
    "{P} and {Q} are husband and wife.",
    "{P} shares his life with {Q}.",
]

Q_TEMPLATES = {
    "founder": ("Q: Who founded {C}? A:", "P_of_C"),
    "industry": ("Q: What industry is {C} in? A:", "I_of_C"),
    "city": ("Q: Where is {C} based? A:", "X_of_C"),
    "job": ("Q: What does {P} do for work? A:", "J_of_P"),
    "spouse": ("Q: Who is {P} married to? A:", "Q_of_P"),
    "hop_industry": ("Q: What industry is the company that {P} founded"
                     " in? A:", "I_of_founded"),
    "hop_city": ("Q: Where is the company that {P} founded based? A:",
                 "X_of_founded"),
    "hop_job": ("Q: What does the spouse of {P} do for work? A:",
                "J_of_partner"),
}
QTYPES_1 = ["founder", "industry", "city", "job", "spouse"]
QTYPES_2 = ["hop_industry", "hop_city", "hop_job"]


def nonce_person(rng):
    return (rng.choice(SYL_A) + rng.choice(SYL_B) + " "
            + rng.choice(SYL_A) + rng.choice(SYL_B))


def nonce_company(rng):
    return rng.choice(SYL_A) + rng.choice(SYL_B) + " " \
        + rng.choice(CO_SUFFIX)


def build_vocab(path="vocab_text_icl.json"):
    """Compact vocab: the world touches ~2k of gpt2's 50k pieces; a
    50k softmax OOMs a 6 GB GPU for no reason. BPE segmentation is
    UNCHANGED (names stay multi-token) — ids are just reindexed.
    Enumerates all template text and a large name sample; unseen
    pieces at runtime map to UNK (rate logged, expected ~0)."""
    global _REMAP, _NVOCAB
    import os
    if os.path.exists(path):
        with open(path) as f:
            ids = json.load(f)
    else:
        rng = random.Random(7)
        texts = []
        fillers = dict(P="Aa Bb", C="Cc Dd", I=INDUSTRIES[0],
                       X=CITIES[0], J=PROFESSIONS[0], Q="Ee Ff")
        for bank in (T_FOUNDED, T_INDUSTRY, T_BASED, T_WORKS,
                     T_PARTNER):
            for t in bank:
                texts.append(t.format(**fillers))
        for tpl, _ in Q_TEMPLATES.values():
            texts.append(tpl.format(**fillers))
        texts += [" " + w for w in INDUSTRIES + PROFESSIONS + CITIES]
        texts += [w for w in INDUSTRIES + PROFESSIONS + CITIES]
        for _ in range(30000):
            n = nonce_person(rng)
            texts += [n, " " + n]
        for _ in range(8000):
            n = nonce_company(rng)
            texts += [n, " " + n]
        ids = sorted({i for t in texts for i in ENC.encode(t)})
        with open(path, "w") as f:
            json.dump(ids, f)
    _REMAP = {g: i + 2 for i, g in enumerate(ids)}   # 0=PAD 1=UNK
    _NVOCAB = len(ids) + 2
    return _NVOCAB


def enc_c(text):
    return [_REMAP.get(i, UNK) for i in ENC.encode(text)]


class Lifetime:
    """One lifetime's fact graph + document stream (token ids)."""

    def __init__(self, rng, n_p=6, n_c=4, stmts=2, holdout_tpl=True,
                 n_quiz1=6, n_quiz2=6):
        self.rng = rng
        self.persons = []
        seen = set()
        while len(self.persons) < n_p:
            p = nonce_person(rng)
            if p not in seen:
                seen.add(p)
                self.persons.append(p)
        self.companies = []
        while len(self.companies) < n_c:
            c = nonce_company(rng)
            if c not in seen:
                seen.add(c)
                self.companies.append(c)
        self.founder = {c: self.persons[i]
                        for i, c in enumerate(self.companies)}
        self.industry = {c: rng.choice(INDUSTRIES)
                         for c in self.companies}
        self.city = {c: rng.choice(CITIES) for c in self.companies}
        self.job = {p: rng.choice(PROFESSIONS) for p in self.persons}
        pairs = list(self.persons)
        rng.shuffle(pairs)
        self.partner = {}
        for a, b in zip(pairs[0::2], pairs[1::2]):
            self.partner[a] = b
            self.partner[b] = a
        # facts: (fid, text_variants, canonical)
        self.facts = []
        lim = -1 if holdout_tpl else None
        for c in self.companies:
            p = self.founder[c]
            self.facts.append([t.format(P=p, C=c)
                               for t in T_FOUNDED[:lim]])
            self.facts.append([t.format(C=c, I=self.industry[c])
                               for t in T_INDUSTRY[:lim]])
            self.facts.append([t.format(C=c, X=self.city[c])
                               for t in T_BASED[:lim]])
        for p in self.persons:
            self.facts.append([t.format(P=p, J=self.job[p])
                               for t in T_WORKS[:lim]])
        done = set()
        for a in self.persons:
            if a in self.partner and a not in done:
                b = self.partner[a]
                done.add(a)
                done.add(b)
                self.facts.append([t.format(P=a, Q=b)
                                   for t in T_PARTNER[:lim]])
        # fact index maps for aux targets
        self.fid_founded = {c: i * 3 for i, c in enumerate(self.companies)}
        self.fid_industry = {c: i * 3 + 1
                             for i, c in enumerate(self.companies)}
        self.fid_city = {c: i * 3 + 2
                         for i, c in enumerate(self.companies)}
        base = 3 * n_c
        self.fid_job = {p: base + i for i, p in enumerate(self.persons)}
        self.fid_partner = {}
        k = base + n_p
        done = set()
        for a in self.persons:
            if a in self.partner and a not in done:
                b = self.partner[a]
                self.fid_partner[a] = k
                self.fid_partner[b] = k
                done.add(a)
                done.add(b)
                k += 1
        self.n_facts = k
        # stream: fact docs (stmts distinct paraphrases each), shuffled
        self.docs = []          # (tokens, kind, fid, aux1, aux2, ans_span)
        for fid, variants in enumerate(self.facts):
            tpls = rng.sample(variants, min(stmts, len(variants)))
            for text in tpls:
                self.docs.append((enc_c(text), "fact", fid,
                                  -1, -1, None))
        rng.shuffle(self.docs)
        # quiz
        self.quiz = []
        for _ in range(n_quiz1):
            qt = rng.choice(QTYPES_1)
            self.quiz.append(self._make_q(qt))
        for _ in range(n_quiz2):
            qt = rng.choice(QTYPES_2)
            self.quiz.append(self._make_q(qt))
        self.docs += self.quiz

    def _make_q(self, qt):
        rng = self.rng
        tpl, _ = Q_TEMPLATES[qt]
        if qt in ("founder", "industry", "city"):
            c = rng.choice(self.companies)
            text = tpl.format(C=c)
            ans = {"founder": self.founder[c],
                   "industry": self.industry[c],
                   "city": self.city[c]}[qt]
            aux1 = {"founder": self.fid_founded[c],
                    "industry": self.fid_industry[c],
                    "city": self.fid_city[c]}[qt]
            aux2 = aux1
        elif qt == "job":
            p = rng.choice(self.persons)
            text = tpl.format(P=p)
            ans = self.job[p]
            aux1 = aux2 = self.fid_job[p]
        elif qt == "spouse":
            p = rng.choice([x for x in self.persons
                            if x in self.partner])
            text = tpl.format(P=p)
            ans = self.partner[p]
            aux1 = aux2 = self.fid_partner[p]
        elif qt in ("hop_industry", "hop_city"):
            c = rng.choice(self.companies)
            p = self.founder[c]
            text = tpl.format(P=p)
            ans = (self.industry[c] if qt == "hop_industry"
                   else self.city[c])
            aux1 = self.fid_founded[c]
            aux2 = (self.fid_industry[c] if qt == "hop_industry"
                    else self.fid_city[c])
        else:                                   # hop_job
            p = rng.choice([x for x in self.persons
                            if x in self.partner])
            q = self.partner[p]
            text = tpl.format(P=p)
            ans = self.job[q]
            aux1 = self.fid_partner[p]
            aux2 = self.fid_job[q]
        q_ids = enc_c(text)
        a_ids = enc_c(" " + ans)
        toks = q_ids + a_ids
        span = (len(q_ids), len(toks))          # answer positions
        return (toks, "q_" + qt, -1, aux1, aux2, span)


def build_batch(B, device, rng, stmts=2, holdout_tpl=False):
    """Returns tensors for B lifetimes: toks (B,D,L), kinds, fids,
    aux1/aux2 (B,D), ans_mask (B,D,L), plus per-doc qtype labels.
    All lifetimes share doc-count layout (same n_p/n_c/quiz sizes)."""
    lts = [Lifetime(rng, stmts=stmts, holdout_tpl=holdout_tpl)
           for _ in range(B)]
    D = len(lts[0].docs)
    toks = torch.full((B, D, L_DOC), PAD, dtype=torch.long)
    fids = torch.full((B, D), -1, dtype=torch.long)
    aux1 = torch.full((B, D), -1, dtype=torch.long)
    aux2 = torch.full((B, D), -1, dtype=torch.long)
    ans_mask = torch.zeros(B, D, L_DOC, dtype=torch.bool)
    sub_last = torch.zeros(B, D, dtype=torch.long)  # last name tok pos
    kinds = [[None] * D for _ in range(B)]
    for b, lt in enumerate(lts):
        assert len(lt.docs) == D, "lifetime layout mismatch"
        for d, (ids, kind, fid, a1, a2, span) in enumerate(lt.docs):
            ids = ids[:L_DOC]
            toks[b, d, :len(ids)] = torch.tensor(ids)
            fids[b, d] = fid
            aux1[b, d] = a1
            aux2[b, d] = a2
            kinds[b][d] = kind
            if span is not None:
                s, e = span
                e = min(e, L_DOC)
                ans_mask[b, d, s:e] = True
                sub_last[b, d] = s - 1          # pos before answer
    return (toks.to(device), fids.to(device), aux1.to(device),
            aux2.to(device), ans_mask.to(device), sub_last.to(device),
            kinds, lts[0].n_facts)


# ---------------------------------------------------------------- model
from toy_stream_icl import Block, Book, ReadHead  # reuse organs


class TextStreamLM(nn.Module):
    def __init__(self, d=256, layers=6, heads=8, use_book=True,
                 vocab=50257):
        super().__init__()
        self.d, self.use_book = d, use_book
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)   # gpt2-style; tied
        pos = torch.zeros(L_DOC, d)
        t = torch.arange(L_DOC).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / d))
        pos[:, 0::2] = torch.sin(t * div)
        pos[:, 1::2] = torch.cos(t * div)
        # match the 0.02-scale embedding regime (pos must not drown it)
        self.register_buffer("pos", pos * 0.02)
        m = torch.full((L_DOC, L_DOC), float("-inf"))
        self.register_buffer("mask", torch.triu(m, diagonal=1))
        self.blocks = nn.ModuleList(Block(d, heads)
                                    for _ in range(layers))
        self.read1 = ReadHead(d) if use_book else None
        self.read2 = ReadHead(d) if use_book else None
        self.ri = (2, layers - 1)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight          # tied

    def forward(self, toks, book=None):
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
        logits = self.head(self.norm(x)).reshape(B, N, L, -1)
        shp = (B, N, L, -1)
        return logits, \
            (w1.reshape(shp) if w1 is not None else None), \
            (w2.reshape(shp) if w2 is not None else None)


# ------------------------------------------------------------- lifetime
def run_batch(model, batch, K, device, arm, aux_w):
    (toks, fids, aux1, aux2, ans_mask, sub_last, kinds, n_facts) = batch
    B, D, L = toks.shape
    book = None
    if model.use_book:
        book = Book(B, K, model.d, device)
        if arm == "frozen":
            book.counts[:] = 1.0
            book.pays = torch.randn_like(book.pays)
    is_fact = fids >= 0
    if model.use_book and arm in ("live", "oracle", "live-theta"):
        with torch.no_grad():
            emb = model.emb(toks)                      # (B,D,L,d)
            npad = (toks != PAD).unsqueeze(-1).float()
            mean = (emb * npad).sum(2) / npad.sum(2).clamp(min=1)
            for d_i in range(D):
                if not bool(is_fact[:, d_i].any()):
                    continue
                kv = F.normalize(mean[:, d_i], dim=-1)
                pv = mean[:, d_i]
                if arm == "live-theta":
                    book.write(kv, pv)
                else:
                    fid = fids[:, d_i].clamp(min=0)
                    ar = torch.arange(B, device=device)
                    n = book.counts[ar, fid] + 1
                    lr = (1.0 / n.clamp(max=book.cap)).unsqueeze(-1)
                    book.keys[ar, fid] += lr * (kv - book.keys[ar, fid])
                    book.pays[ar, fid] += lr * (pv - book.pays[ar, fid])
                    book.counts[ar, fid] = n
    logits, w1, w2 = model(toks, book=book)
    # LM loss everywhere (the model must learn English-from-templates)
    tgt = toks[:, :, 1:]
    keep = tgt != PAD
    lm = F.cross_entropy(
        logits[:, :, :-1].reshape(-1, logits.shape[-1])[keep.reshape(-1)],
        tgt.reshape(-1)[keep.reshape(-1)])
    loss = lm
    # answer CE (extra weight) + aux on question docs
    amask = ans_mask[:, :, 1:] & keep
    if bool(amask.any()):
        ans_ce = F.cross_entropy(
            logits[:, :, :-1].reshape(-1, logits.shape[-1])
            [amask.reshape(-1)],
            tgt.reshape(-1)[amask.reshape(-1)])
        loss = loss + 2.0 * ans_ce
    if model.use_book and arm in ("live", "oracle") and aux_w > 0 \
            and w1 is not None:
        is_q = aux1 >= 0
        if bool(is_q.any()):
            ar_b, ar_d = torch.nonzero(is_q, as_tuple=True)
            pos = sub_last[ar_b, ar_d]
            s1 = aux1[ar_b, ar_d]
            s2 = aux2[ar_b, ar_d]
            p1 = w1[ar_b, ar_d, pos, :].gather(
                1, s1.unsqueeze(1)).squeeze(1)
            p2 = w2[ar_b, ar_d, pos, :].gather(
                1, s2.unsqueeze(1)).squeeze(1)
            loss = loss - aux_w * (torch.log(p1 + 1e-9)
                                   + torch.log(p2 + 1e-9)).mean()
    # metrics
    stats = {}
    with torch.no_grad():
        pred = logits[:, :, :-1].argmax(-1)
        ok_tok = (pred == tgt) | ~amask
        per_doc_ok = ok_tok.all(-1) & (amask.any(-1))
        for fam, prefix in ((QTYPES_1, "hop1"), (QTYPES_2, "hop2")):
            hits, tot = 0, 0
            for b in range(B):
                for d_i in range(D):
                    k = kinds[b][d_i]
                    if k and k.startswith("q_") and k[2:] in fam:
                        tot += 1
                        hits += int(per_doc_ok[b, d_i])
            stats[prefix] = hits / max(tot, 1)
        if book is not None:
            stats["used"] = float((book.counts > 0)
                                  .float().sum(1).mean())
            stats["merges"] = float(book.merges.mean())
        stats["lm_loss"] = float(lm)
    return loss, stats


def train(model, steps, B, K, lr, device, tag, arm, aux_anneal,
          stmts, log_every=100, log_fn=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=0.01)
    model.train()
    rng = random.Random(1234)
    t0 = time.time()
    for step in range(1, steps + 1):
        aux_w = max(0.0, 1.0 - step / max(steps * aux_anneal, 1)) \
            if (model.use_book and aux_anneal > 0) else 0.0
        batch = build_batch(B, device, rng, stmts=stmts)
        loss, st = run_batch(model, batch, K, device, arm, aux_w)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % log_every == 0:
            print(f"[{tag}] step {step:5d}  loss {loss.item():.3f}  "
                  f"lm {st['lm_loss']:.3f}  hop1 {st['hop1']:.3f}  "
                  f"hop2 {st['hop2']:.3f}  aux_w {aux_w:.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if log_fn:
                log_fn({f"{tag}/loss": loss.item(),
                        f"{tag}/hop1": st["hop1"],
                        f"{tag}/hop2": st["hop2"],
                        f"{tag}/step": step})


@torch.no_grad()
def eval_arm(model, B, K, device, arm, batches, stmts,
             holdout_tpl=False, seed=999):
    model.eval()
    rng = random.Random(seed)
    agg = {}
    for _ in range(batches):
        batch = build_batch(B, device, rng, stmts=stmts,
                            holdout_tpl=holdout_tpl)
        _, st = run_batch(model, batch, K, device, arm, 0.0)
        for k, v in st.items():
            agg.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--aux-anneal", type=float, default=0.5,
                    help="0 disables aux entirely")
    ap.add_argument("--stmts", type=int, default=2)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--eval-batches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-prefix", type=str, default="")
    ap.add_argument("--load-prefix", type=str, default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str,
                    default="neocore-stream")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"device={device}", flush=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"text-s{args.steps}-d{args.d}",
                         config=vars(args))
    log_fn = (lambda x: run.log(x)) if run else None

    nv = build_vocab()
    print(f"compact vocab: {nv} pieces (of 50257 gpt2)", flush=True)
    kw = dict(d=args.d, layers=args.layers, vocab=nv)
    m_book = TextStreamLM(use_book=True, **kw).to(device)
    m_dense = TextStreamLM(use_book=False, **kw).to(device)
    if args.load_prefix:
        m_book.load_state_dict(torch.load(
            f"{args.load_prefix}_book.pt"))
        m_dense.load_state_dict(torch.load(
            f"{args.load_prefix}_dense.pt"))
        print("loaded", flush=True)
    else:
        train(m_book, args.steps, args.batch, args.k, args.lr, device,
              "book ", "live", args.aux_anneal, args.stmts,
              log_fn=log_fn)
        train(m_dense, args.steps, args.batch, args.k, args.lr,
              device, "dense", "live", 0, args.stmts, log_fn=log_fn)
    if args.save_prefix:
        torch.save(m_book.state_dict(),
                   f"{args.save_prefix}_book.pt")
        torch.save(m_dense.state_dict(),
                   f"{args.save_prefix}_dense.pt")

    arms = {}
    for arm, mdl in (("live", m_book), ("live-theta", m_book),
                     ("frozen", m_book), ("dense", m_dense)):
        arms[arm] = eval_arm(mdl, args.eval_batch, args.k, device,
                             arm, args.eval_batches, args.stmts)
    arms["live-holdout-tpl"] = eval_arm(
        m_book, args.eval_batch, args.k, device, "live",
        args.eval_batches, args.stmts, holdout_tpl=True)

    print("\nEXACT-match accuracy (multi-token answers; floors: "
          "names ~0, closed sets ~12.5):")
    print("  metric  " + "  ".join(f"{n:>16s}" for n in arms))
    for m in ("hop1", "hop2"):
        row = "  ".join(f"{arms[n][m] * 100:16.1f}" for n in arms)
        print(f"  {m:>6s}  {row}")
    print(f"\nbook used (live) {arms['live'].get('used', 0):.1f}; "
          f"live-theta used {arms['live-theta'].get('used', 0):.1f} "
          f"merges {arms['live-theta'].get('merges', 0):.2f}",
          flush=True)

    if run:
        for a, st in arms.items():
            for k, v in st.items():
                run.summary[f"{a}_{k}"] = v
        with open("text_summary.json", "w") as f:
            json.dump(arms, f, indent=2)
        import wandb as wb
        art = wb.Artifact(f"text-icl-{run.id}", type="results")
        art.add_file("text_summary.json")
        if args.save_prefix:
            art.add_file(f"{args.save_prefix}_book.pt")
            art.add_file(f"{args.save_prefix}_dense.pt")
        run.log_artifact(art).wait()
        run.finish()
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
