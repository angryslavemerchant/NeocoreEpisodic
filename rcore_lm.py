"""Rung 1 of the REASONING-CORE LM (2026-07-22 pivot; spec frozen in
CLAUDE.md entry point + POI session close).

A general small LM with the missing middle timescale built as
architecture:

  archive   the model's OWN lower layers (v3 unified stack, Ibanis):
            sliding-window causal attention only — their per-token
            states are the archive, retained for the whole stream
            (unbounded at inference), never touched by the upper
            stack directly. Trained end to end: long-range gradient
            reaches them only through being selected.
  core      a selective FILTER, not a store (v2, Ibanis's mid-build
            correction; selection over superposition — the vision-era
            law). At every C-token chunk boundary the K-token BUFFER
            is reselected from the whole archive by QK scores whose
            queries come from the filter's own state (n_q learned
            query vectors, updated each boundary from the current
            buffer — top-down attention). Sampled exact-K (Gumbel
            top-K, learnable tau; gradient to the scorer through the
            sigmoid gate on selected content — the policy validated
            in train_pixel_icl). The buffer holds RAW selected token
            states; accumulation lives in the archive (eviction
            loses nothing), so free per-boundary reselection = gaze.
  decoder   from-scratch causal LM that attends ONLY
            [W-token local window || K-token buffer] — never the
            cache (free-decoder law: admission is the sole long-range
            route). Cross-reads at two depths, small-gain init.

Train chunked (admission queries frozen per chunk from the boundary
core state); per-token re-admission is the inference mode (later
rung). W >= C so window+core cover the whole past with no hole.

Arms: learned-gaze / RANDOM-GAZE (the conscience) / dense twin (full
causal attention, no core — the standard recipe). Toy adds
oracle-gaze (admit exactly fact tokens) and nocore (window only).

PRE-REGISTERED BAR (Ibanis): (a) story perplexity ~parity with the
dense twin; (c) recall/composition wins on the stream-world probes;
(b) admission maps are the instrument, not the goal. KILL CRITERION:
learned <= random gaze after ignition-tuned training.

World: 80% packed TinyStories, 20% fact-streams — stream_text_v2's
bank world at GPT-2 real vocab, docs concatenated into ONE continuous
stream (TinyStories prefix pads to T). Question gap >> W forces
recall through the core. Bisect curricula built in: mid-stream
questions only after --sq-warmup of training (D2 law); no abstention
at rung 1.

TOY GATE (--toy, local, $0 — pre-registered, must pass before any
rental):
  leak-causality   perturb token p => logits < p unchanged
  leak-window      M=0 (core inert): perturb p => logits > p+W
                   unchanged (decoder provably cannot reach the cache)
  grad-flow        scorer (q_proj/key_proj/tau) receives gradient
  dissociation     oracle >= 85 query acc; nocore <= 12 (floor);
                   learned >= 2x random; learned fact-gaze >= 2x
                   random's fact-gaze
Toy world: per-stream random name->val bindings (unsmearable), facts
in the first chunks, queries beyond the window.

SMOKE LOG:
  v0 (1000 steps, global random toy encoder): leakage+grad PASS;
     dissociation FAIL — oracle 4.1 = chance WITH perfect gaze
     (0.977). Diagnosis: random GLOBAL encoder attention dilutes the
     name->val binding out of the fact token's state (~1/250); the
     circuit had nothing to store. Fix: local-window (4) toy encoder
     (mirrors a trained encoder's local concentration) + 2000 steps
     + boost 8.
  v1 (register design + local encoder, 2000 steps): FAIL but
     unblocked — oracle 17.0 / learned 14.3 / random 9.0 / nocore
     4.1, still climbing; learned fact-gaze 0.305 vs random 0.142
     (the scorer learns informativeness even here). Registers form
     the circuit SLOWLY (all registers attend the same admitted
     soup; nothing forces specialization — the slot problem).
     Superseded by v2 mid-build (Ibanis): core = FILTER; buffer =
     the K selected tokens themselves; registers deleted.
  v2 (buffer design, 2000 steps): FAIL on magnitude, shape CORRECT —
     oracle 41.1 climbing all run / learned 15.9 / random 9.0 /
     nocore 4.1 = chance; learned gaze 0.290 >= 2x random PASSES;
     tau drifts DOWN in learned (1.0->0.84: commitment grows with
     scorer quality). Buffer >> registers (41 vs 17 same budget).
     Rate limiter diagnosed as read-path gain (cross init 0.005,
     scorer grads ~1e-8). v3: cross init 0.02 + 4000-step horizon.
  v3 (4000 steps, local): oracle 64.7 / learned 32.2 / random 12.4 /
     nocore 4.1 — all still climbing; init change was NOT the rate
     limiter (identical curve to v2 through 2k). Continuous slow
     circuit growth, not a phase jump.
  v3b-cloud (8000 steps, m68306 PRO 6000, ~80 s/arm): oracle 74.2 /
     LEARNED 55.3 (gaze 0.466, tau -> 0.38: hard commitment) /
     random 15.1. 3/4 checks PASS incl. learned >= max(2x random,
     40); only oracle>=85 short — slow asymptote (43/65/74 at
     2k/4k/8k), a toy-decoder convergence-speed issue, not a
     circuit-validity issue. Kill-criterion signal decisive at toy
     scale: learned 3.7x random. Next: 20k-step gate horizon
     (checks unchanged).
  REAL RUNS 1-2 (frozen-archive config, 2026-07-23): KILL CRITERION
     FIRED — learned ~= random ~= s2 on far recall (~17%) at both
     signal mixes; smoking gun = randgaze invariance (learned model
     scores identically with a random buffer: the reader never used
     the buffer). S2 gaze aimed 3x better (0.35 vs 0.11) with zero
     recall payoff. Dense twin: 86.6 h1_far / 68.5 held-out para /
     38.3 h2_far — full attention + trainable representations
     ignite lookup easily; core arms still win story ppl (7.14 vs
     7.30). Diagnosis: frozen off-task archive states are not
     retrievably decodable + whisper gradient. => v3 unified stack
     (this version): archive = own windowed lower layers, trained.
  20k-cloud GATE RESOLUTION: oracle PLATEAUS at 74.7 (74.2@8k) =
     the TOY'S information ceiling, not the circuit's — the toy
     archive is a RANDOM encoder; some name->val hashes are not
     decodable at d=64 regardless of selection. learned 60.7 (81%
     of ceiling, still climbing, gaze 0.481); random 26.2. All
     structural checks PASS across every iteration. The oracle>=85
     bar was unreachable by rig construction; overridden with this
     diagnosis on record. Real run launched WITHOUT further toy
     iterations (the real encoder is trained — no such ceiling).
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

EOT = 50256
NL = 198          # '\n' in GPT-2 BPE


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CrossRead(nn.Module):
    """Decoder cross-attention into the chunk's core state. Small-gain
    output init: near-no-op at start but differentiable everywhere
    (a hard zero init would cut the scorer's gradient path)."""

    def __init__(self, d, heads):
        super().__init__()
        self.nx = nn.LayerNorm(d)
        self.nc = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        # v2 gate: std 0.005 starved the read path (scorer grads
        # ~1e-8; oracle climbed but slowly). Standard gain ignites.
        nn.init.normal_(self.attn.out_proj.weight, std=0.02)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x, core):
        c = self.nc(core)
        a, _ = self.attn(self.nx(x), c, c, need_weights=False)
        return x + a


class RCoreLM(nn.Module):
    """policy: learned | random | oracle. use_core=False + full_attn
    => the dense twin; use_core=False + window => nocore floor.

    v3 (Ibanis, unified stack): ONE decoder-only model. The first
    low_layers are the archive-builders — SLIDING-WINDOW attention
    only (full attention there would smuggle long-range info up the
    residual stream past the core; globality law). The sparsity
    core sits in the middle: the gaze selects K lower-stack token
    states from the whole past at each chunk boundary. The upper
    layers attend [window || buffer] via cross-reads. Dense twin =
    the SAME stack, full attention, no core — exact param match.
    Archive states train end to end; long-range gradient reaches
    the lower stack ONLY through being selected.

    v2 (kept): the core is a FILTER, not a store — the buffer IS
    the K selected token states (selection over superposition); the
    filter's persistent state is n_q query vectors updated from
    what it gathered, expressible only through selection."""

    def __init__(self, vocab, d=256, layers=8, low_layers=3,
                 heads=8, max_t=2048, window=64, chunk=64, k_buf=48,
                 n_query=8, policy="learned", use_core=True,
                 full_attn=False, cross_at=None, s_rounds=1,
                 w_low=None):
        super().__init__()
        assert policy in ("learned", "random", "oracle")
        assert max_t % chunk == 0
        assert 0 < low_layers < layers
        assert window >= chunk - 1 or full_attn or not use_core
        self.d, self.W, self.C = d, window, chunk
        # archive layers may use a NARROWER window (local summaries;
        # wide windows dilute a fact out of its own token's state)
        self.Wl = w_low if w_low else window
        self.K, self.n_query = k_buf, n_query
        # s_rounds: gaze passes per boundary — select, update queries
        # from what was found, reselect (weight-shared; the pixel-era
        # recursive-admission result, transplanted). Decoder consumes
        # the FINAL round's buffer. S=1 == single-pass gaze.
        self.S = s_rounds
        self.gaze_on = True
        self.policy, self.use_core, self.full_attn = \
            policy, use_core, full_attn
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos = nn.Parameter(torch.randn(1, max_t, d) * 0.02)
        self.low = nn.ModuleList(Block(d, heads)
                                 for _ in range(low_layers))
        self.high = nn.ModuleList(Block(d, heads)
                                  for _ in range(layers - low_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.scale = 1.0 / math.sqrt(d)
        self._masks = {}
        if use_core:
            n_high = layers - low_layers
            self.cross_at = tuple(cross_at) if cross_at \
                else (1, n_high - 1)
            self.enc_proj = nn.Linear(d, d)
            self.key_ln = nn.LayerNorm(d)
            self.key_proj = nn.Linear(d, d)     # per-token, no mixing
            # softplus(0.5413) = 1.0 — learnable sampling temperature
            self.raw_tau = nn.Parameter(torch.tensor(0.5413))
            # the filter's state: learned queries + buffer-conditioned
            # update (the validated pixel-ICL admission stack)
            self.query0 = nn.Parameter(
                torch.randn(1, n_query, d) * 0.02)
            self.q_update = nn.MultiheadAttention(d, heads,
                                                  batch_first=True)
            self.q_norm = nn.LayerNorm(d)
            # chunk-0 buffer (no candidates yet): learned empty bank
            self.empty_buf = nn.Parameter(
                torch.randn(1, k_buf, d) * 0.02)
            self.cross = nn.ModuleList(
                CrossRead(d, heads) for _ in self.cross_at)

    @property
    def tau(self):
        return F.softplus(self.raw_tau) + 1e-3

    @property
    def tau_val(self):
        return float(self.tau.detach())

    def _mask(self, T, device, w=None):
        """Bool mask (True = blocked) — dtype-safe under autocast.
        NOTE: stacked local layers give the decoder an effective
        reach of sum-of-layer-windows; anything farther is provably
        core-only (the leak test + the `far` question split both
        bound it by layers*W)."""
        w = self.W if w is None else w
        key = (T, device, w, self.full_attn)
        if key not in self._masks:
            i = torch.arange(T, device=device)
            if self.full_attn:
                keep = i.unsqueeze(0) <= i.unsqueeze(1)
            else:
                j = i.unsqueeze(0)
                ii = i.unsqueeze(1)
                keep = (j <= ii) & (j >= ii - w)
            self._masks[key] = ~keep
        return self._masks[key]

    def core_pass(self, enc_h, oracle_bias=None, collect=False,
                  gaze_override=None):
        """The gaze: at each chunk boundary, reselect the K-token
        buffer from ALL archive tokens < t*C (causal at chunk
        grain). Returns the buffer trajectory (B, nch, K, d) — raw
        selected token states, not summaries."""
        B, T, _ = enc_h.shape
        nch = T // self.C
        empty = self.empty_buf.expand(B, -1, -1)
        if not self.gaze_on:            # leak test: decoder sees a
            return empty.unsqueeze(1).expand(  # constant bank only
                -1, nch, -1, -1), []
        e = self.enc_proj(enc_h)
        kk = self.key_proj(self.key_ln(enc_h))
        pol = gaze_override or self.policy
        q = self.query0.expand(B, -1, -1)
        bufs, admits = [empty], []
        for t in range(1, nch):
            n = t * self.C
            for _r in range(self.S):
                s = torch.einsum("bqd,bnd->bqn",
                                 q, kk[:, :n]) * self.scale
                s = s.max(1).values / self.tau
                if pol == "random":
                    key = torch.rand_like(s.detach().float())
                elif pol == "oracle":
                    key = oracle_bias[:, :n] \
                        + torch.rand_like(s.detach().float())
                else:
                    # Gumbel top-K == sampling w/o replacement from
                    # softmax(s); stochastic at eval too — the policy
                    gum = -torch.log(-torch.log(
                        torch.rand_like(s.detach().float())
                        .clamp_(1e-9, 1 - 1e-9)))
                    key = s.detach().float() + gum
                k = min(self.K, n)
                idx = key.topk(k, dim=1).indices
                sel = e.gather(
                    1, idx.unsqueeze(-1).expand(-1, -1, self.d))
                if pol == "learned":
                    gate = torch.sigmoid(
                        s.gather(1, idx)).unsqueeze(-1)
                    sel = sel * (1 + gate)  # scorer/tau gradient path
                if k < self.K:              # pad early buffers
                    sel = torch.cat(
                        [sel, empty[:, :self.K - k]], dim=1)
                # filter-state update: queries see what they gathered
                dq, _ = self.q_update(q, sel, sel, need_weights=False)
                q = self.q_norm(q + dq)
            bufs.append(sel)
            if collect:
                admits.append(idx.detach().cpu())
        return torch.stack(bufs, dim=1), admits

    def forward(self, toks, oracle_bias=None,
                collect=False, gaze_override=None):
        B, T = toks.shape
        x = self.emb(toks) + self.pos[:, :T]
        mask = self._mask(T, toks.device)
        mlow = self._mask(T, toks.device, self.Wl)
        for blk in self.low:
            x = blk(x, mlow)          # archive-builder (windowed)
        cores = None
        admits = []
        if self.use_core:
            assert T % self.C == 0
            nch = T // self.C
            cores, admits = self.core_pass(
                x, oracle_bias, collect, gaze_override)
        ci = 0
        for li, blk in enumerate(self.high):
            x = blk(x, mask)
            if self.use_core and li in self.cross_at:
                xc = x.reshape(B * nch, self.C, self.d)
                cc = cores.reshape(B * nch, self.K, self.d)
                x = self.cross[ci](xc, cc).reshape(B, T, self.d)
                ci += 1
        logits = self.head(self.norm(x))
        return logits, admits


# ---------------------------------------------------------------------------
# Toy world + gate
# ---------------------------------------------------------------------------

T_TOY, C_TOY, W_TOY = 256, 32, 32
FACT_TK, Q_TK = 3, 4
NAMES = (250, 275)
VALS = (275, 300)
V_TOY = 300


_TOY_PERM = None


def _toy_perm():
    """Fixed Markov successor over filler tokens: the toy stream has
    LEARNABLE structure, so the trained archive layer gets a real LM
    signal to stabilize on (pure-noise fillers made the unified
    stack's archive representation a random walk — v4 gate fix)."""
    global _TOY_PERM
    if _TOY_PERM is None:
        g = random.Random(31337)
        p = list(range(10, 250))
        g.shuffle(p)
        _TOY_PERM = {10 + i: p[i] for i in range(240)}
    return _TOY_PERM


def build_toy_batch(B, rng, device, n_fact=5, n_q=3):
    """Streams of length T+1: Markov fillers + n_fact
    [FACT name val] triples in the first 3 chunks + n_q [Q name val]
    queries beyond the window. name->val random PER STREAM
    (unsmearable); filler transitions FIXED across streams
    (smearable — that is the point: the lower layer learns them)."""
    perm = _toy_perm()
    toks = torch.empty(B, T_TOY + 1, dtype=torch.long)
    for b in range(B):
        cur = rng.randrange(10, 250)
        row = []
        for _ in range(T_TOY + 1):
            row.append(cur)
            cur = perm[cur] if rng.random() < 0.85 \
                else rng.randrange(10, 250)
        toks[b] = torch.tensor(row)
    fact_mask = torch.zeros(B, T_TOY + 1, dtype=torch.bool)
    ans_tgt = torch.zeros(B, T_TOY, dtype=torch.bool)
    queries = []
    for b in range(B):
        names = rng.sample(range(*NAMES), n_fact)
        vals = [rng.randrange(*VALS) for _ in range(n_fact)]
        # facts end <= 62; queries start >= 192: gap 130 > 3*W = 96
        # (the decoder's stacked-window reach) — core is the only route
        f_slots = [3 * k for k in rng.sample(range(20), n_fact)]
        for i, f0 in enumerate(f_slots):
            toks[b, f0], toks[b, f0 + 1], toks[b, f0 + 2] = \
                FACT_TK, names[i], vals[i]
            fact_mask[b, f0:f0 + 3] = True
        q_slots = [3 * k for k in rng.sample(range(60, 84), n_q)]
        picks = rng.sample(range(n_fact), n_q)
        for qi, q0 in zip(picks, q_slots):
            toks[b, q0], toks[b, q0 + 1], toks[b, q0 + 2] = \
                Q_TK, names[qi], vals[qi]
            ans_tgt[b, q0 + 1] = True      # target idx of the val token
            queries.append((b, q0 + 2))
    oracle_bias = torch.where(fact_mask[:, :T_TOY], 8.0, 0.0)
    return (toks.to(device), fact_mask.to(device),
            ans_tgt.to(device), queries, oracle_bias.to(device))


def toy_model(arm, device, seed=0):
    # k_buf=16 >= the 15 fact tokens: oracle's ceiling stays ~100.
    # Unified stack v4: 1 archive layer (w_low=8 — local summaries)
    # + 3 upper (matches the decoder depth that ignited in v2/v3);
    # reach = 8 + 3*32 = 104 < the 130-token question gap.
    torch.manual_seed(seed)
    return RCoreLM(V_TOY, d=64, layers=4, low_layers=1, heads=4,
                   max_t=T_TOY, window=W_TOY, chunk=C_TOY, k_buf=16,
                   n_query=4, w_low=8,
                   policy={"oracle": "oracle", "random": "random",
                           "ofrozen": "oracle"}
                   .get(arm, "learned"),
                   use_core=(arm != "nocore"),
                   cross_at=(1, 2)).to(device)


def toy_step(model, batch, boost=8.0):
    toks, fact_mask, ans_tgt, queries, ob = batch
    inp, tgt = toks[:, :-1], toks[:, 1:]
    logits, admits = model(inp, oracle_bias=ob)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                         tgt.reshape(-1), reduction="none")
    w = 1.0 + boost * ans_tgt.reshape(-1).float()
    loss = (ce * w).sum() / w.sum()
    return loss, logits, admits


@torch.no_grad()
def toy_eval(model, rng, device, batches=8, B=64, collect=False):
    model.eval()
    hits = tot = 0
    gaze_num = gaze_den = 0.0
    for _ in range(batches):
        batch = build_toy_batch(B, rng, device)
        toks, fact_mask, _, queries, ob = batch
        inp = toks[:, :-1]
        logits, admits = model(inp, oracle_bias=ob,
                               collect=collect)
        pred = logits.argmax(-1)
        for b, p in queries:
            hits += int(pred[b, p - 1] == toks[b, p])
            tot += 1
        if collect and admits:
            fm = fact_mask[:, :T_TOY].cpu()
            for idx in admits:
                gaze_num += fm.gather(1, idx).float().sum().item()
                gaze_den += idx.numel()
    model.train()
    return 100.0 * hits / max(tot, 1), \
        (gaze_num / max(gaze_den, 1e-9))


def toy_leak_tests(device):
    """All on a random-init learned model, CPU-tolerant."""
    model = toy_model("learned", device)
    rng = random.Random(0)
    batch = build_toy_batch(4, rng, device)
    toks, _, _, _, ob = batch
    inp = toks[:, :-1]
    p = 170
    inp2 = inp.clone()
    inp2[:, p] = (inp2[:, p] + 1) % 249 + 1

    def run(x):
        torch.manual_seed(123)          # freeze the gumbel draw
        with torch.no_grad():
            return model(x, oracle_bias=ob)[0]
    l1, l2 = run(inp), run(inp2)
    causal = (l1[:, :p] - l2[:, :p]).abs().max().item()
    ok_causal = causal < 1e-4

    # window leak: gaze off (buffer = constant learned bank) =>
    # positions beyond the STACKED window reach (all layers * W)
    # must not move — the upper stack provably cannot reach the
    # cache, and the lower stack is windowed by construction
    model.gaze_on = False
    p2 = 40
    reach = (len(model.low) + len(model.high)) * W_TOY
    inp3 = inp.clone()
    inp3[:, p2] = (inp3[:, p2] + 1) % 249 + 1
    l3, l4 = run(inp), run(inp3)
    far = (l3[:, p2 + reach + 1:] - l4[:, p2 + reach + 1:]) \
        .abs().max().item()
    ok_window = far < 1e-4
    model.gaze_on = True

    # grad flow to the scorer
    loss, _, _ = toy_step(model, batch)
    loss.backward()
    gq = model.query0.grad.abs().sum().item()
    gk = model.key_proj.weight.grad.abs().sum().item()
    gt = abs(model.raw_tau.grad.item())
    ok_grad = gq > 0 and gk > 0 and gt > 0
    print(f"[leak] causality max-diff {causal:.2e} "
          f"{'PASS' if ok_causal else 'FAIL'}")
    print(f"[leak] window   max-diff {far:.2e} "
          f"{'PASS' if ok_window else 'FAIL'}")
    print(f"[leak] grad-flow q {gq:.2e} k {gk:.2e} tau {gt:.2e} "
          f"{'PASS' if ok_grad else 'FAIL'}")
    return ok_causal and ok_window and ok_grad


def toy_main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TOY GATE device={device}")
    leaks_ok = toy_leak_tests(device)

    arms = args.toy_arms.split(",")
    res, gaze = {}, {}
    for arm in arms:
        model = toy_model(arm, device, seed=args.seed)
        if arm == "ofrozen":
            # bisect probe: oracle gaze over a lower layer FROZEN at
            # random init — plumbing test with v2-like archive
            for p_ in model.low.parameters():
                p_.requires_grad = False
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=0.01)
        rng = random.Random(args.seed + 1)
        t0 = time.time()
        for step in range(1, args.toy_steps + 1):
            batch = build_toy_batch(32, rng, device)
            loss, _, _ = toy_step(model, batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0 or step == 1:
                acc, _ = toy_eval(model, random.Random(99),
                                  device, batches=2)
                tau = model.tau_val if model.use_core else 0.0
                print(f"[{arm:7s}] step {step:4d} loss {loss:.3f} "
                      f"qacc {acc:5.1f} tau {tau:.2f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        acc, gz = toy_eval(model, random.Random(1234), device,
                           batches=8, collect=True)
        res[arm], gaze[arm] = acc, gz
    print("\nTOY VERDICT (query acc %, chance 4.0):")
    for a in arms:
        print(f"  {a:7s} {res[a]:6.1f}   fact-gaze {gaze[a]:.3f}")
    checks = [("leakage", leaks_ok)]
    if "oracle" in res:
        checks.append(("oracle>=85", res["oracle"] >= 85))
    if "nocore" in res:
        checks.append(("nocore<=12", res["nocore"] <= 12))
    if "learned" in res and "random" in res:
        checks.append(("learned>=max(2x random, 40)",
                       res["learned"] >= max(2 * res["random"], 40)))
        checks.append(("gaze learned>=2x random",
                       gaze["learned"] >= 2 * gaze["random"]))
    ok = all(v for _, v in checks)
    for name, v in checks:
        print(f"  {'PASS' if v else 'FAIL'}  {name}")
    print(f"TOY GATE {'PASSED' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# Real world: TinyStories + fact streams at GPT-2 vocab
# ---------------------------------------------------------------------------

class _IdMap(dict):
    def get(self, k, default=None):
        return k


def setup_world(args):
    import stream_text_v2 as SW
    SW.load_bank()
    SW._REMAP = _IdMap()
    SW._NVOCAB = 50257
    SW.PAD = EOT
    SW.UNKNOWN_IDS = SW.enc_c(" unknown")
    SW.N_C, SW.N_P = args.n_c, args.n_p
    # mid-stream question gap must clear the decoder's stacked-window
    # reach (layers*W tokens); MIN_GAP is in DOCS (~13 tok/doc)
    SW.MIN_GAP = args.min_gap
    return SW


def build_story_pool(n_tokens, split, cache):
    if os.path.exists(cache):
        pool = np.load(cache, mmap_mode="r")
        if len(pool) >= n_tokens:
            return pool
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("roneneldan/TinyStories", split=split)
    out = []
    t0 = time.time()
    for i, row in enumerate(ds):
        out.extend(enc.encode_ordinary(row["text"]))
        out.append(EOT)
        if len(out) >= n_tokens:
            break
        if i % 200000 == 0:
            print(f"  tokenizing {split}: {len(out)/1e6:.0f}M "
                  f"({time.time()-t0:.0f}s)", flush=True)
    arr = np.array(out[:n_tokens], dtype=np.uint16)
    np.save(cache, arr)
    print(f"  pool {split}: {len(arr)/1e6:.1f}M tokens "
          f"({time.time()-t0:.0f}s)", flush=True)
    return arr


def story_slice(pool, rng, n):
    i = rng.randrange(0, len(pool) - n - 1)
    return np.asarray(pool[i:i + n]).astype(np.int64).tolist()


def build_fact_stream(SW, rng, T, pool, bank_part="train",
                      n_stream_q=8, n_quiz=10, stmts=2,
                      filler_frac=0.3, far_gap=384):
    """One continuous stream: [story prefix | lifetime docs]. Returns
    (ids len T+1, questions [(kind, s, e, far)], fact_mask len T+1).
    far = the question starts > far_gap tokens after the last
    statement of every supporting fact — provably beyond the
    decoder's stacked-window reach, answerable ONLY through the
    core. Near questions are graded separately (window-contaminated
    for the core arms; still honest for dense)."""
    lt = SW.Lifetime(rng, bank_part=bank_part, stmts=stmts,
                     filler_frac=filler_frac, n_stream_q=n_stream_q,
                     n_quiz=n_quiz, abstain_frac=0.0)
    docs = list(lt.docs)
    total = sum(len(d[0]) + 1 for d in docs)
    while total > T - 8 and docs:
        d = docs[-1]                    # trailing quiz questions first
        if not (d[1] or "").startswith("q_"):
            break
        total -= len(d[0]) + 1
        docs.pop()
    assert total <= T - 8, f"world too big: {total} > {T-8}"
    need = T + 1 - total
    ids = story_slice(pool, rng, need - 1) + [EOT] if need > 1 \
        else [EOT] * need
    questions, fact_spans = [], []
    fact_end = {}
    spans_by_fid = {}
    for (doc_ids, kind, fid, a1, a2, span, apos) in docs:
        base = len(ids)
        ids.extend(doc_ids)
        ids.append(NL)
        if kind == "fact":
            fact_spans.append((base, base + len(doc_ids)))
            fact_end[fid] = base + len(doc_ids)
            spans_by_fid.setdefault(fid, []).append(
                (base, base + len(doc_ids)))
        if span is not None and kind.startswith("q_"):
            s, e = span
            sup = max(fact_end.get(a1, 0), fact_end.get(a2, 0))
            sup_spans = spans_by_fid.get(a1, []) \
                + spans_by_fid.get(a2, [])
            questions.append((kind, base + s, base + e,
                              base - sup > far_gap, sup_spans))
    assert len(ids) == T + 1
    fmask = torch.zeros(T + 1, dtype=torch.bool)
    for s, e in fact_spans:
        fmask[s:e] = True
    return ids, questions, fmask


def build_real_batch(SW, B, T, rng, pool, fact_frac, device,
                     bank_part="train", n_stream_q=8, n_quiz=10,
                     stmts=2, filler_frac=0.3, all_fact=False,
                     far_gap=384):
    toks = torch.empty(B, T + 1, dtype=torch.long)
    fact_mask = torch.zeros(B, T + 1, dtype=torch.bool)
    ans_tgt = torch.zeros(B, T, dtype=torch.bool)
    questions = []           # (b, kind, s, e, far)
    n_fact = B if all_fact else int(round(B * fact_frac))
    for b in range(B):
        if b < n_fact:
            ids, qs, fm = build_fact_stream(
                SW, rng, T, pool, bank_part, n_stream_q, n_quiz,
                stmts, filler_frac, far_gap)
            toks[b] = torch.tensor(ids)
            fact_mask[b] = fm
            for kind, s, e, fr, sup in qs:
                e = min(e, T)
                ans_tgt[b, s - 1:e - 1] = True
                questions.append((b, kind, s, e, fr, sup))
        else:
            toks[b] = torch.tensor(story_slice(pool, rng, T + 1))
    return (toks.to(device), fact_mask, ans_tgt.to(device),
            questions)


GKEYS = ("h1_far", "h1_near", "h2_far", "h2_near")


def grade(logits, toks, questions):
    """Teacher-forced exact match over multi-token answer spans,
    split by whether the question is beyond decoder reach (far)."""
    pred = logits.argmax(-1)
    agg = {k: [0, 0] for k in GKEYS}
    for b, kind, s, e, fr, _sup in questions:
        ok = bool((pred[b, s - 1:e - 1] == toks[b, s:e]).all())
        key = ("h1_" if kind == "q_h1" else "h2_") \
            + ("far" if fr else "near")
        agg[key][0] += int(ok)
        agg[key][1] += 1
    return agg


def buffer_hit(admits, questions, chunk):
    """THE PRIMARY GAZE METRIC (Ibanis reframe: 'can the model learn
    where to look?'): for each far question, was ANY token of its
    supporting fact statements in the buffer the decoder read while
    answering — selection quality, independent of the reader."""
    hits = tot = 0
    if not admits:
        return 0.0, 0
    for b, kind, s, e, fr, sup in questions:
        if not fr or not sup:
            continue
        u = s // chunk               # question's chunk index
        if u < 1 or u - 1 >= len(admits):
            continue
        idx = admits[u - 1][b].tolist()
        tot += 1
        hits += int(any(s0 <= i < e0 for i in idx
                        for s0, e0 in sup))
    return hits / max(tot, 1), tot


def gaze_stats(admits, fact_mask, n_fact_rows, chunk):
    """Fraction of admitted archive tokens inside fact statements
    (fact rows only) + mean recency (admitted pos / candidate count,
    1.0 = always the newest tokens)."""
    if not admits or n_fact_rows == 0:
        return 0.0, 0.0
    fm = fact_mask[:n_fact_rows, :-1]
    num = den = rec = 0.0
    for t, idx in enumerate(admits, start=1):
        sub = idx[:n_fact_rows]
        num += fm.gather(1, sub).float().sum().item()
        den += sub.numel()
        rec += (sub.float() / (t * chunk)).mean().item()
    return num / max(den, 1e-9), rec / max(len(admits), 1e-9)


def train_real_arm(model, SW, pool, args, arm, device, log_fn):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)))
    model.train()
    rng = random.Random(1234 + args.seed)
    far_gap = args.layers * args.w
    t0 = time.time()
    for step in range(1, args.steps + 1):
        sq = 0 if step <= args.steps * args.sq_warmup \
            else args.n_stream_q
        toks, fmask, ans_tgt, qs = build_real_batch(
            SW, args.batch, args.t, rng, pool, args.fact_frac,
            device, n_stream_q=sq, n_quiz=args.n_quiz,
            stmts=args.stmts, filler_frac=args.filler_frac,
            far_gap=far_gap)
        inp, tgt = toks[:, :-1], toks[:, 1:]
        collect = (step % args.log_every == 0)
        with torch.autocast(device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            logits, admits = model(inp, collect=collect)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                tgt.reshape(-1), reduction="none")
            w = 1.0 + args.ans_boost * ans_tgt.reshape(-1).float()
            loss = (ce * w).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if collect or step == 1:
            g = grade(logits, toks, qs)
            n_fr = int(round(args.batch * args.fact_frac))
            gz, rec = gaze_stats(admits, fmask, n_fr, args.c)
            tau = model.tau_val if model.use_core else 0.0
            h1 = g["h1_far"][0] / max(g["h1_far"][1], 1)
            h2 = g["h2_far"][0] / max(g["h2_far"][1], 1)
            print(f"[{arm:7s}] step {step:5d} loss {loss:.3f} "
                  f"h1f {h1:.2f} h2f {h2:.2f} gaze {gz:.3f} "
                  f"rec {rec:.2f} tau {tau:.2f} sq {sq} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if log_fn:
                log_fn({f"{arm}/loss": float(loss),
                        f"{arm}/h1_far": h1, f"{arm}/h2_far": h2,
                        f"{arm}/gaze_fact": gz, f"{arm}/tau": tau,
                        f"{arm}/step": step})


@torch.no_grad()
def eval_real(model, SW, pool_val, pool_train, args, device,
              gaze_override=None):
    model.eval()
    out = {}
    # story perplexity on held-out pool
    rng = random.Random(4242)
    ce_sum = ce_n = 0.0
    for _ in range(args.eval_story_batches):
        toks, _, _, _ = build_real_batch(
            SW, args.batch, args.t, rng, pool_val, 0.0, device)
        inp, tgt = toks[:, :-1], toks[:, 1:]
        with torch.autocast(device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            logits, _ = model(inp, gaze_override=gaze_override)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            tgt.reshape(-1))
        ce_sum += float(ce)
        ce_n += 1
    out["lm_loss"] = ce_sum / max(ce_n, 1)
    out["ppl"] = math.exp(out["lm_loss"])
    # fact-stream probes (train + holdout templates)
    far_gap = args.layers * args.w
    for part in ("train", "hold"):
        rng = random.Random(999)
        agg = {k: [0, 0] for k in GKEYS}
        gz_sum = rec_sum = gz_n = 0.0
        for _ in range(args.eval_fact_batches):
            toks, fmask, _, qs = build_real_batch(
                SW, args.batch, args.t, rng, pool_train, 1.0,
                device, bank_part=part,
                n_stream_q=args.n_stream_q, n_quiz=args.n_quiz,
                stmts=args.stmts, filler_frac=args.filler_frac,
                all_fact=True, far_gap=far_gap)
            inp = toks[:, :-1]
            with torch.autocast(device_type="cuda",
                                dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                logits, admits = model(
                    inp, collect=True,
                    gaze_override=gaze_override)
            g = grade(logits, toks, qs)
            for k in agg:
                agg[k][0] += g[k][0]
                agg[k][1] += g[k][1]
            gz, rec = gaze_stats(admits, fmask, args.batch, args.c)
            bh, bh_n = buffer_hit(admits, qs, args.c)
            gz_sum += gz
            rec_sum += rec
            gz_n += 1
            bh_sum = out.get(f"_bh{part}", 0.0) + bh * bh_n
            bhn_sum = out.get(f"_bhn{part}", 0) + bh_n
            out[f"_bh{part}"], out[f"_bhn{part}"] = bh_sum, bhn_sum
        sfx = "" if part == "train" else "_hold"
        for k in GKEYS:
            out[f"{k}{sfx}"] = agg[k][0] / max(agg[k][1], 1)
        out[f"nq{sfx}"] = sum(agg[k][1] for k in GKEYS)
        out[f"bufhit{sfx}"] = out.pop(f"_bh{part}", 0.0) \
            / max(out.pop(f"_bhn{part}", 1), 1)
        if part == "train":
            out["gaze_fact"] = gz_sum / max(gz_n, 1)
            out["gaze_rec"] = rec_sum / max(gz_n, 1)
    model.train()
    return out


def real_main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    SW = setup_world(args)
    # probe one stream for length sanity
    probe, pq, _ = build_fact_stream(
        SW, random.Random(0), args.t,
        np.zeros(args.t * 2, dtype=np.uint16),
        n_stream_q=args.n_stream_q, n_quiz=args.n_quiz,
        stmts=args.stmts, filler_frac=args.filler_frac,
        far_gap=args.layers * args.w)
    nfar = sum(1 for q in pq if q[3])
    print(f"device={device} fact-stream probe: len {len(probe)} "
          f"questions {len(pq)} ({nfar} far)", flush=True)

    pool = build_story_pool(args.pool_tokens, "train",
                            "stories_train.npy")
    pool_val = build_story_pool(args.val_pool_tokens, "validation",
                                "stories_val.npy")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"rcore1-s{args.steps}-t{args.t}",
                         config=vars(args))
    log_fn = (lambda x: run.log(x)) if run else None

    arm_cfg = {
        "learned": dict(policy="learned", use_core=True,
                        full_attn=False),
        "learned-s2": dict(policy="learned", use_core=True,
                           full_attn=False, s_rounds=2),
        "learned-s3": dict(policy="learned", use_core=True,
                           full_attn=False, s_rounds=3),
        "random": dict(policy="random", use_core=True,
                       full_attn=False),
        "dense": dict(policy="learned", use_core=False,
                      full_attn=True),
        "nocore": dict(policy="learned", use_core=False,
                       full_attn=False),
    }
    models, results = {}, {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        torch.manual_seed(args.seed)
        model = RCoreLM(50257, d=args.d, layers=args.layers,
                        low_layers=args.low_layers,
                        heads=args.heads, max_t=args.t,
                        window=args.w, chunk=args.c, k_buf=args.k,
                        n_query=args.n_query, w_low=args.w_low,
                        **arm_cfg[arm]).to(device)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"\n=== arm {arm}: {n_par/1e6:.1f}M trainable params",
              flush=True)
        train_real_arm(model, SW, pool, args, arm, device, log_fn)
        models[arm] = model
        if args.save_prefix:
            torch.save(model.state_dict(),
                       f"{args.save_prefix}_{arm}.pt")
        results[arm] = eval_real(model, SW, pool_val, pool,
                                 args, device)
    for a in list(models):
        if a.startswith("learned"):
            results[f"{a}-randgaze"] = eval_real(
                models[a], SW, pool_val, pool, args,
                device, gaze_override="random")

    print("\n=== RCORE RUNG-1 RESULTS")
    names = list(results)
    keys = ("bufhit", "bufhit_hold", "ppl", "h1_far", "h2_far",
            "h1_near", "h2_near", "h1_far_hold", "h2_far_hold",
            "gaze_fact", "gaze_rec")
    print("  metric        " + "  ".join(f"{n:>16s}" for n in names))
    for k in keys:
        row = "  ".join(f"{results[n].get(k, float('nan')):16.3f}"
                        for n in names)
        print(f"  {k:>12s}  {row}")
    if "learned" in results and "random" in results:
        lr_, rr = results["learned"], results["random"]
        print(f"\nKILL CHECK learned-vs-random: "
              f"dppl {rr['ppl']-lr_['ppl']:+.3f}  "
              f"dh1_far {lr_['h1_far']-rr['h1_far']:+.3f}  "
              f"dh2_far {lr_['h2_far']-rr['h2_far']:+.3f}")

    if run:
        for a, st in results.items():
            for k, v in st.items():
                run.summary[f"{a}_{k}"] = v
        with open("rcore1_summary.json", "w") as f:
            json.dump(results, f, indent=1)
        import wandb as wb
        art = wb.Artifact(f"rcore1-{run.id}", type="results")
        art.add_file("rcore1_summary.json")
        if args.save_prefix:
            for arm in models:
                art.add_file(f"{args.save_prefix}_{arm}.pt")
        run.log_artifact(art).wait()
        run.finish()
    print("RUN_COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--toy-steps", type=int, default=2000)
    ap.add_argument("--toy-arms", type=str,
                    default="oracle,learned,random,nocore")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--t", type=int, default=2048)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--n-query", type=int, default=8)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--low-layers", type=int, default=3)
    ap.add_argument("--w-low", type=int, default=0)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--fact-frac", type=float, default=0.2)
    ap.add_argument("--ans-boost", type=float, default=2.0)
    ap.add_argument("--sq-warmup", type=float, default=0.35)
    ap.add_argument("--n-stream-q", type=int, default=8)
    ap.add_argument("--n-quiz", type=int, default=10)
    ap.add_argument("--stmts", type=int, default=2)
    ap.add_argument("--filler-frac", type=float, default=0.3)
    ap.add_argument("--n-c", type=int, default=3)
    ap.add_argument("--n-p", type=int, default=6)
    ap.add_argument("--min-gap", type=int, default=45)
    ap.add_argument("--pool-tokens", type=int, default=120_000_000)
    ap.add_argument("--val-pool-tokens", type=int,
                    default=2_000_000)
    ap.add_argument("--arms", type=str,
                    default="learned,random,dense")
    ap.add_argument("--eval-story-batches", type=int, default=8)
    ap.add_argument("--eval-fact-batches", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-prefix", type=str, default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str,
                    default="neocore-rcore")
    args = ap.parse_args()
    if args.preflight:
        # one-instance pipeline: toy gate first; only a PASS rolls
        # into the real run (a FAIL exits nonzero -> instance stays
        # alive for inspection per run_training.sh)
        if not toy_main(args):
            import sys
            sys.exit(1)
        real_main(args)
    elif args.toy:
        toy_main(args)
    else:
        real_main(args)


if __name__ == "__main__":
    main()
