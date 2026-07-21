"""Toy smoke: test-time codebook consolidation (2026-07-20).

Question under test: can a hardcoded, capacity-capped, EMA-edited
codebook turn episodic recurrence into usable knowledge AT TEST TIME
(frozen weights, no gradients)?

Structure (the lifetime harness):
- A LIFETIME = L episodes over the SAME 6 novel classes. Labels are
  re-permuted every episode (binding stays unsmearable); classes recur
  across episodes (recurrence is what only the codebook can capture:
  weights can't — eval classes are held out; cards can't — wiped per
  episode).
- Episode = unchanged binding game: 6-image stream, one fused card per
  image (card = code ⊕ label here), per-step probes, architectural
  cosine read head + aux CE on the read attention.
- Codebook: K slots per lifetime, reset at lifetime start, updated ONLY
  by a hardcoded assign-and-running-mean rule (online k-means = literal
  gradient descent on distortion, no backprop; count-capped so it stays
  plastic). Snap-for-write, snap-mix-for-read.

Toy world: class = random unit prototype in R^d_in; instance = proto +
noise scaled so ONE glance is an unreliable class estimate (same-class
instance cos ~ 1/(1+sigma^2)) but the running mean over a lifetime is
clean. Consolidation headroom is built into the world.

Arms (dissociation expected):
  live    EMA codebook on at eval               -> rises across episode
                                                   index within lifetime
  frozen  same model, codebook never updates    -> flat, <= live
  oracle  codebook = true encoded class means   -> ceiling, flat-high
  nocode  separately trained model, cards from  -> flat at the one-shot
          raw percepts (current architecture)      matching level

Success criteria: (a) machinery trains (read_hit >> chance); (b) live's
within-lifetime curve RISES on held-out classes with frozen weights;
(c) live > nocode by late lifetime and closes toward oracle; (d) frozen
<= live (the edits, not the slots, carry the effect).

Smoke iteration log:
1. v0 (snap to nearest of K random-init codes): machinery trains;
   held-out dissociation oracle 78 >> live 55 ~= nocode 53 >> frozen 29
   > chance 16.7, but live's within-lifetime slope only +2.3 (nocode's
   stateless noise floor: +1.5). The edits matter (live >> frozen) yet
   consolidation captures ~none of the oracle's 25-pt headroom.
   Diagnosis: class fragmentation across random-init codes — same-class
   writes/reads snap to different codes. -> v1: novelty-gated code
   birth (constant theta, per the constants law), reads masked to used
   slots, consolidation diagnostics (used-code count, write/read snap
   agreement).
2. v1 (birth, theta=0.45): SMOKE PASSES — held-out live 61.5 -> 64.7
   (+3.2 within-lifetime slope under frozen weights; 67.2 final ep) >>
   nocode 53 flat >> frozen 26 flat; oracle 81-83. First measured
   test-time learning via codebook edits. But diagnostics show theta
   miscalibrated in the LEARNED encoder space (same-class glance cos
   0.33 < theta): fresh codes under-catch their class -> duplicate
   births -> all 16 slots exhaust by ep ~4, write/read agree only ~37.
   The ~15-pt oracle gap likely lives there. -> v1b: theta=0.18
   (between cross 0.01 and same 0.33).
3. v2 (--theta-merge): threshold merging as the REPAIR op for
   miscalibration. Two codes whose cosine exceeds theta_merge combine
   into their count-weighted mean (exact under the running-mean
   algebra: identical to having assigned all their glances to one slot
   from the start), freeing the loser's slot. Maturity dynamic: fresh
   duplicates match each other only ~0.33 (two noisy glances) but
   drift toward the same class center as they feed, so mergeability
   RISES with confidence. Hysteresis: theta_merge > theta_birth, so a
   just-born code (sim < theta_birth to everything) can never
   instantly merge back. The sharp test: deliberately-bad theta=0.45
   WITH merging — if it recovers calibrated-theta performance, the
   rule is robust to the one thing the real system can't control
   (representation drift under a fixed constant).
   [v1b result: theta=0.18 UNDER-births -> two classes captured by one
   code at ep0 (used 3.5/6) — a COLLISION, unrepairable by birth+drag
   (only a split op could undo it); live ~50 flat, no better than
   nocode. THE ASYMMETRY FINDING: over-birth = duplicates = dilution =
   recoverable; under-birth = collisions = permanent. Err toward
   birthing; repair with merge.]
   [v2 result: live 62.4 -> 65.2 (final 67.6) — marginally above v1
   but same shape; merges ~0.5/lifetime, none before ep 4. Bar-merge
   fires too LATE: duplicate codes only reach mutual cos 0.6 after
   ~3-4 feeds each, and each class feeds one glance per episode spread
   over ~2-3 duplicates. Repair arrives as the lifetime ends.]
4. v3 (--merge-mode witness): the GLANCE is the witness. If a write's
   top-2 used codes BOTH exceed the birth bar theta, they must be the
   same class (cross-class codes sit at cos ~0 and cannot both clear
   0.45) -> merge them immediately, then drag. Fires as soon as one
   duplicate matures, episodes before code-to-code similarity does;
   needs NO second threshold — one constant does birth and merge.
   [v3 result: merges finally FIRE (11.8/lifetime from ep 1-2) and the
   climb DIES: live 61.2 -> 61.8 (+0.5) vs v1's +3.2. Churn: each
   merge frees a slot; same-class glances (cos 0.33 < bar 0.45)
   chronically re-birth into it; ~22% of cards forever written from
   n=1 newborn codes = raw-noise quality. THE REREAD: v1's slot
   saturation was never a bug — it was the load-bearing ANNEAL
   (birth-liberal while slots remain = early exploration, avoids
   v1b's collisions; join-forced once full = late consolidation).
   Witness-merge broke the anneal by reopening the birth valve.
   Capacity-exhaustion is a developmental schedule — the tau/diversity
   principle in codebook form.]
5. v4 (--tombstone): make the anneal explicit — a slot freed by merge
   is DEAD for the rest of the lifetime; birth claims only virgin
   slots. Capacity is monotone: <=K births ever, merging shrinks the
   live population toward the true class count, joins get
   progressively forced. Expected: used rises to ~16 then FALLS toward
   ~6; agree climbs late; live keeps v1's anneal AND the cleanup ->
   closes on oracle.
   [v4 result: structure EXACTLY as predicted (used 13.2 -> 7.6, agree
   37 -> 58, merges from ep 1) and live DECLINES 61.4 -> 58.1 — flat
   through ep 5, decaying after. The decline starts AT THE TRAINING
   HORIZON (l-train 6): meta-training only ever saw fragmented books,
   and v4's rule keeps developing the book into a consolidated regime
   the frozen machinery never operated in. v1 dodged this because
   saturation is a FIXED POINT (ep 12 book == ep 6 book); v4's book
   keeps drifting out of distribution. Precondition bite: the frozen
   model must be meta-trained under the protocol it will face —
   including the LATE-lifetime states its own memory rule produces.
   -> v4b: same rule, l-train 12 == l-eval.]
   [v4b result: decline FIXED (flat 54-58 across all 12 eps — the OOD
   diagnosis was right) but the whole curve sits ~7 pts BELOW v1,
   while the diagnostics are the campaign's best (agree 60, used ->
   6.7). Slot tidiness and accuracy have DECOUPLED: cards embed code
   VECTORS, and merges jump the surviving vector mid-episode, staling
   every card written before the merge. v1's slow drags never do this.
   Structural ops fixed the bookkeeping and broke the geometry.
   Structural line closed; the fix must live at the content interface.]
6. v5 (--content soft, Ibanis's proposal): don't snap content to ONE
   code — pass a similarity-softmax MIXTURE over used codes to the
   card and the query. Virtual merging at read time: the blend over
   duplicates approximates the full class mean (what merge computes
   structurally), and write/read meet at the same blended point
   regardless of which duplicate is nominally nearest. No structural
   ops -> no birth-valve churn, no anneal to protect, no regime drift.
   The trade, named: content stops being a SYMBOL (hard code identity
   = the LLM-substrate ambition) and becomes a superposition — benign
   here (cross-class cos ~0 keeps blends unimodal), but it moves the
   discreteness question to the interface. Compromise retained: the
   UPDATE rule stays hard-assign (the book still develops as discrete
   running means); only the content interface softens.
   [v5 result: DECISIVE WIN. live 63.2 -> 69.8 (+6.6 slope, 2x v1's;
   final eps 70.5-70.7, still climbing at ep 11); oracle rose ~5 to
   86-88 (soft lookup helps perfect codes too); gap to oracle 26 ->
   15.5 and shrinking when the lifetime ended. Mechanism confirmed by
   what DIDN'T change: fragmentation fully present (used -> 16, agree
   ~40) but no longer costly — the noise was always at the content
   interface, never in the slot bookkeeping (why v3/v4's tidying never
   paid). Cleanest training of the campaign (loss ~0.35, ~half of any
   hard-snap arm; first perfect training batch). Soft content also
   makes merges ~invariant (count-weighted merge ~= what the mixture
   already computes), reopening structural ops for SYMBOL ECONOMY
   rather than accuracy -> v6: soft + witness + tombstone.]
   [v5 @ l-eval 24 (weights meta-trained on SIX-episode lifetimes):
   63.5 -> 76.8, +12 pts of gradient-free learning, still climbing at
   ep 23; oracle gap halved (25 -> 12). Extrapolation past the
   training horizon is FREE here (unlike v4) because a saturated
   book's mixtures cleaning up is smooth drift, not a regime change.]
7. v6 (soft + witness + tombstone): live 59.9 -> 63.2 — ~6 pts BELOW
   v5 despite structure behaving perfectly (used -> 7.3, agree 62).
   The merge-invariance argument FAILED instructively: under soft
   content a POPULATION of duplicates is a richer estimator than its
   count-weighted collapse (the softmax reweights duplicates per
   glance); merging discards that. Duplicates are an asset at the soft
   interface, a liability at the hard one.

8. v6b (soft + bar-merge, NO tombstone — Ibanis's call): live 61.9 ->
   70.1, slope +8.2 (best of campaign), endpoint edges v5 within
   noise; merges ~1/lifetime, firing only from ep ~4. The bar
   trigger's LAZINESS — fatal under hard content — is exactly right
   under soft: it merges only mature duplicate pairs, whose
   count-weighted collapse sits where the mixture already was
   (~invariant), and otherwise leaves the population for the mixture
   to exploit. Aggressive merging was wrong; CONFIDENT merging is
   free.

VERDICT (smoke phase closed, 2026-07-20): final validated rule set =
hard-assign updates + birth-until-full (capacity exhaustion IS the
explore->commit anneal) + soft mixture content + lazy bar-merge (no
tombstones). One birth constant theta + one merge constant; witness-
merge and tombstones rejected (aggressive structure loses ~6 pts).

REAL-SPACE PRE-CHECK (check_dino_headroom.py, local, $0): premise
measured on held-out IN-100 in frozen DINOv2-S space. Same/cross-class
cos 0.361/0.021 (CLS) — the toy's regime (0.33/0.00) is REAL, not
rigged. 20-way k-shot: proto(1) 77.5 -> proto(25) 94.8 (+17 pts
consolidation headroom; patch-mean: 63 -> 88, +25); prototype beats
k-nearest-exemplar at every k>=2 (+3.2 CLS / +5.0 patch at k=25) while
storing 1 vector vs k. Consolidation beats memory in real space;
noisier percepts -> bigger codebook payoff.

9. v7 (--lam, Ibanis's "codes take up energy"): replace both
   thresholds with ONE constant — DP-means. Objective: distortion +
   lam * (#codes); every rule is a theorem of it. Birth iff min
   squared glance-to-code distance > lam (distortion saved beats
   rent; on normalized vectors this IS the theta rule, theta =
   1 - lam/2). Merge i,j iff (ni*nj/(ni+nj)) * ||ci-cj||^2 < lam —
   the EXACT distortion cost of the collapse, computable from stored
   counts+centroids. Intrinsically confidence-gated: noisy fresh
   duplicates wait, converged mature ones merge — the lazy-merge
   lesson derived instead of hand-tuned. Same open wart (lam is a
   constant in a learned space); the escalation paths are quantile
   self-calibration or per-code variance (Bayesian birth).
   [v7 result: CATASTROPHIC TRAINING FAILURE, mechanism instructive.
   At init the encoder cone's distances (~30) sit under lam=70 ->
   nothing births; and unlike theta-only rules (one-way door — loss
   pressure can spread the geometry past a bad bar and progress
   STICKS), the energy merge is a two-way door: any code the
   spreading geometry births is instantly liquidated (two nearby
   small-count codes = dirt-cheap collapse). The rule that is optimal
   GIVEN a geometry is anti-optimal while the geometry is being
   built. Book pinned at used=1.00 for entire 24-ep lifetimes; live
   30 < nocode 53; even oracle collapsed to 59 (encoder meta-trained
   around a degenerate book never learned code-shaped reads).
   PRINCIPLE: a test-time rule must be developmentally viable, not
   just asymptotically optimal — a fixed price bankrupts the infant
   economy. The fix, if pursued: lam as a fixed QUANTILE of recent
   glance-to-code distances (procedure constant, value tracks the
   geometry).]
10. v7b (--lam-q): quantile rent. Each lifetime records its observed
   glance-to-nearest-code squared distances; lam_t = the q-quantile
   of that record (per lifetime, running). The PROCEDURE is the
   architectural constant; the price floats with the geometry. Free
   consequences: empty record -> rent ~0 -> infant book births
   liberally (the recoverable direction) and CANNOT merge (cost <
   ~0 never fires) — liberal infancy + one-way doors fall out of the
   definition instead of being scheduled. Risk to watch: the
   threshold is now a function of encoder outputs, so meta-training
   could in principle game it (constants-law exposure).
   [v7b result: FAILS, third distinct mechanism — NOVELTY POISONS THE
   RECORD. A fresh lifetime's first observed nearest-distances are
   first-glances-of-new-classes (cross-scale, ~120); the quantile
   activates on that polluted record, rent spikes, every merge looks
   cheap, the book demolishes to ~1 code by ep 2 (used 4.1 -> 1.35),
   then painstakingly rebuilds (9.2 by ep 23) as ordinary join
   distances dilute the record. live 40 -> 49, BELOW nocode (54)
   throughout. The self-reference problem in its pure form: the
   statistic is polluted by the events it governs — novelty inflates
   the rent that then punishes novelty's consequences. Known fix
   (record join distances only, not birth-triggering ones) NOT
   pursued — third patch on the line, epicycle territory.

ENERGY-LINE VERDICT: the objective is right (it derived lazy merging
from first principles) but both implementations failed on
developmental dynamics: fixed rent bankrupts the infant geometry
(v7), self-calibrating rent poisons itself on novelty (v7b). Rules
that price structure must be viable WHILE the geometry/record is
immature, and every adaptive statistic must be insulated from the
decisions it governs. Carry to the real experiment: v6b as the
validated rule; fixed-lam energy as a cheap second arm ONLY there —
frozen DINO percepts have no infancy (the geometry is adult and
measured: same/cross d^2 ~ 82/125 per check_dino_headroom), so v7's
failure mode structurally cannot occur.]
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

N_WAY = 6


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class World:
    def __init__(self, d_in, n_train, n_held, sigma, device, seed=0):
        g = torch.Generator().manual_seed(seed)
        protos = torch.randn(n_train + n_held, d_in, generator=g)
        self.protos = F.normalize(protos, dim=-1).to(device)
        self.d_in, self.sigma = d_in, sigma
        self.train_pool = torch.arange(n_train, device=device)
        self.held_pool = torch.arange(n_train, n_train + n_held,
                                      device=device)

    def sample(self, cls):
        p = self.protos[cls]
        return p + torch.randn_like(p) * (self.sigma / math.sqrt(self.d_in))


def sample_lifetime_classes(pool, E, device):
    idx = torch.rand(E, pool.numel(), device=device).argsort(1)[:, :N_WAY]
    return pool[idx]


def make_episode(world, cls, device):
    """cls: (E, N_WAY) lifetime classes -> one episode's tensors.
    order[:, s] / probe_pos[:, s] = lifetime-class INDEX (0..5) of the
    stream item / probe at step s — diagnostics only, model never sees
    class identity."""
    E = cls.shape[0]
    order = torch.rand(E, N_WAY, device=device).argsort(1)
    stream_cls = cls.gather(1, order)
    labels = torch.rand(E, N_WAY, device=device).argsort(1)  # perm 0..5
    s_hi = torch.arange(1, N_WAY + 1, device=device).float()
    card_t = (torch.rand(E, N_WAY, device=device) * s_hi).long()  # in [0,s]
    probe_cls = stream_cls.gather(1, card_t)
    probe_lab = labels.gather(1, card_t)
    probe_pos = order.gather(1, card_t)
    return (world.sample(stream_cls), labels, world.sample(probe_cls),
            card_t, probe_lab, order, probe_pos)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ToyBinder(nn.Module):
    def __init__(self, d_in, d=64, k=16, use_codes=True, cap=32,
                 theta=0.45, theta_merge=0.0, merge_mode="off",
                 tombstone=False, content="hard", code_temp=10.0,
                 lam=0.0, lam_q=0.0):
        super().__init__()
        self.use_codes, self.k, self.cap, self.theta = use_codes, k, cap, \
            theta
        self.theta_merge = theta_merge          # bar mode only
        self.merge_mode = merge_mode            # off | bar | witness
        self.tombstone = tombstone              # merged-away slots stay dead
        self.content = content                  # hard snap | soft mixture
        self.code_temp = code_temp              # soft mixture sharpness
        self.lam = lam                          # >0: DP-means energy rule
        self.lam_q = lam_q                      # >0: rent = running quantile
        self.enc = nn.Sequential(nn.Linear(d_in, 128), nn.GELU(),
                                 nn.Linear(128, d), nn.LayerNorm(d))
        self.label_emb = nn.Embedding(N_WAY, d)
        self.card = nn.Linear(2 * d, d)     # [content ; label] -> card
        self.qmix = nn.Linear(2 * d, d)     # [percept ; content_q] -> query
        self.read_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_WAY)
        self.raw_temp = nn.Parameter(torch.tensor(10.0))
        self.register_buffer("code_init",
                             F.normalize(torch.randn(k, d), dim=-1))
        self.codes = None                   # per-lifetime state, NOT params
        self.counts = None

    def reset_book(self, E, device):
        self.codes = self.code_init.to(device).unsqueeze(0).repeat(E, 1, 1)
        self.counts = torch.zeros(E, self.k, device=device)
        self.merges = torch.zeros(E, device=device)
        self.dead = torch.zeros(E, self.k, dtype=torch.bool, device=device)
        self.dstats = torch.full((E, 256), float("nan"), device=device)
        self.dn = torch.zeros(E, dtype=torch.long, device=device)

    def _sims(self, v):
        return torch.einsum("ed,ekd->ek", F.normalize(v, dim=-1),
                            F.normalize(self.codes, dim=-1))

    @torch.no_grad()
    def _soft_content(self, v):
        """Similarity-softmax mixture over used codes — virtual merge
        at the content interface; assignment/updates stay hard."""
        sim = self._sims(v.detach())
        used = self.counts > 0
        neg = torch.finfo(sim.dtype).min
        sim_u = sim.masked_fill(~used, neg)
        sim_eff = torch.where(used.any(1, keepdim=True), sim_u, sim)
        w = torch.softmax(sim_eff * self.code_temp, dim=1)
        return torch.einsum("ek,ekd->ed", w, self.codes)

    @torch.no_grad()
    def snap_write(self, v):
        """Hardcoded consolidation: novelty-gated birth + count-capped
        running mean (online k-means = a literal gradient step on
        ||v - c||^2 with lr 1/min(n, cap)); no backprop touches the
        codebook (asserted in main)."""
        v = v.detach()
        if self.lam > 0 or self.lam_q > 0:
            return self._energy_write(v)
        used = self.counts > 0
        sim = self._sims(v)
        neg = torch.finfo(sim.dtype).min
        sim_u = sim.masked_fill(~used, neg)
        best, bestv = sim_u.argmax(1), sim_u.max(1).values
        free = ~used & ~self.dead        # tombstoned slots stay unbirthable
        birth = (bestv < self.theta) & free.any(1)
        first_free = free.float().argmax(1)
        i = torch.where(birth, first_free, best)
        ar = torch.arange(v.shape[0], device=v.device)
        n = self.counts[ar, i] + 1
        lr = (1.0 / n.clamp(max=self.cap)).unsqueeze(-1)
        self.codes[ar, i] += lr * (v - self.codes[ar, i])
        self.counts[ar, i] = n
        if self.merge_mode == "bar":
            # repair op: the just-touched code vs its nearest used
            # neighbor above theta_merge.
            ci = self.codes[ar, i]
            msim = torch.einsum("ed,ekd->ek", F.normalize(ci, dim=-1),
                                F.normalize(self.codes, dim=-1))
            mask = self.counts > 0
            mask[ar, i] = False
            msim = msim.masked_fill(~mask, neg)
            j = msim.argmax(1)
            do = msim.max(1).values > self.theta_merge
            i = self._merge(ar, i, j, do)
        elif self.merge_mode == "witness":
            # the glance is the witness: if the top-2 used codes BOTH
            # cleared the birth bar, they are the same class (cross-
            # class codes cannot both match one glance) -> merge now.
            t2v, t2i = sim_u.topk(2, dim=1)
            do = (~birth) & (t2v[:, 1] > self.theta)
            i = self._merge(ar, i, t2i[:, 1], do)
        if self.content == "soft":
            return self._soft_content(v), i
        return self.codes[ar, i].clone(), i

    @torch.no_grad()
    def _energy_write(self, v):
        """v7: DP-means. ONE constant lam = rent per code, all rules
        theorems of  distortion + lam * #codes.  Birth iff min sq
        distance > lam (distortion saved beats rent). After the write,
        merge the touched code with its nearest used neighbor iff the
        EXACT distortion cost of the collapse — (ni*nj/(ni+nj)) *
        ||ci-cj||^2, closed form from stored counts+centroids — is
        below the rent it frees. Confidence-gating falls out: noisy
        fresh duplicates are far apart (wait), converged mature ones
        are close (merge)."""
        E, dev = v.shape[0], v.device
        ar = torch.arange(E, device=dev)
        used = self.counts > 0
        big = torch.finfo(v.dtype).max
        d2 = ((v.unsqueeze(1) - self.codes) ** 2).sum(-1)
        d2u = d2.masked_fill(~used, big)
        best, bestv = d2u.argmin(1), d2u.min(1).values
        if self.lam_q > 0:
            # v7b: rent = running q-quantile of this lifetime's observed
            # nearest distances; ~0 while the record is thin (liberal
            # infant births, no infant merges)
            lam = torch.where(
                self.dn >= 4,
                torch.nanquantile(self.dstats, self.lam_q, dim=1),
                torch.zeros(E, device=dev))
            has_used = used.any(1)
            slot = (self.dn % 256)
            rows = ar[has_used]
            self.dstats[rows, slot[has_used]] = bestv[has_used]
            self.dn[has_used] += 1
        else:
            lam = torch.full((E,), self.lam, device=dev)
        free = ~used & ~self.dead
        birth = (bestv > lam) & free.any(1)
        i = torch.where(birth, free.float().argmax(1), best)
        n = self.counts[ar, i] + 1
        lr = (1.0 / n.clamp(max=self.cap)).unsqueeze(-1)
        self.codes[ar, i] += lr * (v - self.codes[ar, i])
        self.counts[ar, i] = n
        # energy merge: touched code vs nearest used neighbor
        ci = self.codes[ar, i]
        dj = ((self.codes - ci.unsqueeze(1)) ** 2).sum(-1)
        mask = self.counts > 0
        mask[ar, i] = False
        dj = dj.masked_fill(~mask, big)
        j = dj.argmin(1)
        ni_, nj_ = self.counts[ar, i], self.counts[ar, j]
        cost = ni_ * nj_ / (ni_ + nj_) * dj.gather(
            1, j.unsqueeze(1)).squeeze(1)
        do = (cost < lam) & mask.any(1) & (nj_ > 0)
        i = self._merge(ar, i, j, do)
        if self.content == "soft":
            return self._soft_content(v), i
        return self.codes[ar, i].clone(), i

    @torch.no_grad()
    def _merge(self, ar, i, j, do):
        """Merge codes i and j (count-weighted mean — EXACT under the
        running-mean algebra) where do; loser's slot freed (count 0 ->
        masked from reads/snaps; birth recycles it with lr 1). Returns
        the surviving slot per row."""
        if not do.any():
            return i
        ni = self.counts[ar, i].unsqueeze(-1)
        nj = self.counts[ar, j].unsqueeze(-1)
        merged = (self.codes[ar, i] * ni + self.codes[ar, j] * nj) \
            / (ni + nj)
        keep = torch.where(self.counts[ar, i] >= self.counts[ar, j], i, j)
        drop = torch.where(keep == i, j, i)
        ard = ar[do]
        self.codes[ard, keep[do]] = merged[do]
        self.counts[ard, keep[do]] = (ni + nj).squeeze(-1)[do]
        self.counts[ard, drop[do]] = 0.0
        if self.tombstone:
            self.dead[ard, drop[do]] = True
        self.merges += do.float()
        return torch.where(do, keep, i)

    @torch.no_grad()
    def snap_read(self, v):
        used = self.counts > 0
        sim = self._sims(v.detach())
        neg = torch.finfo(sim.dtype).min
        sim_u = sim.masked_fill(~used, neg)
        sim_eff = torch.where(used.any(1, keepdim=True), sim_u, sim)
        i = sim_eff.argmax(1)
        ar = torch.arange(v.shape[0], device=v.device)
        if self.content == "soft":
            return self._soft_content(v), i
        return self.codes[ar, i].clone(), i

    def run_episode(self, stream_x, labels, probe_x, card_t, probe_lab,
                    order=None, probe_pos=None, update_book=True,
                    code_map=None):
        E, dev = stream_x.shape[0], stream_x.device
        ar = torch.arange(E, device=dev)
        cards = []
        loss = stream_x.new_zeros(())
        stats, agree = {}, []
        for s in range(N_WAY):
            v = self.enc(stream_x[:, s])
            if self.use_codes:
                c, iw = (self.snap_write(v) if update_book
                         else self.snap_read(v))
                if code_map is not None:
                    code_map[ar, order[:, s]] = iw
            else:
                c = v
            cards.append(self.card(
                torch.cat([c, self.label_emb(labels[:, s])], -1)))
            mem = torch.stack(cards, 1)               # (E, s+1, d)
            # probe: fresh instance, no label, read through the cosine head
            u = self.enc(probe_x[:, s])
            if self.use_codes:
                cq, ir = self.snap_read(u)
                if code_map is not None:
                    hit = code_map.gather(1, probe_pos[:, s:s + 1])
                    agree.append((ir == hit.squeeze(1)).float().mean())
            else:
                cq = u
            r = self.qmix(torch.cat([u, cq], -1))
            att_logits = torch.einsum(
                "ed,emd->em", F.normalize(r, dim=-1),
                F.normalize(mem, dim=-1)) * F.softplus(self.raw_temp)
            att = att_logits.softmax(-1)
            read = torch.einsum("em,emd->ed", att, mem)
            logits = self.head(self.read_norm(r + read))
            loss = (loss + F.cross_entropy(logits, probe_lab[:, s])
                    + F.cross_entropy(att_logits.float(), card_t[:, s]))
            if s == N_WAY - 1:
                stats["top1"] = (logits.argmax(-1)
                                 == probe_lab[:, s]).float().mean().item()
                stats["read_hit"] = (att_logits.argmax(-1)
                                     == card_t[:, s]).float().mean().item()
        if agree:
            stats["agree"] = torch.stack(agree).mean().item()
            stats["used"] = (self.counts > 0).float().sum(1).mean().item()
            stats["merges"] = self.merges.mean().item()
        return loss / N_WAY, stats


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train(model, world, pool, steps, E, L, lr, device, tag, log_fn=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        cls = sample_lifetime_classes(pool, E, device)
        model.reset_book(E, device)
        loss = 0.0
        for _ in range(L):
            l, st = model.run_episode(*make_episode(world, cls, device))
            loss = loss + l
        loss = loss / L
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 200 == 0:
            print(f"[{tag}] step {step:5d}  loss {loss.item():.3f}  "
                  f"lastep-top1 {st['top1']:.3f}  read {st['read_hit']:.3f}"
                  f"  temp {F.softplus(model.raw_temp).item():.1f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if log_fn is not None:
                log_fn({f"{tag.strip()}/loss": loss.item(),
                        f"{tag.strip()}/lastep_top1": st["top1"],
                        f"{tag.strip()}/read_hit": st["read_hit"],
                        f"{tag.strip()}/step": step})


@torch.no_grad()
def eval_arm(model, world, pool, arm, E, L, batches, device):
    model.eval()
    acc, read = torch.zeros(L), torch.zeros(L)
    agree, used, merges = torch.zeros(L), torch.zeros(L), torch.zeros(L)
    diag = arm == "live" and model.use_codes
    for _ in range(batches):
        cls = sample_lifetime_classes(pool, E, device)
        model.reset_book(E, device)
        if arm == "oracle":
            xs = world.sample(
                cls.unsqueeze(-1).expand(E, N_WAY, 128).reshape(E, -1))
            means = model.enc(xs).reshape(E, N_WAY, 128, -1).mean(2)
            model.codes[:, :N_WAY] = means
            model.counts[:, :N_WAY] = model.cap
        code_map = (torch.full((E, N_WAY), -1, dtype=torch.long,
                               device=device) if diag else None)
        update = arm == "live"
        for e in range(L):
            _, st = model.run_episode(*make_episode(world, cls, device),
                                      update_book=update, code_map=code_map)
            acc[e] += st["top1"]
            read[e] += st["read_hit"]
            if diag:
                agree[e] += st["agree"]
                used[e] += st["used"]
                merges[e] += st["merges"]
    out = {"acc": acc / batches, "read": read / batches}
    if diag:
        out["agree"], out["used"] = agree / batches, used / batches
        out["merges"] = merges / batches
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--l-train", type=int, default=6)
    ap.add_argument("--l-eval", type=int, default=12)
    ap.add_argument("--d-in", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=1.4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--theta", type=float, default=0.45)
    ap.add_argument("--theta-merge", type=float, default=0.0,
                    help="bar mode's merge threshold; must exceed --theta")
    ap.add_argument("--merge-mode", type=str, default="off",
                    choices=["off", "bar", "witness"])
    ap.add_argument("--tombstone", action="store_true",
                    help="slots freed by merge stay dead this lifetime")
    ap.add_argument("--content", type=str, default="hard",
                    choices=["hard", "soft"])
    ap.add_argument("--code-temp", type=float, default=10.0)
    ap.add_argument("--load-nocode", type=str, default="",
                    help="checkpoint path: skip retraining the (identical "
                         "across versions) nocode baseline")
    ap.add_argument("--load-codebook", type=str, default="",
                    help="checkpoint path: skip codebook-model training "
                         "(eval-only reruns, e.g. longer l-eval)")
    ap.add_argument("--lam", type=float, default=0.0,
                    help=">0: v7 DP-means energy rule (one constant for "
                         "birth AND merge; overrides theta/merge-mode)")
    ap.add_argument("--lam-q", type=float, default=0.0,
                    help=">0: v7b quantile rent — lam = this quantile of "
                         "the lifetime's observed nearest distances")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-train-cls", type=int, default=48)
    ap.add_argument("--n-held-cls", type=int, default=12)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--save-prefix", type=str, default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    world = World(args.d_in, args.n_train_cls, args.n_held_cls, args.sigma,
                  device, seed=args.seed)

    # world difficulty: one glance vs consolidated
    c = sample_lifetime_classes(world.train_pool, 2048, device)[:, 0]
    x1, x2 = world.sample(c), world.sample(c)
    xo = world.sample(sample_lifetime_classes(world.train_pool, 2048,
                                              device)[:, 1])
    print(f"world: same-class glance cos "
          f"{F.cosine_similarity(x1, x2).mean().item():.3f}  cross-class "
          f"{F.cosine_similarity(x1, xo).mean().item():.3f}  "
          f"(consolidation headroom = the gap to ~1.0)")

    if args.merge_mode == "bar":
        assert args.theta_merge > args.theta, \
            "hysteresis: theta_merge must exceed theta (birth bar)"
    model_a = ToyBinder(args.d_in, k=args.k, use_codes=True,
                        theta=args.theta, theta_merge=args.theta_merge,
                        merge_mode=args.merge_mode,
                        tombstone=args.tombstone, content=args.content,
                        code_temp=args.code_temp, lam=args.lam,
                        lam_q=args.lam_q).to(device)
    model_b = ToyBinder(args.d_in, k=args.k, use_codes=False).to(device)
    assert not model_a.code_init.requires_grad, \
        "codebook must be gradient-free"

    if args.load_codebook:
        model_a.load_state_dict(torch.load(args.load_codebook))
        print(f"codebook model loaded from {args.load_codebook}")
    else:
        train(model_a, world, world.train_pool, args.steps, args.batch,
              args.l_train, args.lr, device, "codebook")
    if args.load_nocode:
        model_b.load_state_dict(torch.load(args.load_nocode))
        print(f"nocode baseline loaded from {args.load_nocode}")
    else:
        train(model_b, world, world.train_pool, args.steps, args.batch,
              args.l_train, args.lr, device, "nocode  ")
    if args.save_prefix:
        torch.save(model_a.state_dict(), args.save_prefix + "_codebook.pt")
        torch.save(model_b.state_dict(), args.save_prefix + "_nocode.pt")

    # encoder-space geometry on HELD classes (calibrates theta)
    with torch.no_grad():
        ch = sample_lifetime_classes(world.held_pool, 2048, device)
        u1 = model_a.enc(world.sample(ch[:, 0]))
        u2 = model_a.enc(world.sample(ch[:, 0]))
        uo = model_a.enc(world.sample(ch[:, 1]))
        print(f"encoder space (held classes): same-class cos "
              f"{F.cosine_similarity(u1, u2).mean().item():.3f}  "
              f"cross {F.cosine_similarity(u1, uo).mean().item():.3f}  "
              f"(theta={args.theta})")

    arms = {}
    for arm in ("live", "frozen", "oracle"):
        arms[arm] = eval_arm(model_a, world, world.held_pool, arm,
                             args.eval_batch, args.l_eval,
                             args.eval_batches, device)
    arms["nocode"] = eval_arm(model_b, world, world.held_pool, "nocode",
                              args.eval_batch, args.l_eval,
                              args.eval_batches, device)

    print("\nheld-out classes, frozen weights — final-step probe top1 "
          "(chance 16.7) by episode index within lifetime:")
    names = ["live", "frozen", "oracle", "nocode"]
    print("  ep-idx  " + "  ".join(f"{n:>7s}" for n in names)
          + "   agree    used  merges")
    for e in range(args.l_eval):
        row = "  ".join(f"{arms[n]['acc'][e] * 100:7.2f}" for n in names)
        print(f"  {e:6d}  {row}  {arms['live']['agree'][e] * 100:6.2f}"
              f"  {arms['live']['used'][e]:6.2f}"
              f"  {arms['live']['merges'][e]:6.2f}")
    print("\n  read_hit, same layout:")
    for e in range(args.l_eval):
        row = "  ".join(f"{arms[n]['read'][e] * 100:7.2f}" for n in names)
        print(f"  {e:6d}  {row}")
    for n in names:
        a = arms[n]["acc"] * 100
        print(f"  {n:>7s}: early {a[:3].mean().item():5.2f} -> late "
              f"{a[-3:].mean().item():5.2f}  "
              f"(delta {a[-3:].mean().item() - a[:3].mean().item():+.2f})")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for n, col in zip(names, ("C0", "C1", "C2", "C3")):
            ax.plot(arms[n]["acc"].numpy() * 100, marker="o", color=col,
                    label=n)
        ax.axhline(100 / 6, color="gray", ls=":", label="chance")
        ax.set_xlabel("episode index within lifetime (held-out classes, "
                      "frozen weights)")
        ax.set_ylabel("final-step probe top1 (%)")
        ax.set_title("test-time codebook consolidation — toy smoke")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"curve -> {args.out}")


if __name__ == "__main__":
    main()
