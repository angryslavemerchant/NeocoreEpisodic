# NeocoreEpisodic

Clean-slate continuation of the Neocore program (2026-07-20). The full
2026-07 research log — six pixel-era architectures, the vocabulary
program, the selection gauntlets — lives in the predecessor repo
`angryslavemerchant/NeocoreImagenet` (its CLAUDE.md checkpoints are the
authoritative history). This repo carries only the live code path, the
Vast.ai automation, and the current experiment line.

`POINTS_OF_INTEREST.md` (local-only, gitignored — NEVER commit, repo is
public) is the cross-session thinking log; read it after this file.

## Inherited findings (compressed; details in the old repo)

1. **The globality law**: any component that sees everything dissolves
   the selection question (free decoder / DINO contextualization /
   shallow pre-admission encoder — same failure at every layer).
   Selection only matters when information is forced THROUGH the
   bottleneck: admission over LOCAL pre-mixing tokens, budgeted
   perception, reuse pressure across episodes.
2. **Only architectural constants hold.** Every learned degree of
   freedom that can modulate how much survives gets exploited (exact-K
   is the only selection regime that ever held; soft pressures park
   under thresholds).
3. **Accumulate beats reselect** — measured three independent times
   (pixel-era R7, classification retention, episodic retention chains).
   Memory that must re-win its slot never learns retention.
4. **Static-image classification/reconstruction cannot reward
   selection** (rate-determined, architecture-blind; learned == random
   at every budget; closed permanently 2026-07-19).
5. **Vocabulary stack**: frozen DINOv2-S tokens probe 93.5 on IN-100
   (best pixel-trained encoder ever built here: 35.5); EMA k-means
   codebook K=2048 (artifact vocab-6duv9qzw, codebook_k2048.pt);
   symbolization semantically nearly free (92.1 @ K=8192).
6. **Episodic label-binding (2026-07-19, train_vocab_icl.py) — the
   first task where policies separate.** 6-image episodes, per-episode
   random label permutations (the answer is a binding that exists only
   in the episode — unsmearable); per-step probes; QK admission as
   per-step perception bandwidth over an ACCUMULATING memory; ONE fused
   binding token written per image (label slot's output, co-encoded
   with chosen content); architectural cosine read head (learnable
   temperature) + auxiliary retrieval supervision (CE on read
   attention). Results (best top1, chance 16.7, 20 held-out classes):
   - B4:  random 65.4 > oracle-MI 60.4 > learned 58.4 > static 52.1
   - B16: random 70.0 > learned 67.3 > static 65.0 > oracle-MI 63.0
   - dense (full attention, no bottleneck): **17.6 = CHANCE**
   Readings: (a) free-form attention CANNOT form the binding circuit at
   this scale — the architectural read head IS the capability, not
   scaffolding; (b) learned admission genuinely learned informativeness
   (matched/beat the MI oracle) — first learned-selection success, but
   needs aux supervision + epsilon-greedy (bare gate-whisper gradient
   never bootstraps); (c) **pick DIVERSITY beats pick QUALITY** (+3-7
   pts, measured against an oracle) on a broadly-informative vocabulary
   — the MAE random-masking law at the memory level; (d) top-k and
   uniform-random are two ends of a temperature dial; score-weighted
   SAMPLING is the policy that should dominate across regimes.
7. Toy diagnostics before cloud runs are mandatory and cheap: the ICL
   architecture went through six smoke-caught redesigns at $0 (each
   documented in train_vocab_icl.py's docstring); the oracle-arm
   dissociation (1.00 / 0.34 / 0.32) is what justified the launch.

## COMPLETED 2026-07-20 (overnight run, results below): from-scratch episodic foveated learner

Remove DINO. Learn perception + selection + memory END TO END from raw
pixels under the episodic binding objective. The aux read loss ("two
different images of the same class must produce matching card/summary")
is a contrastive/metric objective in disguise — episodic protocols
(ProtoNets/MatchingNets) are proven to train features from scratch at
exactly this data scale. This is the first full loop with all three
globality-law requirements met: local admission, budgeted perception,
cross-episode reuse pressure.

Design:
- **Tokens**: 256x256 RAM blobs (dataset_ram, unchanged pipeline) ->
  16x16 grid of 16px patches = 256 tokens. Stem = linear embed + 2-3
  PER-PATCH-ONLY layers (LN+MLP; NO cross-patch mixing — locality law;
  the zero-grad locality smoke must still pass).
- **Harness**: unchanged from train_vocab_icl.py — 6-way episodes,
  labels on everything, per-step probes, 80/20 class holdout, fused
  binding writes, cosine read + aux supervision.
- **Admission arms**:
  1. sampled-learned (NEW DEFAULT): admit by SAMPLING from the score
     softmax with learnable temperature (init tau=1) — quality AND
     diversity in one policy. Pure top-k at eval? NO — sample at eval
     too (a stochastic policy is the policy; report mean over the fixed
     val episodes).
  2. topk-learned (the old policy — regime comparison)
  3. random (the standing champion)
  4. dino-stem reference: same everything, frozen DINOv2-S features
     instead of the learned stem — anchors the feature-quality gap
     (there is no MI oracle without codes).
  Skip dense (proven chance) and static.
- **Budget**: B16 first (B4 starves early feature learning); B ∈
  {8, 32} only if B16 differentiates.
- **Schedule**: this is feature learning — longer and gentler than the
  code-lake sweeps. Start: 150 "epochs" x 100 batches x 512 episodes,
  lr 3e-4 cosine, wd 0.05, warmup 5. Expect the binding phase to
  arrive LATE for the scratch arms; do not kill a flat run before
  ~ep 40 (the toy plateau lasted half of training).
- **Measurements**: final-probe top1 + read_hit (primary); admission
  maps across training (WHERE does a system look while learning to
  see — the instrument the whole program has wanted); post-hoc
  attentive probe of stem features on IN-100 classification vs a
  random-init stem baseline (did episodic pressure mint reusable
  features?).
- **Success criteria**: (a) scratch arms well above chance and closing
  toward the dino-stem anchor; (b) sampled-learned >= both topk and
  random (the regime-dominance claim); (c) stem probe >> random-init
  probe.
- **Cost**: ~35-60 s/epoch estimate on a 5090 (256 tokens, B16, 7
  admission+probe passes; heavier than code-lake but images are
  RAM-resident). 4 arms x 150 ep ≈ 4-6 h ≈ $2-3. Smoke locally first:
  shrunken-world toy (icl.N_POS patching pattern) with a synthetic
  pixel lake; oracle-equivalent = signature patches.

### RESULTS (train_pixel_icl.py, wandb project neocore-pix, artifact
pix-icl-1ciu6b6l:v0 — checkpoints + 64 admission maps + summary; run
on m9105, ~13 h, ~$4.3; actual ~65-75 s/epoch)

Toy smoke (local, $0): locality zero-grad PASS; dissociation oracle
94.3 >> topk 61.9 > sampled 32.8 > random 22.3 (chance 16.7).

Cloud (best top1, chance 16.7, 20 held-out classes; stem probe =
post-hoc attentive IN-100 probe on frozen stem, 400 img/class subset):

    arm       best_top1   stem_probe
    sampled     39.60       37.90
    random      39.11       38.86
    topk        31.84       36.52     (peaked ~ep41 then DEGRADED to 27)
    dino        81.01        (93.5 known)
    randinit      —         21.66

Readings:
1. **The full loop works from pixels**: all scratch arms are far above
   chance with zero pretrained components — perception + selection +
   memory learned end to end under episodic binding alone.
2. **Success criteria**: (a) above chance yes, closing toward dino NO
   (39.6 vs 81.0 — the gap is the next frontier); (b) sampled >= both
   topk and random CONFIRMED (39.60 > 39.11 >> 31.84), though the
   sampled-vs-random margin is narrow (+0.5); (c) stem probe >>
   randinit probe CONFIRMED (+16-17 pts) — episodic pressure MINTS
   features. 37.9-38.9 also beats the best pixel-era encoder (35.5).
3. **Topk actively collapses from scratch** (31.8 peak -> 27 final):
   greedy selection + learned features co-collapse into a narrow loop;
   sampling is what keeps the feature-learning diet diverse. The
   temperature dial finding, now with a mechanism.
4. **tau drifts UP in the scratch sampled arm** (1.00 -> 1.22): the
   learned policy chooses MORE diversity when features are weak; the
   dino arm's tau stays ~1.0. Diversity demand scales inversely with
   feature quality.
5. **random's stem probes best** (38.86 > 37.90 sampled): broadest
   coverage = richest feature diet, but its bindings retrieve slightly
   worse. Selection quality and feature curriculum are separable
   objectives — first direct evidence.
6. **Admission maps** (the program's wanted instrument): scratch arms
   forage as a covering code with mild object affinity; the dino arm
   visibly clusters ON objects (birds, keyboards, faces). Feature
   quality is directly visible as foveation quality.
7. **The anchor is generous**: DINO tokens are ViT-contextualized
   (each token saw the whole image), so 81.0 includes a globality
   subsidy a per-patch stem can never match; the honest scratch
   ceiling is below it. Closing the gap is the loop-machinery agenda
   (re-entrant perception), not just a bigger stem.

## COMPLETED 2026-07-20 (same-day, local, $0): test-time codebook
## consolidation — toy smoke (toy_codebook_icl.py)

New direction (Ibanis): move learning from backprop into TEST-TIME
codebook development — a persistent store edited by a hardcoded rule
(online k-means: assign + count-capped running mean = literal gradient
descent on distortion, no backprop), meta-trained so codebook edits are
the only path to reward. Harness: LIFETIME = L episodes over the SAME 6
held-out classes; labels re-permuted per episode (binding unsmearable);
classes recur across episodes (recurrence capturable ONLY by the
codebook — weights never saw the classes, cards wipe per episode).
Two-stage read: percept -> code ("what is this") -> card ("what was it
called here"). Metric: final-step probe top1 vs EPISODE INDEX within
lifetime, frozen weights — a rising curve IS gradient-free learning.
Arms: live (edits on) / frozen (edits off) / oracle (codes = true
encoded class means) / nocode (cards from raw percepts = current
architecture).

Seven-version iteration (~2.5 h wall, all local; full log in the
script docstring): v0 random-init snapping (fragmentation; live ~=
nocode) -> v1 novelty-gated birth (WORKS: 61->67, but over-births) ->
v1b lower bar (collisions, ~nocode) -> v2 bar-merge (never fires) ->
v3 witness-merge (fires, breaks the anneal) -> v4/v4b tombstones
(structure perfect, accuracy worse) -> v5 soft mixture content
(Ibanis's proposal: pass a similarity-softmax MIXTURE over codes to
cards/queries instead of the nearest code) — DECISIVE:

    arm      ep0 -> ep11 (l-eval 12)     ep23 (l-eval 24, same weights)
    live       63.2 -> 69.8 (+6.6)         76.8 (+12 net, still rising)
    nocode     53 flat                     53 flat
    frozen     ~22                         ~22
    oracle     86-88                       87

Readings:
1. **Test-time learning via codebook development is real and
   substantial**: +12 pts over a 24-episode lifetime with weights
   FROZEN, 4x past the meta-training horizon (6-ep lifetimes), no
   plateau; oracle gap halved. The codebook is the only moving part.
2. **The birth/collision asymmetry**: over-birthing (dup codes) =
   dilution = recoverable; under-birthing (two classes in one code) =
   permanent (no split op). Err toward birthing.
3. **Capacity exhaustion IS the anneal**: birth-liberal while slots
   remain (exploration, avoids collisions) -> join-forced when full
   (consolidation). v1's accidental fixed point; every attempt to
   "improve" it structurally (v3/v4/v6 merge+tombstone) lost accuracy.
4. **Interface beats structure**: slot bookkeeping (used-codes,
   write/read agreement) DECOUPLES from accuracy; the fragmentation
   cost was always vector noise at the content interface, fixed by the
   soft mixture, not by tidying slots. Under soft content a duplicate
   POPULATION is a richer estimator than its merged collapse —
   duplicates are an asset at the soft interface.
5. **Meta-train under the full deployment protocol**: v4's accuracy
   decayed starting EXACTLY at the training-horizon episode (its rule
   kept developing the book into regimes training never produced); v5
   extrapolates freely because saturated-book drift is smooth.
6. Validated design (v6b, Ibanis's config): hard-assign updates +
   birth-until-full + soft mixture content + LAZY bar-merge, no
   tombstones — live 61.9 -> 70.1, best slope (+8.2). The bar
   trigger's lateness (fatal under hard content) is correct under
   soft: it merges only mature duplicate pairs, where the collapse is
   ~mixture-invariant. Aggressive structure (witness/tombstone) loses
   ~6 pts; CONFIDENT structure is free. Open wart: theta is a fixed
   threshold in a learned space (calibrated post hoc).

REAL-SPACE PRE-CHECK (check_dino_headroom.py, local, $0): the premise
measured on held-out IN-100 in frozen DINOv2-S space, before renting
anything. Same/cross-class cos 0.361/0.021 (CLS) — the toy's noise
regime (0.33/0.00) is DINO's actual regime, not a rig. 20-way k-shot:
prototype-of-k 77.5 (k=1) -> 94.8 (k=25), +17 pts consolidation
headroom (patch-mean percepts: 63 -> 88, +25); prototype beats
k-nearest-exemplar at every k>=2 (+3.2 CLS / +5.0 patch at k=25)
while storing 1 vector vs k. Consolidation beats memory in real
feature space; noisier percepts -> larger codebook payoff (percept
pooling is now an informed design knob).

Same-evening addendum (v7/v7b, Ibanis's "codes take up energy"): the
DP-means energy rule — ONE constant lam = rent per code; birth iff
distortion saved > rent, merge iff exact collapse cost (ni*nj/(ni+nj)
* ||ci-cj||^2) < rent — derives lazy merging from first principles
but FAILED twice on developmental dynamics: fixed rent bankrupts the
infant encoder geometry (v7: book pinned at 1 code, even oracle
collapsed — the model meta-trained around a degenerate store);
quantile-adaptive rent poisons itself on novelty (v7b: early record =
cross-class first-glances -> rent spike -> self-demolition, rebuild
too slow). Principle: test-time rules must be viable while the
geometry/record is immature, and adaptive statistics must be
insulated from the decisions they govern. Rent IS the granularity
dial of the ontology (rate-distortion Lagrangian; K becomes emergent,
lam sweeps = "at what grain does the world want to be known") — the
right formulation, parked until the geometry is frozen.

REAL-DATA RESULT (train_dino_codebook_icl.py, wandb neocore-codebook
run 6ej88ypk + verified artifact, m9105 5090, ~40 min, ~$0.35): pooled
DINOv2-S patch-mean percepts (encoded on-instance, 529 img/s), v6b
rule, 6-way lifetimes over the 20 held-out classes, L-eval 24:

    live 75.8 FLAT (delta -0.03) > nocode 73.0 flat >> frozen 26.9;
    oracle 84.8; agree ~74; used RISES 5.3 -> 13.7 all lifetime,
    merges 1.9 total; raw percepts same/cross cos 0.404/0.097;
    trained encoder space 0.552/0.014.

Readings:
1. The codebook is real and load-bearing on real data — live beats
   nocode by +2.8 everywhere and frozen-random codes crater to 27 —
   but its entire value arrives by EPISODE 0. No within-lifetime
   climb: the toy's rising curve did NOT transfer.
2. Why: the toy's climb was a property of its noise regime. At 6-way
   with a trained encoder (same-class cos 0.552 in encoder space),
   ONE glance already lands ~a code's-worth from the class direction;
   the first write captures nearly all consolidation value; further
   feeds add nothing at 6-way margins.
3. The oracle gap (9 pts) is a persistent SELF-ORGANIZATION gap, not
   a not-enough-episodes gap: used rises all lifetime — real classes
   are multimodal, the book fragments at VIEW grain (new views keep
   birthing), small-n view-codes never mature into the class means
   the oracle has. Granularity, again (rent!).
4. The pre-check's +25-pt headroom was measured at 20-WAY on raw
   percepts; the trained 6-way harness compresses it to oracle -
   nocode = 11.8, of which self-organization captures 2.8.

20-WAY FOLLOW-UP (overnight 2026-07-21, runs 63has9td/5j31u64g +
verified artifacts, m108899 Ryzen-5090, ~$0.60 incl. 5 failed boots):
(1) COLD 20-way never trains: 1/20 read-chance starves the bootstrap
whisper (principle #6, third independent sighting) — book pinned at 1
code, encoder cone TIGHTENED (0.95/0.84), while the nocode pathway
trained to 1.000 on the same task. Fix that works: --warm-start-run
(partial-load the 6-way-grown geometry, reinit label_emb/head/
code_init) — N-WAY IS A CURRICULUM HORIZON. (2) Warm-started 20-way
result (chance 5.0): live 46.9 flat (+0.3) > nocode 45.3 > frozen
9.9; oracle 56.9; used 13 -> 31.7/32 (view-grain fragmentation,
saturates ~ep 15); NO post-saturation climb (eps 16-23 flat).

LINE CLOSED 2026-07-21 (Ibanis): the toy proved what it needed to —
a codebook can learn at test time via handcrafted write/merge rules.
The recognition-benchmark framing stops here; codebooks continue in
the composition/proto-symbol direction (see POINTS_OF_INTEREST, the
reader thread). Unrun decomposition idea worth keeping: a raw-cosine
arm in any future harness eval separates machinery-generalization
tax from consolidation gap (at 20-way the trained circuit lost ~30
pts vs raw prototype cosine — the binding plumbing, not the book,
was the weakest component).

Verdict across both N: the codebook's value on real percepts is
INSTANT (+2-3 over exemplar cards, frozen craters) and the ~10-pt
oracle gap is CONSTANT — a persistent self-organization (grain)
deficit, not an episode-count deficit. Both rescue hypotheses for a
within-lifetime climb (fine discrimination, capacity-anneal) are
dead on real data. The toy's climb was driven by per-glance noise
that DINO percepts at this grain simply don't have; the meta-trained
encoder squeezes ~all identity into one glance and co-adapts AWAY
the need for accumulation. Open levers, untested: grain (theta/K
sweep targeting class-grain codes), percept noising (make one glance
insufficient by construction — occlusion/crop percepts), and tasks
whose QUERIES need multi-view knowledge (the "reader asks easy
questions" diagnosis — single-lookup probes may never grade
accumulation). Ops: vast --thresholds per-launch gate override +
thresholds_hf_light.json (Drive-bank rate-limited; jpeg floor
irrelevant for decode-once jobs); m9105 dropped a contract (caution);
m140318 known-bad; m108899 Ryzen NL = new known-good (3 clean
provisions, 851 img/s encode, full run in ~11 min).

## STARTED 2026-07-21: rung 1 — lifetime lexicon acquisition
## (toy_lexicon_icl.py; the proto-symbol / reader program)

Design (both DINO-campaign lessons built in): grammar permanent,
LEXICON per-lifetime (32 forms -> random meanings; unsmearable);
8 episodes x (4 studied pairs + 6 two-primitive composition queries);
cross-episode queries answerable ONLY through a persistent KEY-VALUE
book (key = form emb, payload = meaning emb — the pointer-array
factorization), written by the v6b economy, read as soft mixture
ADDED to the query token's residual stream inside a 3-layer d=96
transformer. Arms: live / frozen / oracle / ctx-all (whole lifetime's
pairs in context — the meta-seq2seq/MLC regime at lifetime scale) /
ctx-cap / episodic (current episode only).

v0/v1 results (local $0 + m61489 i9-14900K PRO-6000, 9 min, ~$0.20,
wandb neocore-lex run inqhogit + verified artifact):

    live/oracle: 100.0 cross, 100.0 within, 100.0 exact (book model
      perfect by step 200; filing flawless — used == studied, zero
      merges/collisions across all lifetimes)
    ctx-all:  ~25 within / ~0 cross at 2k steps AND at 8k steps with
      lr 5e-4 (~1M quiz examples) — never learns lookup, only answer
      SHAPE (25% = pad-structure knowledge)
    episodic: same (~25/~3); frozen: floor.

Readings:
1. The book-as-a-LAYER works completely: retrieval + composition
   through a residual-stream codebook trains ~instantly and solves
   lifetime-scale lexicon binding perfectly.
2. Attention-based in-context lookup FAILS TO IGNITE at this scale
   even at 4x budget — the dense-arm lesson in its third domain
   (vision episodic, 20-way bootstrap, now symbolic composition):
   free-form attention does not discover the lookup circuit that the
   architectural read path provides for free. >=40x learnability gap
   (book perfect at 200 steps; baselines nowhere at 8000).
3. Honest caveats for the eventual comparison: baselines are small
   (3L/d96) and the literature (meta-seq2seq, MLC) trains bigger
   models longer — the claim is about LEARNABILITY AT EQUAL SMALL
   BUDGET, not impossibility. Ignition experiments (deeper/wider
   baseline, curriculum) are the obvious next arm if the comparison
   needs teeth.

v2 — SYSTEMATICITY SPLIT (--holdout, the add-jump analog; run
29mqgtje, m61489, ~10 min, ~$0.20): forms 0-7 never meet TWICE in
any training lifetime; eval unconstrained. Result: NOVEL-combo
100.0 at every episode (book and oracle arms; baselines at their
usual floor). The factorization does not LEARN systematic
generalization — it makes the failure inexpressible: the book
delivers meanings frame-blind, the grammar is slot-uniform, and
neither organ can represent "this combination is new." First result
of the program aimed at an external benchmark axis (SCAN add-jump)
and it lands at ceiling by construction.

v3 — LENGTH split (train 2-word queries only, eval 3-word; fixed
sinusoidal positions so the grade lands on the circuit, not on
untrained slot embeddings; run ku961rhj, ~8 min, ~$0.15):
PRODUCTIVITY IS NOT FREE. live/oracle LEN3 ~57 per-position vs ~27
pad-floor, 0.0 exact everywhere. Decomposition is clean: oracle ==
live, so retrieval is length-blind (the book delivered the third
word's meaning) — the failure is the COMPOSER, a fixed-depth
two-group template that cannot unroll a third group. Tonight's
contrast, factored on the architecture's seam: systematicity is a
LEXICON property (solved by construction, 100.0); productivity is a
COMPOSER property (fails, like everyone's). Repair direction = the
program's own thesis: composition = re-reading — an ITERATIVE
composer (autoregressive emission = the minimal re-entrant loop)
should length-generalize by construction. That is the next surgery.

Also standing (Ibanis): an EXEMPLAR-STORE arm (append-only encoded
key/value pairs, same soft read, no economy) joins the arm set from
the noise version onward — in the deterministic world it is
degenerate-identical to the book (writes are append-only there:
used == studied, zero merges), so the current wins are attributable
to the vector-store INTERFACE alone; noise/capacity/drift are where
interface and economy separate (the proto-vs-exemplar decomposition,
symbolic edition).

Next candidates: iterative composer (productivity repair — the
loop's first mandated appearance); noisy demonstrations + exemplar
arm; lexicon > K; function words per-lifetime (rung 2); then
SCAN-proper under the meta-protocol.

## Local environment (Windows)

- No `python` on PATH. Project env is the `ToastEnv` conda env:
  `& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe"` and
  `& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\Scripts\vastai.exe"`.
- Training does NOT run locally — Vast.ai GPUs only. Local = smokes,
  toys, analysis. Set `$env:PYTHONUTF8="1"` for launch.py AND for any
  `vastai logs` call (unicode progress bars crash the CLI under
  cp1252 — the crash looks like "no logs"). In Monitor shells use
  `$TEMP` (via `cygpath`), not `$TMPDIR` (unset).
- wandb projects: `neocore-icl` (episodic era), `neocore` /
  `neocore-cls` / `asfnet*` are the old eras.

## Git rules (repo is PUBLIC)

- NEVER commit `vast/secrets.env`, `.vast/`, or `POINTS_OF_INTEREST.md`
  (all gitignored). **NEVER `git add -A` — explicit file lists only**
  (a private file was once published and had to be force-purged).
- No history rewrites without explicit user authorization.
- Cloud instances clone THIS repo from GitHub (branch `master`) — cloud-
  side changes (onstart.sh, thresholds, train scripts) only take effect
  after push.

## Vast.ai runbook (vast/README.md + BLUEPRINT.md for details)

```powershell
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py search --profile 5090
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py launch --train-script <script> --train-args "..."   # hedge 3 default
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py status | logs | destroy | pull
```

- **Profiles** (user directive 2026-07-17): `5090` workhorse, HARD CAP
  $0.38/hr; `6000` ($1.2 cap) for user-requested same-day results;
  `b200` user-approved only. cpu_ram >= 48 GB everywhere; EPYC fine.
- **Offer selection is a judgment call**: run `search`, apply
  `vast/OFFER_JUDGEMENT.md` (mid-price band, known-bad list, check
  status_msg when logs stay empty — DNS-dead hosts show `curl: (6)`
  there). Hedge race (`--hedge 3`, default) for lottery conditions;
  single `--offer` picks for diagnosis.
- **Health gate**: instances benchmark themselves at boot
  (vast/benchmark.py vs thresholds.json — incl. the Drive-bank probe)
  and self-destroy when unhealthy. Failed RUNS keep the instance alive
  (`AWAITING_PULL` -> `launch.py pull`, then destroy). RUN_COMPLETE ->
  verified wandb artifact -> self-destroy ~30 s later: **an instance
  vanishing is only a failure if wandb does NOT show the run finished —
  always cross-check before re-renting or accusing a machine.**
- **Data**: Google Drive bank (dedicated storage account, NOT the
  user's personal Drive; token = RCLONE_DRIVE_TOKEN in secrets.env,
  forwarded base64) serves the 2.35 GiB jpeg tar; 25 GB RAM blobs and
  the DINO lake rebuild locally from it (~6 min boot-to-training on a
  healthy host). wandb is NOT acceptable for dataset storage (free
  plan). Publish/refresh bank with vast/upload_bank.py (256M chunks).
- Boot flow gotchas: Vast stop->start re-runs the provision command and
  WIPES the repo clone — do not "stop to preserve state"; onstart
  symlinks caches into /workspace so same-machine restarts survive.
- Known-bad machines: `.vast/blacklist.json` + the list in
  OFFER_JUDGEMENT.md. 2026-07-19 network event: California-EPYC pool at
  9-20 mbps to BOTH HF and Drive; gate blocks it; may have cleared —
  let the gate decide.
