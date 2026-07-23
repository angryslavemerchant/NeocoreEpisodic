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

v4 — ITERATIVE COMPOSER (toy_composer_icl.py; run snu6ucah, ~7 min,
~$0.15): walker (hardcoded cursor/counter/advance walk; learned only
emission head + modifier interpreter via exact expected-CE over
repeat combos) vs AR (same tower+book, causal token-by-token
emission). Both trained on 2-primitive queries only:

               len2        len3        len4
    walker   100/100     100/100     100/100
    ar       100/100      69/0        45/0    (per-position/exact)

PRODUCTIVITY BY CONSTRUCTION. The rung-1 story closes symmetric:
systematicity came free when the lexicon became a circuit's contents
(v2); productivity comes free when iteration becomes a circuit's
walk (v4). Sequential emission alone does NOT suffice (AR decays
classically) — what matters is WHO OWNS the iteration. Fifth domain
for the build-the-circuit law. The walker's modifier interpreter is
rung 2's smallest specimen: TWICE trained as an OPERATION (counter
:= 2), not a meaning — learned in ~30 steps.

v5 — SCAN-LITE, FROZEN KERNEL (toy_scan_icl.py; kernel local 1 s $0;
both-arms run k6mjg1iw ~10 min ~$0.20): single-clause SCAN family
(verbs/dirs/opposite/around/counting, ALL per-lifetime; frames V..VSDC).
Kernel arm: ZERO trained parameters — atoms filed in the book,
struct/count words learned by EXACT HYPOTHESIS ELIMINATION from one
demo (enumerable program space; opposite/around are candidate
PROGRAMS over the same repeat/emit kernel — NO new ops), execution by
frozen templates. Result: 100.0 exact on every frame, including
verbs never demonstrated inside any structure (add-jump at full
strength), in one second, gradient-free end to end. AR contrast
(3L/d96, full 120-token study stream in context, 4000 steps
teacher-forced meta-training): train loss 0.13 but greedy exact
28.2 (V) -> 10.9 (VD) -> 7.3 (VDC) -> 4.3 (VSDC) — learned local
imitation, not execution; degrades monotonically with structure.
Epistemic status: kernel 100 is near-tautological (correct algorithm
in a deterministic world); the FINDINGS are (a) the kernel expressed
a second grammar family with no new operations (first counter-
evidence to the brittleness objection), (b) one-demo word learning
incl. structural words, (c) the separation-of-concerns gap measured:
computing acquisition vs optimizing toward it.

v6 — NOISE + CONNECTORS (toy_scan_noise_icl.py, local, 2 s, $0):
connectors PASS the frozen-kernel discipline (swap-bit program over
order-blind concat; no new op; 90 -> 100 as tie-demos wash out);
noise restores the within-lifetime LEARNING CURVE via vote
consolidation (p=0.3: 42 -> 95 over 6 rounds) — exactly where the
DINO diagnosis located it (where one observation stops sufficing).
Caught + fixed a silent gather-broadcast bug via the pre-registered
clean-world 100s.

v7 — CAPACITY ECONOMICS (eval-only sweep on the v3 lexicon
checkpoint, local, $0): K in {24,16,12,8} drawers for 32 words:
live economy 95.7/81.3/68.8/51.1 vs oracle-truncation 75.9/50.1/
38.2/25.9 (exactly K/32). The economy beats truncation ~2:1 at every
pressure: forced joins blend payloads and the soft read + tower
recover information BEYOND one-word-per-slot — superposition as the
emergent overflow strategy. Zero-shot (model never saw overflow in
meta-training) -> learned-superposition headroom if trained scarce.

v8 — SCAN-PROPER, REAL DATASET (scan_proper.py, 2026-07-21, local,
2 s total, $0; data/scan gitignored, re-fetch via curl from
github.com/brendenlake/SCAN): brittleness bout #3, pulled forward at
Ibanis's request ("when are we moving to an actual dataset?"). First
external benchmark of the program. Word ROLES are no longer given —
the learner infers role AND program for all 13 real words by exact
elimination over a computation-shaped hypothesis space (VERB=emission
seq incl. empty; DIR=atom; STRUCT in {d^k v, (d v)^k, k<=4} — 8
programs; COUNT in {2,3,4}; CONN=swap bit) under an authored clause
template V [S] [D] [C], clause [N] clause. Pre-registered: oracle
schema 100 on train; no new ops admitted; 100/100/100 predicted.

    split          train    TEST (exact match)   published seq2seq
    simple         100.0    100.0  (n=4182)         ~99.8
    addprim_jump   100.0    100.0  (n=7706)         ~1.2
    length         100.0    100.0  (n=3920)         ~13.8

All predictions held; zero contradictions; no op #4 needed (around =
(d v)^4, opposite = d^2 v, turn = EMPTY-emission verb, after = swap
bit — all discovered, not named). Acquisition: 15-18 conclusive
demos from the natural corpus; sorted-shortest prefix sweep: 8 demos
= 6/13 words, 24 = 11/13 (25.9), 32 = 13/13 -> 100.0 on the full
add-jump test set. Zero parameters, zero gradients, ~1 s per split.
Findings beyond the score: (a) RANDOM 320-demo subsets score 0.0 —
the one-unknown-per-demo eliminator needs the corpus's short-command
curriculum ladder (bootstrap-whisper law, symbolic edition; sighting
#4); joint multi-demo inference (= the amortized-inference build) is
what would remove that dependence. (b) The famous add-jump split is
trivialized by the pointer-array factorization exactly as v2
predicted: jump's 1467 train occurrences are ALL isolated, and it
composes perfectly anyway. Epistemic status: like v5, near-
tautological GIVEN the schema fits — the finding is that the same
8-program kernel from the toys fit the real grammar with no new
operations, and that the length split (the field's wall) is
length-blind by construction.

## STARTED 2026-07-21/22 (overnight): THE STREAM WORLD — the program's
## goal named, and v0 built (toy_stream_icl.py)

The goal, converged with Ibanis after the SCAN wins deflated ("what
did we do past a soft-updating codebook?"): **the missing middle
timescale**. The standard recipe (token mixing + dense + AR) has two
knowledge stores — weights (learn over training, then freeze; new
knowledge smears in destructively) and context (holds anything,
retains nothing, quadratic rent) — and NOTHING between. Field check
(2026-07-21, sources in POI): DeepSeek MoE routers ARE codebooks
(expert centroids + a handcrafted rent-like load-balancing bias);
PEER = million tiny experts behind product keys; Meta's Memory
Layers at Scale = codebook layers replacing FFN at 128B params
beating dense at 2x compute; DKVB = sparse codebook continual
learning. ALL are carved by backprop then FROZEN at deployment.
Nobody ships write rules. The economy is the piece we hold. Target
demo: frozen-weight LM + writable codebook layer acquires
compositional knowledge from a stream, permanently, beyond any
context window; dense/frozen-book twins can't. "Learning that
doesn't end when training does." The kernel/SCAN line stands as
instruments; the machinery agenda continues INSIDE a model (Ibanis:
"a machine not a brain" — corrected course).

v0 (toy_stream_icl.py, built+run overnight; local gate + 2 cloud
runs, wandb neocore-stream vh0tirc1 clean / kbinqoq4 noise, both
verified artifacts, ~$1.1 incl. 2 failed boots): 6L d256 causal LM,
DOCUMENT-LOCAL attention (context = the sentence, by construction),
one v6b book read at two depths (two hops = composition;
read-output zero-init), writes on fact sentences (key = typed
subject embedding theta=0.75, payload = mean token embedding), aux
retrieval CE annealed to ZERO mid-training, per-lifetime entities/
relations/attributes (16 entities, unsmearable). Questions: lookup
(rel/attr) + 2-hop composition (attr of A's partner — facts filed
documents apart). Arms: live / exemplar-FIFO (append-only ring) /
frozen / oracle / dense twin.

    quiz acc (rel/attr/comp)   clean 16-ent      noise p=0.3, 4 stmts
    live (v6b economy)         100/100/100       94.3/93.7/89.1
    exemplar-FIFO              100/100/100       46.5/54.1/28.3
    oracle                     100/100/100       100/100/100
    frozen                     ~chance           ~chance
    dense twin                 5.2/16.7/16.9     5.2/15.7/16.0  (chance)

    capacity sweep (live vs exemplar, quiz comp):  K=64  48  32  24  16
    clean  live   100 100 100  75  55     exemplar  100  50  29  25  19
    noise  live    90  89  89  66  51     exemplar   28  23  19  18  17

Readings:
1. **The middle timescale exists in v0**: a frozen-weight LM answers
   2-hop compositional questions about facts no context window ever
   held — through the rule-written book alone. Dense twin: chance
   forever (no channel, by construction). Aux wheels annealed to 0
   with no wobble in BOTH worlds: the circuit stands without
   supervision at deployment AND at the end of meta-training.
2. **CORRECTED 2026-07-22 (Ibanis's catch, verified by re-eval)**:
   the noise-world exemplar collapse was CAPACITY STARVATION (128
   statements into 64 slots), not a consolidation win. Exemplar with
   K=128 TIES live under noise (94/93/88 vs 94/93/90): the soft read
   over duplicate entries performs the same averaging/vote as the
   running mean — interface-over-economy, third sighting. The
   economy's real earned claim: SAME accuracy at 1/4 the storage
   (32 slots vs 128) — compression, not accuracy. Requirements
   track facts vs statements; that part stands.
3. **Graceful degradation under forced joins**: live at K=16 (half
   the fact count) still ~2x exemplar everywhere — the v7 lexicon
   superposition result, third domain.
4. Live holds ceiling until K < #facts exactly (100 at K=32=facts,
   drop at 24): the economy's dedup makes capacity requirements
   track FACTS, not statements; exemplar's track statements (dies at
   K=48 < 64 statements even clean).
5. Ops: m122781 gate-killed at bench (fine); m58908 "success,
   running" but ZERO log output 30 min = silent zombie, destroyed
   manually — new failure mode for the ledger; m55313 + m144477
   (Estonia) clean fast runs (~0.35 s/step at d256/6L).

v1 relaxations queued (the wheels, in removal order): learned writer
head on hidden states (not raw embeddings); write timing decided by
novelty gate (no harness fact-flags); interleaved fact/question
streams (true within-lifetime curve); paraphrase templates; then the
REAL milestone — a small real-text LM (TinyStories-class) with one
FFN swapped for the writable book, meta-trained on entity-renamed
streams: the Meta-Memory-Layers comparison with deployment writes.

AUX ABLATION (2026-07-22, local, $0, Ibanis asked "could it exist
without the aux?"): trained the toy from scratch with aux_w=0 from
step 1. SELF-IGNITES: hop-1 lookup 100 by step 100 unsupervised;
composition plateaus ~30-45 for 300 steps then PHASE-JUMPS to ~100
at step 400 (induction-head-shaped transition). The aux buys speed,
not existence — at this scale, with recipe writes. Caveat: the
learned-writer version reintroduces a write-read chicken-and-egg;
aux stays as ignition insurance there, with writer-init-at-recipe as
the aux-free alternative to try first.

REAL-TEXT CAMPAIGN STAGE 1 (2026-07-22, stream_text_icl.py): GPT-2
BPE pipeline built per the 4-layer dataset architecture agreed with
Ibanis (fact graph -> slot templates -> assembly -> probes; nonce
names are MULTI-TOKEN spans — the honest hard mode; held-out
paraphrase templates as an eval split; two-hop chains person ->
company -> industry/city and person -> spouse -> profession).
Answers graded teacher-forced exact-match over multi-token spans.
v1 wheels: harness-slot filing (fact-id), doc-mean-embedding
content; live-theta arm instruments how far bag-of-embedding
novelty-gate filing gets at BPE level (expected to degrade —
informative, not gating). RESULTS (local, 2500 steps then 6000):
gate PASSED — at 6k steps live 82.6 hop1 / 99.2 hop2 vs dense
7.8/13.1 (floor) and frozen ~1/2; HELD-OUT PARAPHRASES FREE (83.1/
99.0 == trained); two-hop essentially solved, hop1 limited by
5-token-exact name answers and still climbing at 6k. live-theta
(bag-of-embedding novelty filing) CLOSED most of its gap with
training (44/59 at 2.5k -> 78/93 at 6k): the embedding geometry
organizes to make even naive filing workable — but the learned
writer stays justified for paraphrase-robust keys. Book model's LM
loss 0.29 vs dense 1.17: the memory also makes the stream itself
predictable (the perplexity signature, visible in miniature). Plan (Ibanis-approved sequencing): this
smoke -> co-design LLM template bank -> graft into pretrained
124M (LoRA ladder: r16 -> r64 -> unfreeze-middle -> full FT) ->
five-arm experiment (writable / frozen-at-eval / Meta-regime
backprop-slots / dense / long-context / LoRA-per-stream). Est.
~$10/full pass, 2-3 nights total.

IGNITION-CONDITIONS BISECT (2026-07-22 evening, 7 cloud runs on PRO
6000s, ~$3.5 total, wandb neocore-stream bisectA/B1/B2/C/D/D1/D2 +
textv2.0-2.2 failures): v2 world (bank templates, 55 facts, fillers,
interleaved questions, abstention) failed to ignite THREE times;
incremental guesses (aux length, abstention curriculum, IDF pooling
+ subject-hook aux) all failed; diagnose_v2.py acquitted key
geometry (same-entity cos ~0.42 either template regime). Factorial
bisect verdicts (all single-variable, 2.5-3.5k steps each, h1/h2 at
book-phase end):
    A  v1-parity world            85/97  IGNITES
    B1 +full 44-template bank     74/92  ignites (slower)
    B2 +full 55-fact world        82/94  ignites (slower)
    C  diversity x size           ~50s@3.5k  ignites (slower still)
    D1 +30% fillers               ignites
    D2 +16 MID-STREAM QUESTIONS   16/21  SUPPRESSED 5x  << THE WALL
    D  full stream machinery      13/17  fails (+abstain basin)
LAW (bootstrap-whisper, ignition edition): partial-book retrieval —
questions asked while the book is still filling — is the hardest
regime and PREVENTS the read circuit from forming if present from
step one; the same regime is harmless curriculum once the circuit
exists. Remedy shipped: stream-question warmup (35%) + abstention
warmup (60%) — hard regimes arrive after ignition. Also: 'unknown'
as a universally-valid answer is a degenerate basin pre-ignition;
answer-side floors (dense) ~2-5/8-15 from closed-set guessing.
v2.3 assembled gate (all axes + both curricula, 6k steps) launched
same evening. Ops: two Vast phantom contracts (create reports
success:False but creates anyway — always status-check + destroy
after failed create; cause: launching onto an occupied machine).

## RUNG-1 CAMPAIGN DAY 1 (2026-07-22/23, rcore_lm.py; ~$15 total,
## 2 real runs + 8 toy gates + ~6 phantom/zombie ops events)

Ibanis made THREE mid-build corrections, each now load-bearing:
(1) core = FILTER not registers (buffer = the K selected token
states themselves; selection-over-superposition, vision law);
(2) UNIFIED STACK — one decoder-only model, lower windowed layers
ARE the archive-builders, core mid-stack, upper layers see
[window || buffer] only; no pretrained/frozen parts;
(3) PURPOSE reframe: the experiment grades "can the model learn
WHERE TO LOOK" (bufhit@question = primary metric, built); recall
is downstream. Dense "sees the answer" trivially — context, not
competitor.

RESULTS (wandb neocore-rcore, artifacts verified):
- Frozen-archive rung 1 (TinyStories-8M encoder): KILL CRITERION
  FIRED both runs — learned ~= random ~= s2 on far recall; smoking
  gun = RANDGAZE INVARIANCE (same recall with a random buffer: the
  reader never used the buffer at all). S2 gaze aimed 3x better
  (0.35 vs 0.11) with zero recall payoff. Dense twin: 86.6 far-h1 /
  68.5 held-out-paraphrase / 38.3 h2 — full attention + trainable
  representations ignite lookup easily at 18M/8k steps. Core arms
  WIN story ppl in both runs (6.71 vs 7.10; 7.14 vs 7.30).
- Unified-stack toy gates v3-v6: reader blocked by ARCHIVE DRIFT —
  new law sighting (bootstrap-whisper, representation edition):
  **the read circuit must form BEFORE its substrate starts
  moving.** Bisect proof: frozen-at-init lower layer -> reader
  climbs (chance->30); training lower layer -> chance forever.
  low-freeze curriculum: circuit survives the unfreeze and keeps
  improving (oracle 31, plateau = toy world ceiling). Learned-gaze
  bootstrap still dead even with annealed gaze-aux (v6) — the
  three-way chicken-and-egg (archive<->gaze<->reader) does not
  self-ignite with generic interfaces.
- Ops: --thresholds takes vast/-prefixed path (bare name crashed
  benchmark -> GATE_FAILED self-destroyed 3 healthy boxes);
  offer-id lists go stale in ~15 min (4 phantom contracts — always
  re-search before launch); m37505 one-strike silent zombie;
  parallel single-arm instances work well (~13 min/arm on PRO 6000).

RESOLVED (same day, design conversation with Ibanis): the A-D patch
menu was superseded by a ground-up re-derivation. **The agreed base
architecture is frozen in RCORE_SPEC.md (v1)** — read it before
touching rcore_lm.py. Headlines: in-line core (selected positions
route THROUGH a small dense-attention reasoning core, everything
else flows F->G untouched; core=identity == dense EXACTLY); gaze
computed by the core's own gaze tokens (self-directed selection,
loops S times, accumulate-vs-replace as an arm pair); serialization
analysis (chain length = persistence grain; per-token viable at
~2-5x with big batch + CUDA graphs, chunked ~1.5-3x = workhorse);
restrictions (windows on F and G) are what MAKE the core
load-bearing and arrive as later additions/curriculum, training
starts dense-identical (easy gradients — the drift-law answer);
primary metric = bufhit@question (gaze-first purpose). NEXT: build
ladder step 1-2 (base impl + identity test + dense-parity sanity),
rcore_lm.py needs a substantial rewrite to the spec.

## SIMPLECORE CAMPAIGN (2026-07-23 afternoon; simplecore.py;
## RCORE_SPEC.md v1 is the frozen design — READ IT FIRST)

Built to Ibanis's re-derived architecture in one afternoon: dense
F(3) -> in-line core (K=32 working set, 4 gaze tokens whose outputs
ARE the next loop's queries, S=2 loops, accumulate) -> dense G(5);
core output delivered as K slot tokens per 64-token boundary inside
ONE interleaved causal sequence; Flamingo-style gates (content gain
tanh(0.1), column logit bias -4) so slots enter near-silent; reduced
vocab 24,495 (13.5M params); identity test BIT-EXACT (masked slots
== dense). NO scaffolding anywhere (no boost/aux/curricula —
pressure knobs are diagnosis tools now, per Ibanis).

Base-stage results (wandb neocore-simplecore, all artifacts
verified; plain 20%-fact mix, 8k steps, T=2048):

    arm                     ppl     late-training     gaze/bufhit
    dense twin             9.83     stable            —
    core ungated          10.93     spike@7.4k        0.38/0.34 (!)
    CORE GATED             7.98     SMOOTH            0.00/0.00
    inert registers       28.84     COLLAPSE@>4k      —
    (gated core + random gaze at eval: ppl 7.99 — content
     irrelevant at INFERENCE)

Readings (n=1 each — seed-1 replications of gated-core and
registers launched, verify before trusting):
1. UNGATED run: gaze SELF-ORGANIZED toward facts unpaid (0.38 vs
   0.196 neutral; bufhit 0.34 vs 0.22 random) — first spontaneous
   where-to-look signal of the program. Its late loss spike
   co-timed with the gaze surge = channel coming online unmanaged.
2. GATED core beat dense by 19% ppl, smooth throughout; but its
   gaze went sink-like (earliest tokens, fact-gaze 0.000) and
   randgaze-at-eval ties => the win is NOT retrieval content.
3. Registers control (same slots, static content) COLLAPSED late =>
   the simple furniture/register-token story is rejected; adaptive
   core content appears to STABILIZE the slot channel during
   training even though it is not read for prediction at eval.
   Weird, novel-shaped, needs the seed replications.
4. Overhead: gated core 0.31 s/step vs dense 0.14 (2.2x, in the
   predicted envelope; registers 0.18 => core sessions ~= 0.13).
Ops: price-FLOOR offers are the lemon cluster (4 dense boots lost
to it; keep-alive autopsy: "PRO 6000" at 75 bf16 TFLOPS, 218 MB/s
disk). Shop mid-band ($1.00-1.10, rel >= 0.994). Offer lists stale
in ~15 min. m37505 one-strike zombie. wandb run names now
uniquified (two same-name runs confused a chart read).

SEED-1 REPLICATIONS (same evening; both artifacts verified):
    core gated s1 6.27 / registers s1 6.32 (no collapse; s0's
    registers collapse = seed lottery, ~1-in-4 slot-run hazard).
FINAL BASE-STAGE VERDICT (replication-backed): (1) slot content is
a PASSENGER — core == registers within seed, randgaze ties at eval,
bufhit ~0; ALL ppl gains are the register/furniture effect (large:
6.3-8.0 vs dense 9.83). (2) The stabilizer story from s0 died on
replication — replicate before believing. (3) Recall at the
unrestricted base is fluency-scaled GUESSING (core s1 "scores"
h1 0.20 with a provably empty retrieval channel) — no memory
claims exist until restrictions kill guessing. All as RCORE_SPEC
section 5 predicted for the base.

UNGATED 16k EXTENSION (2026-07-23 eve, ~$3, killed by user call at
~8.2k steps; wandb neocore-simplecore sc-simplecore-s16000-*): tested
"maybe the 8k gaze surge was undertrained." VERDICT: the opposite —
the ungated channel is a BOOM-BUST oscillator that ends in
detonation. Movie (batch 24, gate-init 5.0/bias-init 0.0 =
reproduced pre-gate config via new --gate-init/--bias-init flags):
gaze climbed to ~0.5 by step 3.4k (above the 8k run's 0.38 peak) ->
collapsed to 0.03 at ~5.2k with loss creep 1.68->1.9 -> partial
rebuild -> LOSS BLOW-UP at 7.8k (1.89 -> 4.46, recall zeroed).
Reading: predator-prey co-adaptation through an undamped channel —
gaze sharpens, slot statistics shift faster than G tracks, G's
cheapest fix is forcing bland selection, cycle repeats and
amplifies. The 8k run's "late surge" was cycle 1 caught mid-boom.
The Flamingo gate is load-bearing DAMPING, not cosmetics; ungated
config closed permanently. Ops: batch 96 OOMs a 96GB card at step 1
— the float32 CE logits path (~19GB + grads at B=96, 24.5k vocab)
is the VRAM hog; B<=64 without a chunked-CE rewrite. Fat-batch
exchange-rate question (B64 twin) launched but destroyed unresolved
at the quits call.

NEXT: (a) restriction stage per RCORE_SPEC section 5 (window G
then F — necessity arrives; twins can finally diverge on
retrieval; gaze gets its first real job); (b) tame the slot-channel
seed lottery (gate schedule / slot norm); (c) 16k-step runs for the
GATED config (ungated's 16k verdict is in above; gated 16k still
unwatched); (d) cosine tail as channel-opening stabilizer arm.

## NEXT SESSION ENTRY POINT — REASONING-CORE LM (2026-07-22 pivot;
## full interview-resolved spec in POINTS_OF_INTEREST session-close)

The programs merged. Build rung 1 of the reasoning-core LM: a
general small LM (10-30M, TinyStories-class corpus) with:
- frozen pretrained causal encoder (unbounded KV cache = the archive)
- K persistent core REGISTERS (32-64), residually updated by
  per-token-admitted encoder tokens (train chunked: admission
  queries from chunk-boundary core state; infer per-token)
- admission scored by core-state-modulated QK, sampled exact-K
- decoder attends ONLY [local window || core] — never the cache
  (free-decoder law: admission is the sole long-range route)
- arms: learned-gaze / RANDOM-GAZE (the conscience) / matched-
  compute dense / (later) gated-state baseline
- bar: ppl ~parity with dense + recall/composition wins on
  stream-world EVAL PROBES; kill if learned <= random gaze
- S=1 core first; S=2 iff admission-thrash in the maps
- expect ignition fights: apply the bisect curricula (hard regimes
  arrive after circuit formation), aux-with-anneal available
Open receipt (parked, low priority): the 124M graft never produced
its table (3 OOMs -> per-chunk fix landed -> final attempt zombied).
Memory-line v2.3 gate result stands as that campaign's checkpoint.

STANDING ALTERNATE — AMORTIZED INFERENCE (propose-verify-file, spec
at v8): the symbolic line's next build if the program returns there.

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
