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
