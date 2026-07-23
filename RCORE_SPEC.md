# Reasoning-Core LM — Base Specification (v1, frozen 2026-07-23)

Agreed between Ibanis and the operating agent after rung-1 day 1
(kill verdicts + toy-gate campaign in CLAUDE.md; design conversation
resolved this document). This is the BASE. Restrictions and training
curricula are additions layered on top, each as its own decision.

## 1. Topology (the base object)

A standard dense decoder-only transformer, cut once.

    tokens -> F (lower dense layers)
           -> CORE (inserted at the cut)
           -> G (upper dense layers) -> next-token head

- F and G are ordinary causal transformer stacks. Nothing bespoke.
- The core is an IN-LINE module: it selects some positions' F-states,
  computes on them, and its outputs serve G. All non-selected
  positions flow F->G untouched.
- **Identity property (the design's anchor): if the core computes
  nothing, the model IS the dense transformer — exactly.** Every
  deviation from dense is therefore attributable to the core.

## 2. The core

A small dense transformer over a working set, with self-directed
selection. Per session:

    repeat S times (the loop):
      1. gaze tokens emit queries
      2. queries score ALL F-states (keys = per-token projection,
         precomputed once, no mixing)
      3. top/sampled-k positions admitted; their raw F-states join
         the working set
      4. core's dense attention layers mix
         [ working set || gaze tokens ] together
    after S loops: member states are the core's output for this
    session's stretch of the stream

- **Gaze tokens (mechanism 2, agreed)**: dedicated learned tokens
  that ride the core's own attention layers; their OUTPUT states are
  the next loop's queries. The gaze is computed by the same
  reasoning that mixes the members — it can express "what is
  MISSING from the set," not just "what resembles it." Gaze tokens
  CARRY across loops (the gaze remembers where it already looked).
- The working set IS the core's state. No registers, no slots,
  nothing held that is not a chosen token's (evolving) state.
- The selected set is SHARED across the positions a session serves
  (one gaze for the whole present) — this, not per-token retrieval,
  is what makes it a working-memory architecture.

### Arm pair to test (both, per Ibanis):
- **accumulate**: admit k=K/S per loop into a growing set (the
  pixel-era recursive-admission shape; validated R1->R7 monotone).
- **replace**: reselect the full K each loop.

### Deferred core decisions (explicitly open):
- masking inside the core (free vs causal-by-source-position)
- exact k, K, S, core depth

## 3. How G consumes the core (the two-sided picture)

Conceptually: the core edits the past — selected positions' states
are upgraded in place. Mechanically (parallel hardware cannot hold
two versions of one position in one attention pass): the stream is
partitioned into stretches; each stretch's positions receive their
session's output set as their view of the core's work. The in-line
picture is how to THINK; the per-stretch view is how it compiles.

## 4. Serialization (why cadence is the only real training cost)

Training parallelism exists wherever computation depends only on
known inputs. F is one parallel pass; G (given core outputs) is one
parallel pass. The ONLY serial part is the core chain: session u's
gaze depends on session u-1's computed state.

    wall-clock = chain length x per-link latency (kernel launches)

- per-token persistence: ~T x S links — naively 10-30x dense cost;
  recoverable to ~2-5x via large batch (chain cost is
  batch-independent) + CUDA-graph capture. VIABLE as an arm.
- per-chunk persistence (C~64): ~T/C x S links — ~1.5-3x dense.
  The workhorse cadence.
- inference: generation is already serial; per-token core sessions
  ride the existing loop at negligible cost. Train chunked / infer
  fine is legitimate but is a measured distribution shift, not a
  free lunch (evaluate trained model at several cadences).

## 5. Restrictions (the additions that make the core load-bearing)

In the base, the core is a redundant shortcut: dense F pre-globalizes
every state and dense G can look anything up itself. The
architecture only becomes NON-trivial through denials:

1. F not-global (else its states smuggle the whole past up the
   residual stream) — sets what one archive entry MEANS.
2. G's cross-position attention not-global (else it bypasses the
   core) — sets what "long-range" means.

Sliding windows are the simplest denial shape. With both denials,
the core's picks are the ONLY long-range route and "where to look"
becomes gradeable. Restriction schedule = a later decision; the
dense-identical base trains first (easy gradients, no ignition
warfare), restrictions layered after — possibly as a curriculum
(anneal windows in), possibly as arms.

## 6. Purpose and primary metric (Ibanis's reframe, 2026-07-23)

The experiment grades: **can the model learn WHERE TO LOOK?**
- primary: bufhit@question — at question time, were the supporting
  fact's tokens in the working set (vs random-selector baseline)
- secondary: recall (memory available downstream of solved gaze),
  ppl vs the dense twin (which is the K->all limit of this model,
  context not competitor)
- instruments: admission maps, tau/temperature dynamics, gaze-token
  attention patterns

## 7. Empirical context this design answers (day-1 findings)

- Frozen-archive rung 1: killed by its own criterion; randgaze
  invariance showed the reader never used the buffer.
- Drift law (bisect-proven): a learned reader cannot form against a
  moving substrate; formed circuits survive later movement. The
  base's easy-gradient regime (dense-identical start) plus
  restrictions-as-curriculum is the architectural answer — ignition
  happens while the model is still nearly dense.
- Buffer churn: stochastic reselection destabilizes readers;
  selection noise budget belongs to the consuming organ. Top-k+eps
  during ignition, sampling later, is the default policy schedule.

## 8. Build ladder

1. Base implementation + identity test (core=identity == dense,
   bit-exact) — the new leak-test analog.
2. Dense-identical training sanity (core live, no restrictions):
   must match dense ppl; core outputs may be ignored — fine.
3. Restriction annealing experiments (toy first, $0/cheap):
   window G, then F, watch where the core becomes load-bearing.
4. Full arms at real scale: learned / random-gaze / dense-twin /
   accumulate-vs-replace / cadence ablation.
