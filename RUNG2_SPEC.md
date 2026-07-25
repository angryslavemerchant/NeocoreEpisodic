# RUNG 2 — THE FALLING LINE (spec v1, drafted 2026-07-25 night)

The north-star experiment: a frozen-weight LM whose loss on a stream
keeps falling because it is REMEMBERING. No questions, no fact
flags, no aux — next-token loss is the only teacher and the only
grade. This is a DATASET SPEC plus a measurement contract; the
architecture (graft + continuous select-writer) is already decided.

## 1. Claim under test

Deployment-time memory pays rent in perplexity: a model with a
rule-written book gets measurably less surprised by a stream as the
stream goes on, beyond anything attention can explain, with weights
frozen. Kill condition included (section 5) — if memory cannot earn
perplexity on prose, the premise fails at this register and we want
that verdict now.

## 2. World (the bulk of the spec)

Generated narrative prose (prose_bank register), recurring renamed
entities, document-local attention as always. Per lifetime:

- **Entities**: ~10 persons + 5 companies (nonce, multi-token),
  attributes drawn per lifetime (job, city, employer, partner,
  industry, product — the existing fact graph).
- **Lifetime length**: 400-800 docs x ~40-56 tokens = 20-40k tokens.
  Hundreds of documents past any attention span.
- **Doc types**:
  1. INTRODUCTION docs (~15%): a fact stated in narrative prose
     (prose_bank statement narratives, extended bank).
  2. RECURRENCE docs (~25%): the load-bearing invention. Prose whose
     continuation is UNPREDICTABLE from local context but PREDICTABLE
     given the earlier fact. Template shape: neutral setup mentioning
     the entity, then an attribute-REALIZING span:
       "By six {P} was already gone. The hospital pages a surgeon
        early." (predictable iff works_as={surgeon} was filed)
       "The flight home felt endless. {P} never sleeps well until
        the plane touches down in {home_city}."
     Rules: (a) the realizing span's tokens are the GRADE TOKENS;
     (b) >=4 attribute values must be locally plausible (the local
     context must not leak the answer); (c) realizing templates are
     DISJOINT from introduction templates (no phrasing echo);
     (d) held-out realizing templates for the honesty split.
  3. DISTRACTOR docs (~25%): entity-mentioning prose, no facts
     (prose_bank distractors, extended).
  4. GENERIC FILLERS (~35%): existing filler bank.
- **Recurrence spacing**: gaps from introduction to each recurrence
  sampled log-uniform from 5 to full-lifetime distance. Gap-bucketed
  loss = the RETENTION CURVE, free instrument.
- **Each fact recurs 2-5 times** across the lifetime (recurrences of
  the same fact use different realizing templates).
- Bank build: extend prose_bank.json with a `realize_*` category per
  relation (~10 train + 3 hold each) + more distractors. Same
  agent-generate + validate pipeline as templates_bank.

## 3. Model and twins

- **live**: graft + CONTINUOUS SELECT-WRITER (per CLAUDE.md NEXT
  BUILD: single-selected-token percepts via learned scorer,
  sliding-window candidacy, top-M=6 write attempts per chunk,
  novelty-gated, no flags at any stage).
- **frozen**: same trained model, junk book at eval (channel
  load-bearing control).
- **dense**: unmodified GPT-2 fine-tuned on the same streams
  (attention-only floor; document-local so it CANNOT remember).
- later/optional: long-context twin (stream in context — the
  competitor arm; separate build, full-attention variant).

## 4. Training vs measurement (the frozen-weights discipline)

- Meta-train on generated lifetimes (entities re-randomized per
  lifetime, unsmearable) with plain LM loss. bf16 + compile.
- THE FALLING LINE IS MEASURED AT EVAL ONLY: frozen weights,
  held-out entity streams, fresh book per lifetime. Any within-
  lifetime improvement is attributable to the book alone.

## 5. Measurement contract (pre-registered)

- **Headline plot**: mean NLL on GRADE TOKENS (attribute-realizing
  spans) bucketed by doc position in lifetime (8 buckets), live vs
  dense vs frozen. SUCCESS = live falls monotonically-ish and ends
  >=0.5 nats below dense on late buckets while dense stays flat.
  KILL = live's curve statistically flat / ends within noise of
  dense: memory does not pay perplexity rent on prose.
- **Secondary**: NLL vs recurrence gap (retention curve — does the
  book hold value at full-lifetime range where context cannot);
  held-out realizing templates (paraphrase honesty); book occupancy
  + which tokens the selector writes (the instrument: what does a
  model choose to remember when nobody tells it).
- **Controls**: frozen must crater toward dense; grade-token NLL on
  FIRST occurrences (nothing to remember) must be equal across arms
  (leak check).
- Diagnostic-only QA quiz (never trained, never headline): optional
  small probe set for interpretability.

## 6. Cost and order

1. Bank extension + world builder + leak checks (grade tokens
   locally unpredictable: verify dense-only NLL high on grade spans
   at first AND later occurrences) — local, $0.
2. Continuous select-writer build (graft_cwriter.py) + CPU smoke.
3. One PRO 6000 run per arm, ~4k steps, bf16+compile: ~$1.5/arm.
   Consumer-CPU boxes per OFFER_JUDGEMENT.
4. Read the line. If it falls: scale lifetime length until it
   stops falling (the capacity/rent frontier becomes the next rung).
