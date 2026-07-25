"""LOOPED READER over the writable book (2026-07-25 late; Ibanis:
"lets just see if it does anything").

Replaces the two fixed-depth ReadHeads with ONE tied read + ONE
shared transformer block, looped S times after block 5:

    blocks 0-5 -> [ read(book) -> inject -> shared block ] x S
               -> blocks 6-11 -> logits

Depth of composition becomes a RUNTIME DIAL (S), not architecture.
The experiment (composer-v4 protocol, neural edition): train on
1/2-hop questions ONLY at S=3; evaluate 3-HOP questions (never seen
in training, impossible for the fixed 2-read pipeline by
construction) at S=3/4/5. Any 3-hop success is unambiguously the
loop's. Bank world (store quality proven: fixed-pipeline h2=96.7),
learned writer carried over unchanged from graft_writer.

3-hop chain: partner(P)=P2 -> works_at(P2)=C -> industry/based_in(C).
New kind "q_h3"; Q3_FRAC module knob controls its eval-time mix
(0.0 during training — the productivity split).

Loop head = read1 (tied across iterations, wo zero-init: loop starts
silent). Shared block warm-started from a copy of block 6, trained
at lr_base like any GPT-2 layer. Aux hooks unchanged: w1 = first
iteration's read map (subject slot), w2 = last iteration's (answer
slot) — the 2-hop aux wiring works without modification.
"""

import argparse
import copy
import json
import random
import time

import torch
import torch.nn as nn

import stream_text_v2 as W
from stream_text_v2 import CHUNK, Q1, Q2, N_C
from graft_writer import (GraftWriterLM, WriterView, run_batch_writer,
                          train_writer, eval_writer, _bf16_ok, _IdMap)
from graft_gpt2 import GraftLM  # noqa: F401  (import chain)

Q3_FRAC = 0.0

Q3_TEMPLATES = {
    "q3_industry": [
        "In what industry does the company employing {P}'s partner operate?",
        "What is the line of business of the company where {P}'s partner works?",
        "The company that employs {P}'s partner operates in which industry?",
    ],
    "q3_city": [
        "In which city is the company employing {P}'s partner headquartered?",
        "Where is the company that {P}'s partner works for based?",
        "The employer of {P}'s partner is based in which city?",
    ],
}


def install_q3():
    base_lt = W.Lifetime

    class Q3Lifetime(base_lt):
        def _make_q(self, B, abstain_frac):
            rng = self.rng
            if Q3_FRAC > 0 and rng.random() < Q3_FRAC:
                p = rng.choice([q for q in self.persons
                                if q in self.partner])
                sp = self.partner[p]
                c = self.employer[sp]
                qt = rng.choice(list(Q3_TEMPLATES))
                text = rng.choice(Q3_TEMPLATES[qt]).format(P=p)
                ans = (self.industry[c] if qt == "q3_industry"
                       else self.city[c])
                f1 = self.fid[("partner", frozenset((p, sp)))]
                f2 = self.fid[("works_at", sp)]
                f3 = self.fid[(("industry" if qt == "q3_industry"
                                else "based_in"), c)]
                ids = W.enc_c(text)
                a = W.enc_c(" " + ans)
                span = (len(ids), len(ids) + len(a))
                apos = self._subj_pos(ids, p)
                return (ids + a, "q_h3", -1, f1, f2, span, apos,
                        [f1, f2, f3])
            return super()._make_q(B, abstain_frac)

    W.Lifetime = Q3Lifetime


class GraftLoopLM(GraftWriterLM):
    def __init__(self, K=96, s_loops=3, name="gpt2"):
        super().__init__(K=K, name=name, read_depths=(5, 11))
        self.s_loops = s_loops
        # shared loop block: architecture clone of a GPT-2 block,
        # warm-started from block 6 (easy gradients at hookup)
        self.loop_block = copy.deepcopy(self.tr.h[6])
        self.read2 = None  # loop replaces the second fixed read

    def new_params(self):
        ps = [p for m in (self.read1, self.key_head, self.pay_head)
              for p in m.parameters()]
        return ps  # loop_block joins the BASE lr group

    def forward_docs_writer(self, toks, book):
        B, N, L = toks.shape
        pos = torch.arange(L, device=toks.device)
        x = self.tr.wte(toks.reshape(B * N, L)) + self.tr.wpe(pos)
        view = WriterView(book, self.key_head, self.pay_head)
        w_first = w_last = None
        percepts = None
        for li, blk in enumerate(self.tr.h):
            out = blk(x)
            x = out[0] if isinstance(out, tuple) else out
            if li == self.ri[0]:
                with torch.no_grad():
                    h = x.reshape(B, N, L, self.d)
                    npad = (toks != W.PAD).unsqueeze(-1).float()
                    wt = self._idf[toks].unsqueeze(-1) * npad \
                        if self._idf is not None else npad
                    percepts = (h * wt).sum(2) \
                        / wt.sum(2).clamp(min=1e-6)
                for _ in range(self.s_loops):
                    xb = x.reshape(B, N * L, self.d)
                    xb, w = self.read1(xb, view)
                    x = xb.reshape(B * N, L, self.d)
                    out = self.loop_block(x)
                    x = out[0] if isinstance(out, tuple) else out
                    if w_first is None:
                        w_first = w
                    w_last = w
        x = self.tr.ln_f(x)
        logits = self.lm_head(x).reshape(B, N, L, -1)
        shp = (B, N, L, -1)
        return logits, \
            (w_first.reshape(shp) if w_first is not None else None), \
            (w_last.reshape(shp) if w_last is not None else None), \
            percepts


def main():
    global Q3_FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--s", type=int, default=3)
    ap.add_argument("--lr-base", type=float, default=3e-5)
    ap.add_argument("--lr-new", type=float, default=1e-3)
    ap.add_argument("--aux-anneal", type=float, default=0.4)
    ap.add_argument("--stmts", type=int, default=2)
    ap.add_argument("--filler-frac", type=float, default=0.3)
    ap.add_argument("--eval-batch", type=int, default=24)
    ap.add_argument("--eval-batches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-prefix", type=str, default="")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str,
                    default="neocore-stream")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    W.load_bank()
    W._REMAP = _IdMap()
    W._NVOCAB = 50257
    W.PAD = 50256
    W.UNKNOWN_IDS = W.enc_c(" unknown")
    install_q3()
    W.build_idf()
    print(f"device={device} LOOPED reader S={args.s}", flush=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"graftloop-s{args.steps}"
                              f"-{int(time.time()) % 100000}",
                         config=vars(args))
    log_fn = (lambda x: run.log(x)) if run else None

    model = GraftLoopLM(K=args.k, s_loops=args.s)
    model = model.to(device)
    if args.compile:
        for bi in range(len(model.tr.h)):
            model.tr.h[bi] = torch.compile(model.tr.h[bi])
        model.loop_block = torch.compile(model.loop_block)
        print("torch.compile: blocks + loop block", flush=True)
    model._idf = W._IDF.to(device) if W._IDF is not None else None
    n_new = sum(p.numel() for p in model.new_params())
    n_loop = sum(p.numel() for p in model.loop_block.parameters())
    print(f"loop graft: {n_new/1e6:.1f}M head params + "
          f"{n_loop/1e6:.1f}M loop block", flush=True)

    # train: 1/2-hop ONLY (Q3_FRAC=0 — the productivity split)
    train_writer(model, args.steps, args.batch, args.k, args.lr_base,
                 args.lr_new, device, args.aux_anneal, args.stmts,
                 args.filler_frac, log_fn=log_fn)
    if args.save_prefix:
        torch.save(model.state_dict(), f"{args.save_prefix}_loop.pt")

    results = {}
    results["livew"] = eval_writer(model, args.eval_batch, args.k,
                                   device, "livew", args.eval_batches,
                                   args.stmts, args.filler_frac)
    results["theta0.9"] = eval_writer(
        model, args.eval_batch, args.k, device, "livew-theta",
        args.eval_batches, args.stmts, args.filler_frac, theta=0.9)
    Q3_FRAC = 0.5
    for s_eval in (args.s, args.s + 1, args.s + 2):
        model.s_loops = s_eval
        results[f"q3_S{s_eval}"] = eval_writer(
            model, args.eval_batch, args.k, device, "livew",
            args.eval_batches, args.stmts, args.filler_frac)
    model.s_loops = args.s
    Q3_FRAC = 0.0

    print("\n=== LOOPED-READER RESULTS (exact match %):")
    names = list(results)
    print("  metric  " + "  ".join(f"{n:>10s}" for n in names))
    for m in ("h1", "h2", "h3", "abstain", "lm_loss", "used"):
        row = "  ".join(
            f"{results[n].get(m, 0) * (1 if m in ('lm_loss', 'used') else 100):10.2f}"
            for n in names)
        print(f"  {m:>7s}  {row}")

    if run:
        for a, st in results.items():
            for k, v in st.items():
                run.summary[f"{a}_{k}"] = v
        with open("graft_loop_summary.json", "w") as f:
            json.dump(results, f, indent=1)
        import wandb as wb
        art = wb.Artifact(f"graftloop-{run.id}", type="results")
        art.add_file("graft_loop_summary.json")
        if args.save_prefix:
            art.add_file(f"{args.save_prefix}_loop.pt")
        run.log_artifact(art).wait()
        run.finish()
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
