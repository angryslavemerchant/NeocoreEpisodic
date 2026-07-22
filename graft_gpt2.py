"""Phase B: graft the writable book into PRETRAINED GPT-2 (124M).
The campaign's first real-model experiment (2026-07-22, authorized
end-to-end by Ibanis).

Architecture: HF GPT-2 small, custom forward over its 12 blocks with
two ReadHeads (same organ as every toy) inserted after blocks 5 and
11, reading one shared per-lifetime Book. Everything fine-tunes
(GPT-2 params at low lr, new organs at high lr); the BOOK CONTENTS
are never touched by backprop in the live arm — written by the v6b
rules from doc-mean PRETRAINED embeddings.

World: stream_text_v2's bank-driven interleaved streams, run at
GPT-2's REAL vocabulary (identity remap, PAD = eot). Multi-token
nonce names, paraphrase holdouts, fillers, abstention — unchanged.

Arms (the experiment's table):
  live       writable book, v6b-rule writes at read time   (ours)
  metabook   book slots are nn.Parameters trained by backprop and
             FROZEN at eval — the Meta-Memory-Layers regime; under
             per-lifetime randomization it cannot store the
             entities: that failure IS the claim, measured
  dense      same GPT-2 fine-tuned, no book
  frozen     live model, garbage book at eval (reads load-bearing?)
  live-theta novelty-gate filing (no harness slots)
  + holdout-template and high-filler evals of live

Trainings: --train-arms live,dense,metabook (sequential, one
process, one wandb run, verified artifact)."""

import argparse
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import stream_text_v2 as W
from stream_text_v2 import run_batch, build_batch, eval_arm
from toy_stream_icl import ReadHead


class _IdMap(dict):
    def get(self, k, default=None):
        return k


class ParamBook(nn.Module):
    """Meta-regime book: slots are trained parameters, shared across
    all lifetimes, frozen at eval. Read interface matches Book."""

    def __init__(self, K, d):
        super().__init__()
        self.keys = nn.Parameter(F.normalize(torch.randn(K, d), dim=-1))
        self.pays = nn.Parameter(torch.zeros(K, d))
        self.code_temp = 12.0
        self.K = K
        self._B = 1

    def bind(self, B):
        self._B = B
        return self

    @property
    def counts(self):
        return torch.ones(self._B, self.K,
                          device=self.keys.device)

    def read(self, q):
        sim = torch.einsum("btd,kd->btk", F.normalize(q, dim=-1),
                           F.normalize(self.keys, dim=-1))
        w = torch.softmax(sim * self.code_temp, dim=-1)
        return torch.einsum("btk,kd->btd", w, self.pays), w

    @torch.no_grad()
    def write(self, kv, pv, exemplar=False):
        # metabook never rule-writes; return dummy slots
        return torch.zeros(kv.shape[0], dtype=torch.long,
                           device=kv.device)


class GraftLM(nn.Module):
    def __init__(self, use_book=True, metabook=False, K=96,
                 name="gpt2", read_depths=(5, 11)):
        super().__init__()
        from transformers import GPT2LMHeadModel
        gpt = GPT2LMHeadModel.from_pretrained(name)
        self.tr = gpt.transformer
        self.lm_head = gpt.lm_head
        self.d = self.tr.wte.weight.shape[1]
        self.emb = self.tr.wte
        self.use_book = use_book
        self.metabook = metabook
        self.ri = read_depths
        self.read1 = ReadHead(self.d) if use_book else None
        self.read2 = ReadHead(self.d) if use_book else None
        self.param_book = (ParamBook(K, self.d) if metabook else None)

    def new_params(self):
        ps = []
        for m in (self.read1, self.read2, self.param_book):
            if m is not None:
                ps += list(m.parameters())
        return ps

    def forward_docs(self, toks, book=None):
        B, N, L = toks.shape
        pos = torch.arange(L, device=toks.device)
        x = self.tr.wte(toks.reshape(B * N, L)) + self.tr.wpe(pos)
        w1 = w2 = None
        for li, blk in enumerate(self.tr.h):
            out = blk(x)
            x = out[0] if isinstance(out, tuple) else out
            if self.use_book and book is not None and li in self.ri:
                xb = x.reshape(B, N * L, self.d)
                if li == self.ri[0]:
                    xb, w1 = self.read1(xb, book)
                else:
                    xb, w2 = self.read2(xb, book)
                x = xb.reshape(B * N, L, self.d)
        x = self.tr.ln_f(x)
        logits = self.lm_head(x).reshape(B, N, L, -1)
        shp = (B, N, L, -1)
        return logits, \
            (w1.reshape(shp) if w1 is not None else None), \
            (w2.reshape(shp) if w2 is not None else None)


def patched_run_batch(model, batch, K, device, arm, aux_w, **kw):
    """metabook: inject the trained ParamBook instead of a fresh Book
    (frozen-at-eval by construction: no rule writes exist for it)."""
    if arm == "metabook":
        book = model.param_book.bind(batch[0].shape[0])
        orig = W.Book
        try:
            W.Book = lambda *a, **k: book
            return run_batch(model, batch, K, device, "metabook-x",
                             aux_w, **kw)
        finally:
            W.Book = orig
    return run_batch(model, batch, K, device, arm, aux_w, **kw)


def train_arm(model, arm, steps, B, K, lr_base, lr_new, device, tag,
              aux_anneal, stmts, filler_frac, log_every=50,
              log_fn=None, abstain_frac=0.12, abstain_warmup=0.6,
              stream_q_warmup=0.35):
    new_ids = {id(p) for p in model.new_params()}
    base = [p for p in model.parameters() if id(p) not in new_ids]
    opt = torch.optim.AdamW(
        [{"params": base, "lr": lr_base},
         {"params": model.new_params(), "lr": lr_new}],
        weight_decay=0.01)
    model.train()
    rng = random.Random(1234)
    t0 = time.time()
    for step in range(1, steps + 1):
        aux_w = max(0.0, 1.0 - step / max(steps * aux_anneal, 1)) \
            if (model.use_book and aux_anneal > 0
                and arm == "live") else 0.0
        af = abstain_frac if step > steps * abstain_warmup else 0.0
        sq = None if step > steps * stream_q_warmup else 0
        batch = build_batch(B, device, rng, stmts=stmts,
                            filler_frac=filler_frac, abstain_frac=af,
                            n_stream_q=sq)
        opt.zero_grad()
        # per-chunk backward inside run_batch (whole-lifetime graph
        # OOMs 96 GB at GPT-2 scale; chunk-local losses make it exact)
        loss, st = patched_run_batch(model, batch, K, device, arm,
                                     aux_w, do_backward=True)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % log_every == 0:
            print(f"[{tag}] step {step:5d}  loss {loss.item():.3f}  "
                  f"lm {st['lm_loss']:.3f}  h1 {st['h1']:.3f}  "
                  f"h2 {st['h2']:.3f}  abst {st['abstain']:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if log_fn:
                log_fn({f"{tag}/loss": loss.item(),
                        f"{tag}/h1": st["h1"], f"{tag}/h2": st["h2"],
                        f"{tag}/step": step})


@torch.no_grad()
def eval_graft(model, B, K, device, arm, batches, stmts, filler_frac,
               bank_part="train", seed=999):
    model.eval()
    rng = random.Random(seed)
    agg = {}
    for _ in range(batches):
        batch = build_batch(B, device, rng, bank_part=bank_part,
                            stmts=stmts, filler_frac=filler_frac)
        _, st = patched_run_batch(model, batch, K, device, arm, 0.0)
        for k, v in st.items():
            if k != "curve":
                agg.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--lr-base", type=float, default=3e-5)
    ap.add_argument("--lr-new", type=float, default=1e-3)
    ap.add_argument("--aux-anneal", type=float, default=0.4)
    ap.add_argument("--stmts", type=int, default=2)
    ap.add_argument("--filler-frac", type=float, default=0.3)
    ap.add_argument("--train-arms", type=str,
                    default="live,dense,metabook")
    ap.add_argument("--eval-batch", type=int, default=24)
    ap.add_argument("--eval-batches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-prefix", type=str, default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str,
                    default="neocore-stream")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # world at GPT-2's real vocabulary
    W.load_bank()
    W._REMAP = _IdMap()
    W._NVOCAB = 50257
    W.PAD = 50256
    W.UNKNOWN_IDS = W.enc_c(" unknown")
    W.build_idf()
    print(f"device={device} real-vocab graft; unknown="
          f"{W.UNKNOWN_IDS} idf built", flush=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"graft124m-s{args.steps}",
                         config=vars(args))
    log_fn = (lambda x: run.log(x)) if run else None

    models = {}
    for arm in args.train_arms.split(","):
        arm = arm.strip()
        print(f"\n=== training arm: {arm}", flush=True)
        m = GraftLM(use_book=(arm != "dense"),
                    metabook=(arm == "metabook"),
                    K=args.k).to(device)
        train_arm(m, arm, args.steps, args.batch, args.k,
                  args.lr_base, args.lr_new, device, arm,
                  args.aux_anneal, args.stmts, args.filler_frac,
                  log_fn=log_fn)
        models[arm] = m
        if args.save_prefix:
            torch.save(m.state_dict(),
                       f"{args.save_prefix}_{arm}.pt")

    results = {}
    mb = models.get("live")
    if mb is not None:
        for arm in ("live", "live-theta", "frozen"):
            results[arm] = eval_graft(mb, args.eval_batch, args.k,
                                      device, arm, args.eval_batches,
                                      args.stmts, args.filler_frac)
        results["live-holdout"] = eval_graft(
            mb, args.eval_batch, args.k, device, "live",
            args.eval_batches, args.stmts, args.filler_frac,
            bank_part="hold")
        results["live-highfill"] = eval_graft(
            mb, args.eval_batch, args.k, device, "live",
            args.eval_batches, args.stmts, 0.6)
    if "dense" in models:
        results["dense"] = eval_graft(models["dense"],
                                      args.eval_batch, args.k, device,
                                      "dense", args.eval_batches,
                                      args.stmts, args.filler_frac)
    if "metabook" in models:
        results["metabook"] = eval_graft(models["metabook"],
                                         args.eval_batch, args.k,
                                         device, "metabook",
                                         args.eval_batches,
                                         args.stmts,
                                         args.filler_frac)

    print("\n=== GRAFT RESULTS (exact match %):")
    names = list(results)
    print("  metric  " + "  ".join(f"{n:>13s}" for n in names))
    for m in ("h1", "h2", "abstain", "lm_loss"):
        row = "  ".join(
            f"{results[n][m] * (1 if m == 'lm_loss' else 100):13.2f}"
            for n in names)
        print(f"  {m:>7s}  {row}")

    if run:
        for a, st in results.items():
            for k, v in st.items():
                run.summary[f"{a}_{k}"] = v
        with open("graft_summary.json", "w") as f:
            json.dump(results, f, indent=1)
        import wandb as wb
        art = wb.Artifact(f"graft124m-{run.id}", type="results")
        art.add_file("graft_summary.json")
        if args.save_prefix:
            for arm in models:
                art.add_file(f"{args.save_prefix}_{arm}.pt")
        run.log_artifact(art).wait()
        run.finish()
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
