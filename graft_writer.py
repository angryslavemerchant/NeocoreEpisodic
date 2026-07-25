"""Learned WRITER for the 124M graft (2026-07-25, authorized by
Ibanis after the graft table landed).

The graft table's two warts indict the writer: holdout paraphrases
63->24 (harness-slot keys don't transfer phrasing-robustly) and
live-theta 0.0 (raw GPT-2 doc-mean geometry doesn't support naive
novelty filing). This script replaces the recipe writer with two
small learned heads while keeping the v6b economy and the zero-
gradient-at-deployment story intact.

DESIGN (the per-chunk-backward-compatible gradient path):
- The Book's slots store running-mean PERCEPTS (IDF-weighted mean of
  block-5 hidden states, detached) in .pays; .keys holds an EMA of
  normalize(key_head(percept)) used ONLY by the economy's decisions
  (novelty/nudge/merge).
- Every chunk forward builds a fresh WriterView: keys/pays are
  RECOMPUTED through key_head/pay_head from the stored (detached)
  percepts, WITH graph. The heads therefore sit inside every chunk's
  loss graph — gradients reach them chunk-locally, so the per-chunk
  backward (the GPT-2-scale OOM fix) stays exact. The store itself
  remains non-differentiable.
- Writes happen AFTER the chunk's forward (harvesting its hidden
  states); safe because CHUNK=16 < MIN_GAP=25 — no question ever
  needs a same-chunk fact.
- Meta-training files by harness flags (slot = fact id; the aux's
  slot map comes free). The autonomy grade is the livew-theta EVAL:
  filing by Book.write's novelty rule in the LEARNED key space.
  Yesterday's number to beat: 0.0. Holdout number to beat: 23.8.
  Ceiling (flag-filed recipe writer): 63.0/33.0.

Deployment story unchanged: heads are frozen weights at eval; writes
remain rule-governed; no gradients at deployment.
"""

import argparse
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import stream_text_v2 as W
from stream_text_v2 import build_batch, CHUNK
from toy_stream_icl import Book
from graft_gpt2 import GraftLM, _IdMap


def _bf16_ok(device):
    return device == "cuda" and torch.cuda.is_bf16_supported()


class LoRAWrap(nn.Module):
    """Minimal LoRA around a transformers Conv1D (weight (nin, nout),
    forward = x @ W + b). Base frozen by the caller."""

    def __init__(self, base, r, alpha=32):
        super().__init__()
        self.base = base
        nin, nout = base.weight.shape
        self.A = nn.Parameter(torch.randn(nin, r) * 0.01)
        self.Bm = nn.Parameter(torch.zeros(r, nout))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.Bm) * self.scale


def add_lora(model, r):
    """Freeze ALL GPT-2 params; wrap attn+mlp projections with LoRA.
    Read/write heads stay trainable (they are not part of tr)."""
    for p in model.tr.parameters():
        p.requires_grad_(False)
    for p in model.lm_head.parameters():
        p.requires_grad_(False)
    lora_ps = []
    for blk in model.tr.h:
        for parent, name in ((blk.attn, "c_attn"), (blk.attn, "c_proj"),
                             (blk.mlp, "c_fc"), (blk.mlp, "c_proj")):
            wrap = LoRAWrap(getattr(parent, name), r)
            setattr(parent, name, wrap)
            lora_ps += [wrap.A, wrap.Bm]
    return lora_ps


class WriterView:
    """Per-chunk differentiable view of the book: transforms stored
    percepts through the writer heads WITH graph. Interface matches
    Book.read."""

    def __init__(self, book, key_head, pay_head):
        self.code_temp = book.code_temp
        self.used = book.counts > 0
        # clone: the post-forward in-place writes must not touch
        # autograd's saved copy (same rule as Book.read's clone)
        p = book.pays.detach().clone()
        self.keys = F.normalize(key_head(p), dim=-1)
        self.pays = pay_head(p)

    def read(self, q):
        sim = torch.einsum("btd,bkd->btk", F.normalize(q, dim=-1),
                           self.keys)
        used = self.used.unsqueeze(1)
        neg = torch.finfo(sim.dtype).min
        sim_u = sim.masked_fill(~used, neg)
        sim_eff = torch.where(used.any(-1, keepdim=True), sim_u, sim)
        w = torch.softmax(sim_eff * self.code_temp, dim=-1)
        return torch.einsum("btk,bkd->btd", w, self.pays), w


class GraftWriterLM(GraftLM):
    def __init__(self, K=96, name="gpt2", read_depths=(5, 11)):
        super().__init__(use_book=True, metabook=False, K=K,
                         name=name, read_depths=read_depths)
        d = self.d

        def mlp():
            return nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                 nn.Linear(d, d))
        self.key_head = mlp()
        self.pay_head = mlp()
        self._idf = None  # set in main after build_idf

    def new_params(self):
        ps = super().new_params()
        ps += list(self.key_head.parameters())
        ps += list(self.pay_head.parameters())
        return ps

    def forward_docs_writer(self, toks, book):
        """Returns logits, w1, w2, percepts (B,N,d — detached,
        IDF-pooled block-ri[0] states, computed PRE-read1 so the
        percept is book-independent)."""
        B, N, L = toks.shape
        pos = torch.arange(L, device=toks.device)
        x = self.tr.wte(toks.reshape(B * N, L)) + self.tr.wpe(pos)
        view = WriterView(book, self.key_head, self.pay_head)
        w1 = w2 = None
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
            if li in self.ri:
                xb = x.reshape(B, N * L, self.d)
                if li == self.ri[0]:
                    xb, w1 = self.read1(xb, view)
                else:
                    xb, w2 = self.read2(xb, view)
                x = xb.reshape(B * N, L, self.d)
        x = self.tr.ln_f(x)
        logits = self.lm_head(x).reshape(B, N, L, -1)
        shp = (B, N, L, -1)
        return logits, \
            (w1.reshape(shp) if w1 is not None else None), \
            (w2.reshape(shp) if w2 is not None else None), \
            percepts


def run_batch_writer(model, batch, K, device, arm, aux_w,
                     do_backward=False, loss_norm=40.0, theta=0.75):
    """Mirror of stream_text_v2.run_batch with the writer path:
    post-forward writes from hidden-state percepts; reads through the
    per-chunk WriterView. arm in (livew, livew-theta, frozen)."""
    (toks, fids, aux1, aux2, ans_mask, q_pos, s_pos, kinds,
     n_facts) = batch
    B, D, L = toks.shape
    book = Book(B, K, model.d, device, theta=theta)
    if arm == "frozen":
        book.counts[:] = 1.0
        book.pays = torch.randn_like(book.pays)
    slots = torch.zeros(B, n_facts, dtype=torch.long, device=device)
    loss = torch.zeros((), device=device)
    nq = 0
    stats = {k: [0.0, 0] for k in ("h1", "h2", "h3", "abstain")}
    lm_sum, lm_n = 0.0, 0
    for c0 in range(0, D, CHUNK):
        chunk_loss = torch.zeros((), device=device)
        c1 = min(c0 + CHUNK, D)
        sub = toks[:, c0:c1]
        logits, w1, w2, percepts = model.forward_docs_writer(sub, book)
        tgt = sub[:, :, 1:]
        keep = tgt != W.PAD
        if bool(keep.any()):
            lm = F.cross_entropy(
                logits[:, :, :-1].reshape(-1, logits.shape[-1])
                [keep.reshape(-1)],
                tgt.reshape(-1)[keep.reshape(-1)])
            chunk_loss = chunk_loss + lm
            lm_sum += float(lm.detach())
            lm_n += 1
        amask = ans_mask[:, c0:c1, 1:] & keep
        if bool(amask.any()):
            ans_ce = F.cross_entropy(
                logits[:, :, :-1].reshape(-1, logits.shape[-1])
                [amask.reshape(-1)],
                tgt.reshape(-1)[amask.reshape(-1)])
            chunk_loss = chunk_loss + 2.0 * ans_ce
        with torch.no_grad():
            pred = logits[:, :, :-1].argmax(-1)
            okt = (pred == tgt) | ~amask
            doc_ok = okt.all(-1) & amask.any(-1)
        for j in range(c1 - c0):
            d_i = c0 + j
            any_q = False
            for b in range(B):
                k = kinds[b][d_i]
                if k is None or not k.startswith("q_"):
                    continue
                any_q = True
                hit = float(doc_ok[b, j])
                key = ("abstain" if k == "q_abstain"
                       else ("h1" if k == "q_h1"
                             else ("h3" if k == "q_h3" else "h2")))
                stats[key][0] += hit
                stats[key][1] += 1
            if any_q:
                nq += 1
        if arm != "frozen" and aux_w > 0 and w1 is not None:
            acol1 = aux1[:, c0:c1]
            is_q = acol1 >= 0
            if bool(is_q.any()):
                bb, dd = torch.nonzero(is_q, as_tuple=True)
                pos = q_pos[:, c0:c1][bb, dd]
                spos = s_pos[:, c0:c1][bb, dd]
                s1 = slots[bb, acol1[bb, dd]]
                s2 = slots[bb, aux2[:, c0:c1][bb, dd]]
                p1 = w1[bb, dd, spos, :].gather(
                    1, s1.unsqueeze(1)).squeeze(1)
                p2 = w2[bb, dd, pos, :].gather(
                    1, s2.unsqueeze(1)).squeeze(1)
                chunk_loss = chunk_loss - aux_w * (
                    torch.log(p1 + 1e-9)
                    + torch.log(p2 + 1e-9)).mean()
        if do_backward:
            if chunk_loss.requires_grad:
                (chunk_loss / loss_norm).backward()
            loss = loss + chunk_loss.detach()
        else:
            loss = loss + chunk_loss
        # writes AFTER forward+loss: harvest this chunk's percepts.
        # autocast OFF: the book's economy runs fp32 (finfo(dtype).min
        # sentinels in Book.write overflow bf16), and percepts are
        # cast up so stored means never mix precisions.
        if arm in ("livew", "livew-theta", "livew-all"):
            with torch.no_grad(), \
                    torch.autocast(device_type="cuda", enabled=False):
                percepts = percepts.float()
                for j in range(c1 - c0):
                    fcol = fids[:, c0 + j]
                    ok = fcol >= 0
                    # livew-all: the gate sees EVERY doc (incl.
                    # distractor prose) and decides alone — the
                    # fully-autonomous grade. Others: fact docs only.
                    if arm != "livew-all" and not bool(ok.any()):
                        continue
                    u = percepts[:, j]
                    kv = F.normalize(model.key_head(u), dim=-1)
                    if arm in ("livew-theta", "livew-all"):
                        i = book.write(kv, u)
                        slots[ok, fcol[ok]] = i[ok]
                    else:
                        # flag filing, masked to fact rows only
                        arr = torch.arange(B, device=device)[ok]
                        ii = fcol[ok]
                        n = book.counts[arr, ii] + 1
                        lr = (1.0 / n.clamp(max=book.cap)) \
                            .unsqueeze(-1)
                        book.keys[arr, ii] += lr * (kv[ok]
                                                    - book.keys[arr, ii])
                        book.pays[arr, ii] += lr * (u[ok]
                                                    - book.pays[arr, ii])
                        book.counts[arr, ii] = n
                        slots[ok, fcol[ok]] = ii
    out = {k: v[0] / max(v[1], 1) for k, v in stats.items()}
    out["lm_loss"] = lm_sum / max(lm_n, 1)
    out["used"] = float((book.counts > 0).float().sum(1).mean())
    return loss / max(nq, 1), out


def train_writer(model, steps, B, K, lr_base, lr_new, device,
                 aux_anneal, stmts, filler_frac, log_every=50,
                 log_fn=None, abstain_frac=0.12, abstain_warmup=0.6,
                 stream_q_warmup=0.35):
    lora_ps = getattr(model, "lora_ps", [])
    skip = {id(p) for p in model.new_params()} \
        | {id(p) for p in lora_ps}
    base = [p for p in model.parameters()
            if id(p) not in skip and p.requires_grad]
    groups = [{"params": base, "lr": lr_base},
              {"params": model.new_params(), "lr": lr_new}]
    if lora_ps:
        groups.append({"params": lora_ps,
                       "lr": getattr(model, "lr_lora", 3e-4)})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    model.train()
    rng = random.Random(1234)
    t0 = time.time()
    for step in range(1, steps + 1):
        aux_w = max(0.0, 1.0 - step / max(steps * aux_anneal, 1)) \
            if aux_anneal > 0 else 0.0
        af = abstain_frac if step > steps * abstain_warmup else 0.0
        sq = None if step > steps * stream_q_warmup else 0
        batch = build_batch(B, device, rng, stmts=stmts,
                            filler_frac=filler_frac, abstain_frac=af,
                            n_stream_q=sq)
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=_bf16_ok(device)):
            loss, st = run_batch_writer(model, batch, K, device,
                                        "livew", aux_w,
                                        do_backward=True)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % log_every == 0:
            print(f"[livew] step {step:5d}  loss {loss.item():.3f}  "
                  f"lm {st['lm_loss']:.3f}  h1 {st['h1']:.3f}  "
                  f"h2 {st['h2']:.3f}  abst {st['abstain']:.3f}  "
                  f"used {st['used']:.1f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if log_fn:
                log_fn({"livew/loss": loss.item(),
                        "livew/h1": st["h1"], "livew/h2": st["h2"],
                        "livew/step": step})


@torch.no_grad()
def eval_writer(model, B, K, device, arm, batches, stmts, filler_frac,
                bank_part="train", seed=999, theta=0.75):
    model.eval()
    rng = random.Random(seed)
    agg = {}
    for _ in range(batches):
        batch = build_batch(B, device, rng, bank_part=bank_part,
                            stmts=stmts, filler_frac=filler_frac)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=_bf16_ok(device)):
            _, st = run_batch_writer(model, batch, K, device, arm,
                                     0.0, theta=theta)
        for k, v in st.items():
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
    ap.add_argument("--world", type=str, default="bank",
                    choices=("bank", "prose"))
    ap.add_argument("--lora", type=int, default=0,
                    help="LoRA rank; 0 = full fine-tune")
    ap.add_argument("--lr-lora", type=float, default=3e-4)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the GPT-2 blocks (cloud only;"
                         " needs triton)")
    ap.add_argument("--stmts", type=int, default=2)
    ap.add_argument("--filler-frac", type=float, default=0.3)
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

    W.load_bank()
    W._REMAP = _IdMap()
    W._NVOCAB = 50257
    W.PAD = 50256
    W.UNKNOWN_IDS = W.enc_c(" unknown")
    if args.world == "prose":
        import stream_prose
        stream_prose.setup()
    W.build_idf()
    print(f"device={device} learned-writer graft "
          f"world={args.world} lora={args.lora}", flush=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"graftw-s{args.steps}"
                              f"-{int(time.time()) % 100000}",
                         config=vars(args))
    log_fn = (lambda x: run.log(x)) if run else None

    model = GraftWriterLM(K=args.k)
    if args.lora > 0:
        model.lora_ps = add_lora(model, args.lora)
        model.lr_lora = args.lr_lora
        n_lora = sum(p.numel() for p in model.lora_ps)
        print(f"LoRA r{args.lora}: {n_lora/1e6:.2f}M adapter params, "
              f"base FROZEN", flush=True)
    model = model.to(device)
    if args.compile:
        # compile per-block: chunk shapes vary only in N (docs per
        # chunk), so variants stay few; heads/reads are tiny, skip
        for bi in range(len(model.tr.h)):
            model.tr.h[bi] = torch.compile(model.tr.h[bi])
        print("torch.compile: 12 GPT-2 blocks compiled", flush=True)
    model._idf = W._IDF.to(device) if W._IDF is not None else None
    n_new = sum(p.numel() for p in model.new_params())
    print(f"writer graft: {n_new/1e6:.1f}M new params", flush=True)
    train_writer(model, args.steps, args.batch, args.k, args.lr_base,
                 args.lr_new, device, args.aux_anneal, args.stmts,
                 args.filler_frac, log_fn=log_fn)
    if args.save_prefix:
        torch.save(model.state_dict(), f"{args.save_prefix}_livew.pt")

    results = {}
    results["livew"] = eval_writer(model, args.eval_batch, args.k,
                                   device, "livew", args.eval_batches,
                                   args.stmts, args.filler_frac)
    for th in (0.6, 0.75, 0.9):
        results[f"theta{th}"] = eval_writer(
            model, args.eval_batch, args.k, device, "livew-theta",
            args.eval_batches, args.stmts, args.filler_frac, theta=th)
        results[f"all{th}"] = eval_writer(
            model, args.eval_batch, args.k, device, "livew-all",
            args.eval_batches, args.stmts, args.filler_frac, theta=th)
    results["holdout"] = eval_writer(
        model, args.eval_batch, args.k, device, "livew",
        args.eval_batches, args.stmts, args.filler_frac,
        bank_part="hold")
    results["highfill"] = eval_writer(
        model, args.eval_batch, args.k, device, "livew",
        args.eval_batches, args.stmts, 0.6)
    results["frozen"] = eval_writer(
        model, args.eval_batch, args.k, device, "frozen",
        args.eval_batches, args.stmts, args.filler_frac)

    print("\n=== LEARNED-WRITER RESULTS (exact match %):")
    names = list(results)
    print("  metric  " + "  ".join(f"{n:>11s}" for n in names))
    for m in ("h1", "h2", "abstain", "lm_loss", "used"):
        row = "  ".join(
            f"{results[n].get(m, 0) * (1 if m in ('lm_loss', 'used') else 100):11.2f}"
            for n in names)
        print(f"  {m:>7s}  {row}")

    if run:
        for a, st in results.items():
            for k, v in st.items():
                run.summary[f"{a}_{k}"] = v
        with open("graft_writer_summary.json", "w") as f:
            json.dump(results, f, indent=1)
        import wandb as wb
        art = wb.Artifact(f"graftw-{run.id}", type="results")
        art.add_file("graft_writer_summary.json")
        if args.save_prefix:
            art.add_file(f"{args.save_prefix}_livew.pt")
        run.log_artifact(art).wait()
        run.finish()
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
