"""CONTINUOUS SELECT-WRITER + the FALLING LINE (RUNG2_SPEC.md).

Writer (the last wheel off — no flags at ANY stage):
- every doc is a write candidate; the v6b novelty rule decides
- book slots store the doc's LAST-8 token states (a window BLOCK,
  no averaging); a learned SELECTOR + key/pay heads render blocks
  into keys/payloads, recomputed per chunk => exact chunk-local
  gradients for selection AND rendering (select-not-pool with a
  differentiable picker)
Training: plain next-token CE. No aux, no answer boost, no
abstention, no questions. The only teacher is perplexity.
Gap curriculum (ignition insurance, bisect law): recurrence gaps
capped small early, annealed to full range by half of training.

Measurement (pre-registered in RUNG2_SPEC):
- grade-token NLL on recurrence docs, bucketed by lifetime position
  (8) and by gap (4) — the falling line and the retention curve
- recur_ctrl docs (no intro exists) = leak control, must match
  across arms; write-kind mix = what the model chooses to remember
Arms: --arm live | dense. frozen = eval of live with junk book.
"""

import argparse
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import stream_text_v2 as W
import stream_fall
from stream_text_v2 import build_batch, CHUNK
from toy_stream_icl import Book
from graft_writer import GraftWriterLM, _bf16_ok, _IdMap
from graft_gpt2 import GraftLM

W_WIN = 8
GAP_BUCKETS = (16, 48, 128)  # edges; 4 buckets
POS_BUCKETS = 8


class CWView:
    """Per-chunk differentiable view: window blocks -> selector
    softmax -> percept -> key/pay heads. Grad to selector + heads."""

    def __init__(self, book, selector, key_head, pay_head, d):
        self.code_temp = book.code_temp
        self.used = book.counts > 0
        blocks = book.pays.detach().clone() \
            .reshape(*book.pays.shape[:2], W_WIN, d)
        live = blocks.abs().sum(-1) > 1e-6            # (B,K,W)
        sc = selector(blocks).squeeze(-1)             # (B,K,W)
        sc = sc.masked_fill(~live, float("-inf"))
        sc = torch.where(live.any(-1, keepdim=True), sc,
                         torch.zeros_like(sc))
        sw = torch.softmax(sc, dim=-1)
        percept = torch.einsum("bkw,bkwd->bkd", sw, blocks)
        self.keys = F.normalize(key_head(percept), dim=-1)
        self.pays = pay_head(percept)

    def read(self, q):
        sim = torch.einsum("btd,bkd->btk", F.normalize(q, dim=-1),
                           self.keys)
        used = self.used.unsqueeze(1)
        neg = torch.finfo(sim.dtype).min
        sim_u = sim.masked_fill(~used, neg)
        sim_eff = torch.where(used.any(-1, keepdim=True), sim_u, sim)
        w = torch.softmax(sim_eff * self.code_temp, dim=-1)
        return torch.einsum("btk,bkd->btd", w, self.pays), w


class GraftCWriterLM(GraftWriterLM):
    def __init__(self, K=96, name="gpt2"):
        super().__init__(K=K, name=name, read_depths=(5, 11))
        self.selector = nn.Linear(self.d, 1)

    def new_params(self):
        return super().new_params() \
            + list(self.selector.parameters())

    def forward_docs_writer(self, toks, book):
        B, N, L = toks.shape
        pos = torch.arange(L, device=toks.device)
        x = self.tr.wte(toks.reshape(B * N, L)) + self.tr.wpe(pos)
        view = (CWView(book, self.selector, self.key_head,
                       self.pay_head, self.d)
                if book is not None else None)
        w1 = w2 = None
        windows = None
        for li, blk in enumerate(self.tr.h):
            out = blk(x)
            x = out[0] if isinstance(out, tuple) else out
            if li == self.ri[0]:
                with torch.no_grad():
                    h = x.reshape(B, N, L, self.d)
                    npad = toks != W.PAD                  # (B,N,L)
                    # last-W non-pad positions per doc
                    idx = torch.cumsum(npad.long(), dim=-1)
                    n_tok = idx[..., -1:]                  # (B,N,1)
                    want = n_tok - torch.arange(
                        W_WIN, 0, -1, device=toks.device).view(1, 1, -1)
                    ok = want >= 0
                    pos_of = torch.zeros(B, N, L + 1,
                                         dtype=torch.long,
                                         device=toks.device)
                    ar = torch.arange(L, device=toks.device)
                    pos_of.scatter_(2, idx * npad.long(),
                                    ar.view(1, 1, -1)
                                    .expand(B, N, -1) * npad.long())
                    gi = pos_of.gather(2, (want + 1).clamp(min=0))
                    windows = h.gather(
                        2, gi.unsqueeze(-1).expand(-1, -1, -1,
                                                   self.d))
                    windows = windows * ok.unsqueeze(-1)
            if book is not None and li in self.ri:
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
            windows


def make_book(model, B, K, device, junk=False):
    book = Book(B, K, model.d, device)
    book.pays = torch.zeros(B, K, W_WIN * model.d, device=device)
    if junk:
        book.counts[:] = 1.0
        book.pays = torch.randn_like(book.pays)
    return book


def run_batch_fall(model, batch, K, device, arm,
                   do_backward=False, loss_norm=40.0, collect=False):
    (toks, fids, aux1, aux2, ans_mask, q_pos, s_pos, kinds,
     n_facts) = batch
    B, D, L = toks.shape
    use_book = arm in ("live", "frozen")
    book = make_book(model, B, K, device, junk=(arm == "frozen")) \
        if use_book else None
    loss = torch.zeros((), device=device)
    lm_sum, lm_n = 0.0, 0
    grade = []          # (pos_bucket, gap, nll, is_ctrl)
    wkinds = {}
    nch = 0
    for c0 in range(0, D, CHUNK):
        c1 = min(c0 + CHUNK, D)
        sub = toks[:, c0:c1]
        if use_book:
            logits, _, _, windows = model.forward_docs_writer(sub,
                                                              book)
        else:
            logits, _, _ = model.forward_docs(sub, book=None)
            windows = None
        tgt = sub[:, :, 1:]
        keep = tgt != W.PAD
        tok_nll = F.cross_entropy(
            logits[:, :, :-1].reshape(-1, logits.shape[-1]),
            tgt.reshape(-1), reduction="none"
        ).reshape(B, c1 - c0, L - 1)
        if bool(keep.any()):
            lm = tok_nll[keep].mean()
            loss_c = lm
            lm_sum += float(lm.detach())
            lm_n += 1
        else:
            loss_c = torch.zeros((), device=device)
        if do_backward and loss_c.requires_grad:
            (loss_c / loss_norm).backward()
        loss = loss + loss_c.detach()
        nch += 1
        # grade-token collection (no_grad)
        with torch.no_grad():
            amask = ans_mask[:, c0:c1, 1:] & keep
            for j in range(c1 - c0):
                d_i = c0 + j
                for b in range(B):
                    k = kinds[b][d_i]
                    if k not in ("recur", "recur_ctrl"):
                        continue
                    m = amask[b, j]
                    if not bool(m.any()):
                        continue
                    nll = float(tok_nll[b, j][m].mean())
                    pb = min(d_i * POS_BUCKETS // max(D, 1),
                             POS_BUCKETS - 1)
                    gap = int(aux2[b, d_i])
                    grade.append((pb, gap, nll,
                                  k == "recur_ctrl"))
            # writes: EVERY doc with content, post-forward, no flags
            if use_book and arm == "live" and windows is not None:
                with torch.autocast(device_type="cuda",
                                    enabled=False):
                    wb = windows.float()
                    live_m = wb.abs().sum(-1) > 1e-6      # (B,N,W)
                    any_tok = live_m.any(-1)              # (B,N)
                    sc = model.selector(wb).squeeze(-1) \
                        .masked_fill(~live_m, float("-inf"))
                    sc = torch.where(any_tok.unsqueeze(-1), sc,
                                     torch.zeros_like(sc))
                    sw = torch.softmax(sc, dim=-1)
                    perc = torch.einsum("bnw,bnwd->bnd", sw, wb)
                    kvs = F.normalize(model.key_head(perc), dim=-1)
                    for j in range(c1 - c0):
                        if not bool(any_tok[:, j].any()):
                            continue
                        book.write(kvs[:, j],
                                   wb[:, j].reshape(B, -1))
                        if collect:
                            kk = kinds[0][c0 + j] or "pad"
                            wkinds[kk] = wkinds.get(kk, 0) + 1
    out = {"lm_loss": lm_sum / max(lm_n, 1), "grade": grade,
           "wkinds": wkinds}
    if book is not None:
        out["used"] = float((book.counts > 0).float().sum(1).mean())
    return loss / max(nch, 1), out


def summarize(grades):
    pos = [[] for _ in range(POS_BUCKETS)]
    gapb = [[] for _ in range(len(GAP_BUCKETS) + 1)]
    ctrl = []
    for pb, gap, nll, is_ctrl in grades:
        if is_ctrl:
            ctrl.append(nll)
            continue
        pos[pb].append(nll)
        gi = sum(gap > e for e in GAP_BUCKETS)
        gapb[gi].append(nll)
    f = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return ([f(x) for x in pos], [f(x) for x in gapb], f(ctrl))


def train(model, args, device, arm, log_fn=None):
    lora_ps = getattr(model, "lora_ps", [])
    skip = {id(p) for p in model.new_params()} if arm == "live" \
        else set()
    base = [p for p in model.parameters()
            if id(p) not in skip and p.requires_grad]
    groups = [{"params": base, "lr": args.lr_base}]
    if arm == "live":
        groups.append({"params": model.new_params(),
                       "lr": args.lr_new})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    model.train()
    rng = random.Random(1234 + args.seed)
    t0 = time.time()
    full_cap = 400
    for step in range(1, args.steps + 1):
        # gap curriculum: cap 24 -> uncapped by half of training
        frac = min(1.0, step / max(args.steps * args.gap_anneal, 1))
        stream_fall.GAP_CAP = None if frac >= 1.0 else \
            int(24 * (full_cap / 24) ** frac)
        batch = build_batch(args.batch, device, rng,
                            stmts=args.stmts,
                            filler_frac=args.filler_frac)
        opt.zero_grad()
        with torch.autocast(device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=_bf16_ok(device)):
            loss, st = run_batch_fall(model, batch, args.k, device,
                                      arm, do_backward=True)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 50 == 0:
            pos_c, _, ctrl = summarize(st["grade"])
            early = pos_c[0]
            late = pos_c[-1]
            print(f"[{arm}] step {step:5d}  lm {st['lm_loss']:.3f}  "
                  f"grade e {early:.2f} l {late:.2f} "
                  f"ctrl {ctrl:.2f}  used {st.get('used', 0):.0f}  "
                  f"cap {stream_fall.GAP_CAP}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if log_fn:
                log_fn({f"{arm}/lm": st["lm_loss"],
                        f"{arm}/grade_late": late,
                        f"{arm}/grade_early": early,
                        f"{arm}/step": step})


@torch.no_grad()
def evaluate(model, args, device, arm, bank_part="train", seed=999):
    model.eval()
    stream_fall.GAP_CAP = None
    rng = random.Random(seed)
    grades, wk, used = [], {}, []
    lm = []
    for _ in range(args.eval_batches):
        batch = build_batch(args.eval_batch, device, rng,
                            bank_part=bank_part, stmts=args.stmts,
                            filler_frac=args.filler_frac)
        with torch.autocast(device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=_bf16_ok(device)):
            _, st = run_batch_fall(model, batch, args.k, device,
                                   arm, collect=True)
        grades += st["grade"]
        lm.append(st["lm_loss"])
        for k, v in st.get("wkinds", {}).items():
            wk[k] = wk.get(k, 0) + v
        if "used" in st:
            used.append(st["used"])
    pos_c, gap_c, ctrl = summarize(grades)
    return {"pos_curve": pos_c, "gap_curve": gap_c, "ctrl": ctrl,
            "lm_loss": sum(lm) / len(lm),
            "used": (sum(used) / len(used)) if used else 0,
            "wkinds": wk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=str, default="live",
                    choices=("live", "dense"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--lr-base", type=float, default=3e-5)
    ap.add_argument("--lr-new", type=float, default=1e-3)
    ap.add_argument("--gap-anneal", type=float, default=0.5)
    ap.add_argument("--stmts", type=int, default=2)
    ap.add_argument("--filler-frac", type=float, default=0.35)
    ap.add_argument("--eval-batch", type=int, default=16)
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
    stream_fall.setup()
    W.build_idf()
    print(f"device={device} FALLING LINE arm={args.arm}", flush=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"fall-{args.arm}-s{args.steps}"
                              f"-{int(time.time()) % 100000}",
                         config=vars(args))
    log_fn = (lambda x: run.log(x)) if run else None

    if args.arm == "live":
        model = GraftCWriterLM(K=args.k).to(device)
    else:
        model = GraftLM(use_book=False).to(device)
    if args.compile:
        for bi in range(len(model.tr.h)):
            model.tr.h[bi] = torch.compile(model.tr.h[bi])
        print("torch.compile on", flush=True)
    model._idf = W._IDF.to(device) if W._IDF is not None else None

    train(model, args, device, args.arm, log_fn=log_fn)
    if args.save_prefix:
        torch.save(model.state_dict(),
                   f"{args.save_prefix}_{args.arm}.pt")

    results = {args.arm: evaluate(model, args, device, args.arm)}
    results[f"{args.arm}-hold"] = evaluate(model, args, device,
                                           args.arm,
                                           bank_part="hold")
    if args.arm == "live":
        results["frozen"] = evaluate(model, args, device, "frozen")

    print("\n=== FALLING LINE RESULTS (grade-token NLL):")
    for name, r in results.items():
        pc = " ".join(f"{v:5.2f}" for v in r["pos_curve"])
        gc = " ".join(f"{v:5.2f}" for v in r["gap_curve"])
        print(f"  {name:>12s} pos[{pc}]  gap[{gc}]  "
              f"ctrl {r['ctrl']:5.2f}  lm {r['lm_loss']:.3f}  "
              f"used {r['used']:.0f}")
        if r["wkinds"]:
            print(f"  {'':>12s} writes: {r['wkinds']}")

    if run:
        for a, st in results.items():
            for k, v in st.items():
                if k != "wkinds":
                    run.summary[f"{a}_{k}"] = v
            run.summary[f"{a}_wkinds"] = json.dumps(st["wkinds"])
        with open("fall_summary.json", "w") as f:
            json.dump(results, f, indent=1)
        import wandb as wb
        art = wb.Artifact(f"fall-{run.id}", type="results")
        art.add_file("fall_summary.json")
        if args.save_prefix:
            art.add_file(f"{args.save_prefix}_{args.arm}.pt")
        run.log_artifact(art).wait()
        run.finish()
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
