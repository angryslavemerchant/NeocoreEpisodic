"""
From-scratch episodic foveated learner (2026-07-20): remove DINO. Learn
perception + selection + memory END TO END from raw pixels under the
episodic binding objective of train_vocab_icl.py.

Why this can work at all: the aux read loss ("two different images of
the same class must produce matching binding/summary vectors") is a
contrastive/metric objective in disguise, and episodic protocols
(ProtoNets/MatchingNets) are proven to train features from scratch at
exactly this data scale. This is the first full loop with all three
globality-law requirements met: LOCAL admission (per-patch-only stem,
no cross-patch mixing before the bottleneck), BUDGETED perception
(exact-B admission per step), cross-episode REUSE pressure (80/20
class holdout; features must transfer to unseen bindings).

Tokens: 256x256 RAM blobs (dataset_ram, unchanged pipeline) -> 16x16
grid of 16px patches = 256 tokens. Stem = linear embed + per-patch-only
LN+MLP layers. NO cross-patch mixing (locality law) — enforced by the
zero-grad locality smoke (--toy runs it). The stem is gradient-
checkpointed in training: 12 stem passes/step x saved MLP activations
at batch 512 otherwise cost ~13 GB on a 32 GB card.

Harness: unchanged from train_vocab_icl.py — 6-way episodes, labels on
everything (per-episode random label permutation), per-step probes,
80/20 class holdout, ONE fused binding token written per image,
architectural cosine read head + auxiliary retrieval supervision.

Admission arms (--arms):
  sampled  NEW DEFAULT: admit by SAMPLING from the score softmax with
           learnable temperature (init tau=1; Gumbel top-k = exact
           Plackett-Luce sampling without replacement). Quality AND
           diversity in one policy. Sampling at EVAL too — a stochastic
           policy IS the policy; we report the mean over fixed val
           episodes. tau learns through the gate (scores are divided by
           tau before both sampling and the sigmoid gate).
  topk     the old learned policy (epsilon-greedy explore) — regime
           comparison.
  random   the standing champion.
  dino     same everything, frozen DINOv2-S features instead of the
           learned stem — anchors the feature-quality gap. (No MI
           oracle exists without codes.)
Dense (proven chance) and static are skipped.

Measurements: final-probe top1 + read_hit (primary); admission maps
across training (WHERE does a system look while learning to see);
post-hoc attentive probe of stem features on IN-100 classification vs
a random-init stem baseline (did episodic pressure mint reusable
features?).

Toy smoke (--toy, local, $0): shrunken world — 64x64 synthetic images,
4x4 grid of 16px patches. Every patch is iid uint8 noise EXCEPT one
signature patch per image: a near-solid class color at a random grid
position. Class identity lives only in that color; the DETECTOR
("smooth patch = informative") is a local, generalizable feature, so
held-out classes with novel colors stay solvable — selection must be
content-based, not memorized. Oracle arm scores patches by negative
intra-patch variance (the known-informativeness reference). Expected
dissociation at tight budget (label + 2 of 16): oracle >> {sampled,
topk} > random ~ chance-ish. Also runs the stem locality zero-grad
check.
"""

import argparse
import gc
import math
import os
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

N_STREAM = 6
N_LABELS = 6
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
D_DINO = 384


class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio=3.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x):
        y = self.n1(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class PixelStem(nn.Module):
    """Per-patch-only perception: linear embed + depth x (LN+MLP).
    NO cross-patch mixing — the locality law. Verified by the zero-grad
    smoke: d out[:, i] / d in[:, j] == 0 for j != i."""

    def __init__(self, patch_dim, d, depth=2, mlp_ratio=3.0):
        super().__init__()
        self.patch_dim = patch_dim
        self.embed = nn.Linear(patch_dim, d)
        h = int(d * mlp_ratio)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(),
                          nn.Linear(h, d))
            for _ in range(depth))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                         # (B, P, patch_dim)
        x = self.embed(x)
        for l in self.layers:
            x = x + l(x)
        return self.norm(x)


class DinoStem(nn.Module):
    """Frozen DINOv2-S features in, learned per-token projection out.
    Same locality property (per-token ops only)."""

    def __init__(self, d):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(D_DINO),
                                  nn.Linear(D_DINO, d), nn.LayerNorm(d))

    def forward(self, x):                         # (B, P, 384) float
        return self.proj(x)


def patchify(u8, patch):
    """(B,3,H,W) uint8 on device -> (B, P, 3*patch*patch) normalized
    float. Pure data prep — the stem is the first learned op."""
    B, C, H, W = u8.shape
    g = H // patch
    x = u8.float()
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1) * 255
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1) * 255
    x = (x - mean) / std
    x = x.reshape(B, C, g, patch, g, patch).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, g * g, C * patch * patch)


class EpisodeSampler:
    """Index-level episode construction (source-agnostic: the same
    indices address the pixel blob and the DINO lake, which are built
    in the same image order — arms therefore see identical episodes)."""

    def __init__(self, labels, class_ids):
        labels = labels.long().cpu()
        self.class_ids = class_ids.long().cpu()
        idx_lists = [torch.nonzero(labels == int(c), as_tuple=True)[0]
                     for c in self.class_ids]
        assert len(idx_lists) >= N_STREAM, "need >= 6 classes"
        self.counts = torch.tensor([len(t) for t in idx_lists])
        assert int(self.counts.min()) >= 2, "need >=2 imgs per class"
        pad = torch.zeros(len(idx_lists), int(self.counts.max()),
                          dtype=torch.long)
        for i, t in enumerate(idx_lists):
            pad[i, :len(t)] = t
        self.pad = pad

    def sample(self, n_ep, generator=None):
        g = generator
        nc = len(self.counts)
        slot_cls = torch.rand(n_ep, nc, generator=g).argsort(1)[:, :N_STREAM]
        cnt = self.counts[slot_cls]
        r = (torch.rand(n_ep, N_STREAM, generator=g) * cnt).long()
        r = torch.minimum(r, cnt - 1)
        stream_idx = self.pad[slot_cls, r]                    # (E,6)
        perm = torch.rand(n_ep, N_STREAM, generator=g).argsort(1)
        targets = (torch.rand(n_ep, N_STREAM, generator=g)
                   * torch.arange(1, N_STREAM + 1)).long()
        t_cls = slot_cls.gather(1, targets)
        t_cnt = self.counts[t_cls]
        r_sup = r.gather(1, targets)
        r2 = (torch.rand(n_ep, N_STREAM, generator=g)
              * (t_cnt - 1).clamp(min=1)).long()
        r2 = r2 + (r2 >= r_sup).long()
        r2 = torch.minimum(r2, t_cnt - 1)
        query_idx = self.pad[t_cls, r2]                       # (E,6)
        y = perm.gather(1, targets)
        return stream_idx, perm, query_idx, y, targets


class EpisodePrefetcher:
    """Background thread: sample episode indices, gather rows from the
    CPU-resident source tensor, ship to GPU on a side stream — overlaps
    the ~1.2 GB/batch of episode data with the model's compute."""

    def __init__(self, sampler, source, batch, n_batches, device, seed):
        self.sampler, self.source = sampler, source
        self.batch, self.n_batches = batch, n_batches
        self.device = torch.device(device)
        self.seed = seed
        self.epoch = 0

    def _produce(self, q):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        stream = torch.cuda.Stream(self.device)
        try:
            for _ in range(self.n_batches):
                si, pm, qi, y, ts = self.sampler.sample(self.batch, g)
                flat = torch.cat([si, qi], dim=1).reshape(-1)
                rows = self.source[flat]                      # CPU gather
                with torch.cuda.stream(stream):
                    rows = rows.to(self.device)
                    meta = tuple(t.to(self.device)
                                 for t in (pm, y, ts))
                    ev = torch.cuda.Event()
                    ev.record(stream)
                q.put((rows, meta, ev))
        finally:
            q.put(None)

    def __iter__(self):
        q = queue.Queue(maxsize=2)
        t = threading.Thread(target=self._produce, args=(q,), daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            rows, (pm, y, ts), ev = item
            torch.cuda.current_stream(self.device).wait_event(ev)
            rows = rows.reshape(self.batch, 2 * N_STREAM,
                                *self.source.shape[1:])
            yield rows[:, :N_STREAM], pm, rows[:, N_STREAM:], y, ts
        t.join()
        self.epoch += 1


class PixelICL(nn.Module):
    """Foveated episodic binder over raw pixels: per-patch stem, QK
    admission with memory across a stream, cosine read head. Harness
    identical to train_vocab_icl.ICLModel; only the tokenizer changed."""

    def __init__(self, n_pos: int, patch_dim: int, stem: str = "pixel",
                 policy: str = "sampled", modulate: bool = True,
                 budget: int = 16, d: int = 256, heads: int = 8,
                 n_query: int = 8, deep: int = 6, stem_depth: int = 2,
                 explore_frac: float = 0.125, ckpt_stem: bool = True):
        super().__init__()
        assert policy in ("sampled", "topk", "random", "oracle")
        self.oracle_fn = None      # toy only: (raw_step)->(E,n_pos) scores
        self.n_pos, self.policy, self.modulate = n_pos, policy, modulate
        self.budget, self.n_query = budget, n_query
        self.explore_frac = explore_frac
        self.stem_kind = stem
        self.ckpt_stem = ckpt_stem and stem == "pixel"
        self.patch = int(math.sqrt(patch_dim // 3))
        self.stem = PixelStem(patch_dim, d, depth=stem_depth) \
            if stem == "pixel" else DinoStem(d)
        # softplus(0.5413) = 1.0 — learnable sampling temperature
        self.raw_tau = nn.Parameter(torch.tensor(0.5413))
        self.pos = nn.Parameter(torch.randn(1, n_pos, d) * 0.02)
        self.label_emb = nn.Embedding(N_LABELS, d)
        self.label_pos = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mem_emb = nn.Parameter(torch.zeros(1, 1, d))
        self.key_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.query0 = nn.Parameter(torch.randn(1, n_query, d) * 0.02)
        self.q_update = nn.MultiheadAttention(d, heads, batch_first=True)
        self.q_norm = nn.LayerNorm(d)
        self.deep = nn.ModuleList(Block(d, heads) for _ in range(deep))
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(d)
        self.read_norm = nn.LayerNorm(d)
        # cosine read with learnable temperature (see train_vocab_icl)
        self.read_temp = nn.Parameter(torch.tensor(10.0))
        self.head = nn.Linear(d, N_LABELS)
        self.scale = 1.0 / math.sqrt(d)

    @property
    def tau(self):
        return F.softplus(self.raw_tau) + 1e-3

    @property
    def tau_val(self):
        return float(self.tau.detach())

    def _run_deep(self, x):
        for b in self.deep:
            x = b(x)
        return x

    def _pool_vec(self, tok):
        q = self.pool_q.expand(tok.shape[0], -1, -1)
        p, _ = self.pool_attn(q, tok, tok, need_weights=False)
        return self.pool_norm(p.squeeze(1))

    def _score(self, q, pool):
        k = self.key_proj(pool)                   # per-token, no mixing
        s = torch.einsum("bqd,bnd->bqn", q, k) * self.scale
        s = s.max(dim=1).values
        if self.policy == "sampled":
            s = s / self.tau                      # tau learns via the gate
        return s

    def _admit(self, s, force_idx=None, oracle_key=None):
        # label token ALWAYS admitted (architecturally salient supervision,
        # identical across arms — smoke-verified necessity in the code-lake
        # harness); B-1 content tokens chosen by the policy.
        if self.policy == "random":
            key = torch.rand_like(s)
        elif self.policy == "oracle":
            key = oracle_key + torch.rand_like(s) * 1e-3
        elif self.policy == "sampled":
            # Gumbel top-k == sampling w/o replacement from softmax(s);
            # stochastic in eval too — the stochastic policy IS the policy
            gumbel = -torch.log(-torch.log(
                torch.rand_like(s).clamp_(1e-9, 1 - 1e-9)))
            key = s.detach().float() + gumbel
        else:                                     # topk
            key = s.detach().float()
        n_force = 0
        if force_idx is not None:
            n_force = force_idx.shape[1]
            key = key.scatter(1, force_idx, float("-inf"))
        n_exp = 0
        if (self.training and self.policy == "topk"
                and self.explore_frac > 0):
            n_exp = max(1, int(self.budget * self.explore_frac))
        top_g = key.topk(self.budget - n_force - n_exp, dim=1).indices
        parts = [top_g]
        if force_idx is not None:
            parts.insert(0, force_idx)
        if n_exp:
            rnd = torch.rand_like(key)
            rnd.scatter_(1, top_g, -1.0)
            if force_idx is not None:
                rnd.scatter_(1, force_idx, -1.0)
            parts.append(rnd.topk(n_exp, dim=1).indices)
        return torch.cat(parts, dim=1)

    def _tokens(self, raw, ep_label=None):
        """raw: (E,3,H,W) uint8 [pixel stem] or (E,P,384) fp16 [dino]."""
        if self.stem_kind == "pixel":
            with torch.no_grad():
                feats = patchify(raw, self.patch)
            if self.ckpt_stem and self.training:
                emb = checkpoint(self.stem, feats, use_reentrant=False)
            else:
                emb = self.stem(feats)
        else:
            emb = self.stem(raw.float())
        tok = emb + self.pos
        if ep_label is None:
            return tok
        lt = self.label_emb(ep_label).unsqueeze(1) + self.label_pos
        return torch.cat([tok, lt], dim=1)        # (E, P+1, d)

    def forward(self, stream_raw, stream_labels, query_raw, targets,
                collect=False):
        E = stream_raw.shape[0]
        dev = stream_raw.device
        q = self.query0.expand(E, -1, -1)
        mem = None
        logits_all = []
        inst = {"admit": []} if collect else {}
        aux = torch.zeros((), device=dev)
        for s in range(N_STREAM):
            tok = self._tokens(stream_raw[:, s], stream_labels[:, s])
            sc = self._score(q, tok)              # (E, P+1)
            force = torch.full((E, 1), self.n_pos, dtype=torch.long,
                               device=dev)
            ok_s = None
            if self.policy == "oracle":
                ok_s = F.pad(self.oracle_fn(stream_raw[:, s]), (0, 1),
                             value=float("-inf"))
            top = self._admit(sc, force_idx=force, oracle_key=ok_s)
            if collect:
                inst["admit"].append(top.detach().cpu())
            sel = tok.gather(
                1, top.unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
            gate = torch.sigmoid(sc.gather(1, top)).unsqueeze(-1)
            x = sel * (1 + gate)
            if mem is not None:
                x = torch.cat([x, mem], dim=1)
            h = self._run_deep(x)
            new = h[:, :1] + self.mem_emb         # ONE fused binding token
            mem = new if mem is None else torch.cat([mem, new], dim=1)
            if self.modulate:
                dq, _ = self.q_update(q, mem, mem, need_weights=False)
                q = self.q_norm(q + dq)

            # probe: perceive the query under the budget, read ALL memory
            # through the architectural cosine head (the read circuit is
            # built in; only representations are learned). Probes write
            # nothing.
            tokq = self._tokens(query_raw[:, s])
            scq = self._score(q, tokq)
            ok_q = None if self.policy != "oracle" \
                else self.oracle_fn(query_raw[:, s])
            topq = self._admit(scq, oracle_key=ok_q)
            selq = tokq.gather(
                1, topq.unsqueeze(-1).expand(-1, -1, tokq.shape[-1]))
            gq = torch.sigmoid(scq.gather(1, topq)).unsqueeze(-1)
            hq = self._run_deep(selq * (1 + gq))
            p = self._pool_vec(hq)
            att_logits = torch.einsum(
                "bd,bmd->bm", F.normalize(p, dim=-1),
                F.normalize(mem, dim=-1)) * self.read_temp
            att = torch.softmax(att_logits, dim=-1)
            read = torch.einsum("bm,bmd->bd", att, mem)
            logits_all.append(self.head(self.read_norm(p + read)))
            # auxiliary retrieval supervision (CE on read attention) —
            # unsupervised matching saturates on a fixed slot
            aux = aux + F.cross_entropy(att_logits.float(), targets[:, s])
            if s == N_STREAM - 1:
                with torch.no_grad():
                    inst["read_hit"] = float(
                        (att.argmax(-1) == targets[:, -1]).float().mean())
                    inst["read_conf"] = float(att.max(-1).values.mean())
        inst["aux_loss"] = aux / N_STREAM
        return torch.stack(logits_all, dim=1), inst


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

@torch.no_grad()
def admission_figure(model, fig_images_u8, fig_batch, out_png, n_ep=4):
    """WHERE does the system look: stream images of fixed val episodes
    with admitted patches (label token excluded) outlined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sr, pm, qr, y, ts = fig_batch
    _, inst = model(sr, pm, qr, ts, collect=True)
    admits = inst["admit"]                        # 6 x (n_ep, B)
    g = int(math.sqrt(model.n_pos))
    fig, axes = plt.subplots(n_ep, N_STREAM,
                             figsize=(2.1 * N_STREAM, 2.1 * n_ep))
    axes = np.atleast_2d(axes)
    for e in range(n_ep):
        for s in range(N_STREAM):
            img = fig_images_u8[e, s].permute(1, 2, 0).cpu().numpy()
            ax = axes[e, s]
            ax.imshow(img)
            px = img.shape[0] // g
            for t in admits[s][e].tolist():
                if t >= model.n_pos:
                    continue                      # label token
                r, c = divmod(t, g)
                ax.add_patch(plt.Rectangle((c * px, r * px), px, px,
                                           fill=False, edgecolor="red",
                                           linewidth=1.2))
            ax.axis("off")
    fig.suptitle(f"{model.policy}/{model.stem_kind} B={model.budget} "
                 f"— admitted patches per step")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


class AttentiveProbe(nn.Module):
    def __init__(self, d, heads=8, n_classes=100):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, tok):
        q = self.q.expand(tok.shape[0], -1, -1)
        pooled, _ = self.attn(q, tok, tok)
        return self.head(self.norm(pooled.squeeze(1)))


@torch.no_grad()
def _extract_stem_features(stem, images_u8, sel, patch, device, batch=512):
    """(len(sel), P, d) fp16 CPU features through a frozen stem."""
    out = None
    for i in range(0, len(sel), batch):
        idx = sel[i:i + batch].to(images_u8.device)
        u8 = images_u8[idx].to(device)
        f = stem(patchify(u8, patch)).half().cpu()
        if out is None:
            out = torch.empty(len(sel), *f.shape[1:], dtype=torch.float16)
        out[i:i + batch] = f
    return out


def stem_probe(stem, train_images, train_labels, val_images, val_labels,
               patch, device, n_per_class=400, epochs=25, batch=1024,
               tag="stem"):
    """Post-hoc attentive probe on frozen stem features: did episodic
    pressure mint reusable IN-100 features? Compare against a
    random-init stem. (Subsampled per class — a within-script
    comparison, NOT comparable to the DINO probe table.)"""
    stem = stem.eval()
    g = torch.Generator().manual_seed(7)
    y_all = train_labels.long()
    sel = []
    for c in range(100):
        idx = torch.nonzero(y_all == c, as_tuple=True)[0]
        sel.append(idx[torch.randperm(len(idx), generator=g)[:n_per_class]])
    sel = torch.cat(sel)
    ftr = _extract_stem_features(stem, train_images, sel, patch, device)
    fva = _extract_stem_features(stem, val_images,
                                 torch.arange(val_images.shape[0]),
                                 patch, device)
    ytr = y_all[sel]
    yva = val_labels.long().to(device)
    probe = AttentiveProbe(ftr.shape[-1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0.0
    for ep in range(epochs):
        probe.train()
        order = torch.randperm(len(sel))
        for i in range(0, len(sel), batch):
            b = order[i:i + batch]
            logits = probe(ftr[b].to(device).float())
            loss = F.cross_entropy(logits, ytr[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        probe.eval()
        correct = tot = 0
        with torch.no_grad():
            for i in range(0, fva.shape[0], batch):
                pred = probe(fva[i:i + batch].to(device).float()).argmax(1)
                correct += int((pred == yva[i:i + batch]).sum())
                tot += pred.shape[0]
        best = max(best, 100 * correct / tot)
    print(f"[probe {tag}] best top1 {best:.2f}", flush=True)
    return best


# ---------------------------------------------------------------------------
# Training one arm
# ---------------------------------------------------------------------------

ARM_CFG = {
    "sampled": {"policy": "sampled", "modulate": True, "stem": "pixel"},
    "topk":    {"policy": "topk",    "modulate": True, "stem": "pixel"},
    "random":  {"policy": "random",  "modulate": False, "stem": "pixel"},
    "dino":    {"policy": "sampled", "modulate": True, "stem": "dino"},
    "oracle":  {"policy": "oracle",  "modulate": False, "stem": "pixel"},
}


def gather_val_batch(source_dev, ep, device):
    si, pm, qi, y, ts = ep
    return (source_dev[si.to(device)], pm.to(device),
            source_dev[qi.to(device)], y.to(device), ts.to(device))


def train_arm(arm, model, train_source, samp_tr, val_eps, val_source_dev,
              fig_images, device, args, run_dir):
    import wandb
    os.environ.pop("WANDB_RUN_ID", None)
    name = f"PIX_{arm}_B{args.budget}"
    run = wandb.init(project=args.wandb_project, name=name,
                     config={"arm": arm,
                             **{k: v for k, v in vars(args).items()
                                if isinstance(v, (int, float, str))}},
                     reinit=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    warm = torch.optim.lr_scheduler.LinearLR(opt, 1e-2, 1.0, args.warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.num_epochs - args.warmup, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos],
                                                  [args.warmup])
    fetch = EpisodePrefetcher(samp_tr, train_source, args.batch_size,
                              args.batches_per_epoch, device,
                              seed=args.seed * 1000)
    fig_batch = gather_val_batch(val_source_dev, tuple(
        v[:4] for v in val_eps[0]), device)
    best = 0.0
    for ep in range(args.num_epochs):
        model.train()
        t0, losses = time.time(), []
        for sr, pm, qr, y, ts in fetch:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, inst = model(sr, pm, qr, ts)
                loss = (F.cross_entropy(logits.flatten(0, 1), y.flatten())
                        + args.aux_w * inst["aux_loss"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        model.eval()
        correct = tot = 0
        insts = []
        probe_acc = np.zeros(N_STREAM)
        slot_hit, slot_n = np.zeros(N_STREAM), np.zeros(N_STREAM)
        with torch.no_grad():
            for ep_idx in val_eps:
                sr, pm, qr, y, ts = gather_val_batch(val_source_dev,
                                                     ep_idx, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, inst = model(sr, pm, qr, ts)
                pred = logits.argmax(-1)
                ok_all = pred == y
                probe_acc += ok_all.float().sum(0).cpu().numpy()
                ok = ok_all[:, -1]
                correct += int(ok.sum()); tot += y.shape[0]
                insts.append(inst)
                ts_np = ts[:, -1].cpu().numpy()
                ok_np = ok.cpu().numpy()
                for sl in range(N_STREAM):
                    m = ts_np == sl
                    slot_hit[sl] += ok_np[m].sum(); slot_n[sl] += m.sum()
        acc = 100 * correct / tot
        best = max(best, acc)
        log = {"val_top1": acc, "train_loss": float(np.mean(losses)),
               "epoch": ep, "epoch_sec": time.time() - t0,
               "tau": model.tau_val}
        for sl in range(N_STREAM):
            log[f"acc_slot{sl}"] = float(100 * slot_hit[sl]
                                         / max(slot_n[sl], 1))
            log[f"acc_probe{sl}"] = float(100 * probe_acc[sl] / tot)
        for key in ("read_hit", "read_conf"):
            log[key] = float(np.mean([b[key] for b in insts]))
        log["val_aux"] = float(np.mean(
            [float(b["aux_loss"]) for b in insts]))
        if ep % args.fig_every == 0 or ep == args.num_epochs - 1:
            png = run_dir / f"admission_{name}_ep{ep:03d}.png"
            try:
                admission_figure(model, fig_images, fig_batch, png)
                log["admission_map"] = wandb.Image(str(png))
            except Exception as e:
                print(f"[fig] failed: {e}", flush=True)
        run.log(log)
        if ep % 10 == 0 or ep == args.num_epochs - 1:
            print(f"[{name}] ep {ep + 1}/{args.num_epochs} "
                  f"top1 {acc:.2f} (best {best:.2f}) "
                  f"hit {log['read_hit']:.2f} tau {log['tau']:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    run.summary["best_top1"] = best
    torch.save(model.state_dict(), run_dir / f"{name}.pt")
    return run, best


# ---------------------------------------------------------------------------
# Toy smoke: shrunken world with color-signature patches
# ---------------------------------------------------------------------------

def locality_smoke(device):
    """d stem_out[:, i] / d input[:, j] must be ZERO for j != i."""
    stem = PixelStem(768, 64, depth=2).to(device).double()
    x = torch.randn(1, 16, 768, dtype=torch.float64, device=device,
                    requires_grad=True)
    out = stem(x)
    out[0, 3].sum().backward()
    g = x.grad.abs().sum(dim=-1)[0]               # (16,)
    leak = float(g.sum() - g[3])
    assert leak == 0.0, f"LOCALITY VIOLATED: cross-patch grad {leak}"
    print("[smoke] locality zero-grad: PASS", flush=True)


def make_toy_world(n_classes=20, per_class=20, img=64, patch=16, seed=0):
    """Every patch iid uint8 noise EXCEPT one signature patch per image:
    a near-solid class color at a random grid position. The detector
    ('smooth patch') is local and generalizes; the identity (color) is
    class-specific."""
    g = torch.Generator().manual_seed(seed)
    grid = img // patch
    colors = torch.randint(30, 226, (n_classes, 3), generator=g)
    n = n_classes * per_class
    images = torch.randint(0, 256, (n, 3, img, img), generator=g,
                           dtype=torch.uint8)
    labels = torch.arange(n_classes).repeat_interleave(per_class)
    sig_pos = torch.randint(0, grid * grid, (n,), generator=g)
    jitter = torch.randint(-10, 11, (n, 3, patch, patch), generator=g)
    for i in range(n):
        r, c = divmod(int(sig_pos[i]), grid)
        block = (colors[labels[i]].view(3, 1, 1) + jitter[i]).clamp(0, 255)
        images[i, :, r * patch:(r + 1) * patch,
               c * patch:(c + 1) * patch] = block.to(torch.uint8)
    return images, labels, sig_pos


def toy_smoke(args, device):
    locality_smoke(device)
    img, patch = 64, 16
    n_pos = (img // patch) ** 2                   # 16
    budget = 3                                    # label + TWO of 16
    images, labels, _ = make_toy_world(img=img, patch=patch, seed=0)
    val_images, val_labels, _ = make_toy_world(img=img, patch=patch,
                                               seed=1)
    split = torch.randperm(20, generator=torch.Generator().manual_seed(42))
    samp_tr = EpisodeSampler(labels, split[:13])
    samp_va = EpisodeSampler(val_labels, split[13:])
    g = torch.Generator().manual_seed(123)
    vi = samp_va.sample(512, g)
    images_dev = images.to(device)
    val_images_dev = val_images.to(device)
    val_batch = gather_val_batch(val_images_dev, vi, device)

    def oracle_fn(raw):                           # (E,3,64,64) uint8
        p = patchify(raw, patch)                  # (E,16,768)
        return -p.std(dim=-1)                     # smooth = signature

    results = {}
    for arm in ("oracle", "sampled", "topk", "random"):
        cfg = ARM_CFG[arm]
        torch.manual_seed(args.seed)
        model = PixelICL(n_pos, 3 * patch * patch, stem="pixel",
                         policy=cfg["policy"], modulate=cfg["modulate"],
                         budget=budget, d=128, heads=4, n_query=4,
                         deep=3, stem_depth=2, ckpt_stem=False).to(device)
        model.oracle_fn = oracle_fn
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=0.05)
        samp_g = torch.Generator().manual_seed(9)
        best = 0.0
        for ep in range(args.toy_epochs):
            model.train()
            for _ in range(30):
                si, pm, qi, y, ts = samp_tr.sample(128, samp_g)
                sr = images_dev[si.to(device)]
                qr = images_dev[qi.to(device)]
                pm, y, ts = (t.to(device) for t in (pm, y, ts))
                logits, inst = model(sr, pm, qr, ts)
                loss = (F.cross_entropy(logits.flatten(0, 1), y.flatten())
                        + args.aux_w * inst["aux_loss"])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                logits, inst = model(val_batch[0], val_batch[1],
                                     val_batch[2], val_batch[4])
                acc = float((logits.argmax(-1)[:, -1]
                             == val_batch[3][:, -1]).float().mean()) * 100
            best = max(best, acc)
            if ep % 5 == 0 or ep == args.toy_epochs - 1:
                print(f"[toy {arm}] ep {ep + 1}/{args.toy_epochs} "
                      f"top1 {acc:.1f} (best {best:.1f}) "
                      f"hit {inst['read_hit']:.2f} "
                      f"tau {model.tau_val:.2f}", flush=True)
        results[arm] = best
    print("TOY_RESULTS " + str({k: round(v, 1) for k, v in
                                results.items()}), flush=True)
    chance = 100.0 / N_LABELS
    assert results["oracle"] > 2.2 * chance, \
        "oracle arm never learned — harness is broken"
    print("[smoke] toy world: DONE (inspect the dissociation above)",
          flush=True)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", type=str, nargs="+",
                   default=["sampled", "random", "topk", "dino"])
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--batches_per_epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--aux_w", type=float, default=0.5)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--deep", type=int, default=6)
    p.add_argument("--stem_depth", type=int, default=2)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--n_val_episodes", type=int, default=2048)
    p.add_argument("--fig_every", type=int, default=10)
    p.add_argument("--probe_per_class", type=int, default=400)
    p.add_argument("--lake_dir", type=str, default="./data/dino_lake")
    p.add_argument("--jpeg_cache_dir", type=str, default="./jpeg_cache")
    p.add_argument("--dataset_name", type=str,
                   default="clane9/imagenet-100")
    p.add_argument("--dataset_cache_dir", type=str, default="./data")
    p.add_argument("--wandb_project", type=str, default="neocore-pix")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--toy", action="store_true",
                   help="local smoke: locality check + shrunken world")
    p.add_argument("--toy_epochs", type=int, default=25)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.toy:
        toy_smoke(args, device)
        return

    assert device == "cuda", "real runs need a GPU"
    from dataset_ram import ensure_ram_cache
    cfg = SimpleNamespace(jpeg_cache_dir=args.jpeg_cache_dir,
                          dataset_name=args.dataset_name,
                          dataset_cache_dir=args.dataset_cache_dir)
    train_blob_path, val_blob_path = ensure_ram_cache(cfg)

    need_dino = "dino" in args.arms
    if need_dino and not (Path(args.lake_dir) / "train_tokens.pt").exists():
        import train_vocab as tv
        tv.stage0(args, device)

    run_dir = Path("runs") / "PIX_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)
    (Path("runs") / "LATEST").write_text(str(run_dir))

    # class holdout — identical split to the code-lake ICL experiment
    split = torch.randperm(100, generator=torch.Generator().manual_seed(42))
    print(f"holdout val classes: {sorted(split[80:].tolist())}", flush=True)

    val_blob = torch.load(val_blob_path, map_location="cpu", mmap=True)
    val_images_cpu = val_blob["images"].contiguous()      # ~1 GB
    val_labels = val_blob["labels"].clone()
    del val_blob
    samp_va = EpisodeSampler(val_labels, split[80:])
    g = torch.Generator().manual_seed(123)
    val_eps = [samp_va.sample(args.batch_size, g)
               for _ in range(max(args.n_val_episodes
                                  // args.batch_size, 1))]
    fig_images = val_images_cpu[
        val_eps[0][0][:4].reshape(-1)].reshape(
        4, N_STREAM, 3, 256, 256).to(device)

    pixel_arms = [a for a in args.arms if ARM_CFG[a]["stem"] == "pixel"]
    results, probes = {}, {}
    patch_dim = 3 * args.patch * args.patch
    n_pos = (256 // args.patch) ** 2

    if pixel_arms:
        print("[data] materializing train blob into RAM (~25 GB)...",
              flush=True)
        blob = torch.load(train_blob_path, map_location="cpu", mmap=True)
        train_images = blob["images"].contiguous()
        train_labels = blob["labels"].clone()
        del blob
        samp_tr = EpisodeSampler(train_labels, split[:80])
        val_images_dev = val_images_cpu.to(device)        # ~1 GB VRAM

        for arm in pixel_arms:
            cfg_a = ARM_CFG[arm]
            torch.manual_seed(args.seed)
            model = PixelICL(n_pos, patch_dim, stem="pixel",
                             policy=cfg_a["policy"],
                             modulate=cfg_a["modulate"],
                             budget=args.budget, d=args.d, deep=args.deep,
                             stem_depth=args.stem_depth).to(device)
            run, best = train_arm(arm, model, train_images, samp_tr,
                                  val_eps, val_images_dev, fig_images,
                                  device, args, run_dir)
            results[f"PIX_{arm}_B{args.budget}"] = best
            pr = stem_probe(model.stem, train_images, train_labels,
                            val_images_dev, val_labels, args.patch,
                            device, n_per_class=args.probe_per_class,
                            tag=arm)
            probes[arm] = pr
            run.summary["stem_probe_top1"] = pr
            run.finish()
            print(f"=== PIX_{arm}_B{args.budget}: best top1 {best:.2f} "
                  f"stem_probe {pr:.2f}", flush=True)
            del model
            torch.cuda.empty_cache()

        # random-init reference: same stem architecture, no training
        torch.manual_seed(args.seed + 777)
        rand_stem = PixelStem(patch_dim, args.d,
                              depth=args.stem_depth).to(device)
        probes["randinit"] = stem_probe(
            rand_stem, train_images, train_labels, val_images_dev,
            val_labels, args.patch, device,
            n_per_class=args.probe_per_class, tag="randinit")
        del rand_stem, train_images, val_images_dev
        gc.collect()
        torch.cuda.empty_cache()

    if need_dino:
        print("[data] materializing DINO lake into RAM (~25 GB)...",
              flush=True)
        lake = Path(args.lake_dir)
        tr = torch.load(lake / "train_tokens.pt", map_location="cpu",
                        mmap=True)
        train_tokens = tr["tokens"].contiguous()
        tr_labels = tr["labels"].clone()
        del tr
        va = torch.load(lake / "val_tokens.pt", map_location="cpu",
                        mmap=True)
        val_tokens_dev = va["tokens"].contiguous().to(device)  # ~1 GB
        del va
        samp_tr_d = EpisodeSampler(tr_labels, split[:80])

        cfg_a = ARM_CFG["dino"]
        torch.manual_seed(args.seed)
        model = PixelICL(n_pos, patch_dim, stem="dino",
                         policy=cfg_a["policy"], modulate=cfg_a["modulate"],
                         budget=args.budget, d=args.d,
                         deep=args.deep).to(device)
        run, best = train_arm("dino", model, train_tokens, samp_tr_d,
                              val_eps, val_tokens_dev, fig_images,
                              device, args, run_dir)
        results[f"PIX_dino_B{args.budget}"] = best
        run.finish()
        print(f"=== PIX_dino_B{args.budget}: best top1 {best:.2f}",
              flush=True)
        del model, train_tokens, val_tokens_dev
        gc.collect()
        torch.cuda.empty_cache()

    # summary + verified artifact (self-destroy is gated on this)
    import json
    import wandb
    summary = {"results": results, "stem_probes": probes}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    os.environ.pop("WANDB_RUN_ID", None)
    run = wandb.init(project=args.wandb_project, name="PIX_sweep_summary",
                     reinit=True)
    for k, v in results.items():
        run.summary[k] = v
    for k, v in probes.items():
        run.summary[f"probe_{k}"] = v
    art = wandb.Artifact(f"pix-icl-{run.id}", type="cls_sweep")
    art.add_dir(str(run_dir))
    run.log_artifact(art)
    art.wait()
    print("ARTIFACT_VERIFIED")
    run.finish()
    print("SWEEP_RESULTS " + json.dumps(summary))


if __name__ == "__main__":
    main()
