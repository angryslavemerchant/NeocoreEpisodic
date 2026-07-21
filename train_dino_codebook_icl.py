"""REAL-DATA lifetime codebook consolidation (2026-07-20, same-day as
the toy campaign): the toy_codebook_icl harness run on REAL IN-100
percepts — frozen DINOv2-S pooled vectors (encode_dino_percepts.py
cache) instead of the synthetic prototype+noise world. Everything
validated in the toy carries over verbatim (model, v6b rule set,
lifetime harness, arms); ONLY the world is real now.

Why this is the right scale-up: the campaign's design conversation
(with Ibanis) explicitly dropped token-level admission for this line —
"full attention over the patch grid -> consolidating single-vector
codes" — to isolate the memory axis. A pooled DINO percept per image
IS that design. Selection/loop machinery stays in the pixel-ICL line.

The measured stakes (check_dino_headroom.py, same classes, same
features): +17 pts (CLS) / +25 pts (patch-mean) of consolidation
headroom between 1-glance and 25-glance class knowledge; prototypes
beat exemplar memory at every k>=2. The within-lifetime live curve
harvests that budget or fails to — either is informative.

Setup mirrors the toy: lifetimes of L episodes over the SAME 6 classes
drawn from the harness's seed-42 class split (80 meta-train / 20 held
out); labels permuted per episode; cards wiped per episode; the
persistent codebook is the only cross-episode channel at eval (frozen
weights). Default rule = v6b (soft mixture content + lazy bar-merge).
Arms: live / frozen / oracle / nocode.

Percepts default to patch-mean (--rep patch): noisier one-glance
identity (63% 1-shot vs CLS 77%) = larger consolidation payoff = the
regime where the codebook has the most to prove.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import toy_codebook_icl as icl
from toy_codebook_icl import (ToyBinder, eval_arm,
                              sample_lifetime_classes, train)


class RealWorld:
    """Drop-in replacement for the toy World: sample(cls) returns a
    random real image's pooled DINO percept of that class."""

    def __init__(self, cache_path, rep, device):
        d = torch.load(cache_path, map_location="cpu")
        vecs, labs = d[rep].float(), d["label"].long()
        counts = torch.bincount(labs, minlength=100)
        table = torch.zeros(100, int(counts.max()), dtype=torch.long)
        for c in range(100):
            idx = (labs == c).nonzero(as_tuple=True)[0]
            table[c, :idx.numel()] = idx
        self.vecs = vecs.to(device)
        self.counts = counts.to(device)
        self.table = table.to(device)
        split = torch.randperm(100,
                               generator=torch.Generator().manual_seed(42))
        self.train_pool = split[:80].sort().values.to(device)
        self.held_pool = split[80:].sort().values.to(device)
        self.d_in = vecs.shape[1]

    def sample(self, cls):
        shape = cls.shape
        flat = cls.reshape(-1)
        r = (torch.rand(flat.numel(), device=flat.device)
             * self.counts[flat]).long()
        return self.vecs[self.table[flat, r]].reshape(*shape, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, default="dino_percepts_vits14.pt")
    ap.add_argument("--rep", type=str, default="patch",
                    choices=["patch", "cls"])
    ap.add_argument("--n-way", type=int, default=6,
                    help="classes per lifetime/episode (6-way saturates "
                         "one-shot on DINO percepts — run 6ej88ypk; finer "
                         "discrimination is where consolidation pays)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--l-train", type=int, default=12)
    ap.add_argument("--l-eval", type=int, default=24)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--theta", type=float, default=0.45)
    ap.add_argument("--theta-merge", type=float, default=0.6)
    ap.add_argument("--merge-mode", type=str, default="bar",
                    choices=["off", "bar", "witness"])
    ap.add_argument("--content", type=str, default="soft",
                    choices=["hard", "soft"])
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--save-prefix", type=str, default="")
    ap.add_argument("--load-codebook", type=str, default="")
    ap.add_argument("--load-nocode", type=str, default="")
    ap.add_argument("--wandb", action="store_true",
                    help="stream to wandb + upload a VERIFIED artifact "
                         "(REQUIRED on cloud: the instance self-destroys "
                         "on exit 0 and takes local files with it)")
    ap.add_argument("--wandb_project", type=str, default="neocore-codebook")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    icl.N_WAY = args.n_way      # module global read at call time throughout

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"dino-{args.rep}-{args.n_way}way-"
                              f"{args.merge_mode}-{args.content}",
                         config=vars(args))
    log_fn = (lambda d: run.log(d)) if run else None

    if not os.path.exists(args.cache):
        print(f"cache {args.cache} missing — encoding IN-100 through "
              "DINOv2-S first", flush=True)
        from encode_dino_percepts import encode
        encode(args.cache, batch=128)

    world = RealWorld(args.cache, args.rep, device)
    print(f"world: {world.vecs.shape[0]} real percepts ({args.rep}), "
          f"held classes {world.held_pool.tolist()}", flush=True)

    # raw-percept geometry (should match check_dino_headroom)
    c = sample_lifetime_classes(world.held_pool, 2048, device)
    x1, x2 = world.sample(c[:, 0]), world.sample(c[:, 0])
    xo = world.sample(c[:, 1])
    print(f"raw percepts: same-class cos "
          f"{F.cosine_similarity(x1, x2).mean().item():.3f}  cross "
          f"{F.cosine_similarity(x1, xo).mean().item():.3f}", flush=True)

    kw = dict(d=args.d, k=args.k, theta=args.theta,
              theta_merge=args.theta_merge)
    model_a = ToyBinder(world.d_in, use_codes=True,
                        merge_mode=args.merge_mode, content=args.content,
                        lam=args.lam, **kw).to(device)
    model_b = ToyBinder(world.d_in, use_codes=False, **kw).to(device)

    if args.load_codebook:
        model_a.load_state_dict(torch.load(args.load_codebook))
        print(f"codebook model loaded from {args.load_codebook}",
              flush=True)
    else:
        train(model_a, world, world.train_pool, args.steps, args.batch,
              args.l_train, args.lr, device, "codebook", log_fn=log_fn)
    if args.load_nocode:
        model_b.load_state_dict(torch.load(args.load_nocode))
        print(f"nocode baseline loaded from {args.load_nocode}",
              flush=True)
    else:
        train(model_b, world, world.train_pool, args.steps, args.batch,
              args.l_train, args.lr, device, "nocode  ", log_fn=log_fn)
    if args.save_prefix:
        torch.save(model_a.state_dict(), args.save_prefix + "_codebook.pt")
        torch.save(model_b.state_dict(), args.save_prefix + "_nocode.pt")

    with torch.no_grad():
        u1, u2, uo = (model_a.enc(x1), model_a.enc(x2), model_a.enc(xo))
        print(f"encoder space (held): same-class cos "
              f"{F.cosine_similarity(u1, u2).mean().item():.3f}  cross "
              f"{F.cosine_similarity(u1, uo).mean().item():.3f}  "
              f"(theta={args.theta})", flush=True)

    arms = {}
    for arm in ("live", "frozen", "oracle"):
        arms[arm] = eval_arm(model_a, world, world.held_pool, arm,
                             args.eval_batch, args.l_eval,
                             args.eval_batches, device)
    arms["nocode"] = eval_arm(model_b, world, world.held_pool, "nocode",
                              args.eval_batch, args.l_eval,
                              args.eval_batches, device)

    print(f"\nREAL DATA — held-out IN-100 classes, frozen weights — "
          f"final-step probe top1 (chance {100 / args.n_way:.1f}) "
          f"by episode index:")
    names = ["live", "frozen", "oracle", "nocode"]
    print("  ep-idx  " + "  ".join(f"{n:>7s}" for n in names)
          + "   agree    used  merges")
    for e in range(args.l_eval):
        row = "  ".join(f"{arms[n]['acc'][e] * 100:7.2f}" for n in names)
        print(f"  {e:6d}  {row}  {arms['live']['agree'][e] * 100:6.2f}"
              f"  {arms['live']['used'][e]:6.2f}"
              f"  {arms['live']['merges'][e]:6.2f}")
    print("\n  read_hit, same layout:")
    for e in range(args.l_eval):
        row = "  ".join(f"{arms[n]['read'][e] * 100:7.2f}" for n in names)
        print(f"  {e:6d}  {row}")
    for n in names:
        a = arms[n]["acc"] * 100
        print(f"  {n:>7s}: early {a[:3].mean().item():5.2f} -> late "
              f"{a[-3:].mean().item():5.2f}  "
              f"(delta {a[-3:].mean().item() - a[:3].mean().item():+.2f})")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for n, col in zip(names, ("C0", "C1", "C2", "C3")):
            ax.plot(arms[n]["acc"].numpy() * 100, marker="o", color=col,
                    label=n)
        ax.axhline(100 / 6, color="gray", ls=":", label="chance")
        ax.set_xlabel("episode index within lifetime (held-out IN-100 "
                      "classes, frozen weights)")
        ax.set_ylabel("final-step probe top1 (%)")
        ax.set_title(f"real-data codebook consolidation — {args.rep} "
                     f"percepts, {args.merge_mode}/{args.content}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"curve -> {args.out}", flush=True)

    if run:
        import wandb
        for n in names:
            a = arms[n]["acc"] * 100
            run.summary[f"{n}_early"] = a[:3].mean().item()
            run.summary[f"{n}_late"] = a[-3:].mean().item()
        results = {n: {k: (v.tolist() if torch.is_tensor(v) else v)
                       for k, v in arms[n].items()} for n in names}
        with open("results.json", "w") as f:
            json.dump({"args": vars(args), "arms": results}, f, indent=1)
        if args.out:
            run.log({"lifetime_curve": wandb.Image(args.out)})
        art = wandb.Artifact(f"dino-codebook-{run.id}", type="results")
        art.add_file("results.json")
        if args.out:
            art.add_file(args.out)
        for p in (args.save_prefix + "_codebook.pt",
                  args.save_prefix + "_nocode.pt"):
            if args.save_prefix and os.path.exists(p):
                art.add_file(p)
        run.log_artifact(art)
        art.wait()          # VERIFIED upload before the instance self-destroys
        print("ARTIFACT_VERIFIED", flush=True)
        run.finish()


if __name__ == "__main__":
    main()
