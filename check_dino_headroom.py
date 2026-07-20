"""Consolidation-headroom pre-check in real DINO space (2026-07-20).

Before building the real-data lifetime-codebook experiment, measure
whether its premise holds on IN-100: does AVERAGING k glances of a
held-out class (prototype = what a codebook code becomes) beat
REMEMBERING k glances (nearest-exemplar = what the nocode/episodic
baseline does)? The toy's world was rigged so averaging is optimal;
this is the un-rigged version, computed with zero training.

- Classes: the harness's exact holdout — randperm(100, seed 42)[80:].
- Features: frozen DINOv2-S, both CLS token and mean patch tokens.
- Stats: same-class vs cross-class cosine (the toy quoted 0.33/0.00;
  DINO's gap sizes the denoising headroom).
- Curves over k in {1,2,4,8,16,25}: 20-way classification of 25 eval
  images/class using (a) cosine-to-mean-of-k prototypes, (b) max
  cosine over k stored exemplars. proto(k) - nn(k) growing with k =
  consolidation beats memory in DINO space = the codebook experiment
  is justified. proto(25) - proto(1) = total headroom.

Local, $0: HF val split (only held classes kept), ~1k images through
DINOv2-S on the 2060.
"""

import argparse
import io

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
import pyarrow.parquet as pq
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--n-proto-pool", type=int, default=25)
    ap.add_argument("--resamples", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    held = sorted(torch.randperm(
        100, generator=torch.Generator().manual_seed(42))[80:].tolist())
    print(f"holdout classes (harness seed-42 split): {held}", flush=True)

    # read the val parquet directly (local scipy is broken and the
    # `datasets` builder insists on patching scipy at import)
    pf = hf_hub_download("clane9/imagenet-100",
                         "data/validation-00000-of-00001.parquet",
                         repo_type="dataset")
    tbl = pq.read_table(pf, columns=["image", "label"])
    imgs_col = tbl.column("image").to_pylist()
    labels_col = tbl.column("label").to_pylist()
    print(f"val split: {len(labels_col)} images", flush=True)

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device).eval()

    # gather per-class images, encode
    by_class = {c: [] for c in held}
    for rec, c in zip(imgs_col, labels_col):
        if c in by_class and len(by_class[c]) < args.per_class:
            by_class[c].append(
                Image.open(io.BytesIO(rec["bytes"])).convert("RGB"))
    n_min = min(len(v) for v in by_class.values())
    print(f"images per class: min {n_min}", flush=True)

    feats = {"cls": [], "patch": []}
    with torch.no_grad():
        for c in held:
            imgs = torch.stack([tf(im) for im in by_class[c]]).to(device)
            out = []
            for i in range(0, imgs.shape[0], args.batch):
                r = model.forward_features(imgs[i:i + args.batch])
                out.append((r["x_norm_clstoken"],
                            r["x_norm_patchtokens"].mean(1)))
            feats["cls"].append(torch.cat([o[0] for o in out]).float())
            feats["patch"].append(torch.cat([o[1] for o in out]).float())
    C = len(held)

    for rep in ("cls", "patch"):
        X = torch.stack([f[:n_min] for f in feats[rep]]).cpu()  # (C,N,D)
        Xn = F.normalize(X, dim=-1)
        N = Xn.shape[1]

        # same-class vs cross-class cosine
        same = torch.einsum("cnd,cmd->cnm", Xn, Xn)
        iu = torch.triu_indices(N, N, offset=1)
        same_mean = same[:, iu[0], iu[1]].mean().item()
        cross = torch.einsum("cnd,kmd->cknm", Xn, Xn)
        mask = ~torch.eye(C, dtype=torch.bool)
        cross_mean = cross[mask].mean().item()
        print(f"\n[{rep}] same-class cos {same_mean:.3f}   "
              f"cross-class cos {cross_mean:.3f}   "
              f"(toy was 0.33 / 0.00)", flush=True)

        # k-shot: consolidate (prototype) vs remember (nearest exemplar)
        pool, evl = Xn[:, :args.n_proto_pool], Xn[:, args.n_proto_pool:]
        ne = evl.shape[1]
        labels = torch.arange(C).unsqueeze(1).expand(C, ne).reshape(-1)
        eflat = evl.reshape(-1, evl.shape[-1])              # (C*ne, D)
        print(f"[{rep}]  k    proto(k)   nn(k)    delta")
        for k in (1, 2, 4, 8, 16, args.n_proto_pool):
            k = min(k, args.n_proto_pool)
            pa, na = [], []
            for _ in range(args.resamples):
                idx = torch.stack([torch.randperm(args.n_proto_pool)[:k]
                                   for _ in range(C)])
                shots = torch.stack([pool[c, idx[c]] for c in range(C)])
                proto = F.normalize(shots.mean(1), dim=-1)   # (C, D)
                pa.append((torch.einsum("ed,cd->ec", eflat, proto)
                           .argmax(1) == labels).float().mean().item())
                sims = torch.einsum("ed,ckd->eck", eflat, shots)
                na.append((sims.max(-1).values.argmax(1)
                           == labels).float().mean().item())
            pm, nm = sum(pa) / len(pa), sum(na) / len(na)
            print(f"[{rep}] {k:3d}   {pm * 100:6.2f}   {nm * 100:6.2f}"
                  f"   {(pm - nm) * 100:+6.2f}", flush=True)


if __name__ == "__main__":
    main()
