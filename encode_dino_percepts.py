"""Encode IN-100 (HF clane9/imagenet-100, parquet direct — local scipy
is broken so `datasets` is bypassed) through frozen DINOv2-S into
pooled per-image percepts, cached to one local .pt:

    {'cls': (N,384) fp16, 'patch': (N,384) fp16 (patch-token mean),
     'label': (N,) int64, 'is_val': (N,) bool}

This is the entire perception stage of the real-data lifetime-codebook
experiment (train_dino_codebook_icl.py): images are seen once, here;
everything downstream operates on these vectors. ~130k images, ~40 min
on the local 2060, ~200 MB cache. NEVER commit the cache.
"""

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_224(rec):
    """PIL-only resize-256 + center-crop-224 (torchvision is deliberately
    absent on the cloud image — it can clobber the tuned torch build)."""
    im = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    im = im.crop((l, t, l + 224, t + 224))
    return torch.from_numpy(
        np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)


def encode(out_path, batch=64, workers=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device).eval().half()

    shards = [f for f in list_repo_files("clane9/imagenet-100",
                                         repo_type="dataset")
              if f.startswith("data/") and f.endswith(".parquet")]
    print(f"{len(shards)} parquet shards", flush=True)

    all_cls, all_patch, all_lab, all_val = [], [], [], []
    t0, done = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        for shard in sorted(shards):
            path = hf_hub_download("clane9/imagenet-100", shard,
                                   repo_type="dataset")
            tbl = pq.read_table(path, columns=["image", "label"])
            recs = tbl.column("image").to_pylist()
            labs = tbl.column("label").to_pylist()
            is_val = "validation" in shard
            for i in range(0, len(recs), batch):
                chunk = recs[i:i + batch]
                imgs = list(ex.map(_load_224, chunk))
                x = torch.stack(imgs).to(device)
                x = ((x - mean) / std).half()
                with torch.no_grad():
                    r = model.forward_features(x)
                all_cls.append(r["x_norm_clstoken"].cpu())
                all_patch.append(r["x_norm_patchtokens"].mean(1).cpu())
                all_lab.extend(labs[i:i + batch])
                all_val.extend([is_val] * len(chunk))
                done += len(chunk)
                if done % 8192 < batch:
                    print(f"{done} images  {done / (time.time() - t0):.0f}"
                          " img/s", flush=True)
            print(f"shard done: {shard} (total {done})", flush=True)

    out = {"cls": torch.cat(all_cls),
           "patch": torch.cat(all_patch),
           "label": torch.tensor(all_lab, dtype=torch.long),
           "is_val": torch.tensor(all_val, dtype=torch.bool)}
    torch.save(out, out_path)
    print(f"saved {out['cls'].shape[0]} percepts -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="dino_percepts_vits14.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    encode(args.out, args.batch, args.workers)


if __name__ == "__main__":
    main()
