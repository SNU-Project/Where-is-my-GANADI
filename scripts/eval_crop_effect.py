"""개 탐지·크롭이 실제로 도움이 되는지 shelter leave-one-out 벤치마크로 검증.

사용:
    python scripts/eval_crop_effect.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from src.data.manifests import load_shelter_split
from src.data.transforms import build_eval_transform
from src.retrieval.color import color_histogram
from src.retrieval.detect import crop_to_dog
from src.retrieval.metrics import evaluate_reid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_reid import load_model  # noqa: E402


def embed_and_hist(paths, model, transform, device, crop: bool):
    feats, hists = [], []
    t0 = time.time()
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        if crop:
            img = crop_to_dog(img, device="cpu")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model(x).cpu().numpy()[0]
        feats.append(f / (np.linalg.norm(f) + 1e-12))
        hists.append(color_histogram(img))
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(paths)} ({time.time()-t0:.0f}s)")
    return np.stack(feats), np.stack(hists)


def main():
    split = load_shelter_split()
    model, transform = load_model("resnet50_imagenet", "checkpoints/E1_resnet50_bnneck.pt")
    from src.models.backbones import get_device
    device = get_device()
    model.to(device)

    q_pids = np.asarray(split.query_pids)
    g_pids = np.asarray(split.gallery_pids)
    q_cams = np.asarray(split.query_camids)
    g_cams = np.asarray(split.gallery_camids)

    for crop in [False, True]:
        tag = "크롭함" if crop else "크롭안함"
        print(f"\n=== {tag} ===")
        q_feats, q_hists = embed_and_hist(split.query_paths, model, transform, device, crop)
        g_feats, g_hists = embed_and_hist(split.gallery_paths, model, transform, device, crop)

        emb_sim = q_feats @ g_feats.T
        color_sim = np.zeros_like(emb_sim)
        for i in range(len(q_hists)):
            color_sim[i] = np.minimum(g_hists, q_hists[i]).sum(axis=1)

        for alpha, label in [(1.0, "임베딩만"), (0.75, "임베딩75+색상25")]:
            final = alpha * emb_sim + (1 - alpha) * color_sim
            m = evaluate_reid(1 - final, q_pids, g_pids, q_cams, g_cams, max_rank=10)
            print(f"  [{tag}/{label}] {m}")


if __name__ == "__main__":
    main()
