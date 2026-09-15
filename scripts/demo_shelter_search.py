"""실제 사용 시나리오 시연: 보호소 사진 한 장을 "실종견 사진"이라 가정하고, 나머지 보호소
갤러리에서 Top-5를 찾아 눈으로 확인한다 (`src/data/manifests.load_shelter_split` 재사용).

사용:
    python scripts/demo_shelter_search.py --checkpoint checkpoints/E1_resnet50_bnneck.pt --n 6
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.data.manifests import load_shelter_split
from src.models.backbones import get_device
from src.retrieval.color import color_histogram
from src.retrieval.extract import extract_features

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_reid import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n", type=int, default=6, help="시연할 쿼리 개수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.75, help="임베딩 가중치 (1.0=색상 미반영)")
    ap.add_argument("--out", default="docs/eda/14_shelter_demo_search.png")
    args = ap.parse_args()

    split = load_shelter_split()
    model, transform = load_model("resnet50_imagenet", args.checkpoint)
    device = get_device()

    q_feats = extract_features(split.query_paths, model, transform, device, 32)
    g_feats = extract_features(split.gallery_paths, model, transform, device, 32)
    q_feats = q_feats / (np.linalg.norm(q_feats, axis=1, keepdims=True) + 1e-12)
    g_feats = g_feats / (np.linalg.norm(g_feats, axis=1, keepdims=True) + 1e-12)
    emb_sim = q_feats @ g_feats.T

    q_hists = np.stack([color_histogram(Image.open(p).convert("RGB")) for p in split.query_paths])
    g_hists = np.stack([color_histogram(Image.open(p).convert("RGB")) for p in split.gallery_paths])
    color_sim = np.zeros_like(emb_sim)
    for i in range(len(q_hists)):
        color_sim[i] = np.minimum(g_hists, q_hists[i]).sum(axis=1)

    final_sim = args.alpha * emb_sim + (1 - args.alpha) * color_sim
    distmat = 1 - final_sim

    g_pids = np.asarray(split.gallery_pids)
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(split.query_paths), size=args.n, replace=False)

    plt.rcParams["font.family"] = "AppleGothic"
    fig, axes = plt.subplots(args.n, 6, figsize=(15, 2.6 * args.n))
    for row, qi in enumerate(idxs):
        order = distmat[qi].argsort()[:5]
        q_pid = split.query_pids[qi]

        ax = axes[row, 0]
        ax.imshow(Image.open(split.query_paths[qi]))
        ax.set_title(f"쿼리\n(개체 {q_pid})", fontsize=8)
        ax.axis("off")

        for col, gi in enumerate(order):
            ax = axes[row, col + 1]
            ax.imshow(Image.open(split.gallery_paths[gi]))
            is_correct = g_pids[gi] == q_pid
            sim = 1 - distmat[qi, gi]
            ax.set_title(f"#{col+1} sim={sim:.2f}{' ✓' if is_correct else ''}",
                         fontsize=8, color="green" if is_correct else "black")
            ax.axis("off")

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=110)
    print(f"[완료] {args.out}")


if __name__ == "__main__":
    main()
