"""갤러리 도메인 밖(스마트폰으로 찍은 실제 반려견 사진 등)의 단일 쿼리 이미지를
데모와 동일한 파이프라인(E1 임베딩 + 색상 블렌딩)으로 검색해 Top-K를 이미지+메타데이터로 보여준다.

사용:
    python scripts/query_real_photo.py --image "/path/to/photo.jpg" --true-breed 푸들
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.models.backbones import get_device
from src.retrieval.color import color_histogram

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_reid import load_model  # noqa: E402

CACHE_DIR = Path("demo_cache")
SHELTER_ROOT = Path("Data/shelter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/E1_resnet50_bnneck.pt")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--true-breed", default=None, help="쿼리 개의 실제 품종(알고 있으면, 참고용)")
    ap.add_argument("--out", default="docs/eda/17_real_photo_query.png")
    ap.add_argument("--cache-dir", default="demo_cache")
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)

    idx = pd.read_csv(cache_dir / "gallery_index.csv")
    meta = pd.read_csv(cache_dir / "gallery_meta.csv")[["desertion_no", "kind_nm", "color_cd"]]
    photo_meta = idx.merge(meta, on="desertion_no", how="left")
    g_feats = np.load(cache_dir / "gallery_embeddings.npy")
    g_hists = np.load(cache_dir / "gallery_color_hists.npy")

    model, transform = load_model("resnet50_imagenet", args.checkpoint)
    device = get_device()
    model.to(device)

    import torch
    q_img = Image.open(args.image).convert("RGB")
    x = transform(q_img).unsqueeze(0).to(device)
    with torch.no_grad():
        q_feat = model(x).cpu().numpy()[0]
    q_feat = q_feat / (np.linalg.norm(q_feat) + 1e-12)
    q_hist = color_histogram(q_img)

    emb_sim = g_feats @ q_feat
    color_sim = np.minimum(g_hists, q_hist).sum(axis=1)
    final = args.alpha * emb_sim + (1 - args.alpha) * color_sim

    order = np.argsort(-final)[: args.topk]

    print(f"[쿼리] {args.image}" + (f" (실제 품종: {args.true_breed})" if args.true_breed else ""))
    print(f"[설정] alpha={args.alpha} (임베딩 {args.alpha:.0%} + 색상 {1-args.alpha:.0%})\n")
    for rank, gi in enumerate(order, 1):
        row = photo_meta.iloc[gi]
        print(f"  #{rank} sim={final[gi]:.3f} (임베딩={emb_sim[gi]:.3f} 색상={color_sim[gi]:.3f}) "
              f"품종={row.kind_nm} 색상={row.color_cd} 공고번호={row.desertion_no}")

    plt.rcParams["font.family"] = "AppleGothic"
    fig, axes = plt.subplots(1, args.topk + 1, figsize=(3 * (args.topk + 1), 3.4))
    axes[0].imshow(q_img)
    title = "쿼리(친구 강아지)"
    if args.true_breed:
        title += f"\n실제품종:{args.true_breed}"
    axes[0].set_title(title, fontsize=9)
    axes[0].axis("off")
    for col, gi in enumerate(order):
        row = photo_meta.iloc[gi]
        ax = axes[col + 1]
        ax.imshow(Image.open(SHELTER_ROOT / row.relpath))
        ax.set_title(f"#{col+1} sim={final[gi]:.2f}\n{row.kind_nm}/{row.color_cd}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f"\n[완료] {args.out}")


if __name__ == "__main__":
    main()
