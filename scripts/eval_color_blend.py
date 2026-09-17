"""색상 히스토그램을 섞었을 때 실제 보호소 leave-one-out 벤치마크에서 Rank-1/mAP가
나빠지지 않는지 확인하고, 여러 alpha 값을 비교해 적절한 값을 고른다.

사용:
    python scripts/eval_color_blend.py --checkpoint checkpoints/E1_resnet50_bnneck.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.data.manifests import load_shelter_split
from src.models.backbones import get_device
from src.retrieval.color import color_histogram, histogram_similarity
from src.retrieval.extract import extract_features
from src.retrieval.metrics import evaluate_reid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_reid import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/E1_resnet50_bnneck.pt")
    args = ap.parse_args()

    split = load_shelter_split()
    model, transform = load_model("resnet50_imagenet", args.checkpoint)
    device = get_device()

    print("[info] 임베딩 추출 중...")
    q_feats = extract_features(split.query_paths, model, transform, device, 32)
    g_feats = extract_features(split.gallery_paths, model, transform, device, 32)
    q_feats = q_feats / (np.linalg.norm(q_feats, axis=1, keepdims=True) + 1e-12)
    g_feats = g_feats / (np.linalg.norm(g_feats, axis=1, keepdims=True) + 1e-12)
    emb_sim = q_feats @ g_feats.T  # (nq, ng) cosine

    print("[info] 색상 히스토그램 계산 중...")
    q_hists = np.stack([color_histogram(Image.open(p)) for p in split.query_paths])
    g_hists = np.stack([color_histogram(Image.open(p)) for p in split.gallery_paths])
    # 히스토그램 교집합을 행렬로: sum_k min(q[k], g[k]) -> 벡터화
    color_sim = np.zeros_like(emb_sim)
    for i in range(len(q_hists)):
        color_sim[i] = np.minimum(g_hists, q_hists[i]).sum(axis=1)

    g_pids = np.asarray(split.gallery_pids)
    q_pids = np.asarray(split.query_pids)
    q_cams = np.asarray(split.query_camids)
    g_cams = np.asarray(split.gallery_camids)

    for alpha in [1.0, 0.85, 0.75, 0.6, 0.5]:
        final_sim = alpha * emb_sim + (1 - alpha) * color_sim
        distmat = 1 - final_sim
        m = evaluate_reid(distmat, q_pids, g_pids, q_cams, g_cams, max_rank=10)
        tag = "임베딩만" if alpha == 1.0 else f"alpha={alpha}"
        print(f"[{tag:10s}] {m}")


if __name__ == "__main__":
    main()
