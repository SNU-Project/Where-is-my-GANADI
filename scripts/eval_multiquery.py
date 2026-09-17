"""쿼리 사진 여러 장을 평균 임베딩으로 합쳐서 검색하면(멀티샷 쿼리) 실제로 도움이 되는지
DogFaceNet test로 검증 — "사용자가 사진 5장까지 올릴 수 있게 하면 어떨까?" 아이디어의 근거 확보용.

방법: 개체당 이미지가 K+1장 이상인 개체만 골라, K장을 쿼리(L2정규화 임베딩 평균 -> 재정규화)로
쓰고 나머지를 갤러리로 삼는 leave-K-out 방식. K=1(기존 단일 쿼리)과 K=2..5를 비교.

사용:
    python scripts/eval_multiquery.py --checkpoint checkpoints/E1_resnet50_bnneck_breedpretrain_triplet.pt
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.transforms import build_eval_transform
from src.models.backbones import get_device
from src.retrieval.extract import extract_features
from src.retrieval.metrics import evaluate_reid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_reid import load_model  # noqa: E402

DFN_ROOT = Path("Data/DogFaceNet")
MANIFEST = "metadata/dogfacenet_manifest.csv"


def build_split(k: int, seed: int = 123):
    df = pd.read_csv(MANIFEST)
    test_df = df[df.split_group == "test"].reset_index(drop=True)
    rng = random.Random(seed)

    q_groups, g_paths, g_pids, g_cams = [], [], [], []
    global_cam = 0
    for dog_id, id_rows in test_df.groupby("dog_id"):
        rows = id_rows.to_dict("records")
        if len(rows) < k + 1:
            continue  # 갤러리에 최소 1장은 남아야 함
        rng.shuffle(rows)
        query_rows, gallery_rows = rows[:k], rows[k:]
        q_groups.append(([DFN_ROOT / r["relpath"] for r in query_rows], dog_id))
        for r in gallery_rows:
            g_paths.append(DFN_ROOT / r["relpath"])
            g_pids.append(dog_id)
            g_cams.append(global_cam)
            global_cam += 1
    return q_groups, g_paths, g_pids, g_cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/E1_resnet50_bnneck_breedpretrain_triplet.pt")
    ap.add_argument("--ks", default="1,2,3,5")
    args = ap.parse_args()

    model, transform = load_model("resnet50_imagenet", args.checkpoint)
    device = get_device()
    model.to(device)

    for k in [int(x) for x in args.ks.split(",")]:
        q_groups, g_paths, g_pids, g_cams = build_split(k)
        n_ids = len(q_groups)
        all_q_paths = [p for paths, _ in q_groups for p in paths]
        feats = extract_features(all_q_paths, model, transform, device, 32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

        q_feats, q_pids, q_cams = [], [], []
        idx = 0
        for gi, (paths, dog_id) in enumerate(q_groups):
            n = len(paths)
            group_feats = feats[idx:idx + n]
            idx += n
            avg = group_feats.mean(axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-12)
            q_feats.append(avg)
            q_pids.append(dog_id)
            q_cams.append(-1 - gi)  # 갤러리 camid와 절대 안 겹치는 음수값 (제외 규칙 비활성화)
        q_feats = np.stack(q_feats)

        g_feats = extract_features(g_paths, model, transform, device, 32)
        g_feats = g_feats / (np.linalg.norm(g_feats, axis=1, keepdims=True) + 1e-12)

        distmat = 1 - (q_feats @ g_feats.T)
        m = evaluate_reid(distmat, np.asarray(q_pids), np.asarray(g_pids),
                           np.asarray(q_cams), np.asarray(g_cams), max_rank=10)
        print(f"[K={k}장 평균 쿼리] 개체수={n_ids} {m}")


if __name__ == "__main__":
    main()
