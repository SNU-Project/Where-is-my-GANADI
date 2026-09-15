"""Re-ID 평가 러너: 백본으로 쿼리/갤러리 특징을 뽑아 CMC/mAP를 계산한다.

E0(학습 없음, ImageNet 특징) 기본 실행:
    python scripts/eval_reid.py --dataset mpdd
    python scripts/eval_reid.py --dataset dogfacenet
    python scripts/eval_reid.py --dataset both

학습된 체크포인트(E1 등) 평가:
    python scripts/eval_reid.py --dataset both --checkpoint checkpoints/E1_resnet50_bnneck.pt --exp-name E1_ce_bnneck

결과는 metadata/results.csv 에 누적 기록된다 (실험명, 데이터셋, 지표).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.manifests import load_dogfacenet_split, load_mpdd_split, load_shelter_split
from src.data.transforms import build_eval_transform
from src.models.backbones import BNNeckModel, ProtoEmbedder, build_backbone, get_device
from src.retrieval.extract import extract_features
from src.retrieval.metrics import compute_distmat, evaluate_reid

RESULTS_CSV = Path("metadata/results.csv")
RESULTS_FIELDS = [
    "exp_name", "dataset", "backbone", "rank1", "rank5", "rank10", "mAP",
    "num_valid_queries", "num_queries", "seconds",
]


def load_model(backbone_name: str, checkpoint: str | None):
    if checkpoint is None:
        return build_backbone(backbone_name)
    ckpt = torch.load(checkpoint, map_location="cpu")
    if "num_classes" in ckpt:  # E1: BNNeckModel (분류기 포함)
        model = BNNeckModel(num_classes=ckpt["num_classes"], pretrained=False)
    else:  # E2: ProtoEmbedder (분류기 없음, 임베딩만)
        model = ProtoEmbedder(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step_key = "epoch" if "epoch" in ckpt else "episode"
    print(f"[info] 체크포인트 로드: {checkpoint} ({step_key}={ckpt.get(step_key)}, "
          f"학습 시 DogFaceNet val Rank-1={ckpt.get('dfn_val_rank1'):.4f})")
    return model, build_eval_transform()


def run_one(dataset: str, backbone_name: str, exp_name: str, batch_size: int = 32,
            checkpoint: str | None = None):
    print(f"\n=== {exp_name} on {dataset} ===")
    t0 = time.time()

    if dataset == "mpdd":
        split = load_mpdd_split()
    elif dataset == "dogfacenet":
        split = load_dogfacenet_split()
    elif dataset == "shelter":
        split = load_shelter_split()
    else:
        raise ValueError(dataset)

    print(f"[info] query={len(split.query_paths)}  gallery={len(split.gallery_paths)}")

    model, transform = load_model(backbone_name, checkpoint)
    device = get_device()
    print(f"[info] device={device}")

    q_feats = extract_features(split.query_paths, model, transform, device, batch_size)
    g_feats = extract_features(split.gallery_paths, model, transform, device, batch_size)

    distmat = compute_distmat(q_feats, g_feats)
    import numpy as np

    metrics = evaluate_reid(
        distmat,
        np.asarray(split.query_pids),
        np.asarray(split.gallery_pids),
        np.asarray(split.query_camids),
        np.asarray(split.gallery_camids),
        max_rank=10,
    )
    elapsed = time.time() - t0
    print(metrics)
    print(f"[info] 소요 시간: {elapsed:.1f}s")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if write_header:
            writer.writeheader()
        row = metrics.as_dict()
        row.update({
            "exp_name": exp_name,
            "dataset": dataset,
            "backbone": backbone_name,
            "seconds": round(elapsed, 1),
        })
        writer.writerow(row)
    print(f"[완료] {RESULTS_CSV}에 기록")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mpdd", "dogfacenet", "shelter", "both"], default="both")
    ap.add_argument("--backbone", default="resnet50_imagenet")
    ap.add_argument("--exp-name", default="E0_imagenet_baseline")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--checkpoint", default=None, help="학습된 체크포인트 경로 (예: checkpoints/E1_resnet50_bnneck.pt)")
    args = ap.parse_args()

    datasets = ["mpdd", "dogfacenet"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_one(ds, args.backbone, args.exp_name, args.batch_size, args.checkpoint)


if __name__ == "__main__":
    main()
