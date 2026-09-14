"""E1: ResNet50 + BNNeck + CrossEntropy 파인튜닝 (MPDD + DogFaceNet 통합 학습).

- 학습 데이터: MPDD train(저해상도 제외) + DogFaceNet train, 전역 ID로 통합 (src/data/train_dataset.py)
- 클래스 불균형 대응: WeightedRandomSampler (역빈도 가중치)
- 검증(두 가지 성격):
    - MPDD val (train과 같은 개체, closed-set) -> 분류 정확도로 "학습이 수렴하는지" 확인
    - DogFaceNet val (train과 분리된 개체, open-set) -> Rank-1로 "새 개체 일반화" 확인, 체크포인트 선택 기준
- 최고 체크포인트: checkpoints/E1_resnet50_bnneck.pt (gitignore 처리됨, 용량 문제로 커밋 안 함)
- 로그: metadata/train_log_E1.csv

사용:
    python scripts/train_embed.py --epochs 15
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.manifests import load_dogfacenet_split
from src.data.quality import class_balance_weights
from src.data.train_dataset import build_combined_train_set, build_mpdd_val_closed_set
from src.data.transforms import build_train_transform, build_eval_transform
from src.models.backbones import BNNeckModel, get_device
from src.retrieval.extract import extract_features
from src.retrieval.metrics import compute_distmat, evaluate_reid

CKPT_DIR = Path("checkpoints")
LOG_CSV = Path("metadata/train_log_E1.csv")


@torch.no_grad()
def evaluate_mpdd_val_closed_set(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _, logits = model.forward_with_logits(imgs)
        loss_sum += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_dogfacenet_val_openset(model, transform, device, batch_size):
    split = load_dogfacenet_split(group="val")
    q_feats = extract_features(split.query_paths, model, transform, device, batch_size)
    g_feats = extract_features(split.gallery_paths, model, transform, device, batch_size)
    distmat = compute_distmat(q_feats, g_feats)
    return evaluate_reid(
        distmat,
        np.asarray(split.query_pids), np.asarray(split.gallery_pids),
        np.asarray(split.query_camids), np.asarray(split.gallery_camids),
        max_rank=10,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    print(f"[info] device={device}")

    train_transform = build_train_transform()
    eval_transform = build_eval_transform()

    train_ds, num_classes, label_map = build_combined_train_set(train_transform)
    mpdd_val_ds = build_mpdd_val_closed_set(eval_transform, label_map)
    print(f"[info] train: {len(train_ds)}장 / {num_classes}개체, MPDD val(closed-set): {len(mpdd_val_ds)}장")

    weights = class_balance_weights(train_ds.labels)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers)
    mpdd_val_loader = DataLoader(mpdd_val_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)

    model = BNNeckModel(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    log_rows = []
    best_rank1 = -1.0
    best_path = CKPT_DIR / "E1_resnet50_bnneck.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += len(labels)
        scheduler.step()

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        val_loss, val_acc = evaluate_mpdd_val_closed_set(model, mpdd_val_loader, device)
        dfn_val_metrics = evaluate_dogfacenet_val_openset(model, eval_transform, device, args.batch_size)

        elapsed = time.time() - t0
        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
              f"MPDD val_loss={val_loss:.3f} val_acc={val_acc:.3f} | "
              f"DogFaceNet val {dfn_val_metrics} | {elapsed:.1f}s")

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "mpdd_val_loss": val_loss, "mpdd_val_acc": val_acc,
            "dfn_val_rank1": dfn_val_metrics.rank1, "dfn_val_map": dfn_val_metrics.mAP,
            "seconds": round(elapsed, 1),
        })

        if dfn_val_metrics.rank1 > best_rank1:
            best_rank1 = dfn_val_metrics.rank1
            torch.save({"model": model.state_dict(), "num_classes": num_classes,
                        "epoch": epoch, "dfn_val_rank1": best_rank1}, best_path)
            print(f"  [저장] 새 최고 DogFaceNet val Rank-1={best_rank1:.4f} -> {best_path}")

    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n[완료] 로그: {LOG_CSV}, 최고 체크포인트: {best_path} (DogFaceNet val Rank-1={best_rank1:.4f})")


if __name__ == "__main__":
    main()
