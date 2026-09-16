"""1단계(E3): ResNet50을 Stanford Dogs(120품종) 분류로 먼저 파인튜닝해, 순수 ImageNet 대신
"개 품종을 구분하는 법"을 이미 아는 백본에서 identity 학습을 시작하게 한다.

배경: 실제 사진(스마트폰, 가정집 배경) 쿼리 진단에서, 지금 E1(ImageNet init + identity CE만)은
품종/색깔 유사성을 사실상 배우지 못한다는 게 확인됨 — identity 손실만으로는 "비슷하게 생긴 개"를
구분할 이유가 전혀 없기 때문. Stanford Dogs는 MPDD/DogFaceNet보다 훨씬 다양한 촬영 조건·품종을
담고 있어, 이 사전학습 단계가 도메인 다양성과 품종 신호를 동시에 주입하는 역할을 한다.

산출물: checkpoints/breed_pretrained_resnet50_features.pt
  -> src/models/backbones.py의 BNNeckModel(init_backbone_path=...)로 이어서 identity 파인튜닝
     (scripts/train_embed.py --init-backbone checkpoints/breed_pretrained_resnet50_features.pt)

사용:
    python scripts/pretrain_breed_backbone.py --epochs 15
"""
import argparse
import io
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

from src.data.transforms import build_eval_transform, build_train_transform
from src.models.backbones import get_device

PARQUET_DIR = Path("Data/StanfordDogs/parquet")
CKPT_OUT = Path("checkpoints/breed_pretrained_resnet50_features.pt")
LOG_CSV = Path("metadata/train_log_breed_pretrain.csv")


class ParquetImageDataset(Dataset):
    """HF parquet(image bytes + label 컬럼)을 그대로 메모리에 올려 쓰는 단순 Dataset.
    2만 장 규모라 디코딩된 PIL 대신 원본 바이트만 들고 있다가 __getitem__에서 디코딩(메모리 절약)."""

    def __init__(self, df: pd.DataFrame, transform):
        self.img_bytes = df["pixel_values"].apply(lambda x: x["bytes"] if isinstance(x, dict) else x).tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self.img_bytes[i])).convert("RGB")
        return self.transform(img), self.labels[i]


def load_parquet_splits():
    train_df = pd.concat([
        pd.read_parquet(PARQUET_DIR / "train_0.parquet"),
        pd.read_parquet(PARQUET_DIR / "train_1.parquet"),
    ], ignore_index=True)
    test_df = pd.read_parquet(PARQUET_DIR / "test_0.parquet")
    return train_df, test_df


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    print(f"[info] device={device}")

    train_df, test_df = load_parquet_splits()
    num_classes = int(pd.concat([train_df["label"], test_df["label"]]).max()) + 1
    print(f"[info] train={len(train_df)}장 test={len(test_df)}장 품종수={num_classes}")

    train_ds = ParquetImageDataset(train_df, build_train_transform())
    test_ds = ParquetImageDataset(test_df, build_eval_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    features = nn.Sequential(*list(backbone.children())[:-1])  # avgpool까지, fc 제외
    classifier = nn.Linear(2048, num_classes)
    model = nn.Sequential(features, nn.Flatten(1), classifier).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    log_rows = []
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        scheduler.step()

        train_loss, train_acc = loss_sum / total, correct / total
        test_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
              f"test_acc={test_acc:.3f} | {elapsed:.1f}s")
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                          "test_acc": test_acc, "seconds": round(elapsed, 1)})

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(features.state_dict(), CKPT_OUT)
            print(f"  [저장] 새 최고 test_acc={best_acc:.4f} -> {CKPT_OUT}")

    import csv
    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n[완료] 로그: {LOG_CSV}, 최고 백본: {CKPT_OUT} (품종분류 test_acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
