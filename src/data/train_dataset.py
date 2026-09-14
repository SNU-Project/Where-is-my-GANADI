"""E1/E2 학습용 데이터셋: MPDD + DogFaceNet train을 하나의 ID 공간으로 합친다.

두 데이터셋은 서로 다른 실제 개체 집합이라 (겹치는 개체 없음) ID 네임스페이스만
분리하면 그대로 합쳐서 학습할 수 있다 (D1.6에서 결정한 내용).
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

MPDD_ROOT = Path("Data/Multi-pose dog dataset/pytorch")
DFN_ROOT = Path("Data/DogFaceNet")


@dataclass
class LabeledImage:
    path: Path
    global_label: int
    source: str  # "mpdd" | "dogfacenet"


class IdentityImageDataset(Dataset):
    """(이미지, 전역 라벨) 쌍. WeightedRandomSampler용 라벨 리스트도 제공."""

    def __init__(self, items: list, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item.path).convert("RGB")
        return self.transform(img), item.global_label

    @property
    def labels(self) -> list:
        return [it.global_label for it in self.items]


def _load_mpdd(split_name: str):
    df = pd.read_csv("metadata/multipose_dog_manifest_clean.csv")
    if split_name == "train":
        df = df[(df.split == "train") & df.usable_for_training]
    else:
        df = df[df.split == split_name]
    return df, MPDD_ROOT


def _load_dogfacenet(split_name: str):
    df = pd.read_csv("metadata/dogfacenet_manifest_clean.csv")
    df = df[df.split_group == split_name]
    return df, DFN_ROOT


def build_combined_train_set(transform):
    """MPDD train + DogFaceNet train -> 전역 라벨로 합친 학습 데이터셋.

    반환: (dataset, num_classes, label_map)
    label_map: {(source, 원본 dog_id): 전역 라벨} — 나중에 val/test 매핑 확인용으로 남겨둠.
    """
    mpdd_df, mpdd_root = _load_mpdd("train")
    dfn_df, dfn_root = _load_dogfacenet("train")

    label_map = {}
    items = []
    for _, r in mpdd_df.iterrows():
        key = ("mpdd", r.dog_id)
        if key not in label_map:
            label_map[key] = len(label_map)
        items.append(LabeledImage(mpdd_root / r.relpath, label_map[key], "mpdd"))
    for _, r in dfn_df.iterrows():
        key = ("dogfacenet", r.dog_id)
        if key not in label_map:
            label_map[key] = len(label_map)
        items.append(LabeledImage(dfn_root / r.relpath, label_map[key], "dogfacenet"))

    ds = IdentityImageDataset(items, transform)
    return ds, len(label_map), label_map


def build_mpdd_val_closed_set(transform, label_map: dict):
    """MPDD val(=train과 같은 95 ID)을 학습 때 쓴 전역 라벨로 매핑 -> CE loss/accuracy 모니터링용.
    label_map에 없는(=train에서 저해상도로 빠진) ID의 val 이미지는 제외한다.
    """
    df, root = _load_mpdd("val")
    items = []
    for _, r in df.iterrows():
        key = ("mpdd", r.dog_id)
        if key in label_map:
            items.append(LabeledImage(root / r.relpath, label_map[key], "mpdd"))
    return IdentityImageDataset(items, transform)
