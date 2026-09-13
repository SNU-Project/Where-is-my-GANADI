"""이미지 경로 리스트 -> 임베딩 특징 행렬 (배치 추론)."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImagePathDataset(Dataset):
    def __init__(self, paths: list, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def extract_features(
    paths: list,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
) -> np.ndarray:
    model.eval().to(device)
    ds = ImagePathDataset(paths, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    feats = []
    for batch in dl:
        batch = batch.to(device)
        out = model(batch)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)
