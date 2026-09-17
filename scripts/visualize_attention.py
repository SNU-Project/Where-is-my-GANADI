"""정성분석용 활성화 히트맵 — 임베딩이 이미지의 어느 영역에 반응하는지 시각화.

방법: BNNeckModel.features의 마지막 conv 특징맵(GAP 직전, 2048채널×H×W)에서 채널 방향
L2 norm을 구해 공간적 "반응 강도" 히트맵을 만든다. 진짜 Grad-CAM(분류 타깃에 대한 역전파)은
open-set 검색 모델엔 고정 타깃이 없어 적용이 애매하므로, 임베딩 크기 기반의 간단하고 해석하기
쉬운 대안을 쓴다 — "이 위치가 최종 임베딩 크기에 얼마나 기여하는지"를 직접 보여준다.

사용:
    python scripts/visualize_attention.py --image path/to/img.jpg --out docs/eda/xx.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.transforms import build_eval_transform
from src.models.backbones import get_device

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_reid import load_model  # noqa: E402


def activation_heatmap(model, img: Image.Image, transform, device):
    x = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_map = model.features[:-1](x)  # (1, 2048, H, W) -- avgpool 직전
        heat = feat_map.pow(2).sum(dim=1, keepdim=True).sqrt()  # 채널 L2 norm -> (1,1,H,W)
        heat = F.interpolate(heat, size=img.size[::-1], mode="bilinear", align_corners=False)
    heat = heat[0, 0].cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
    return heat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/E1_resnet50_bnneck_breedpretrain_triplet.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    model, transform = load_model("resnet50_imagenet", args.checkpoint)
    device = get_device()
    model.to(device)

    img = Image.open(args.image).convert("RGB")
    heat = activation_heatmap(model, img, transform, device)

    plt.rcParams["font.family"] = "AppleGothic"
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    axes[0].imshow(img)
    axes[0].set_title("원본")
    axes[0].axis("off")
    axes[1].imshow(img)
    axes[1].imshow(heat, cmap="jet", alpha=0.5)
    axes[1].set_title("임베딩 활성화 히트맵\n(밝을수록 임베딩에 기여 큼)")
    axes[1].axis("off")
    if args.title:
        fig.suptitle(args.title)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"[완료] {args.out}")


if __name__ == "__main__":
    main()
