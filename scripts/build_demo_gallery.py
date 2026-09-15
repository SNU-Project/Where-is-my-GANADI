"""데모 앱용 갤러리 임베딩을 미리 계산해 캐시한다 (앱 시작마다 재계산하지 않도록).

사용:
    python scripts/build_demo_gallery.py --checkpoint checkpoints/E1_resnet50_bnneck.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

from src.models.backbones import get_device
from src.retrieval.color import color_histogram
from src.retrieval.extract import extract_features

CACHE_DIR = Path("demo_cache")
SHELTER_ROOT = Path("Data/shelter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/E1_resnet50_bnneck.pt")
    ap.add_argument("--manifest", default="metadata/shelter_manifest_clean.csv")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_reid import load_model  # noqa: E402

    df = pd.read_csv(args.manifest)

    rows = []
    for _, r in df.iterrows():
        photos = [p for p in str(r.photo_local_paths).split("|") if p]
        for p in photos:
            rows.append({"desertion_no": r.desertion_no, "relpath": p})
    photo_df = pd.DataFrame(rows)
    print(f"[info] 갤러리 사진 {len(photo_df)}장 / 공고 {photo_df.desertion_no.nunique()}건")

    model, transform = load_model("resnet50_imagenet", args.checkpoint)
    device = get_device()
    paths = [SHELTER_ROOT / p for p in photo_df.relpath]
    feats = extract_features(paths, model, transform, device, args.batch_size)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)  # L2 정규화 저장

    print("[info] 색상 히스토그램 계산 중 (임베딩만으로는 안 잡히는 색깔 유사도 보완, D5 피드백 반영)...")
    hists = np.stack([color_histogram(Image.open(p).convert("RGB")) for p in paths])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / "gallery_embeddings.npy", feats)
    np.save(CACHE_DIR / "gallery_color_hists.npy", hists)
    photo_df.to_csv(CACHE_DIR / "gallery_index.csv", index=False)
    df.to_csv(CACHE_DIR / "gallery_meta.csv", index=False)
    print(f"[완료] {CACHE_DIR}/gallery_embeddings.npy ({feats.shape}), "
          f"gallery_color_hists.npy ({hists.shape}), gallery_index.csv, gallery_meta.csv")


if __name__ == "__main__":
    main()
