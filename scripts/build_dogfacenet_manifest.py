"""DogFaceNet(HuggingFace: dimidagd/DogFaceNet_224resize) 다운로드 + 매니페스트 CSV 생성.

- 이미지: 이미 224x224로 정렬된 강아지 얼굴 크롭, 1,393개체 / 8,363장
- 공식 train/test 분할이 없어 개체(ID) 기준 80/20 분할을 직접 만든다 (open-set, ID 겹침 없음).
- 승인/신청 불필요, MIT 라이선스.

사용:
    python scripts/build_dogfacenet_manifest.py
"""
import csv
import random
from pathlib import Path

from datasets import load_dataset

OUT_ROOT = Path("Data/DogFaceNet")
IMG_DIR = OUT_ROOT / "images"
MANIFEST = Path("metadata/dogfacenet_manifest.csv")
TEST_ID_RATIO = 0.2
SEED = 42


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    print("[info] Hugging Face에서 DogFaceNet 다운로드 중...")
    ds = load_dataset("dimidagd/DogFaceNet_224resize", split="train")
    label_names = ds.features["label"].names
    print(f"[info] 총 {len(ds)}장, {len(label_names)}개체")

    # ID 기준 open-set split
    rng = random.Random(SEED)
    ids = list(range(len(label_names)))
    rng.shuffle(ids)
    n_test = int(len(ids) * TEST_ID_RATIO)
    test_ids = set(ids[:n_test])

    rows = []
    per_id_counter = {}
    for i, ex in enumerate(ds):
        label = ex["label"]
        per_id_counter[label] = per_id_counter.get(label, 0) + 1
        seq = per_id_counter[label]
        fname = f"{label}_{seq}.jpg"
        split_group = "test" if label in test_ids else "train"
        fpath = IMG_DIR / fname
        if not fpath.exists():
            img = ex["image"].convert("RGB")
            img.save(fpath, quality=95)
        w, h = ex["image"].size
        rows.append({
            "split_group": split_group,
            "filename": fname,
            "relpath": f"images/{fname}",
            "dog_id": label,
            "width": w,
            "height": h,
        })
        if (i + 1) % 1000 == 0:
            print(f"[info] {i + 1}/{len(ds)} 처리")

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split_group", "filename", "relpath", "dog_id", "width", "height"])
        writer.writeheader()
        writer.writerows(rows)

    n_train_ids = len(ids) - n_test
    print(f"\n[완료] {MANIFEST} ({len(rows)}행)")
    print(f"  train ids={n_train_ids}  test ids={n_test}  (open-set, 서로 겹치지 않음)")
    print(f"  이미지 저장 위치: {IMG_DIR}")


if __name__ == "__main__":
    main()
