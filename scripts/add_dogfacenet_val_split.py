"""DogFaceNet manifest에 val(검증) 세트를 추가한다.

기존엔 train/test 2단계뿐이었다 (test 278 ID는 이미 E0 결과 산출에 쓰여 그대로 유지해야 함).
train 1,115 ID 중 일부(기본 15%)를 떼어 val로 만든다 — test와는 무관하게, train 내부에서만
재분할하므로 기존에 보고한 E0 test 수치는 바뀌지 않는다.

사용:
    python scripts/add_dogfacenet_val_split.py
"""
import random
from pathlib import Path

import pandas as pd

MANIFEST = Path("metadata/dogfacenet_manifest.csv")
VAL_RATIO_OF_TRAIN = 0.15
SEED = 7  # build_dogfacenet_manifest.py의 SEED(42)와 다른 값을 써서 독립적인 재분할임을 명시


def main():
    df = pd.read_csv(MANIFEST)
    if (df.split_group == "val").any():
        print("[skip] 이미 val이 존재합니다. 다시 만들려면 metadata/dogfacenet_manifest.csv를 "
              "scripts/build_dogfacenet_manifest.py로 재생성한 뒤 이 스크립트를 실행하세요.")
        return

    train_ids = sorted(df.loc[df.split_group == "train", "dog_id"].unique())
    rng = random.Random(SEED)
    rng.shuffle(train_ids)
    n_val = int(len(train_ids) * VAL_RATIO_OF_TRAIN)
    val_ids = set(train_ids[:n_val])

    df.loc[df.dog_id.isin(val_ids) & (df.split_group == "train"), "split_group"] = "val"
    df.to_csv(MANIFEST, index=False)

    counts = df.groupby("split_group").dog_id.nunique()
    print(f"[완료] {MANIFEST} 갱신")
    print(counts)
    print(f"train ∩ val ID 중복: "
          f"{len(set(df[df.split_group=='train'].dog_id) & set(df[df.split_group=='val'].dog_id))} "
          f"(0이어야 함)")
    print(f"train ∩ test ID 중복: "
          f"{len(set(df[df.split_group=='train'].dog_id) & set(df[df.split_group=='test'].dog_id))} "
          f"(0이어야 함, test는 원래부터 안 건드림)")


if __name__ == "__main__":
    main()
