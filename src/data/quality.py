"""데이터 품질 점검·처리 유틸: 결측치, 이상치(저해상도/손상 이미지), 클래스 불균형."""
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

MIN_USABLE_SHORT_SIDE = 64  # 이 미만이면 "이상치"로 간주해 학습에서 제외


def check_image_integrity(root: Path, relpaths) -> "pd.Series[bool]":
    """이미지가 실제로 열리는지 확인 (손상 파일 = 이상치)."""
    ok = []
    for rp in relpaths:
        p = root / rp
        try:
            with Image.open(p) as im:
                im.verify()
            ok.append(True)
        except (UnidentifiedImageError, OSError, FileNotFoundError):
            ok.append(False)
    return pd.Series(ok)


def flag_low_resolution(df: pd.DataFrame, min_short_side: int = MIN_USABLE_SHORT_SIDE) -> "pd.Series[bool]":
    short_side = df[["width", "height"]].min(axis=1)
    return short_side < min_short_side


def class_balance_weights(labels) -> list:
    """클래스(개체 ID)별 등장 빈도의 역수로 샘플 가중치를 만든다.
    WeightedRandomSampler에 넣으면 희귀 개체가 학습 중 덜 무시되도록 보정된다.
    """
    counts = Counter(labels)
    return [1.0 / counts[l] for l in labels]
