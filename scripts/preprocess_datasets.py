"""세 데이터셋(MPDD, DogFaceNet, 보호소)에 대해 결측치·이상치·클래스 불균형을 실제로 점검·처리하고,
"전처리 후" 매니페스트(`*_clean.csv`)를 만든다. 과제 제출 요건의 "전처리 후 데이터셋"에 대응.

사용:
    python scripts/preprocess_datasets.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.quality import check_image_integrity, flag_low_resolution, MIN_USABLE_SHORT_SIDE


def preprocess_mpdd():
    print("\n=== MPDD ===")
    root = Path("Data/Multi-pose dog dataset/pytorch")
    df = pd.read_csv("metadata/multipose_dog_manifest.csv")
    n0 = len(df)

    # 이상치 1: 손상되어 안 열리는 이미지
    df["is_readable"] = check_image_integrity(root, df.relpath)
    n_broken = (~df.is_readable).sum()
    print(f"[결측치/손상] 안 열리는 이미지: {n_broken}장 -> 제거")

    # 이상치 2: 짧은 변이 너무 작아 사실상 노이즈에 가까운 이미지
    df["is_low_res"] = flag_low_resolution(df, MIN_USABLE_SHORT_SIDE)
    n_lowres = df.is_low_res.sum()
    print(f"[이상치] 짧은 변 {MIN_USABLE_SHORT_SIDE}px 미만: {n_lowres}장 ({n_lowres/n0*100:.1f}%) -> 학습 제외 플래그")

    df["usable_for_training"] = df.is_readable & ~df.is_low_res
    n_removed = (~df.usable_for_training).sum()

    # 클래스 불균형 확인 (train split 기준)
    train_counts = df[(df.split == "train") & df.usable_for_training].dog_id.value_counts()
    print(f"[클래스 불균형] train ID당 이미지 수: min={train_counts.min()} max={train_counts.max()} "
          f"평균={train_counts.mean():.1f} (불균형비 {train_counts.max()/train_counts.min():.1f}배)")

    out = Path("metadata/multipose_dog_manifest_clean.csv")
    df.to_csv(out, index=False)
    print(f"[완료] {out} (원본 {n0}장 중 {n_removed}장 학습제외 플래그, 나머지는 그대로 보존)")
    return df


def preprocess_dogfacenet():
    print("\n=== DogFaceNet ===")
    root = Path("Data/DogFaceNet")
    df = pd.read_csv("metadata/dogfacenet_manifest.csv")
    n0 = len(df)

    df["is_readable"] = check_image_integrity(root, df.relpath)
    n_broken = (~df.is_readable).sum()
    print(f"[결측치/손상] 안 열리는 이미지: {n_broken}장")

    df["is_low_res"] = flag_low_resolution(df, MIN_USABLE_SHORT_SIDE)
    print(f"[이상치] 짧은 변 {MIN_USABLE_SHORT_SIDE}px 미만: {df.is_low_res.sum()}장 (이미 224x224로 정렬돼 거의 없음)")

    df["usable_for_training"] = df.is_readable & ~df.is_low_res

    class_counts = df[df.split_group == "train"].dog_id.value_counts()
    print(f"[클래스 불균형] train 개체당 이미지 수: min={class_counts.min()} max={class_counts.max()} "
          f"중앙값={class_counts.median():.0f} -> 불균형비 {class_counts.max()/class_counts.min():.0f}배")
    print("  대응: (a) E2 Prototypical Network는 에피소드마다 클래스당 K장을 균일 샘플링하므로 "
          "개체별 총 보유 이미지 수 차이에 원천적으로 덜 민감함")
    print("        (b) E1 CE 파인튜닝은 src.data.quality.class_balance_weights()로 만든 "
          "역빈도 가중치를 WeightedRandomSampler에 사용해 희귀 개체 샘플링 확률을 보정")

    out = Path("metadata/dogfacenet_manifest_clean.csv")
    df.to_csv(out, index=False)
    print(f"[완료] {out}")
    return df


def preprocess_shelter():
    print("\n=== 보호소(animal.go.kr) ===")
    path = Path("Data/shelter/shelter_manifest.csv")
    if not path.exists():
        print("[skip] Data/shelter/shelter_manifest.csv 없음 (fetch_shelter_api.py 먼저 실행 필요)")
        return None
    df = pd.read_csv(path, dtype=str).fillna("")
    n0 = len(df)

    # 결측치 점검: 텍스트 필드가 빈 문자열인 경우
    text_cols = ["kind_nm", "color_cd", "age", "weight", "special_mark", "care_tel"]
    missing_report = {c: int((df[c].str.strip() == "").sum()) for c in text_cols}
    print("[결측치] 컬럼별 빈 값 개수:")
    for c, n in missing_report.items():
        print(f"    {c:15s} {n:4d}/{n0} ({n/n0*100:.1f}%)")

    # 결측치 처리: 표시용 텍스트 필드는 "정보없음"으로 채움 (모델 입력이 아니라 데모 UI 표시용이라 안전)
    for c in ["kind_nm", "color_cd", "age", "weight", "special_mark"]:
        df[c] = df[c].replace("", "정보없음")

    # 이상치/결측 이미지: 사진이 한 장도 안 받아진 공고는 데모 갤러리에 못 씀 -> 제거
    n_no_photo = (df["photo_local_paths"].str.strip() == "").sum()
    df_clean = df[df["photo_local_paths"].str.strip() != ""].copy()
    print(f"[이상치] 사진 다운로드 실패로 이미지가 하나도 없는 공고: {n_no_photo}건 -> 제거")

    # 클래스(품종) 불균형 확인
    breed_counts = df_clean.kind_nm.value_counts()
    print(f"[클래스 불균형] 품종 종류 {len(breed_counts)}개, 최다 '{breed_counts.index[0]}' "
          f"{breed_counts.iloc[0]}건 vs 최소 1건 -> 하드 필터 대신 소프트 신호로만 사용 (기존 결정 유지)")

    out = Path("metadata/shelter_manifest_clean.csv")
    df_clean.to_csv(out, index=False)
    print(f"[완료] {out} ({n0}건 -> {len(df_clean)}건)")
    return df_clean


if __name__ == "__main__":
    preprocess_mpdd()
    preprocess_dogfacenet()
    preprocess_shelter()
