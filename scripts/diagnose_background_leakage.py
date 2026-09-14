"""E0(ImageNet, 학습 없음)가 개체 정체성이 아니라 배경/촬영세션을 외워서 맞히는 건 아닌지 진단.

아이디어: MPDD는 카메라(camera) 컬럼이 촬영 세션(=배경) 대리 지표다.
    - "같은 개체, 다른 카메라" 쌍의 평균 유사도  (Re-ID가 원하는 진짜 신호)
    - "다른 개체, 같은 카메라" 쌍의 평균 유사도  (배경이 만드는 가짜 신호일 위험)
    - "다른 개체, 다른 카메라" 쌍의 평균 유사도  (아무 관련 없는 대조군, 가장 낮아야 정상)

만약 "다른 개체, 같은 카메라"가 "같은 개체, 다른 카메라"에 근접하거나 더 높다면,
E0의 높은 점수가 배경(카메라) 힌트에 크게 의존한다는 뜻이다.

사용:
    python scripts/diagnose_background_leakage.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.manifests import load_mpdd_split
from src.models.backbones import build_backbone, get_device
from src.retrieval.extract import extract_features


def pair_similarity_stats(feats: np.ndarray, pids: np.ndarray, camids: np.ndarray, sample_pairs: int = 200_000):
    n = len(feats)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    sim = feats @ feats.T

    rng = np.random.default_rng(0)
    i = rng.integers(0, n, size=sample_pairs)
    j = rng.integers(0, n, size=sample_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    same_pid = pids[i] == pids[j]
    same_cam = camids[i] == camids[j]

    groups = {
        "같은 개체 · 다른 카메라 (Re-ID 신호)": same_pid & ~same_cam,
        "다른 개체 · 같은 카메라 (배경 위험 신호)": ~same_pid & same_cam,
        "다른 개체 · 다른 카메라 (대조군)": ~same_pid & ~same_cam,
        "같은 개체 · 같은 카메라 (참고, 평가에선 제외되는 쌍)": same_pid & same_cam,
    }
    s = sim[i, j]
    print(f"(샘플 페어 {len(i)}개 기준)\n")
    stats = {}
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        vals = s[mask]
        stats[name] = (vals.mean(), vals.std(), mask.sum())
        print(f"  {name:38s} n={mask.sum():6d}  평균 유사도={vals.mean():.4f}  (표준편차 {vals.std():.4f})")
    return stats


def main():
    split = load_mpdd_split()
    all_paths = split.query_paths + split.gallery_paths
    all_pids = np.asarray(split.query_pids + split.gallery_pids)
    all_camids = np.asarray(split.query_camids + split.gallery_camids)

    model, transform = build_backbone("resnet50_imagenet")
    device = get_device()
    feats = extract_features(all_paths, model, transform, device, batch_size=32)

    print("=== MPDD (query+gallery 합쳐서 분석, E0 특징 기준) ===\n")
    stats = pair_similarity_stats(feats, all_pids, all_camids)

    same_id_diff_cam = stats.get("같은 개체 · 다른 카메라 (Re-ID 신호)")
    diff_id_same_cam = stats.get("다른 개체 · 같은 카메라 (배경 위험 신호)")
    diff_id_diff_cam = stats.get("다른 개체 · 다른 카메라 (대조군)")

    print("\n=== 해석 ===")
    if same_id_diff_cam and diff_id_same_cam:
        gap_signal = same_id_diff_cam[0] - diff_id_diff_cam[0]
        gap_leak = diff_id_same_cam[0] - diff_id_diff_cam[0]
        print(f"진짜 개체 신호 크기(같은개체-대조군): {gap_signal:+.4f}")
        print(f"배경(카메라) 유출 크기(같은카메라-대조군): {gap_leak:+.4f}")
        if gap_leak > gap_signal * 0.5:
            print("[경고] 배경(카메라)이 개체 신호의 절반 이상 크기로 유사도를 끌어올리고 있음 "
                  "-> E0 성능이 배경 힌트에 상당히 의존할 위험이 있음.")
        else:
            print("[안심] 배경(카메라) 유출 크기가 개체 신호에 비해 작음 "
                  "-> E0 성능이 배경보다는 실제 개체 특징에서 나온다고 볼 수 있음.")


if __name__ == "__main__":
    main()
