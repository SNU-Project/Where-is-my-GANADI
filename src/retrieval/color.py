"""색상 히스토그램 유사도 — 임베딩만으로는 "정답 아닌 후보들의 순서"가 그럴듯하게
정렬되지 않는 문제(D5 이후 피드백)를 보완한다. 딥러닝 임베딩은 진짜 정답(같은 개체)을
찾아내는 데는 강하지만, 오답 후보끼리의 순위는 학습 신호가 없어 색깔이 달라도 높게 나올 수 있다.
"""
import numpy as np
from PIL import Image

HUE_BINS = 16
SAT_BINS = 8


def color_histogram(img: Image.Image, size: int = 64) -> np.ndarray:
    """HSV 2D(색상×채도) 히스토그램. 채도를 같이 써서 흰색/검정/회색(무채색)과
    실제 유채색을 구분한다. 밝기(V)는 조명 차이에 민감해 제외."""
    hsv = np.asarray(img.convert("HSV").resize((size, size)))
    h, s = hsv[..., 0].ravel(), hsv[..., 1].ravel()
    hist, _, _ = np.histogram2d(h, s, bins=[HUE_BINS, SAT_BINS], range=[[0, 255], [0, 255]])
    hist = hist / (hist.sum() + 1e-9)
    return hist.ravel()


def histogram_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """히스토그램 교집합(intersection) — 0(완전 다름)~1(완전 동일)."""
    return float(np.minimum(h1, h2).sum())


def blended_score(embedding_sim: np.ndarray, color_sim: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """최종 점수 = alpha*임베딩유사도 + (1-alpha)*색상유사도.
    alpha를 높게 둬 임베딩(개체 식별 능력)을 주로 따르되, 색상으로 명백한 오답을 눌러준다."""
    return alpha * embedding_sim + (1 - alpha) * color_sim
