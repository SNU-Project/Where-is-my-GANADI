"""색상 히스토그램 유사도 — 임베딩만으로는 "정답 아닌 후보들의 순서"가 그럴듯하게
정렬되지 않는 문제(D5 이후 피드백)를 보완한다. 딥러닝 임베딩은 진짜 정답(같은 개체)을
찾아내는 데는 강하지만, 오답 후보끼리의 순위는 학습 신호가 없어 색깔이 달라도 높게 나올 수 있다.
"""
import numpy as np
from PIL import Image

HUE_BINS = 12
SAT_BINS = 6
VAL_BINS = 6


def color_histogram(img: Image.Image, size: int = 64) -> np.ndarray:
    """HSV 3D(색상×채도×밝기) 히스토그램.

    원래는 밝기(V)를 뺐었다("조명 차이에 민감해서") — 그런데 채도(S)만으로는 흰색/회색/검정색이
    전부 "저채도"로 뭉쳐 서로 구분이 안 됨(실사용 사진 진단에서 확인: 회색 푸들 쿼리가 흰색 개들에게
    전부 밀림). V를 다시 넣어 무채색 사이의 밝기 차이(흰색 vs 회색 vs 검정)를 구분하게 한다."""
    hsv = np.asarray(img.convert("HSV").resize((size, size)))
    h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    hist, _ = np.histogramdd([h, s, v], bins=[HUE_BINS, SAT_BINS, VAL_BINS],
                              range=[[0, 255], [0, 255], [0, 255]])
    hist = hist / (hist.sum() + 1e-9)
    return hist.ravel()


def histogram_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """히스토그램 교집합(intersection) — 0(완전 다름)~1(완전 동일)."""
    return float(np.minimum(h1, h2).sum())


def blended_score(embedding_sim: np.ndarray, color_sim: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """최종 점수 = alpha*임베딩유사도 + (1-alpha)*색상유사도.
    alpha를 높게 둬 임베딩(개체 식별 능력)을 주로 따르되, 색상으로 명백한 오답을 눌러준다."""
    return alpha * embedding_sim + (1 - alpha) * color_sim
