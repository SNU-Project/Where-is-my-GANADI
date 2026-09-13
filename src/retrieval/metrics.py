"""Re-ID 평가지표: CMC (Rank-k) / mAP.

Market-1501 프로토콜을 따른다 (Zheng et al., 2015):
쿼리와 (pid, camid)가 모두 같은 갤러리 항목은 "같은 카메라에서 찍힌 자기 자신"으로 간주해
평가에서 제외한다. camid를 이미지마다 전부 다르게 주면(예: 행 인덱스) 이 제외 규칙이
사실상 비활성화된다 — 카메라 개념이 없는 데이터셋(DogFaceNet 등)에 사용.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ReidMetrics:
    rank1: float
    rank5: float
    rank10: float
    mAP: float
    num_valid_queries: int
    num_queries: int

    def as_dict(self) -> dict:
        return {
            "rank1": self.rank1,
            "rank5": self.rank5,
            "rank10": self.rank10,
            "mAP": self.mAP,
            "num_valid_queries": self.num_valid_queries,
            "num_queries": self.num_queries,
        }

    def __str__(self) -> str:
        return (
            f"Rank-1={self.rank1:.4f}  Rank-5={self.rank5:.4f}  Rank-10={self.rank10:.4f}  "
            f"mAP={self.mAP:.4f}  (유효 쿼리 {self.num_valid_queries}/{self.num_queries})"
        )


def compute_distmat(query_feats: np.ndarray, gallery_feats: np.ndarray) -> np.ndarray:
    """L2 정규화된 특징 기준 코사인 거리(1 - cosine similarity) 행렬. shape (nq, ng)."""
    q = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-12)
    g = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-12)
    sim = q @ g.T
    return 1.0 - sim


def evaluate_reid(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 10,
) -> ReidMetrics:
    """Market-1501 스타일 CMC/mAP 계산.

    각 쿼리마다:
      1) 갤러리를 거리순으로 정렬
      2) (pid, camid)가 쿼리와 완전히 같은 항목 제거 (자기 자신/동일 카메라 중복 컷)
      3) 남은 랭킹에서 첫 정답 등장 위치로 CMC 갱신, precision-recall로 AP 계산
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            # 이 쿼리 개체가 갤러리에 (다른 카메라로도) 아예 없음 -> 평가 대상에서 제외
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        precision_at_k = tmp_cmc / (np.arange(len(raw_cmc)) + 1)
        ap = (precision_at_k * raw_cmc).sum() / num_rel
        all_ap.append(ap)

    if num_valid_q == 0:
        raise RuntimeError(
            "평가 가능한 쿼리가 하나도 없습니다 (모든 쿼리 개체가 갤러리에서 카메라 제외 규칙에 걸림). "
            "camid 부여 방식을 확인하세요."
        )

    all_cmc = np.asarray(all_cmc, dtype=np.float32).sum(axis=0) / num_valid_q
    mAP = float(np.mean(all_ap))

    def rank(k):
        return float(all_cmc[k - 1]) if k <= len(all_cmc) else float(all_cmc[-1])

    return ReidMetrics(
        rank1=rank(1),
        rank5=rank(5),
        rank10=rank(10),
        mAP=mAP,
        num_valid_queries=num_valid_q,
        num_queries=num_q,
    )
