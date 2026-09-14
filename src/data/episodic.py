"""Prototypical Network 에피소드(N-way K-shot) 샘플러.

DogFaceNet 개체당 이미지가 2~41장(중앙값 5장)으로 극히 적어(D1.7 참고), 고정 K/Q는 대부분의
클래스를 에피소드에서 아예 못 쓰게 만든다. 그래서 클래스마다 "가진 만큼만" 쓰는 적응형 샘플링을 쓴다:
  - support: 최대 k_max장 (2장짜리 클래스는 1장만 — 그래도 참여는 가능)
  - query: 남은 이미지 중 최대 q_max장

주의(v1 -> v2 수정 이력): 처음엔 support를 무조건 1장으로 고정했더니, 프로토타입(support
임베딩 평균)이 사실상 이미지 1장짜리라 잡음이 심해 500 에피소드 학습 중 val Rank-1이 에피소드
100 이후 오히려 떨어지는 과적합이 나타났다 (D4 로그 참고). support를 최대 5장까지 늘려 프로토타입을
안정시키도록 수정했다 — 원 논문(Snell et al.)의 5-shot 설정에 더 가까워짐.
"""
import random
from collections import defaultdict
from pathlib import Path


def group_by_label(items) -> dict:
    """[LabeledImage, ...] -> {global_label: [path, ...]}"""
    pools = defaultdict(list)
    for it in items:
        pools[it.global_label].append(it.path)
    return pools


class EpisodeSampler:
    def __init__(self, class_pools: dict, n_way: int = 20, k_max: int = 5, q_max: int = 4, seed: int = 0):
        # 에피소드를 구성하려면 최소 2장(support 1 + query 1) 필요
        self.pools = {label: paths for label, paths in class_pools.items() if len(paths) >= 2}
        self.n_way = min(n_way, len(self.pools))
        self.k_max = k_max
        self.q_max = q_max
        self.rng = random.Random(seed)
        self.labels = sorted(self.pools.keys())

    def sample(self):
        """반환: support_items, query_items — 각각 [(path, episode_local_class_idx), ...]"""
        classes = self.rng.sample(self.labels, self.n_way)
        support_items, query_items = [], []
        for local_idx, label in enumerate(classes):
            paths = self.pools[label][:]
            self.rng.shuffle(paths)
            n_avail = len(paths)
            k = min(self.k_max, n_avail - 1)  # query가 최소 1장은 남도록
            support = paths[:k]
            q_n = min(n_avail - k, self.q_max)
            query = paths[k:k + q_n]
            support_items += [(p, local_idx) for p in support]
            query_items += [(p, local_idx) for p in query]
        return support_items, query_items
