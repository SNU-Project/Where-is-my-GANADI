"""Prototypical Network 에피소드(N-way K-shot) 샘플러.

DogFaceNet 개체당 이미지가 2~41장(중앙값 5장)으로 극히 적어(D1.7 참고), 고정 K/Q는 대부분의
클래스를 에피소드에서 아예 못 쓰게 만든다. 그래서 클래스마다 "가진 만큼만" 쓰는 적응형 샘플링을 쓴다:
  - support: 항상 1장 (2장짜리 클래스도 참여 가능)
  - query: 남은 이미지 중 최대 q_max장 (많이 가진 클래스는 더 많이 기여)
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
    def __init__(self, class_pools: dict, n_way: int = 20, q_max: int = 4, seed: int = 0):
        # 에피소드를 구성하려면 최소 2장(support 1 + query 1) 필요
        self.pools = {label: paths for label, paths in class_pools.items() if len(paths) >= 2}
        self.n_way = min(n_way, len(self.pools))
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
            support = paths[:1]
            q_n = min(len(paths) - 1, self.q_max)
            query = paths[1:1 + q_n]
            support_items += [(p, local_idx) for p in support]
            query_items += [(p, local_idx) for p in query]
        return support_items, query_items
