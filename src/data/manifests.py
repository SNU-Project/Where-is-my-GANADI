"""MPDD / DogFaceNet 매니페스트 CSV -> (이미지 경로, pid, camid) 쿼리/갤러리 리스트.

Re-ID 평가는 (경로 리스트, pid 배열, camid 배열)의 query/gallery 쌍이 있으면 되므로,
두 데이터셋을 이 공통 형태로 맞춰서 `src/retrieval/metrics.evaluate_reid`에 그대로 넣는다.
"""
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ReidSplit:
    name: str
    root: Path
    query_paths: list
    query_pids: "list[int]"
    query_camids: "list[int]"
    gallery_paths: list
    gallery_pids: "list[int]"
    gallery_camids: "list[int]"


def load_mpdd_split(
    root: Path = Path("Data/Multi-pose dog dataset/pytorch"),
    manifest_csv: Path = Path("metadata/multipose_dog_manifest.csv"),
) -> ReidSplit:
    """MPDD는 query/gallery 폴더와 실제 카메라 id가 이미 있어 그대로 쓴다."""
    df = pd.read_csv(manifest_csv)

    def pack(split_name):
        sub = df[df.split == split_name]
        paths = [root / p for p in sub.relpath]
        return paths, sub.dog_id.tolist(), sub.camera.tolist()

    q_paths, q_pids, q_cams = pack("query")
    g_paths, g_pids, g_cams = pack("gallery")
    return ReidSplit("MPDD", root, q_paths, q_pids, q_cams, g_paths, g_pids, g_cams)


def load_dogfacenet_split(
    root: Path = Path("Data/DogFaceNet"),
    manifest_csv: Path = Path("metadata/dogfacenet_manifest.csv"),
    seed: int = 123,
) -> ReidSplit:
    """DogFaceNet은 query/gallery 구분이 없어(개체 분류용) 개체(ID) 기준 test 그룹에서
    이미지 1장을 무작위로 뽑아 query로, 나머지를 gallery로 쓰는 leave-one-out 방식을 쓴다.
    camid는 이미지마다 전부 다른 값(전역 인덱스)을 줘서 카메라 기반 제외 규칙을 비활성화한다
    (DogFaceNet에는 카메라 개념이 없음).
    """
    df = pd.read_csv(manifest_csv)
    test_df = df[df.split_group == "test"].reset_index(drop=True)

    rng = random.Random(seed)
    q_paths, q_pids, q_cams = [], [], []
    g_paths, g_pids, g_cams = [], [], []
    global_cam = 0

    for dog_id, group in test_df.groupby("dog_id"):
        rows = group.to_dict("records")
        rng.shuffle(rows)
        query_row, *gallery_rows = rows
        q_paths.append(root / query_row["relpath"])
        q_pids.append(dog_id)
        q_cams.append(global_cam)
        global_cam += 1
        for r in gallery_rows:
            g_paths.append(root / r["relpath"])
            g_pids.append(dog_id)
            g_cams.append(global_cam)
            global_cam += 1

    return ReidSplit("DogFaceNet", root, q_paths, q_pids, q_cams, g_paths, g_pids, g_cams)
