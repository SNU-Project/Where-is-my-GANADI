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
    group: str = "test",
) -> ReidSplit:
    """DogFaceNet은 query/gallery 구분이 없어(개체 분류용) 개체(ID) 기준 그룹(기본 test, 학습 중
    모니터링할 땐 group="val")에서 이미지 1장을 무작위로 뽑아 query로, 나머지를 gallery로 쓰는
    leave-one-out 방식을 쓴다. camid는 이미지마다 전부 다른 값(전역 인덱스)을 줘서 카메라 기반
    제외 규칙을 비활성화한다 (DogFaceNet에는 카메라 개념이 없음).
    """
    df = pd.read_csv(manifest_csv)
    test_df = df[df.split_group == group].reset_index(drop=True)

    rng = random.Random(seed)
    q_paths, q_pids, q_cams = [], [], []
    g_paths, g_pids, g_cams = [], [], []
    global_cam = 0

    for dog_id, id_rows in test_df.groupby("dog_id"):
        rows = id_rows.to_dict("records")
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


def load_shelter_split(
    root: Path = Path("Data/shelter"),
    manifest_csv: Path = Path("metadata/shelter_manifest_clean.csv"),
    seed: int = 7,
) -> ReidSplit:
    """실제 animal.go.kr 공고 사진으로 진짜 사용 시나리오를 재현하는 테스트.

    같은 유기견의 공고 사진이 보통 2장 등록돼 있다(대표사진 popfile1/2) — 이걸 각각 query/gallery로
    나누면 "사용자가 사진 한 장을 올렸을 때, 같은 개의 다른 사진을 찾아내는지"를 실제 지저분한
    현장 사진으로 검증할 수 있다. 학습·검증에 전혀 쓰이지 않은 완전히 새로운 도메인 테스트.
    사진이 1장뿐인 공고(9건)는 query/gallery 쌍을 만들 수 없어 제외한다.
    """
    df = pd.read_csv(manifest_csv)
    rng = random.Random(seed)
    q_paths, q_pids, q_cams = [], [], []
    g_paths, g_pids, g_cams = [], [], []
    global_cam = 0

    for _, r in df.iterrows():
        photos = [p for p in str(r.photo_local_paths).split("|") if p]
        if len(photos) < 2:
            continue
        rng.shuffle(photos)
        pid = r.desertion_no
        q_paths.append(root / photos[0])
        q_pids.append(pid)
        q_cams.append(global_cam)
        global_cam += 1
        for p in photos[1:]:
            g_paths.append(root / p)
            g_pids.append(pid)
            g_cams.append(global_cam)
            global_cam += 1

    return ReidSplit("Shelter", root, q_paths, q_pids, q_cams, g_paths, g_pids, g_cams)
