"""사진에서 "개" 영역만 잘라낸다 — 배경(마당·바닥·사람 팔 등)이 색상 비교와 임베딩을
오염시키는 문제(D5.1 이후 실사용 피드백: "배경 비슷한 사진만 나온다")를 해결하기 위함.

COCO로 사전학습된 가벼운 탐지기(MobileNetV3 기반 Faster R-CNN)를 그대로 쓴다 — COCO의
80개 클래스 중 "dog"(라벨 18)만 본다. 추가 학습이나 다운로드할 별도 데이터셋 없이 torchvision에
이미 포함된 사전학습 가중치만으로 동작한다.
"""
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

COCO_DOG_LABEL = 18
# 개를 고양이/양/말/소/곰 등 비슷한 네발짐승으로 착각하는 경우가 많음이 확인됨(진단 결과).
# 우리 데이터는 100% 개 사진이라는 게 이미 확실하므로, 종 분류는 틀려도 "동물 영역" 위치
# 추정 자체는 맞을 가능성이 높은 이 클래스들도 폭넓게 후보로 받아들인다.
COCO_ANIMAL_LABELS = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25}  # bird~giraffe
MIN_UPSCALE_SIDE = 480  # 저해상도 사진(MPDD 30%가 128px 미만)은 탐지 전에 미리 키워준다
_model = None


def _get_model(device):
    global _model
    if _model is None:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT  # MobileNetV3-320보다 훨씬 강한 백본
        # 기본 box_score_thresh=0.05가 저신뢰도 "개" 후보까지 모델 내부에서 미리 걸러버려서
        # 낮춘다 - 우리 쪽 score_thresh 필터링(레이블만 확인)에서 최종 판단하면 된다.
        _model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.001)
        _model.eval().to(device)
    return _model


def _maybe_upscale(img: Image.Image) -> tuple[Image.Image, float]:
    """짧은 변이 MIN_UPSCALE_SIDE보다 작으면 bicubic으로 키운다. (키운 이미지, 스케일 배율) 반환."""
    w, h = img.size
    short_side = min(w, h)
    if short_side >= MIN_UPSCALE_SIDE:
        return img, 1.0
    scale = MIN_UPSCALE_SIDE / short_side
    return img.resize((int(w * scale), int(h * scale)), Image.BICUBIC), scale


@torch.no_grad()
def crop_to_dog(img: Image.Image, device="cpu", margin: float = 0.12, score_thresh: float = 0.0) -> Image.Image:
    """가장 확신도 높은 "동물" 박스로 크롭. 못 찾으면 원본 그대로 반환(안전한 폴백).

    1순위로 "개" 라벨을 찾되, 없으면 비슷하게 헷갈리는 다른 네발짐승 라벨(고양이/양/말 등,
    COCO_ANIMAL_LABELS)까지 넓혀서 찾는다 — 종 분류는 틀려도 위치 추정은 맞을 때가 많다는
    진단 결과에 따른 것. score_thresh=0.0: 점수가 아무리 낮아도 해당 라벨 중 1등 박스를 쓴다.
    """
    img = img.convert("RGB")
    upscaled, scale = _maybe_upscale(img)

    model = _get_model(device)
    x = TF.to_tensor(upscaled).to(device)
    pred = model([x])[0]

    dog_mask = (pred["labels"] == COCO_DOG_LABEL) & (pred["scores"] >= score_thresh)
    if not dog_mask.any():
        animal_labels = torch.tensor(list(COCO_ANIMAL_LABELS))
        dog_mask = torch.isin(pred["labels"], animal_labels) & (pred["scores"] >= score_thresh)
    if not dog_mask.any():
        return img

    boxes = pred["boxes"][dog_mask]
    scores = pred["scores"][dog_mask]
    best = (boxes[scores.argmax()] / scale).tolist()  # 원본(업스케일 전) 좌표로 환산

    w, h = img.size
    x1, y1, x2, y2 = best
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))
