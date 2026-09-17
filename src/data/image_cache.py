"""디코딩된 이미지 캐시. 에피소드 학습·검증에서 같은 이미지(특히 val set)를 반복해서
읽다 보니 디스크 I/O가 병목이 되는 걸 확인해(D4, E2 v2가 v1보다 훨씬 느려짐) 추가했다.
변환(transform)은 매번 다시 적용하므로 train 증강의 무작위성은 그대로 유지된다.
"""
from pathlib import Path

from PIL import Image

_cache: dict = {}


def load_rgb(path) -> Image.Image:
    key = str(path)
    img = _cache.get(key)
    if img is None:
        img = Image.open(path).convert("RGB")
        img.load()  # 파일 핸들을 바로 닫아도 되도록 픽셀 데이터를 메모리에 확정
        _cache[key] = img
    return img


def cache_info():
    return {"cached_images": len(_cache)}
