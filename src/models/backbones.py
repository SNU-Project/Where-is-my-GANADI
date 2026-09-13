"""임베딩 백본. E0(베이스라인)는 ImageNet 사전학습 ResNet50의 GAP 출력(2048-d)을 그대로 쓴다."""
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ResNet50Embedder(nn.Module):
    """ResNet50 conv stack + GAP -> 2048-d 임베딩. fc(분류) 레이어는 제거."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # avgpool까지
        self.out_dim = 2048
        self.transform = weights.transforms() if weights else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


def build_backbone(name: str = "resnet50_imagenet") -> tuple[nn.Module, object]:
    if name == "resnet50_imagenet":
        model = ResNet50Embedder(pretrained=True)
        return model, model.transform
    raise ValueError(f"알 수 없는 백본: {name}")
