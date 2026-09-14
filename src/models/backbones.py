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


class BNNeckModel(nn.Module):
    """ResNet50 + BNNeck (Luo et al., "Bag of Tricks", 2019).

    학습: backbone -> GAP -> BN(bias 없음) -> bn_feat -> Linear(분류기) -> logits, CE loss.
    추론(검색): 분류기를 버리고 bn_feat(BN 통과 후 임베딩)을 코사인 유사도에 쓴다 —
    BN을 거친 특징이 분류 경계 뿐 아니라 검색(코사인 거리)에도 더 잘 맞는다는 게 원 논문의 핵심 트릭.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.transform = weights.transforms() if weights else None
        self.out_dim = 2048

        self.bnneck = nn.BatchNorm1d(self.out_dim)
        self.bnneck.bias.requires_grad_(False)  # BNNeck 표준: bias 학습 안 함
        self.classifier = nn.Linear(self.out_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor):
        feat = torch.flatten(self.features(x), 1)
        bn_feat = self.bnneck(feat)
        if self.training:
            logits = self.classifier(bn_feat)
            return bn_feat, logits
        return bn_feat

    def forward_with_logits(self, x: torch.Tensor):
        """train/eval 모드와 무관하게 (bn_feat, logits)를 반환 — 닫힌집합(val) 분류 정확도 체크용.
        BatchNorm은 self.training(model.eval() 여부)에 따라 정상적으로 배치/이동평균 통계를 쓴다."""
        feat = torch.flatten(self.features(x), 1)
        bn_feat = self.bnneck(feat)
        logits = self.classifier(bn_feat)
        return bn_feat, logits
