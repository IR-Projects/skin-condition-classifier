"""
Model builder for the skin-condition classifier.

Supports two backbones:
    - "efficientnet_b0"  (primary, Week 3)
    - "resnet50"         (ablation, Week 4)

Interface:
    build_model(backbone: str, num_classes: int, pretrained: bool = True)

Important:
    pretrained=False must NOT trigger a weight download. CI runs without
    network access, and tests call this with pretrained=False.
"""
import torch.nn as nn
from torchvision import models


SUPPORTED_BACKBONES = ("efficientnet_b0", "resnet50")


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Supported: {SUPPORTED_BACKBONES}"
        )

    if backbone == "efficientnet_b0":
        # weights=None  → random init, no download (CI-safe)
        # weights=enum  → download pretrained ImageNet weights
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        # EfficientNet's classifier is a Sequential whose last layer is Linear
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model