import pytest
import torch

from src.model import SUPPORTED_BACKBONES, build_model


@pytest.mark.parametrize("backbone", ["efficientnet_b0", "resnet50"])
def test_model_output_shape(backbone):
    # pretrained=False keeps CI offline (no ImageNet download)
    model = build_model(backbone=backbone, num_classes=8, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert tuple(output.shape) == (2, 8)


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_model(backbone="not_a_real_backbone", num_classes=8, pretrained=False)


def test_supported_backbones_list():
    assert set(SUPPORTED_BACKBONES) == {"efficientnet_b0", "resnet50"}