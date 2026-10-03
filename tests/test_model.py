import torch
from src.model import build_model


def test_model_output_shape():
    model = build_model(num_classes=7)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert tuple(output.shape) == (2, 7)