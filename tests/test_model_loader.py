import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from api import model_loader
from api.main import app
from src.classes import CLASS_NAMES
from src.model import build_model

client = TestClient(app)


def _save_random_checkpoint(path, backbone="efficientnet_b0"):
    """Save a checkpoint with random weights in the same dict format as train.py."""
    model = build_model(backbone, len(CLASS_NAMES), pretrained=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(CLASS_NAMES),
            "backbone": backbone,
            "img_size": 224,
        },
        path,
    )


@pytest.fixture(autouse=True)
def _reset_to_dummy():
    model_loader.use_dummy()
    yield
    model_loader.use_dummy()


def test_dummy_mode_predicts_probabilities():
    assert model_loader.get_info()["mode"] == "dummy"
    probs = model_loader.predict(Image.new("RGB", (64, 64)))
    assert len(probs) == len(CLASS_NAMES)
    assert abs(sum(probs) - 1) < 1e-6


def test_real_mode_with_random_weights_checkpoint(tmp_path):
    ckpt = tmp_path / "random_effnet.pt"
    _save_random_checkpoint(ckpt)

    model_loader.load_model(str(ckpt))

    info = model_loader.get_info()
    assert info["mode"] == "real"
    assert info["model"] == "efficientnet_b0"
    assert model_loader.get_classes() == list(CLASS_NAMES)

    probs = model_loader.predict(Image.new("RGB", (300, 200)))
    assert len(probs) == len(CLASS_NAMES)
    assert abs(sum(probs) - 1) < 1e-5


def test_missing_checkpoint_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_loader.load_model(str(tmp_path / "does_not_exist.pt"))


def test_incomplete_checkpoint_fails_loudly(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"state_dict": {}}, bad)
    with pytest.raises(KeyError):
        model_loader.load_model(str(bad))


def test_health_reports_mode():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mode"] == "dummy"