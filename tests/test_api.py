import io

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from src.classes import CLASS_NAMES

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_image():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="PNG")
    buf.seek(0)
    response = client.post("/predict", files={"file": ("test.png", buf, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in CLASS_NAMES
    assert "model" in body
    assert len(body["probabilities"]) == len(CLASS_NAMES)
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-6


def test_predict_invalid_file():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400