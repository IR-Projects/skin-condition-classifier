import io
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from prometheus_client import Counter, Histogram, make_asgi_app

from api import model_loader

app = FastAPI(title="Skin Condition Classifier")

# Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())

REQUESTS = Counter("predict_requests_total", "Total /predict requests", ["status"])
LATENCY = Histogram("predict_latency_seconds", "Inference latency in seconds")
PREDICTIONS = Counter("predictions_total", "Predictions per class", ["label", "mode"])
CONFIDENCE = Histogram(
    "prediction_confidence",
    "Top-1 confidence of each prediction",
    buckets=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# Maximum upload size: 5 MB
MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg"]
DISCLAIMER = "This is not a medical diagnosis. Consult a doctor."


def _reject(detail: str) -> HTTPException:
    REQUESTS.labels(status="rejected").inc()
    return HTTPException(status_code=400, detail=detail)


@app.get("/health")
def health():
    return {"status": "ok", **model_loader.get_info()}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # 1. File type check
    if file.content_type not in ALLOWED_TYPES:
        raise _reject(f"Invalid file type: {file.content_type}. Allowed types: JPEG, PNG.")

    # 2. Read at most MAX_FILE_SIZE + 1 bytes, then check the size
    data = await file.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise _reject(f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB.")

    # 3. Open the image (corrupted files are rejected)
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise _reject("Invalid or corrupted image file")

    # 4. Model prediction (CPU-bound, so run it outside the event loop)
    start = time.perf_counter()
    probs = await run_in_threadpool(model_loader.predict, image)
    LATENCY.observe(time.perf_counter() - start)

    classes = model_loader.get_classes()
    info = model_loader.get_info()
    best = max(range(len(probs)), key=probs.__getitem__)

    PREDICTIONS.labels(label=classes[best], mode=info["mode"]).inc()
    CONFIDENCE.observe(probs[best])
    REQUESTS.labels(status="ok").inc()

    return {
        "prediction": classes[best],
        "confidence": probs[best],
        "probabilities": dict(zip(classes, probs)),
        "model": info["model"],
        "mode": info["mode"],
        "disclaimer": DISCLAIMER,
    }