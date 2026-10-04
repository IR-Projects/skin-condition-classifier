import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from api.model_loader import MODEL_NAME, predict
from src.classes import CLASS_NAMES

app = FastAPI(title="Skin Condition Classifier")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    probs = predict(image)
    best = max(range(len(probs)), key=probs.__getitem__)
    return {
        "prediction": CLASS_NAMES[best],
        "confidence": probs[best],
        "probabilities": dict(zip(CLASS_NAMES, probs)),
        "model": MODEL_NAME,
        "disclaimer": "This is not a medical diagnosis. Consult a doctor.",
    }