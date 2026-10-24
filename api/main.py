import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from prometheus_client import make_asgi_app

from api.model_loader import MODEL_NAME, predict
from src.classes import CLASS_NAMES

app = FastAPI(title="Skin Condition Classifier")
# Prometheus metrikleri için /metrics endpoint'ini bağla
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Maksimum dosya boyutu: 5 MB
MAX_FILE_SIZE = 5 * 1024 * 1024  
ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg"]

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # 1. Dosya türü kontrolü
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Allowed types: JPEG, PNG."
        )
    
    # 2. Dosyayı oku ve boyut kontrolü yap
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB."
        )

    # 3. Resmi aç (bozuksa hata yakala)
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

    # 4. Model tahmini
    probs = predict(image)
    best = max(range(len(probs)), key=probs.__getitem__)
    
    return {
        "prediction": CLASS_NAMES[best],
        "confidence": probs[best],
        "probabilities": dict(zip(CLASS_NAMES, probs)),
        "model": MODEL_NAME,
        "disclaimer": "This is not a medical diagnosis. Consult a doctor.",
    }