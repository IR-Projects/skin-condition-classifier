import os
import random

import torch
import torch.nn.functional as F
from PIL import Image

from src.classes import CLASS_NAMES
from src.model import build_model
from src.data.transforms import get_eval_transforms

# Varsayılan olarak dummy modda başla
MODEL_NAME = "dummy"
_model = None
_transform = None
_classes = CLASS_NAMES

def _load_model():
    global MODEL_NAME, _model, _transform, _classes
    
    # CI (GitHub Actions) testleri için: MODEL_PATH yoksa dummy modelde kal
    model_path = os.environ.get("MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        print(f"WARNING: MODEL_PATH not set or file not found ({model_path}). Running in dummy mode.")
        MODEL_NAME = "dummy"
        return

    try:
        # Checkpoint'i CPU'ya yükle
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Checkpoint içinden bilgileri çıkar
        backbone = checkpoint.get("backbone", "efficientnet_b0")
        img_size = checkpoint.get("img_size", 224)
        _classes = checkpoint.get("classes", CLASS_NAMES)
        
        # Modeli inşa et ve ağırlıkları yükle
        _model = build_model(backbone=backbone, num_classes=len(_classes), pretrained=False)
        _model.load_state_dict(checkpoint["state_dict"])
        _model.eval() # Inference moduna al
        
        # Görüntü ön işleme (transform) hazırla
        _transform = get_eval_transforms(img_size)
        
        MODEL_NAME = os.path.basename(model_path)
        print(f"SUCCESS: Loaded real model from {model_path} ({backbone})")
        
    except Exception as e:
        print(f"ERROR: Failed to load model from {model_path}. Error: {e}")
        print("Falling back to dummy mode.")
        MODEL_NAME = "dummy"

# Modül import edildiğinde modeli yükle
_load_model()

def predict(image: Image.Image) -> list[float]:
    # Eğer gerçek model yüklenmediyse (dummy mod), eski rastgele davranışı göster
    if MODEL_NAME == "dummy" or _model is None or _transform is None:
        scores = [random.random() for _ in CLASS_NAMES]
        total = sum(scores)
        return [score / total for score in scores]

    # Gerçek model tahmini
    img_tensor = _transform(image).unsqueeze(0)  # (3, 224, 224) -> (1, 3, 224, 224)
    
    with torch.no_grad():
        outputs = _model(img_tensor)
        # Çıktıları olasılıklara (softmax) çevir
        probs = F.softmax(outputs, dim=1).squeeze(0)
        
    return probs.tolist()