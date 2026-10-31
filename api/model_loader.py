"""Model loading and inference for the API.

Modes
-----
dummy: MODEL_PATH is not set (CI, local smoke tests). Returns random probabilities.
real:  MODEL_PATH points to a checkpoint dict saved by entrypoint/train.py with the keys
       state_dict, classes, backbone, img_size.

If MODEL_PATH is set but the file is missing or cannot be loaded, loading fails loudly
(the app does not start) instead of silently serving random predictions.
"""

import logging
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image

from src.classes import CLASS_NAMES
from src.data.transforms import get_eval_transforms
from src.model import build_model

logger = logging.getLogger("skin_api.model")

REQUIRED_KEYS = ("state_dict", "classes", "backbone", "img_size")


def _dummy_state() -> dict:
    return {
        "mode": "dummy",
        "model": None,
        "transform": None,
        "classes": list(CLASS_NAMES),
        "backbone": "dummy",
        "checkpoint": None,
    }


_state = _dummy_state()


def use_dummy() -> None:
    """Switch to dummy mode (random predictions)."""
    global _state
    _state = _dummy_state()


def load_model(model_path: str) -> None:
    """Load a real checkpoint. Raises on any problem, never falls back silently."""
    global _state

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # weights_only=True: only tensors and plain Python types are unpickled (safer).
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    missing = [key for key in REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint {model_path} is missing keys: {missing}")

    classes = list(checkpoint["classes"])
    backbone = checkpoint["backbone"]

    model = build_model(backbone, len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    _state = {
        "mode": "real",
        "model": model,
        "transform": get_eval_transforms(int(checkpoint["img_size"])),
        "classes": classes,
        "backbone": backbone,
        "checkpoint": os.path.basename(model_path),
    }
    logger.warning("Loaded real model %s (%s, %d classes)", model_path, backbone, len(classes))


def init_from_env() -> None:
    """Read MODEL_PATH: set -> real model (must load), unset -> dummy mode."""
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        load_model(model_path)
    else:
        logger.warning("MODEL_PATH is not set: running in DUMMY mode (random predictions).")
        use_dummy()


def get_info() -> dict:
    return {
        "mode": _state["mode"],
        "model": _state["backbone"],
        "checkpoint": _state["checkpoint"],
    }


def get_classes() -> list[str]:
    return list(_state["classes"])


def predict(image: Image.Image) -> list[float]:
    state = _state  # read once, in case the state is swapped during the call

    if state["mode"] == "dummy":
        scores = [random.random() for _ in state["classes"]]
        total = sum(scores)
        return [score / total for score in scores]

    tensor = state["transform"](image).unsqueeze(0)  # (3, H, W) -> (1, 3, H, W)
    with torch.no_grad():
        probs = F.softmax(state["model"](tensor), dim=1).squeeze(0)
    return probs.tolist()


init_from_env()