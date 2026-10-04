import random

from src.classes import CLASS_NAMES

MODEL_NAME = "dummy"


def predict(image):
    # Week 4: replace with real checkpoint inference (efficientnet_b0 / resnet50).
    scores = [random.random() for _ in CLASS_NAMES]
    total = sum(scores)
    return [score / total for score in scores]