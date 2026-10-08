import io
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

SPLITS_DIR = Path("data/splits")
REQUIRED_COLUMNS = {"path", "class"}


def test_split_files_exist():
    for name in ("train.csv", "val.csv", "test.csv"):
        assert (SPLITS_DIR / name).exists(), f"{name} not found in {SPLITS_DIR}"


def test_split_columns():
    df = pd.read_csv(SPLITS_DIR / "train.csv")
    assert REQUIRED_COLUMNS.issubset(set(df.columns)), (
        f"train.csv missing columns. Got: {list(df.columns)}"
    )


def test_split_not_empty():
    for name in ("train.csv", "val.csv", "test.csv"):
        df = pd.read_csv(SPLITS_DIR / name)
        assert len(df) > 0, f"{name} is empty"


def test_classes_in_csv_are_known():
    from src.classes import CLASS_NAMES

    df = pd.read_csv(SPLITS_DIR / "train.csv")
    unknown = set(df["class"].unique()) - set(CLASS_NAMES)
    assert not unknown, f"Unknown classes in train.csv: {unknown}"


def test_eval_transform_on_synthetic_image():
    # DVC-free, network-free: build a synthetic image in memory
    from src.data.transforms import get_eval_transforms

    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(128, 128, 128)).save(buf, format="PNG")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    tf = get_eval_transforms(224)
    tensor = tf(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)