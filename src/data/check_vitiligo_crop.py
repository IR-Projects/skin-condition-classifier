"""
Quick check: does the aggressive transform actually crop vitiligo tighter?

Loads one vitiligo image and shows:
    top row    — 4 samples from the standard train transform (scale 0.8-1.0)
    bottom row — 4 samples from the aggressive train transform (scale 0.5-1.0)

Run:
    python src/data/check_vitiligo_crop.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from transforms import get_train_transforms, get_train_transforms_aggressive
from dataset import SkinLesionDataset, CLASS_TO_IDX

SPLITS = r"C:\Users\TOSHIBA\Desktop\skin_Conditions_AI\data\splits"
CSV = os.path.join(SPLITS, "train.csv")

# Load raw rows (no transform) so we control the transform ourselves
ds = SkinLesionDataset(CSV, transform=None)
vit_idx = ds.df[ds.df["class"] == "vitiligo"].index[0]
path = ds.df.iloc[vit_idx]["path"]
print(f"Using: {path}")

raw = Image.open(path).convert("RGB")

standard_tf = get_train_transforms(224)
aggressive_tf = get_train_transforms_aggressive(224)

def denorm(t):
    arr = t.numpy().transpose(1, 2, 0)
    arr = arr * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(arr, 0, 1)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    axes[0, i].imshow(denorm(standard_tf(raw)))
    axes[0, i].set_title("standard (0.8-1.0)", fontsize=10)
    axes[0, i].axis("off")

for i in range(4):
    axes[1, i].imshow(denorm(aggressive_tf(raw)))
    axes[1, i].set_title("aggressive (0.5-1.0)", fontsize=10)
    axes[1, i].axis("off")

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "vitiligo_crop_comparison.png")
plt.savefig(out, dpi=80)
print(f"Saved: {out}")