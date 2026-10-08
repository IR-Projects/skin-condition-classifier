import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

# sorted alphabetically for easier indexing and consistency across files
# to avoid mistakes in the confusion matrix
CLASS_NAMES = [
    "acne",
    "bcc",
    "eczema",
    "melanoma",
    "nail_psoriasis",
    "nevus",
    "seborrheic_keratosis",
    "vitiligo",
]
# maps class names to indices — useful for converting model outputs to class indices
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
# maps indices to class names — useful for converting model outputs back to class names
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}


class SkinLesionDataset(Dataset):
    def __init__(self, csv_path, transform=None, transform_overrides=None, return_path=False):
        # this reads the split CSV and loads it into a DataFrame
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        # store the transform
        self.transform = transform
        # Dict mapping class name -> transform, for classes that need
        # different augmentation (e.g. vitiligo uses aggressive crop).
        # Falls back to self.transform when a class isn't in the dict.
        self.transform_overrides = transform_overrides or {}
        self.return_path = return_path

        # Sanity-check that every class in the CSV is known
        unknown = set(self.df["class"].unique()) - set(CLASS_NAMES)
        if unknown:
            raise ValueError(f"Unknown classes in {csv_path}: {unknown}")

    def __len__(self):
        # returns the row count
        return len(self.df)

    def __getitem__(self, idx):
        # returns the image and label at the specified index.
        # If return_path is True, it also returns the absolute image path.
        row = self.df.iloc[idx]
        path = row["path"]
        # convert the class string to an integer
        label = CLASS_TO_IDX[row["class"]]

        try:
            # open the image and force 3-channel RGB
            # some datasets have grayscale or CMYK images
            img = Image.open(path).convert("RGB")
        except Exception:
            # corrupt image — return a blank gray image with a fake label.
            # The training loop filters these out via the sentinel value.
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            label = -1

        # pick the transform for this class — override if present,
        # otherwise the default
        tf = self.transform_overrides.get(row["class"], self.transform)
        if tf is not None:
            # augmentation
            # get a randomly cropped, flipped, rotated tensor
            img = tf(img)

        if self.return_path:
            return img, label, path
        return img, label


# Sanity check
# checks if the dataset is loaded correctly and if the transforms are applied
# correctly. It also visualizes some samples from the dataset to ensure that
# the images are being processed as expected.
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transforms import get_eval_transforms

    import matplotlib
    matplotlib.use("Agg")  # headless backend — avoids needing a display
    import matplotlib.pyplot as plt
    import numpy as np

    SPLITS = r"C:\Users\TOSHIBA\Desktop\skin_Conditions_AI\data\splits"
    train_csv = os.path.join(SPLITS, "train.csv")

    tf = get_eval_transforms(224)
    ds = SkinLesionDataset(train_csv, transform=tf)

    print(f"Loaded train dataset: {len(ds)} samples")
    print(f"Classes: {CLASS_NAMES}")

    # Print class distribution
    print("\nClass distribution:")
    print(ds.df["class"].value_counts().sort_index())

    # Load a few samples and inspect shapes
    print("\nLoading first 3 samples:")
    for i in range(3):
        img, label = ds[i]
        print(f"  sample {i}: tensor shape={tuple(img.shape)}, "
              f"label={label} ({IDX_TO_CLASS[label]})")

    # Visual grid: show what transforms are doing.
    # One sample per class so we can eyeball whether each class looks right.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, i in zip(axes.flat, range(len(CLASS_NAMES))):
        target_cls = CLASS_NAMES[i]
        idx = ds.df[ds.df["class"] == target_cls].index[0]
        img, label = ds[idx]
        img_np = img.numpy().transpose(1, 2, 0)
        # Denormalize for display
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(IDX_TO_CLASS[label])
        ax.axis("off")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "dataset_sanity_check.png")
    plt.savefig(out, dpi=80)
    print(f"\nSaved visualization to {out}")