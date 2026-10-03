"""
Patient-level stratified train/val/test split.

Reads data/manifest.csv, writes:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv

Split is 70/15/15. Grouped by group_id (patient_id where available,
lesion_id as fallback, image path for datasets with no patient info).
Every class is stratified so proportions match across splits.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\TOSHIBA\Desktop\skin_Conditions_AI\data"
MANIFEST = os.path.join(DATA_DIR, "manifest_deduped.csv")
OUT_DIR = os.path.join(DATA_DIR, "splits")

RATIOS = (0.70, 0.15, 0.15)
SEED = 42


def grouped_stratified_split(df, label_col, group_col, ratios, seed):
    rng = np.random.default_rng(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["split"] = None

    for cls in sorted(df[label_col].unique()):
        cls_df = df[df[label_col] == cls]
        groups = cls_df[group_col].unique().tolist()
        rng.shuffle(groups)

        group_sizes = cls_df.groupby(group_col).size().to_dict()
        total = len(cls_df)

        targets = [int(round(total * r)) for r in ratios]
        targets[2] = total - targets[0] - targets[1]

        names = ["train", "val", "test"]
        counts = {n: 0 for n in names}

        for g in groups:
            # assign this group to whichever split is furthest below target
            deficits = [targets[i] - counts[names[i]] for i in range(3)]
            idx = int(np.argmax(deficits))
            counts[names[idx]] += group_sizes[g]
            df.loc[(df[label_col] == cls) & (df[group_col] == g), "split"] = names[idx]

    return df


def main():
    print(f"Loading manifest: {MANIFEST}")
    df = pd.read_csv(MANIFEST)
    print(f"Total rows: {len(df)}")

    df = grouped_stratified_split(df, "class", "group_id", RATIOS, SEED)

    # ---- Sanity checks ----
    print("\nClass distribution per split:")
    print(pd.crosstab(df["class"], df["split"]))

    print("\nSplit sizes:")
    print(df["split"].value_counts())

    # Verify no group crosses splits
    leaks = df.groupby("group_id")["split"].nunique()
    n_leaks = int((leaks > 1).sum())
    print(f"\nGroups appearing in more than one split: {n_leaks}")
    if n_leaks > 0:
        print("WARNING: leakage detected — inspect those group_ids.")

    # ---- Save ----
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in ["train", "val", "test"]:
        sub = df[df["split"] == name].drop(columns=["split"])
        out = os.path.join(OUT_DIR, f"{name}.csv")
        sub.to_csv(out, index=False)
        print(f"Wrote {out}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()