"""
Cluster vitiligo images by visual similarity to reduce patient-level leakage.

Problem: vitiligo images from SkinDiseaseBD / SkinDiseaseClassification
have no patient IDs, but appear to come from a small number of patients
with many photos each. split.py treats each image independently, so the
same person can end up in both train and test.

Fix: compute perceptual hashes for vitiligo images only, cluster by
Hamming distance <= THRESHOLD (looser than dedupe's 2, since same-person
photos aren't exact duplicates), and rewrite vitiligo group_ids to
cluster IDs. Then re-run split.py to get a leakage-safe split.

Run:
    python src/data/cluster_vitiligo.py
"""
import os
from multiprocessing import Pool

import pandas as pd
import numpy as np
from PIL import Image
import imagehash

DATA_DIR = r"C:\Users\TOSHIBA\Desktop\skin_Conditions_AI\data"
MANIFEST = os.path.join(DATA_DIR, "manifest_deduped.csv")
OUT = os.path.join(DATA_DIR, "manifest_deduped.csv")  # overwrite

THRESHOLD = 12  # Hamming distance out of 64 bits


def hash_one(path):
    try:
        with Image.open(path) as img:
            return (path, str(imagehash.phash(img.convert("RGB"))))
    except Exception:
        return (path, None)


def union_find(pairs, n):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for a, b in pairs:
        union(a, b)
    return [find(i) for i in range(n)]


def main():
    df = pd.read_csv(MANIFEST)
    vit = df[df["class"] == "vitiligo"].copy()
    print(f"Vitiligo images: {len(vit)}")

    print("Hashing vitiligo images...")
    with Pool(4) as pool:
        results = dict(pool.map(hash_one, vit["path"].tolist(), chunksize=16))

    vit["phash"] = vit["path"].map(results)
    missing = vit["phash"].isna().sum()
    if missing:
        print(f"WARNING: {missing} vitiligo images failed to hash")

    hashable = vit[vit["phash"].notna()].copy()
    hashable["_original_idx"] = hashable.index
    print(f"Finding same-person clusters (Hamming <= {THRESHOLD})...")
    n = len(hashable)
    bits = np.zeros((n, 64), dtype=np.uint8)
    for i, h in enumerate(hashable["phash"]):
        bits[i] = np.unpackbits(np.array([int(h, 16)], dtype=">u8").view(np.uint8))

    pairs = []
    chunk = 500
    for i in range(0, n, chunk):
        for j in range(i, n, chunk):
            block = bits[i:i+chunk, None, :] != bits[None, j:j+chunk, :]
            dists = block.sum(axis=2)
            if i == j:
                upper = np.triu(np.ones_like(dists, dtype=bool), k=1)
                ys, xs = np.where((dists <= THRESHOLD) & upper)
            else:
                ys, xs = np.where(dists <= THRESHOLD)
            for y, x in zip(ys, xs):
                pairs.append((i + y, j + x))

    print(f"  {len(pairs)} same-person pairs found")
    labels = union_find(pairs, n)
    hashable["cluster"] = labels

    # Cluster size distribution
    sizes = hashable.groupby("cluster").size()
    print("\nCluster sizes:")
    print(f"  singletons: {(sizes == 1).sum()}")
    print(f"  clusters >1: {(sizes > 1).sum()}")
    print(f"  largest cluster: {sizes.max()}")
    print(f"  mean size of multi-image clusters: {sizes[sizes > 1].mean():.1f}")

    # Write new group_ids back
    df.loc[hashable["_original_idx"], "group_id"] = (
        "vitiligo_cluster_" + hashable["cluster"].astype(str)
    )
    df.to_csv(OUT, index=False)
    print(f"\nWrote updated manifest to {OUT}")
    print("Next: re-run split.py to regenerate splits with vitiligo grouped.")


if __name__ == "__main__":
    main()