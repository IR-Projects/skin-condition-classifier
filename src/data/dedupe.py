"""
Perceptual-hash dedupe for the manifest.

Purpose:
    Find near-duplicate images that MD5 hashing misses — same image
    resized, recompressed, or re-uploaded across datasets.

What it does:
    1. Reads data/manifest.csv
    2. Computes a perceptual hash (pHash) for every image
    3. Finds pairs whose Hamming distance is <= THRESHOLD
    4. Groups near-duplicates via union-find
    5. Writes a cleaned manifest + a duplicate_groups.json report

What it does NOT do:
    - Same patient, different photos of the same lesion → use group_id
    - Same patient, different lesions → use group_id
    Only your patient-level split handles those.

   
"""
import os
import json
import time
from multiprocessing import Pool
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import imagehash

DATA_DIR = r"C:\Users\TOSHIBA\Desktop\skin_Conditions_AI\data"
MANIFEST_IN = os.path.join(DATA_DIR, "manifest.csv")
MANIFEST_OUT = os.path.join(DATA_DIR, "manifest_deduped.csv")
REPORT_OUT = os.path.join(DATA_DIR, "duplicate_groups.json")

# Hamming distance threshold (bits out of 64).
#   0 = only exact hash matches
#   5 = catches resized/recompressed copies (recommended)
#  10 = aggressive, may over-merge similar-looking distinct images
THRESHOLD = 2

# Which source to prefer when a group spans multiple datasets.
# Higher = more likely to be kept as the "representative".
SOURCE_PRIORITY = {
    "HAM10000": 4,
    "ISIC": 3,
    "SkinDisNet": 3,
    "ACNE04": 2,
    "DermNet": 2,
    "SkinDiseaseBD": 2,
    "SkinDiseaseClassification": 2,
}


def hash_one(path):
    """Compute pHash for a single image. Returns (path, hash_hex) or (path, None)."""
    try:
        with Image.open(path) as img:
            h = imagehash.phash(img.convert("RGB"))
        return (path, str(h))
    except Exception:
        return (path, None)


def compute_hashes(paths, workers=4):
    """Multiprocessing wrapper for hash_one."""
    t0 = time.time()
    with Pool(workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(hash_one, paths, chunksize=64), 1):
            results.append(r)
            if i % 500 == 0 or i == len(paths):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(paths) - i) / rate if rate > 0 else 0
                print(f"  hashed {i}/{len(paths)}  "
                      f"({rate:.0f}/s, ETA {eta:.0f}s)")
    return dict(results)


def find_near_duplicate_pairs(hash_hex_list, threshold):
    """
    Return list of (i, j) index pairs whose hashes are within `threshold`.
    Vectorized via bit array + chunked hamming distance.
    """
    N = len(hash_hex_list)
    bits = np.zeros((N, 64), dtype=np.uint8)
    for i, h in enumerate(hash_hex_list):
        # imagehash gives a 64-bit hex string; unpack into 64 bits
        val = int(h, 16)
        bits[i] = np.unpackbits(np.array([val], dtype=">u8").view(np.uint8))

    pairs = []
    chunk = 2000
    for i in range(0, N, chunk):
        for j in range(i, N, chunk):
            # [chunk, 1, 64] != [1, chunk, 64] -> [chunk, chunk, 64]
            block = bits[i:i + chunk, None, :] != bits[None, j:j + chunk, :]
            dists = block.sum(axis=2)
            
            if i == j:
                # For diagonal blocks: only consider strict upper triangle.
                # Use a mask — do NOT zero dists, or zeros become false matches.
                upper = np.triu(np.ones_like(dists, dtype=bool), k=1)
                ys, xs = np.where((dists <= threshold) & upper)
            else:
                ys, xs = np.where(dists <= threshold)
            for y, x in zip(ys, xs):
                pairs.append((i + y, j + x))
    return pairs


def union_find(pairs, N):
    parent = list(range(N))

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
    return [find(i) for i in range(N)]


def main():
    print(f"Loading manifest: {MANIFEST_IN}")
    df = pd.read_csv(MANIFEST_IN)
    print(f"Total rows: {len(df)}")

    paths = df["path"].tolist()
    print(f"\nComputing pHash for {len(paths)} images...")
    hash_map = compute_hashes(paths, workers=4)

    # Attach hashes; drop any that failed to hash
    df["phash"] = df["path"].map(hash_map)
    failed = df["phash"].isna().sum()
    if failed:
        print(f"\nWARNING: {failed} images failed to hash and are kept as-is.")
    hashable = df[df["phash"].notna()].reset_index(drop=True)
    unhashable = df[df["phash"].isna()].reset_index(drop=True)

    print(f"\nFinding near-duplicate pairs (threshold = {THRESHOLD}/64 bits)...")
    t0 = time.time()
    pairs = find_near_duplicate_pairs(hashable["phash"].tolist(), THRESHOLD)
    print(f"  found {len(pairs)} near-duplicate pairs in {time.time()-t0:.1f}s")

    print("\nGrouping near-duplicates (union-find)...")
    labels = union_find(pairs, len(hashable))
    hashable["group"] = labels

    # For each group with >1 member, pick a representative to keep
    print("Selecting representatives...")
    keep_idx = set()
    dup_groups = []
    for gid, group in hashable.groupby("group"):
        if len(group) == 1:
            keep_idx.add(group.index[0])
            continue
        # Pick representative: highest source priority, then first alphabetically
        group = group.copy()
        group["_prio"] = group["source"].map(SOURCE_PRIORITY).fillna(1)
        group = group.sort_values(["_prio", "path"], ascending=[False, True])
        winner = group.iloc[0]
        keep_idx.add(winner.name)
        dup_groups.append({
            "kept": winner["path"],
            "kept_source": winner["source"],
            "kept_class": winner["class"],
            "dropped": [{"path": r["path"], "source": r["source"],
                         "class": r["class"]} for _, r in group.iloc[1:].iterrows()],
        })

    deduped = hashable.loc[sorted(keep_idx)].drop(columns=["group", "phash"])
    final = pd.concat([deduped, unhashable], ignore_index=True)

    print("\nClass totals — before vs after:")
    before = df["class"].value_counts().sort_index()
    after = final["class"].value_counts().sort_index()
    for cls in before.index:
        b, a = int(before[cls]), int(after.get(cls, 0))
        print(f"  {cls:25s} {b:6d} -> {a:6d}  (-{b-a})")

    print(f"\nTotal images: {len(df)} -> {len(final)}  "
          f"(removed {len(df) - len(final)} near-duplicates)")

    final.to_csv(MANIFEST_OUT, index=False)
    print(f"Saved deduped manifest: {MANIFEST_OUT}")

    with open(REPORT_OUT, "w") as f:
        json.dump({
            "threshold": THRESHOLD,
            "total_before": len(df),
            "total_after": len(final),
            "removed": len(df) - len(final),
            "duplicate_groups": dup_groups,
        }, f, indent=2)
    print(f"Saved report: {REPORT_OUT}")


if __name__ == "__main__":
    main()