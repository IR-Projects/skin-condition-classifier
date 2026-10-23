"""
pack_dataset.py -- shrink + bundle the skin images into ONE zip for Colab/Kaggle.

Why: DVC, Google Drive and Kaggle all pay a cost per FILE, and the raw data is
tens of thousands of full-size photos (some thousands of pixels wide). Training
only needs ~224px, so we resize once here and ship a single small archive.

Run from the project root with the venv active:
    python pack_dataset.py --manifest data/manifest.csv --out data/processed/skin_256.zip

The manifest is a CSV with one row per image and (by default) these columns:
    path   original image path (absolute, or relative to --root)
    label  class name
    split  train / val / test   (whatever names you use become the folder names)
Any extra columns (patient_id, source, ...) are kept in packed_manifest.csv
inside the zip, so you can still audit the split later.
If your columns have other names, use --path-col / --label-col / --split-col.

Output layout inside the zip (works with torchvision ImageFolder, one per split):
    train/melanoma/000123_ISIC_0024306.jpg
    val/...    test/...    packed_manifest.csv

Images are resized so the SHORTER side is --size (default 256), keeping the
aspect ratio. Images already smaller than that are left alone (never upscaled).
Keep your normal training transforms (crop/resize to 224, normalize): this only
makes the source files cheaper to move around.
"""
import argparse
import io
import multiprocessing as mp
import re
import sys
import time
import zipfile
from pathlib import Path

from PIL import Image, ImageOps


def safe(text):
    """Make a string safe to use as a folder/file name inside the zip."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "x"


def process(job):
    """Runs in a worker process. Returns (index, jpeg_bytes_or_None, error_or_None)."""
    idx, path, size, quality = job
    try:
        with Image.open(path) as img:
            # JPEG only: decode at reduced scale. Big speedup on huge photos.
            img.draft("RGB", (size, size))
            img = ImageOps.exif_transpose(img)  # honour camera rotation flag
            img = img.convert("RGB")
            w, h = img.size
            scale = size / min(w, h)
            if scale < 1:  # only ever shrink
                img = img.resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))),
                    Image.Resampling.LANCZOS,
                )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            return idx, buf.getvalue(), None
    except Exception as exc:  # corrupt / missing / unsupported file
        return idx, None, f"{type(exc).__name__}: {exc}"


def main():
    import pandas as pd  # imported here so worker processes start fast

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", required=True, help="CSV with path/label/split columns")
    ap.add_argument("--out", required=True, help="output .zip path")
    ap.add_argument("--root", default="", help="prefix for relative paths in the manifest")
    ap.add_argument("--size", type=int, default=256, help="shorter side in pixels (default 256)")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality (default 92)")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--path-col", default="path")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--split-col", default="split")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    missing = [c for c in (args.path_col, args.label_col, args.split_col) if c not in df.columns]
    if missing:
        sys.exit(
            f"Manifest is missing column(s) {missing}. Columns found: {list(df.columns)}\n"
            "Use --path-col / --label-col / --split-col to point at yours."
        )
    df = df.reset_index(drop=True)

    full_paths = [str(Path(args.root) / p) for p in df[args.path_col]]
    # The row number in the name keeps files from different datasets apart
    # even when their original filenames collide.
    arc_names = [
        f"{safe(s)}/{safe(l)}/{i:06d}_{safe(Path(str(p)).stem)}.jpg"
        for i, (s, l, p) in enumerate(zip(df[args.split_col], df[args.label_col], df[args.path_col]))
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(i, p, args.size, args.quality) for i, p in enumerate(full_paths)]
    ok = [False] * len(df)
    failed = []
    t0 = time.time()
    ctx = mp.get_context("spawn")  # same behaviour on Windows and Linux

    # JPEGs are already compressed, so store them as-is (faster, no size penalty).
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        with ctx.Pool(args.workers) as pool:
            for n, (i, data, err) in enumerate(pool.imap_unordered(process, jobs, chunksize=16), 1):
                if err:
                    failed.append((full_paths[i], err))
                else:
                    zf.writestr(arc_names[i], data)
                    ok[i] = True
                if n % 2000 == 0 or n == len(jobs):
                    print(f"{n}/{len(jobs)} done  ({time.time() - t0:.0f}s)", flush=True)

        mask = pd.Series(ok, index=df.index)
        packed = df[mask].copy()
        packed.insert(0, "packed_path", [arc_names[i] for i in packed.index])
        zf.writestr("packed_manifest.csv", packed.to_csv(index=False))

    print(f"\nPacked {int(mask.sum())} images -> {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    print("\nImages per class and split:")
    print(packed.groupby([args.label_col, args.split_col]).size().unstack(fill_value=0))

    if failed:
        log = out.with_suffix(".failed.txt")
        log.write_text("\n".join(f"{p}\t{e}" for p, e in failed), encoding="utf-8")
        print(f"\n{len(failed)} image(s) could not be read -> listed in {log}")


if __name__ == "__main__":
    main()