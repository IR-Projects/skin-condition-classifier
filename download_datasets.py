"""
Downloads the Tier 1 datasets for the skin condition classifier into data/raw/.

BEFORE RUNNING:
1. Kaggle API key: go to kaggle.com -> Account -> Create New API Token,
   downloads kaggle.json. Place it at C:\\Users\\<you>\\.kaggle\\kaggle.json
2. pip install kaggle isic-cli

WHAT THIS SCRIPT DOES:
- HAM10000: fully automated via Kaggle
- ISIC Archive: fully automated via isic-cli (melanoma, BCC, nevus, seborrheic keratosis)
- DermNet: fully automated via Kaggle mirror (covers eczema, psoriasis, tinea, warts, vitiligo)
- ACNE04: CANNOT be automated. No public API — you must request it from the
  original paper's GitHub repo (search "ACNE04 dataset github"), which
  usually requires an email/form. This script just creates the folder
  and drops a note there so you don't forget.
"""

import subprocess
import sys
from pathlib import Path

RAW = Path("data/raw")
FOLDERS = ["ham10000", "isic_archive", "dermnet", "acne04"]


def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"!! Command failed (exit {result.returncode}) — see error above. Continuing to next dataset.")
    return result.returncode == 0


def main():
    for folder in FOLDERS:
        (RAW / folder).mkdir(parents=True, exist_ok=True)

    # 1. HAM10000 via Kaggle
    print("=" * 60)
    print("1/4 — HAM10000")
    print("=" * 60)
    run(
        ["kaggle", "datasets", "download", "-d", "kmader/skin-cancer-mnist-ham10000",
         "-p", str(RAW / "ham10000"), "--unzip"]
    )

    # 2. ISIC Archive via isic-cli
    print("=" * 60)
    print("2/4 — ISIC Archive (melanoma, BCC, nevus, seborrheic keratosis)")
    print("=" * 60)
    run(
        ["isic", "image", "download",
         "--search", "diagnosis:melanoma OR diagnosis:basal cell carcinoma OR diagnosis:nevus OR diagnosis:seborrheic keratosis",
         str(RAW / "isic_archive")]
    )

    # 3. DermNet via Kaggle mirror
    print("=" * 60)
    print("3/4 — DermNet (eczema, psoriasis, tinea, warts, vitiligo)")
    print("=" * 60)
    run(
        ["kaggle", "datasets", "download", "-d", "shubhamgoel27/dermnet",
         "-p", str(RAW / "dermnet"), "--unzip"]
    )

    # 4. ACNE04 — manual
    print("=" * 60)
    print("4/4 — ACNE04 — MANUAL STEP REQUIRED")
    print("=" * 60)
    note = RAW / "acne04" / "README_MANUAL_DOWNLOAD.txt"
    note.write_text(
        "ACNE04 has no public API/CLI download.\n"
        "Find the original paper's GitHub repo (search 'ACNE04 dataset github') "
        "and follow their request process (usually a form or email to the authors).\n"
        "Once you have the files, place them in this folder.\n"
    )
    print(f"No API available. Instructions written to {note}")

    print("\nDone. Check each folder above for errors before running EDA.")


if __name__ == "__main__":
    main()
