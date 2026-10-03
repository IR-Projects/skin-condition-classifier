"""
Pulls the remaining ISIC Archive categories (BCC, seborrheic keratosis, melanoma).
Run from the project root with the venv active: python isic_remaining.py
"""
import subprocess

pulls = [
    ('diagnosis_3:"Basal cell carcinoma"', "data/raw/isic_archive/bcc"),
    ('diagnosis_3:"Seborrheic keratosis"', "data/raw/isic_archive/seborrheic_keratosis"),
    ('diagnosis_3:"Melanoma, NOS"', "data/raw/isic_archive/melanoma"),
]

for search, outdir in pulls:
    print("=" * 60)
    print(f"Pulling: {search} -> {outdir}")
    print("=" * 60)
    result = subprocess.run(
        ["isic", "image", "download", "--search", search, "--limit", "1500", outdir]
    )
    if result.returncode != 0:
        print(f"!! Failed on {search} (exit {result.returncode}) — check message above.")
    else:
        print(f"Done: {outdir}")

print("\nAll pulls attempted.")
