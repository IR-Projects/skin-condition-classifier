"""
Build a unified manifest of every image we'll train on.

Output: data/manifest.csv with columns:
    path        absolute path to the image
    class       our target class name (8 classes)
    source      which dataset it came from
    patient_id  if the source provides one (else None)
    lesion_id   if the source provides one (else None)
    group_id    what we group by for patient-level splits
                (priority: patient_id > lesion_id > image path)
"""
import os
import glob
import pandas as pd

BASE = r"C:\Users\TOSHIBA\Desktop\skin_Conditions_AI\data\raw"
OUT_DIR = r"C:\Users\TOSHIBA\Desktop\skin_Conditions_AI\data"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    ]


rows = []

# ---------------- ISIC ----------------
isic_path = os.path.join(BASE, "isic_archive")
for class_name in os.listdir(isic_path):
    folder = os.path.join(isic_path, class_name)
    if not os.path.isdir(folder):
        continue

    # Try to pull lesion_id from per-class metadata.csv if it exists
    lesion_map = {}
    meta_path = os.path.join(folder, "metadata.csv")
    if os.path.exists(meta_path):
        try:
            m = pd.read_csv(meta_path)
            if "image" in m.columns and "lesion_id" in m.columns:
                lesion_map = dict(zip(m["image"], m["lesion_id"]))
        except Exception:
            pass

    for fpath in list_images(folder):
        stem = os.path.splitext(os.path.basename(fpath))[0]
        lesion_id = lesion_map.get(stem, lesion_map.get(os.path.basename(fpath)))
        rows.append({
            "path": fpath,
            "class": class_name,
            "source": "ISIC",
            "patient_id": None,
            "lesion_id": lesion_id,
            "group_id": lesion_id if lesion_id else fpath,
        })

# ---------------- HAM10000 ----------------
ham_meta = pd.read_csv(os.path.join(BASE, "ham10000", "HAM10000_metadata.csv"))
dx_map = {"mel": "melanoma", "nv": "nevus", "bcc": "bcc", "bkl": "seborrheic_keratosis"}

def find_ham_image(image_id):
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        cand = os.path.join(BASE, "ham10000", part, image_id + ".jpg")
        if os.path.exists(cand):
            return cand
    return None

for _, r in ham_meta.iterrows():
    cls = dx_map.get(r["dx"])
    if cls is None:
        continue
    fpath = find_ham_image(r["image_id"])
    if fpath is None:
        continue
    pid = r.get("patient_id")
    lesion = r.get("lesion_id")
    if pd.notna(pid):
        group = f"ham_pat_{pid}"
    elif pd.notna(lesion):
        group = f"ham_les_{lesion}"
    else:
        group = fpath
    rows.append({
        "path": fpath,
        "class": cls,
        "source": "HAM10000",
        "patient_id": str(pid) if pd.notna(pid) else None,
        "lesion_id": str(lesion) if pd.notna(lesion) else None,
        "group_id": group,
    })

# ---------------- ACNE04 ----------------
for grade in ["acne0_1024", "acne1_1024", "acne2_1024", "acne3_1024"]:
    for fpath in list_images(os.path.join(BASE, "acne04", grade)):
        rows.append({
            "path": fpath, "class": "acne", "source": "ACNE04",
            "patient_id": None, "lesion_id": None, "group_id": fpath,
        })

# ---------------- Skin Disease Classification Dataset ----------------
sdc = os.path.join(BASE, "Skin Disease Classification Dataset")
for sub, cls in [("acne", "acne"),
                 ("Vitiligo", "vitiligo"),
                 ("Nail_psoriasis", "nail_psoriasis")]:
    for fpath in list_images(os.path.join(sdc, sub)):
        rows.append({
            "path": fpath, "class": cls, "source": "SkinDiseaseClassification",
            "patient_id": None, "lesion_id": None, "group_id": fpath,
        })

# ---------------- DermNet (eczema only) ----------------
for split in ["train", "test"]:
    folder = os.path.join(BASE, "dermnet", split, "Eczema Photos")
    for fpath in list_images(folder):
        rows.append({
            "path": fpath, "class": "eczema", "source": "DermNet",
            "patient_id": None, "lesion_id": None, "group_id": fpath,
        })

# ---------------- SkinDisNet (eczema only, patient-level) ----------------
sdn_dirs = glob.glob(os.path.join(BASE, "SkinDisNet*"))
if sdn_dirs:
    sdn_root = sdn_dirs[0]
    sdn_csvs = glob.glob(os.path.join(sdn_root, "**", "SkinDisNet_Metadata.csv"),
                         recursive=True)
    if sdn_csvs:
        sdn_df = pd.read_csv(sdn_csvs[0])
        sdn_df["Diagnosis"] = sdn_df["Diagnosis"].str.strip()
        ec = sdn_df[sdn_df["Diagnosis"] == "Eczema"]

        # Prefer Preprocessed over Augmented — we do our own augmentation
        for sub in ["Preprocessed", "Augmented"]:
            sub_dir = os.path.join(sdn_root, sub)
            if not os.path.isdir(sub_dir):
                continue
            found_any = False
            for folder_name in os.listdir(sub_dir):
                if "Eczema" not in folder_name:
                    continue
                for fpath in list_images(os.path.join(sub_dir, folder_name)):
                    found_any = True
                    stem = os.path.splitext(os.path.basename(fpath))[0]
                    m = ec[ec["Image_id"] == stem]
                    pid = m.iloc[0]["Patient_id"] if not m.empty else None
                    rows.append({
                        "path": fpath, "class": "eczema", "source": "SkinDisNet",
                        "patient_id": pid, "lesion_id": None,
                        "group_id": f"sdn_{pid}" if pid else fpath,
                    })
            if found_any:
                break  # don't double-count from Preprocessed AND Augmented

# ---------------- SkinDiseaseBD (vitiligo only; tinea dropped) ----------------
sdbd = os.path.join(
    BASE, "SkinDiseaseBD A Dataset of Common Skin Disease Ima", "Updated Images"
)
for fpath in list_images(os.path.join(sdbd, "Vitiligo")):
    rows.append({
        "path": fpath, "class": "vitiligo", "source": "SkinDiseaseBD",
        "patient_id": None, "lesion_id": None, "group_id": fpath,
    })

manifest = pd.DataFrame(rows)

# ---------------- Cap nevus to 2500 across ALL sources ----------------
NEVUS_CAP = 2500
nevus = manifest[manifest["class"] == "nevus"]
other = manifest[manifest["class"] != "nevus"]
print(f"Nevus before cap: {len(nevus)}")
if len(nevus) > NEVUS_CAP:
    nevus = nevus.sample(n=NEVUS_CAP, random_state=42)
    print(f"Nevus after cap:  {len(nevus)}")
manifest = pd.concat([other, nevus], ignore_index=True)

# ---------------- Report + save ----------------
print("\nClass totals after nevus cap:")
print(manifest["class"].value_counts().sort_index())

print("\nBreakdown by class × source:")
print(manifest.groupby(["class", "source"]).size())

out_path = os.path.join(OUT_DIR, "manifest.csv")
manifest.to_csv(out_path, index=False)
print(f"\nManifest saved to: {out_path}")
print(f"Total rows: {len(manifest)}")