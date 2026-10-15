# Skin Condition Classifier

A multi-class skin condition image classifier built with transfer learning in PyTorch and deployed through an MLOps pipeline (data versioning, experiment tracking, containerized serving, CI/CD, and monitoring). MLOps class project.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for diagnosis or treatment decisions. Always consult a qualified doctor about skin concerns.

## Status

Work in progress — Week 3 complete (CI green, Dockerized API working, real model integration planned for Week 4).

## Team

- **İrem Kalınsazlıoğlu** — MLOps / infrastructure
- **Rania Benounane** — data / model

## Tech stack

- Python 3.12
- PyTorch (model training) — EfficientNet-B0 primary, ResNet50 ablation
- FastAPI (serving)
- DVC + Hugging Face Storage Buckets (data versioning)
- MLflow on DagsHub (experiment tracking and model registry)
- Docker (containerization)
- GitHub Actions (CI/CD)
- ruff + pytest (lint / test)

## Project structure

```
skin-condition-classifier/
├── .github/workflows/ci.yml    # CI: ruff + pytest on every push and PR
├── .dvc/                       # DVC remote config (Hugging Face bucket)
├── configs/                    # shared config (num_classes, image_size, etc.)
├── data/
│   ├── raw/                    # images, tracked by DVC (raw.dvc pointer)
│   └── splits/                 # train/val/test CSVs, tracked in git
├── entrypoint/
│   ├── train.py                # train one backbone, log to MLflow
│   └── evaluate.py             # per-class F1, confusion matrix, TenCrop
├── notebooks/                  # exploratory data analysis (local only)
├── scripts/                    # one-off helpers (download_*, log_dummy_run)
├── src/
│   ├── classes.py              # 8 class names, alphabetical
│   ├── tracking.py             # init_tracking() for MLflow
│   ├── model.py                # build_model(backbone, num_classes, pretrained)
│   └── data/                   # data pipeline
│       ├── build_manifest.py
│       ├── dedupe.py
│       ├── split.py
│       ├── dataset.py
│       └── transforms.py
├── api/
│   ├── main.py                 # FastAPI: /health, /predict
│   └── model_loader.py         # checkpoint loader (dummy fallback for CI)
├── tests/                      # pytest suite
├── Dockerfile
├── requirements.txt
├── ruff.toml
├── pytest.ini
└── README.md
```

## Data

Datasets are not stored in Git. They are versioned with DVC, and only the small `.dvc` metadata files are committed.

**Training sources (7):** ACNE04, SkinDiseaseClassification, HAM10000, ISIC, DermNet, SkinDisNet, SkinDiseaseBD.

**Fitzpatrick17k** metadata is included for skin-tone analysis. **DDI** is used only for fairness evaluation (Week 5), not for training.

### Data pipeline

| Step | Script | Output |
|---|---|---|
| 1. Collect | `scripts/download_*.py` | raw images in `data/raw/` |
| 2. Manifest | `src/data/build_manifest.py` | `data/manifest.csv` |
| 3. Dedupe | `src/data/dedupe.py` | `data/manifest_deduped.csv` (pHash, ≤2 bit Hamming) |
| 4. Cluster vitiligo | `src/data/cluster_vitiligo.py` | `group_id` overrides for same-patient vitiligo images |
| 5. Split | `src/data/split.py` | `data/splits/{train,val,test}.csv` (patient-level 70/15/15) |

**Numbers:** 18,126 raw images → 2,623 near-duplicates removed (including 994 melanoma duplicates in ISIC) → **15,503 curated images**.

**Split:** train 10,854 / val 2,328 / test 2,321 — **zero patient-level group leakage** (verified).

## Setup

Requires Python 3.12.

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install "dvc[s3]"
```

### Configure DVC to pull the dataset

Add S3 credentials to `~/.aws/credentials`:

```ini
[hf]
aws_access_key_id = <your HFAK key>
aws_secret_access_key = <your secret>
```

Add path-style addressing to `~/.aws/config`:

```ini
[profile hf]
region = us-east-1
endpoint_url = https://s3.hf.co/Raniabo
s3 =
    addressing_style = path
```

Point DVC at the bucket (local config, gitignored):

```bash
dvc remote modify --local hf endpointurl https://s3.hf.co/Raniabo
dvc remote modify --local hf region us-east-1
dvc remote modify --local hf profile hf
```

Pull the data:

```bash
dvc pull
```

This downloads ~5 GB of images into `data/raw/`. Split CSVs are already in the repo under `data/splits/`.

### Lint and test

```bash
ruff check .
pytest
```

Both run automatically in CI on every push and PR.

## Training

```bash
# EfficientNet-B0 baseline (5 epochs)
python -m entrypoint.train --backbone efficientnet_b0 --epochs 5 --batch-size 32

# ResNet50 ablation (same config for fair comparison)
python -m entrypoint.train --backbone resnet50 --epochs 5 --batch-size 32
```

On a T4 GPU: ~5 min/epoch. On CPU: ~30–50 min/epoch.

Checkpoints save to `checkpoints/<backbone>_best.pt`.

## Evaluation

```bash
python -m entrypoint.evaluate --backbone efficientnet_b0
```

Produces per-class precision, recall, F1, and a confusion matrix using TenCrop test-time augmentation.

## Run the API (local)

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive docs.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service status |
| `POST` | `/predict` | Upload an image file, returns predicted class, confidence, and per-class probabilities |

**Example:**

```bash
curl -X POST http://localhost:8000/predict -F "file=@lesion.jpg"
```

**Response:**

```json
{
  "prediction": "melanoma",
  "confidence": 0.87,
  "probabilities": { "melanoma": 0.87, "nevus": 0.06 },
  "model": "efficientnet_b0",
  "disclaimer": "Not a medical diagnostic tool. Consult a qualified dermatologist."
}
```

Non-image uploads return HTTP 400.

The API currently uses a dummy model (random probabilities). The real checkpoint is wired in during Week 4. Results are not a medical diagnosis.

## Docker

```bash
docker build -t skin-api .
docker run -p 8000:8000 skin-api
```

Then open `http://localhost:8000/docs`.

## MLflow experiment tracking

Shared tracking server (DagsHub): `https://dagshub.com/iremkalinsazlioglu/skin-condition-mlflow.mlflow`

### Setup (once per person)

1. Create a free DagsHub account and ask the repo owner for collaborator access.
2. In DagsHub: **Settings → Tokens → Generate New Token**. Copy it right away.
3. In your terminal (PowerShell), set the environment variables:

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/iremkalinsazlioglu/skin-condition-mlflow.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<your DagsHub username>"
$env:MLFLOW_TRACKING_PASSWORD="<your token>"
```

4. In training code, call `init_tracking()` from `src/tracking.py` before starting a run.

**Never commit tokens or put them in code.** Without these variables, tracking falls back to a local `mlflow.db`.

### Logging conventions

- **Experiment name:** `skin-condition-classifier`
- **Metrics** (logged every epoch, `step=epoch`): `train_loss`, `train_acc`, `val_loss`, `val_acc`
- **Params:** `lr`, `batch_size`, `optimizer`, `model_name`, `pretrained`, `seed`
- **Artifacts:** `best_checkpoint.pt`, confusion matrix on val, per-class F1
- **Suggested run names:** `efficientnet_b0-seed42`, `resnet50-seed42`

Smoke test of the setup: `python -m scripts.log_dummy_run`

## Results

*To be updated after Week 4 evaluation.*

| Backbone | Val Acc | Test Acc | Macro F1 |
|---|---|---|---|
| EfficientNet-B0 | — | — | — |
| ResNet50 | — | — | — |

## Limitations

- **Not a medical device.** No regulatory clearance; must not be used for diagnosis.
- **Residual leakage risk.** Four of the seven source datasets do not expose patient identifiers. Vitiligo was clustered by visual similarity to reduce risk; other classes carry a documented residual risk.
- **DermNet watermark** is visible on some eczema images — a potential shortcut the model could learn.
- **Dataset bias.** Geographic, age, and skin-tone distribution is skewed toward the source populations. Fairness analysis on the DDI dataset is planned for Week 5.
- **No hyperparameter optimization.** Compute budget limited us to a manual sweep; automated search (Optuna) is listed as future work.