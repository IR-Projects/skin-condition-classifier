# Skin Condition Classifier

[![CI](https://github.com/IR-Projects/skin-condition-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/IR-Projects/skin-condition-classifier/actions/workflows/ci.yml)

An 8-class skin condition image classifier built with transfer learning in PyTorch, wrapped in an MLOps pipeline: versioned data (DVC), tracked and reproducible experiments (MLflow on DagsHub), a containerised FastAPI service, CI on every push, and Prometheus metrics. Built for the MLOps course.

> **Disclaimer:** This is an educational project. It is **not a medical device** and must not be used for diagnosis or treatment decisions. Always consult a qualified doctor about skin concerns.

**Classes (alphabetical, this order is fixed everywhere):** `acne`, `bcc`, `eczema`, `melanoma`, `nail_psoriasis`, `nevus`, `seborrheic_keratosis`, `vitiligo`

## Contents

[Status](#status) · [Results](#results) · [How it fits together](#how-it-fits-together) · [Quick start](#quick-start) · [Data](#data) · [Training](#training) · [Reproducibility](#reproducibility) · [Evaluation](#evaluation) · [API](#api) · [Docker and monitoring](#docker-and-monitoring) · [Experiment tracking](#experiment-tracking-mlflow--dagshub) · [CI](#continuous-integration) · [Project structure](#project-structure) · [Limitations](#limitations) · [Roadmap](#roadmap) · [Team](#team)

## Status

**Done**

- Data pipeline: manifest, near-duplicate removal, patient/lesion-aware 70/15/15 split, DVC versioning
- Baseline model: EfficientNet-B0 (ImageNet-pretrained) trained and tracked in MLflow
- Training is seeded and logs its environment (git commit, library versions, GPU)
- Evaluation script: per-class metrics, confusion matrix, per-source breakdown, confidence-threshold sweep
- API that loads the real checkpoint, Dockerfile and docker-compose, `/metrics` endpoint
- CI: lint, tests and Docker build

**In progress / next:** see the [Roadmap](#roadmap).

## Results

EfficientNet-B0, seed 42, 15 epochs; the checkpoint from the best epoch (8) is kept. Numbers below are on the **validation** split. The test split has not been used yet.

| Model | Val accuracy | Val macro-F1 | Val melanoma recall | Test |
|---|---|---|---|---|
| EfficientNet-B0 | 0.9055 | 0.9083 | 0.761 | pending |
| ResNet50 (ablation) | pending | pending | pending | pending |

How to read these:

- The checkpoint was **selected on the validation set**, so these numbers are optimistic. Final numbers will come from the held-out test split, evaluated once per final model.
- Melanoma recall on validation fluctuated a lot between epochs (about 0.60 to 0.86). Macro-F1 alone picks a checkpoint with weaker melanoma recall than some earlier epochs, so both numbers are reported.
- Re-running the baseline reproduced the same per-epoch validation metrics (see [Reproducibility](#reproducibility)).
- Full curves and per-class recall for every run are in [MLflow on DagsHub](https://dagshub.com/iremkalinsazlioglu/skin-condition-mlflow.mlflow).

## How it fits together

```
 raw datasets ──► manifest ──► dedupe ──► patient-aware split ──► data/splits/*.csv  (git)
                                                  │
                                                  ▼
                              pack_dataset.py: resize to 256 px, one zip ──► DVC ──► Hugging Face bucket
                                                  │
                                                  ▼
                       Kaggle / local GPU: entrypoint/train.py  ──►  MLflow (DagsHub): params, metrics, tags
                                                  │
                                                  ▼
                              checkpoint  ──►  entrypoint/evaluate.py  (metrics, confusion matrix, plots)
                                                  │
                                                  ▼
                       FastAPI (api/)  ──►  Docker / docker-compose  ──►  /health  /predict  /metrics
                                                  ▲
                              GitHub Actions: ruff + pytest + docker build
```

## Quick start

Requires **Python 3.12**.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements-dev.txt   # everything: training, evaluation, tests, DVC
ruff check .
pytest
```

`requirements-prod.txt` is the slim set used by the Docker image (API only, CPU PyTorch).

### Run the API with a trained checkpoint

Checkpoints are not stored in git. Put one in a `models/` folder (for example the file from your MLflow run) and point `MODEL_PATH` at it:

```powershell
# PowerShell
$env:MODEL_PATH="models/efficientnet_b0_best.pt"
uvicorn api.main:app --reload
```

```bash
# bash
MODEL_PATH=models/efficientnet_b0_best.pt uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`. If `MODEL_PATH` is missing or the file is not found, the API starts in **dummy mode and returns random probabilities** (this keeps CI offline). `GET /health` shows which one is running: `"model": "dummy"` means no real model was loaded.

## Data

Datasets are not stored in git. Only small `.dvc` pointer files and the split CSVs are committed.

**Training sources (7):** ACNE04, SkinDiseaseClassification, HAM10000, ISIC Archive, DermNet (eczema folder only), SkinDisNet (eczema only), SkinDiseaseBD (vitiligo only).

**Held out from training:** the Fitzpatrick17k metadata (`fitzpatrick17k.csv`) is kept for skin-tone analysis, and the DDI dataset is used only for fairness evaluation, never for training.

> Some source datasets restrict use to academic or non-commercial research (for example ACNE04 and DDI). Check each licence before redistributing images or models trained on them.

### Classes and where they come from

| Class | Train | Val | Test | Sources |
|---|---|---|---|---|
| acne | 923 | 198 | 197 | ACNE04, SkinDiseaseClassification |
| bcc | 1,133 | 243 | 242 | ISIC, HAM10000 |
| eczema | 1,331 | 285 | 285 | DermNet, SkinDisNet |
| melanoma | 1,134 | 243 | 242 | ISIC, HAM10000 |
| nail_psoriasis | 1,446 | 310 | 309 | SkinDiseaseClassification |
| nevus | 1,708 | 366 | 366 | ISIC, HAM10000 |
| seborrheic_keratosis | 1,802 | 386 | 385 | ISIC, HAM10000 |
| vitiligo | 1,377 | 297 | 295 | SkinDiseaseBD, SkinDiseaseClassification |
| **Total** | **10,854** | **2,328** | **2,321** | |

### Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Collect | `download_datasets.py`, `isic_remaining.py` (HAM10000, ISIC, DermNet, ACNE04); the other sources were downloaded manually | raw images in `data/raw/` |
| 2. Manifest | `src/data/build_manifest.py` | `data/manifest.csv` (nevus capped at 2,500 images across sources) |
| 3. Dedupe | `src/data/dedupe.py` | `data/manifest_deduped.csv` (perceptual hash, Hamming distance ≤ 2) |
| 4. Cluster vitiligo | `src/data/cluster_vitiligo.py` | `group_id` overrides for photos that look like the same patient |
| 5. Split | `src/data/split.py` | `data/splits/{train,val,test}.csv` |
| 6. Pack | `scripts/make_split_manifest.py`, `scripts/pack_dataset.py`, `scripts/make_packed_splits.py` | `data/processed/skin_256.zip` and `data/splits_packed/*.csv` |

**Numbers:** 18,126 raw images, 2,623 near-duplicates removed (994 of them melanoma duplicates in ISIC), **15,503 curated images**.

**Split:** stratified by class, grouped so no group is split across train/val/test. A group is a patient where the source provides patient IDs (SkinDisNet), a lesion for HAM10000, a visual cluster for vitiligo, and otherwise a single image. ISIC and HAM10000 share image IDs, so an ISIC image is put in the same group as its HAM10000 copy. Checked on the committed files: no `group_id`, file path or image filename appears in more than one split.

**Packed data:** training uses a single 256 px archive (about 314 MB) instead of tens of thousands of full-size files. The raw folder is about 17.9 GB across 96,875 files, which is why pulling it is slow. Images are resized with the shorter side at 256 px and never upscaled, and phone-rotation (EXIF) metadata is applied.

### Getting the data

```bash
pip install -r requirements-dev.txt          # includes dvc[s3]
```

Add the Hugging Face S3 credentials to `~/.aws/credentials`:

```ini
[hf]
aws_access_key_id = <your HFAK key>
aws_secret_access_key = <your secret>
```

and `~/.aws/config`:

```ini
[profile hf]
region = us-east-1
endpoint_url = https://s3.hf.co/Raniabo
s3 =
    addressing_style = path
```

then point DVC at the bucket (local config, not committed):

```bash
dvc remote modify --local hf endpointurl https://s3.hf.co/Raniabo
dvc remote modify --local hf region us-east-1
dvc remote modify --local hf profile hf
```

Pull the packed training archive and unzip it:

```bash
dvc pull data/processed/skin_256.zip.dvc
unzip -q data/processed/skin_256.zip -d data/packed   # add /packed to data/.gitignore
```

Only run a bare `dvc pull` if you need to rebuild the dataset from the raw images, because it downloads the full ~17.9 GB.

## Training

```bash
# on the packed 256 px images (recommended)
python -m entrypoint.train --backbone efficientnet_b0 \
    --data-dir data/splits_packed --image-root data/packed

# ResNet50 ablation: same settings, different backbone
python -m entrypoint.train --backbone resnet50 \
    --data-dir data/splits_packed --image-root data/packed

# on the full-size raw images (needs the full dvc pull)
python -m entrypoint.train --backbone efficientnet_b0
```

Options: `--epochs` (15), `--batch-size` (32), `--lr` (1e-4), `--seed` (42), `--num-workers` (2), `--checkpoint-dir` (`checkpoints`).

| Setting | Value |
|---|---|
| Backbone | EfficientNet-B0 (ImageNet-pretrained, `IMAGENET1K_V1`); ResNet50 (`IMAGENET1K_V2`) as ablation |
| Input | 224 × 224, ImageNet mean/std normalisation |
| Loss | Cross-entropy with class weights computed from the training counts |
| Optimiser | AdamW, learning rate 1e-4, weight decay 1e-4, cosine schedule |
| Augmentation (train only) | resize 256, random-resized crop (scale 0.8–1.0), flips, rotation ±20°, light colour jitter. Vitiligo uses a tighter crop (scale 0.5–1.0). |
| Evaluation transform | resize 256, centre crop 224 |
| Checkpoint rule | best validation macro-F1; saved as a dict with `state_dict`, `classes`, `backbone`, `img_size` |

Colour jitter is deliberately light because colour carries diagnostic information (for example melanoma variegation).

### Training on Kaggle

The baseline was trained on a Kaggle GPU (about one minute per epoch on the packed data). In the notebook:

1. Add the Hugging Face and DagsHub credentials as Kaggle Secrets and enable Internet.
2. Write the AWS credentials/config files from the secrets, then clone this repo and `pip install "dvc[s3]" mlflow scikit-learn`. Torch and torchvision are already installed on Kaggle.
3. `dvc pull data/processed/skin_256.zip.dvc`, then unzip to `/tmp/skin`.
4. Set `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` from the secrets.
5. Run `python -m entrypoint.train --data-dir data/splits_packed --image-root /tmp/skin --checkpoint-dir /kaggle/working/checkpoints`.

Never print or commit tokens.

## Reproducibility

- **Seeds:** Python, NumPy, PyTorch and CUDA are seeded from `--seed`, and cuDNN runs in deterministic mode.
- **Environment is recorded:** every run logs the `git_commit`, `torch` version and `gpu` as MLflow tags, and the full `pip freeze` as an artifact.
- **Data is versioned:** the exact training archive is identified by the hash in `data/processed/skin_256.zip.dvc`, and the splits are committed.
- **Fixed class order:** `src/classes.py` holds the single class list, and the checkpoint stores it. Evaluation refuses a checkpoint whose class order differs.
- **Observed:** re-running the baseline on Kaggle reproduced the same per-epoch validation metrics.

Exact numbers can still differ across GPU types or library versions, which is why the environment is logged with each run.

## Evaluation

```bash
# sanity check on validation: should match the checkpoint's best val macro-F1
python -m entrypoint.evaluate --checkpoint models/efficientnet_b0_best.pt --split val

# final numbers, once per final model, with 10-crop test-time augmentation
python -m entrypoint.evaluate --checkpoint models/efficientnet_b0_best.pt \
    --split test --tencrop --out-dir reports/test --log-mlflow
```

For the packed data add `--data-dir data/splits_packed --image-root <unzipped folder>`. The default split is `val` on purpose: use `test` only for final numbers and never to choose hyper-parameters or checkpoints.

Files written to `--out-dir` (default `reports/<split>`):

| File | Contents |
|---|---|
| `metrics.json` | accuracy, macro-F1, balanced accuracy, melanoma recall and precision, false-positive rate, calibration error (ECE) |
| `per_class.csv` | precision, recall, F1, support, false positives, false-positive rate per class |
| `confusion_matrix.csv` / `.png` | counts and row-normalised matrix |
| `per_source.csv` | recall per (source dataset, class), to spot dataset shortcuts |
| `threshold_sweep.csv` / `.png` | coverage, accuracy and precision if low-confidence predictions are rejected |
| `predictions.csv` | one row per image with the prediction, confidence and all probabilities |

`--log-mlflow` also uploads these as an evaluation run.

## API

```bash
curl -X POST http://localhost:8000/predict -F "file=@lesion.jpg"
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | `{"status": "ok", "model": "<checkpoint file name or dummy>"}` |
| `POST` | `/predict` | Upload one JPEG or PNG (max 5 MB); returns the predicted class and all probabilities |
| `GET` | `/metrics` | Prometheus-format metrics |
| `GET` | `/docs` | Interactive OpenAPI docs |

Response from `/predict`:

```json
{
  "prediction": "melanoma",
  "confidence": 0.87,
  "probabilities": { "acne": 0.01, "bcc": 0.03, "eczema": 0.00, "melanoma": 0.87,
                     "nail_psoriasis": 0.00, "nevus": 0.06, "seborrheic_keratosis": 0.02, "vitiligo": 0.01 },
  "model": "efficientnet_b0_best.pt",
  "disclaimer": "This is not a medical diagnosis. Consult a doctor."
}
```

Wrong file types, files over 5 MB and corrupted images return HTTP 400. Preprocessing in the API is the same as the evaluation transform (resize 256, centre crop 224, ImageNet normalisation).

## Docker and monitoring

The image is CPU-only and does not contain a model. Mount your checkpoint and set `MODEL_PATH`:

```bash
make docker-build      # docker build -t skin-api .
make docker-run        # mounts ./models and sets MODEL_PATH=models/efficientnet_b0_best.pt
```

or with Compose (adds a health check and restart policy):

```bash
docker compose up      # or: make compose-up / make compose-down
```

Then open `http://localhost:8000/docs`. As with a local run, a missing checkpoint means dummy mode: check `/health`.

`/metrics` exposes standard Prometheus process and runtime metrics. A Prometheus/Grafana stack is not part of this repository yet.

`make help` lists all shortcuts (`lint`, `test`, `run`, `docker-build`, `docker-run`, `compose-up`, `compose-down`).

## Experiment tracking (MLflow + DagsHub)

Shared tracking server: `https://dagshub.com/iremkalinsazlioglu/skin-condition-mlflow.mlflow`

Setup (once per person):

1. Create a free DagsHub account and ask the repo owner for collaborator access.
2. In DagsHub: **Settings → Tokens → Generate New Token**. Copy it immediately.
3. Set these environment variables (PowerShell):

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/iremkalinsazlioglu/skin-condition-mlflow.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<your DagsHub username>"
$env:MLFLOW_TRACKING_PASSWORD="<your token>"
```

**Never commit tokens or put them in code.** Without these variables, tracking falls back to a local `mlflow.db`, and the run will not be visible to the team.

Conventions:

- **Experiment:** `skin-condition-classifier`; **run names** like `efficientnet_b0-seed42`, `resnet50-seed42`
- **Metrics, every epoch:** `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_macro_f1`, and `recall_<class>` for each class
- **Params:** `backbone`, `epochs`, `batch_size`, `lr`, `optimizer`, `scheduler`, `class_weighted`, `pretrained`, `seed`, `num_classes`
- **Tags:** `git_commit`, `torch`, `gpu`
- **Artifacts:** `pip_freeze.txt`, the best checkpoint, and evaluation reports

Smoke test the connection: `python -m scripts.log_dummy_run` (delete the `dummy-run` afterwards).

## Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) runs on pushes and pull requests to `main`:

1. `ruff check .`
2. `pytest` (API, class list, data splits, model builder; offline, no pretrained downloads)
3. `docker build` of the API image, only if the first two pass

Run the same checks locally before every push: `ruff check .`, then `pytest`.

## Project structure

```
skin-condition-classifier/
├── .github/workflows/ci.yml       # ruff, pytest, docker build
├── .dvc/                          # DVC remote config (Hugging Face bucket)
├── api/
│   ├── main.py                    # FastAPI: /health, /predict, /metrics
│   └── model_loader.py            # loads the checkpoint (dummy fallback for CI)
├── configs/config.yaml            # early defaults, not read by the code (see train.py arguments)
├── data/
│   ├── raw.dvc                    # full raw images (DVC pointer)
│   ├── processed/skin_256.zip.dvc # packed 256 px training archive (DVC pointer)
│   ├── splits/                    # train/val/test CSVs (full-size paths)
│   └── splits_packed/             # same splits, paths inside the packed archive
├── entrypoint/
│   ├── train.py                   # train one backbone, log to MLflow
│   └── evaluate.py                # metrics, confusion matrix, per-source, threshold sweep
├── scripts/
│   ├── pack_dataset.py            # resize + zip images
│   ├── make_split_manifest.py     # merge splits into one manifest for packing
│   ├── make_packed_splits.py      # write split CSVs for the packed archive
│   └── log_dummy_run.py           # MLflow connection test
├── src/
│   ├── classes.py                 # the 8 class names, fixed order
│   ├── model.py                   # build_model(backbone, num_classes, pretrained)
│   ├── tracking.py                # init_tracking() for MLflow
│   └── data/                      # build_manifest, dedupe, cluster_vitiligo, split, dataset, transforms
├── tests/                         # pytest suite
├── notebooks/                     # exploratory data analysis (not linted)
├── download_datasets.py           # data download helpers
├── isic_remaining.py
├── fitzpatrick17k.csv             # skin-tone metadata, for fairness analysis
├── Dockerfile  docker-compose.yml  Makefile
├── requirements-prod.txt          # API / Docker
├── requirements-dev.txt           # training, evaluation, tests, DVC
├── ruff.toml  pytest.ini
└── status.md                      # working notes
```

## Limitations

- **Not a medical device.** No regulatory clearance or clinical validation.
- **Image sources differ by class.** Melanoma, nevus, basal cell carcinoma and seborrheic keratosis come only from ISIC and HAM10000, which are mostly dermoscopy images. The other four classes come from different datasets. The model may partly learn "which dataset does this look like" instead of the condition, and ordinary photos of lesions can be misclassified. `per_source.csv` from the evaluation script shows this per source.
- **Residual leakage risk.** Several sources (ACNE04, DermNet eczema, SkinDiseaseBD, SkinDiseaseClassification, ISIC) do not expose patient or lesion identifiers, so photos of the same lesion or person may still land in different splits. Vitiligo was clustered by visual similarity to reduce this.
- **DermNet watermark.** A visible watermark on some eczema images is a possible shortcut.
- **Dataset bias.** Age, geography and skin-tone distribution follow the source datasets. A skin-tone fairness analysis on DDI is planned.
- **Selection on validation.** The checkpoint is chosen on the validation split, so validation scores are optimistic. Report test numbers for final claims.
- **Class weighting trades precision for recall** on smaller classes and can raise their false-positive rate. Check per-class precision, not only accuracy.
- **No automated hyper-parameter search yet.** Current settings are manual.

## Roadmap

- [x] Data pipeline, patient/lesion-aware splits, DVC
- [x] EfficientNet-B0 baseline tracked in MLflow
- [x] API with real model loading, Docker, docker-compose, CI, `/metrics`
- [x] Evaluation script
- [ ] Test-set evaluation of the final model (10-crop)
- [ ] ResNet50 ablation with identical settings
- [ ] Hyper-parameter search with Optuna (nested MLflow runs)
- [ ] MLflow Model Registry; API loads the champion model from the registry
- [ ] DDI skin-tone fairness analysis
- [ ] Grad-CAM visualisations
- [ ] Confidence threshold and an "uncertain" answer; calibration
- [ ] Monitoring dashboard (Prometheus/Grafana)

## Team

- **Rania Benounane**: data and model
- **İrem Kalınsazlıoğlu**: MLOps and infrastructure