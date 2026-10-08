@'
# Skin Condition Classifier

A multi-class skin condition image classifier built with transfer learning in PyTorch and deployed through an MLOps pipeline (data versioning, experiment tracking, containerized serving, CI/CD, and monitoring). MLOps class project.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for diagnosis or treatment decisions. Always consult a qualified doctor about skin concerns.

## Status

Work in progress (Week 3: CI green, Dockerized API working, real model integration planned for Week 4).

## Team

- İrem Kalınsazlıoğlu: MLOps / infrastructure
- Rania Benounane: data / model

## Tech stack

- Python 3.12
- PyTorch (model training)
- FastAPI (serving)
- DVC (data versioning)
- MLflow (experiment tracking and model registry)
- Docker (containerization)
- GitHub Actions (CI/CD)

## Project structure

    src/         data processing and model training code
    api/         FastAPI prediction service
    tests/       automated tests
    data/        datasets (not stored in Git, versioned with DVC)
    notebooks/   exploratory data analysis

## Data

Datasets are **not** stored in Git. They are versioned with DVC, and only the small `.dvc` metadata files are committed. Training sources (7): ACNE04, SkinDiseaseClassification, HAM10000, ISIC, DermNet, SkinDisNet, SkinDiseaseBD. Fitzpatrick17k metadata is included for skin-tone analysis; the DDI dataset is used only for fairness evaluation (Week 5), not for training.

## Setup

Requires Python 3.12.

1. Create and activate a virtual environment:

       python -m venv .venv
       .venv\Scripts\activate

2. Install dependencies: `pip install -r requirements.txt`
3. Pull versioned data (DVC remote access must be configured first): `dvc pull`
4. Run lint and tests: `ruff check .` and `pytest`

## Run the API (local)

    uvicorn api.main:app --reload

Open http://127.0.0.1:8000/docs for the interactive docs.

- `GET /health`: service status
- `POST /predict`: upload an image file, returns the predicted class, confidence and per-class probabilities

The API currently uses a dummy model (random probabilities). The real checkpoint is wired in during Week 4. Results are not a medical diagnosis.

## Docker

    docker build -t skin-api .
    docker run -p 8000:8000 skin-api

Then open http://localhost:8000/docs.

## MLflow experiment tracking

Shared tracking server (DagsHub): https://dagshub.com/iremkalinsazlioglu/skin-condition-mlflow.mlflow

### Setup (once per person)

1. Create a free DagsHub account and ask the repo owner for collaborator access.
2. In DagsHub: Settings > Tokens > Generate New Token. Copy it right away.
3. In your terminal (PowerShell), set the environment variables:

       $env:MLFLOW_TRACKING_URI="https://dagshub.com/iremkalinsazlioglu/skin-condition-mlflow.mlflow"
       $env:MLFLOW_TRACKING_USERNAME="<your DagsHub username>"
       $env:MLFLOW_TRACKING_PASSWORD="<your token>"

4. In training code, call `init_tracking()` from `src/tracking.py` before starting a run.

Never commit tokens or put them in code. Without these variables, tracking falls back to a local `mlflow.db`.

### Logging conventions

- Experiment name: `skin-condition-classifier`
- Metrics, logged every epoch with `step=epoch`: `train_loss`, `train_acc`, `val_loss`, `val_acc`
- Params: `lr`, `batch_size`, `optimizer`, `model_name`, `pretrained`, `seed`
- Artifacts: `best_checkpoint.pt`, confusion matrix on val, per-class F1
- Suggested run names: `efficientnet_b0-seed42`, `resnet50-seed42`

Smoke test of the setup: `python -m scripts.log_dummy_run`
'@ | Out-File -Encoding utf8 README.md