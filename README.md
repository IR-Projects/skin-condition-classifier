# Skin Condition Classifier

A multi-class skin condition image classifier built with transfer learning in PyTorch and deployed through an MLOps pipeline (data versioning, experiment tracking, containerized serving, CI/CD, and monitoring). MLOps class project.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for diagnosis or treatment decisions. Always consult a qualified doctor about skin concerns.

## Status

Work in progress (week 1: repository setup).

## Team

- <Your name>: MLOps / infrastructure
- <Teammate name>: data / model

## Tech stack

- Python 3.12
- PyTorch (model training)
- FastAPI (serving)
- DVC (data versioning)
- MLflow (experiment tracking and model registry)
- Docker (containerization)
- GitHub Actions (CI/CD)

## Project structure

```
src/         data processing and model training code
api/         FastAPI prediction service
tests/       automated tests
data/        datasets (not stored in Git, versioned with DVC)
notebooks/   exploratory data analysis
```

## Data

Datasets are **not** stored in Git. They are versioned with DVC, and only the small `.dvc` metadata files are committed. Sources: ISIC Archive, HAM10000, DermNet, ACNE04, and DDI.

## Setup

Setup instructions will be added once the environment is finalized.

## MLflow experiment tracking

Shared tracking server (DagsHub): https://dagshub.com/iremkalinsazlioglu/skin-condition-mlflow.mlflow

### Setup (once per person)
1. Create a free DagsHub account and ask the repo owner for collaborator access.
2. In DagsHub: Settings > Tokens > Generate New Token. Copy it right away.
3. In your terminal (PowerShell), set the environment variables:
```
   $env:MLFLOW_TRACKING_URI="https://dagshub.com/iremkalinsazlioglu/skin-condition-mlflow.mlflow"
   $env:MLFLOW_TRACKING_USERNAME="<your DagsHub username>"
   $env:MLFLOW_TRACKING_PASSWORD="<your token>"
```
4. In training code, call `init_tracking()` from `src/tracking.py` before starting a run.

Never commit tokens or put them in code. Without these variables, tracking falls back to a local `mlflow.db`.

### Logging conventions
- Experiment name: `skin-condition-classifier`
- Metrics, logged every epoch with `step=epoch`: `train_loss`, `train_acc`, `val_loss`, `val_acc`
- Params: `lr`, `batch_size`, `optimizer`, `model_name`, `pretrained`, `seed`
- Artifacts: `best_checkpoint.pt`, confusion matrix on val, per-class F1
- Suggested run names: `efficientnet_b0-seed42`, `resnet50-seed42`

Smoke test of the setup: `python -m scripts.log_dummy_run`