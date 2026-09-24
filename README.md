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