import mlflow

from src.tracking import init_tracking

init_tracking()

with mlflow.start_run(run_name="dummy-run"):
    mlflow.log_params(
        {
            "lr": 0.001,
            "batch_size": 32,
            "optimizer": "adam",
            "model_name": "dummy",
            "pretrained": False,
            "seed": 42,
        }
    )
    for epoch in range(5):
        mlflow.log_metric("train_loss", 1.0 / (epoch + 1), step=epoch)
        mlflow.log_metric("train_acc", 0.5 + epoch * 0.1, step=epoch)
        mlflow.log_metric("val_loss", 1.1 / (epoch + 1), step=epoch)
        mlflow.log_metric("val_acc", 0.45 + epoch * 0.1, step=epoch)