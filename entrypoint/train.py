"""
Training entry point.

Usage:
    python -m entrypoint.train --backbone efficientnet_b0 --epochs 5 --batch-size 32

Logs per-epoch metrics to MLflow via src/tracking.init_tracking().
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import build_model
from src.tracking import init_tracking
from src.classes import CLASS_NAMES
from src.data.dataset import SkinLesionDataset
from src.data.transforms import get_train_transforms, get_eval_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="efficientnet_b0",
                   choices=["efficientnet_b0", "resnet50"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data/splits")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        # Drop corrupt-image sentinels (label -1)
        mask = labels >= 0
        if mask.sum() == 0:
            continue
        imgs, labels = imgs[mask].to(device), labels[mask].to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += imgs.size(0)

    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        mask = labels >= 0
        if mask.sum() == 0:
            continue
        imgs, labels = imgs[mask].to(device), labels[mask].to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Datasets
    train_ds = SkinLesionDataset(
        os.path.join(args.data_dir, "train.csv"),
        transform=get_train_transforms(224),
    )

    val_ds = SkinLesionDataset(
        os.path.join(args.data_dir, "val.csv"),
        transform=get_eval_transforms(224),
    )

  
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Model
    model = build_model(args.backbone, num_classes=len(CLASS_NAMES),
                        pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # MLflow — B's init_tracking() takes no args and configures the
    # global mlflow module. We then open a run explicitly.
    import mlflow
    run_name = f"{args.backbone}-seed{args.seed}"
    init_tracking()
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "adamw",
        "pretrained": True,
        "seed": args.seed,
        "num_classes": len(CLASS_NAMES),
    })

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    best_path = os.path.join(args.checkpoint_dir, f"{args.backbone}_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}  ({dt:.0f}s)")

        mlflow.log_metrics({
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_acc": vl_acc,
        }, step=epoch)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_path)

    mlflow.log_metric("best_val_acc", best_val_acc)
    mlflow.log_artifact(best_path)
    mlflow.end_run()
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Checkpoint saved: {best_path}")


if __name__ == "__main__":
    main()