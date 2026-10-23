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

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import build_model
from src.tracking import init_tracking
from src.classes import CLASS_NAMES
from src.data.dataset import SkinLesionDataset
from src.data.transforms import (
    get_eval_transforms,
    get_train_transforms,
    get_train_transforms_aggressive,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="efficientnet_b0",
                   choices=["efficientnet_b0", "resnet50"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-root", default=None, help="folder the CSV paths are relative to (default: repo root)")
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
    total_loss, n = 0.0, 0
    ys, ps = [], []
    for imgs, labels in loader:
        mask = labels >= 0
        if mask.sum() == 0:
            continue
        imgs, labels = imgs[mask].to(device), labels[mask].to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        n += imgs.size(0)
        ys += labels.cpu().tolist()
        ps += logits.argmax(1).cpu().tolist()
    acc = float(np.mean(np.array(ys) == np.array(ps)))
    macro_f1 = float(f1_score(ys, ps, average="macro"))
    recalls = recall_score(ys, ps, average=None,
                           labels=list(range(len(CLASS_NAMES))), zero_division=0)
    return total_loss / max(n, 1), acc, macro_f1, recalls


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = SkinLesionDataset(
        os.path.join(args.data_dir, "train.csv"),
        transform=get_train_transforms(224),
        transform_overrides={"vitiligo": get_train_transforms_aggressive(224)},
        root=args.image_root,
    )
    val_ds = SkinLesionDataset(
        os.path.join(args.data_dir, "val.csv"),
        transform=get_eval_transforms(224),
        root=args.image_root,
    )

    # class weights from TRAIN counts only
    counts = train_ds.df["class"].value_counts()
    weights = torch.tensor(
        [len(train_ds) / (len(CLASS_NAMES) * counts[c]) for c in CLASS_NAMES],
        dtype=torch.float32,
    ).to(device)

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    model = build_model(args.backbone, num_classes=len(CLASS_NAMES),
                        pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    import mlflow
    init_tracking()
    mlflow.start_run(run_name=f"{args.backbone}-seed{args.seed}")
    mlflow.log_params({
        "backbone": args.backbone, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "optimizer": "adamw",
        "scheduler": "cosine", "class_weighted": True, "pretrained": True,
        "seed": args.seed, "num_classes": len(CLASS_NAMES),
    })

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_f1 = -1.0
    best_path = os.path.join(args.checkpoint_dir, f"{args.backbone}_best.pt")
    mel = CLASS_NAMES.index("melanoma")

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            vl_loss, vl_acc, vl_f1, rec = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            print(f"Epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f} "
                  f"train_acc={tr_acc:.4f}  val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} "
                  f"val_macroF1={vl_f1:.4f} mel_recall={rec[mel]:.3f}  ({time.time()-t0:.0f}s)")
            mlflow.log_metrics({
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": vl_loss, "val_acc": vl_acc, "val_macro_f1": vl_f1,
                **{f"recall_{c}": float(r) for c, r in zip(CLASS_NAMES, rec)},
            }, step=epoch)
            if vl_f1 > best_f1:
                best_f1 = vl_f1
                torch.save({"state_dict": model.state_dict(), "classes": CLASS_NAMES,
                            "backbone": args.backbone, "img_size": 224}, best_path)
    finally:
        if os.path.exists(best_path):
            mlflow.log_artifact(best_path)
        mlflow.end_run()

    print(f"Best val macro-F1: {best_f1:.4f}")
    print(f"Checkpoint saved: {best_path}")


if __name__ == "__main__":
    main()