"""
Training script for ResNet-50 NSCLC subtype classifier.

Trains on Kaggle LC25000 dataset (LUAD vs LUSC).
Saves best model weights to outputs/models/resnet_best.pth
Saves training curves to outputs/figures/

Usage (from project root, with nsclc env active):
    python src/train_resnet.py
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Allow imports from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.kaggle_dataset import KaggleNSCLCDataset, get_class_weights
from src.models.resnet_encoder import build_model


# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "data_dir"      : "data/raw",
    "model_dir"     : "outputs/models",
    "figures_dir"   : "outputs/figures",
    "results_dir"   : "outputs/results",

    "num_classes"   : 2,
    "embedding_dim" : 256,
    "dropout"       : 0.4,

    "epochs"        : 15,
    "batch_size"    : 32,
    "num_workers"   : 4,
    "learning_rate" : 1e-4,
    "weight_decay"  : 1e-4,

    # Freeze backbone for first N epochs, then unfreeze for fine-tuning
    "freeze_epochs" : 3,

    "seed"          : 42,
    "device"        : "cuda" if torch.cuda.is_available() else "cpu",
}

# ─── Reproducibility ──────────────────────────────────────────────────────────

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
if CONFIG["device"] == "cuda":
    torch.cuda.manual_seed_all(CONFIG["seed"])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_dirs():
    for d in [CONFIG["model_dir"], CONFIG["figures_dir"], CONFIG["results_dir"]]:
        os.makedirs(d, exist_ok=True)


def save_curves(train_losses, val_losses, train_accs, val_accs):
    """Save training/validation loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, "b-o", label="Train loss", markersize=4)
    ax1.plot(epochs, val_losses,   "r-o", label="Val loss",   markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-o", label="Train accuracy", markersize=4)
    ax2.plot(epochs, val_accs,   "r-o", label="Val accuracy",   markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training vs Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CONFIG["figures_dir"], "resnet_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves → {path}")


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(
                f"  Batch [{batch_idx+1}/{len(loader)}]  "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels,
                  [1 if p > 0.5 else 0 for p in all_probs],
                  average="weighted")

    return avg_loss, accuracy, auc, f1, all_labels, all_probs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    make_dirs()
    device = torch.device(CONFIG["device"])
    print(f"\n{'='*60}")
    print(f" NSCLC ResNet-50 Subtype Classifier Training")
    print(f"{'='*60}")
    print(f" Device      : {device}")
    if device.type == "cuda":
        print(f" GPU         : {torch.cuda.get_device_name(0)}")
    print(f" Epochs      : {CONFIG['epochs']}")
    print(f" Batch size  : {CONFIG['batch_size']}")
    print(f" LR          : {CONFIG['learning_rate']}")
    print(f"{'='*60}\n")

    # ── Datasets & loaders ──
    train_dataset = KaggleNSCLCDataset(
        root_dir=CONFIG["data_dir"], split="train", seed=CONFIG["seed"]
    )
    val_dataset = KaggleNSCLCDataset(
        root_dir=CONFIG["data_dir"], split="val", seed=CONFIG["seed"]
    )
    test_dataset = KaggleNSCLCDataset(
        root_dir=CONFIG["data_dir"], split="test", seed=CONFIG["seed"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    # ── Model ──
    model = build_model(
        num_classes=CONFIG["num_classes"],
        pretrained=True,
        embedding_dim=CONFIG["embedding_dim"],
        dropout=CONFIG["dropout"],
    ).to(device)

    # Freeze backbone initially — only train projection + classifier heads
    model.freeze_backbone()
    print("Backbone frozen for first", CONFIG["freeze_epochs"], "epochs.\n")

    # ── Loss with class weights ──
    class_weights = get_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    )

    # ── Training ──
    best_val_auc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, CONFIG["epochs"] + 1):
        # Unfreeze backbone after freeze_epochs
        if epoch == CONFIG["freeze_epochs"] + 1:
            model.unfreeze_backbone()
            # Re-initialise optimiser to include backbone params
            optimizer = optim.AdamW(
                model.parameters(),
                lr=CONFIG["learning_rate"] * 0.1,  # lower LR for fine-tuning
                weight_decay=CONFIG["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CONFIG["epochs"] - CONFIG["freeze_epochs"],
                eta_min=1e-6,
            )
            print(f"\n--- Backbone unfrozen at epoch {epoch} (fine-tuning) ---\n")

        start = time.time()
        print(f"Epoch [{epoch}/{CONFIG['epochs']}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_auc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        elapsed = time.time() - start
        print(
            f"  Train  — Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%\n"
            f"  Val    — Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%  "
            f"AUC: {val_auc:.4f}  F1: {val_f1:.4f}\n"
            f"  Time   — {elapsed:.1f}s\n"
        )

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_path = os.path.join(CONFIG["model_dir"], "resnet_best.pth")
            torch.save({
                "epoch"         : epoch,
                "model_state"   : model.state_dict(),
                "val_auc"       : val_auc,
                "val_acc"       : val_acc,
                "config"        : CONFIG,
            }, save_path)
            print(f"  ✓ New best model saved (AUC: {val_auc:.4f}) → {save_path}\n")

    # ── Test evaluation ──
    print(f"\n{'='*60}")
    print(" Final Test Evaluation")
    print(f"{'='*60}")

    # Load best model for test
    checkpoint = torch.load(
        os.path.join(CONFIG["model_dir"], "resnet_best.pth"),
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc, test_auc, test_f1, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device
    )
    test_preds = [1 if p > 0.5 else 0 for p in test_probs]

    print(f" Test Accuracy : {test_acc:.2f}%")
    print(f" Test AUC      : {test_auc:.4f}")
    print(f" Test F1       : {test_f1:.4f}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=['LUAD','LUSC'])}")

    # Save results
    results = {
        "test_accuracy" : round(test_acc, 4),
        "test_auc"      : round(test_auc, 4),
        "test_f1"       : round(test_f1, 4),
        "best_val_auc"  : round(best_val_auc, 4),
        "config"        : CONFIG,
    }
    results_path = os.path.join(CONFIG["results_dir"], "resnet_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # Save curves
    save_curves(train_losses, val_losses, train_accs, val_accs)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()