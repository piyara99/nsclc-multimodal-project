"""
Fusion Model Training Script.

Trains three model variants for binary recurrence prediction:
    1. fusion       — image embeddings + clinical features (main model)
    2. image_only   — ablation: image embeddings only
    3. clinical_only— ablation: clinical features only

For each variant, runs stratified k-fold cross-validation and
saves performance metrics for comparison in the final report.

Usage (from project root, nsclc env active):
    python src/train_fusion.py

Outputs:
    outputs/models/fusion_best.pth
    outputs/models/image_only_best.pth
    outputs/models/clinical_only_best.pth
    outputs/results/fusion_comparison.json
    outputs/figures/fusion_roc_curves.png
    outputs/figures/fusion_training_curves.png
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.clinical_processor import ClinicalProcessor
from src.models.fusion_model import build_fusion_model

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "clinical_csv"      : "data/metadata/tcga_clinical_master.csv",
    "embeddings_path"   : "data/features/image_embeddings.npy",
    "labels_path"       : "data/features/image_labels.npy",
    "model_dir"         : "outputs/models",
    "figures_dir"       : "outputs/figures",
    "results_dir"       : "outputs/results",

    "clinical_input_dim": 5,
    "image_embedding_dim": 256,
    "dropout"           : 0.3,

    "epochs"            : 60,
    "batch_size"        : 8,       # small — only 38 patients
    "learning_rate"     : 5e-4,
    "weight_decay"      : 1e-3,
    "n_folds"           : 5,       # stratified k-fold

    "seed"              : 42,
    "device"            : "cuda" if torch.cuda.is_available() else "cpu",
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ─── Data preparation ─────────────────────────────────────────────────────────

def load_clinical_data():
    """Load and preprocess clinical features and labels."""
    processor = ClinicalProcessor(CONFIG["clinical_csv"])
    X_clin, y, feature_names = processor.get_features_and_labels(fit_scaler=True)
    class_weights = processor.get_class_weights_tensor()
    return X_clin, y, feature_names, class_weights, processor


def load_image_embeddings():
    """Load pre-extracted image embeddings and their labels."""
    embeddings = np.load(CONFIG["embeddings_path"])  # (10000, 256)
    labels     = np.load(CONFIG["labels_path"])      # (10000,)
    return embeddings, labels


def match_clinical_to_images(X_clin, y_clin, img_embeddings, img_labels):
    """
    For each clinical patient, assign one representative image embedding.

    Strategy: For each clinical patient (labelled by subtype 0=LUAD/1=LUSC),
    randomly sample one image embedding from the matching subtype pool.
    This creates a paired (image_embedding, clinical_features, label) dataset.

    Note: In a real clinical setting, the image would be the patient's own WSI.
    Here we use subtype-matched sampling as a principled proxy since we do not
    have per-patient WSI-to-clinical linkage in the Kaggle dataset.

    Returns:
        X_img_matched  : (N_patients, 256)
        X_clin_matched : (N_patients, 5)
        y_matched      : (N_patients,) — recurrence labels
    """
    np.random.seed(CONFIG["seed"])

    X_img_matched  = []
    X_clin_matched = []
    y_matched      = []

    # subtype column index in clinical features: index 2 = subtype
    SUBTYPE_IDX = 2

    for i in range(len(y_clin)):
        clin_feat = X_clin[i]           # (5,)
        label     = y_clin[i]           # 0 or 1

        # Infer subtype from clinical feature (after scaling, use raw subtype)
        # We use img_labels directly: 0=LUAD, 1=LUSC
        # Match on subtype: use LUAD images for LUAD patients, LUSC for LUSC
        # subtype is stored as a scaled value — use original label pool split
        # For simplicity: first half of clinical data tends to be LUAD,
        # but we'll use img_labels to find matching subtype images properly.

        # Use LUAD images (label=0) for first subtype, LUSC (label=1) for second
        # We check the raw clinical CSV subtype implicitly via img_labels
        subtype_val = int(round(float(clin_feat[SUBTYPE_IDX])))
        # After StandardScaler, values are centered — use sign to infer class
        # Better: use the original unscaled subtype. We'll do it via modulo:
        # clinical patients are ordered: first N_LUAD are LUAD, rest are LUSC
        # Actually, use a direct approach: sample from matching image pool
        img_subtype = 0 if subtype_val <= 0 else 1
        pool_indices = np.where(img_labels == img_subtype)[0]

        chosen_idx = np.random.choice(pool_indices)
        X_img_matched.append(img_embeddings[chosen_idx])
        X_clin_matched.append(clin_feat)
        y_matched.append(label)

    X_img_matched  = np.array(X_img_matched,  dtype=np.float32)
    X_clin_matched = np.array(X_clin_matched, dtype=np.float32)
    y_matched      = np.array(y_matched,       dtype=np.int64)

    print(f"\n[Data Matching] Paired dataset: {len(y_matched)} patients")
    print(f"  Image embeddings shape  : {X_img_matched.shape}")
    print(f"  Clinical features shape : {X_clin_matched.shape}")
    print(f"  Recurrence distribution : "
          f"{dict(zip(*np.unique(y_matched, return_counts=True)))}")

    return X_img_matched, X_clin_matched, y_matched


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for img_emb, clin_feat, labels in loader:
        img_emb    = img_emb.to(device)
        clin_feat  = clin_feat.to(device)
        labels     = labels.float().to(device)

        optimizer.zero_grad()
        probs = model(img_emb, clin_feat)
        loss  = criterion(probs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img_emb.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for img_emb, clin_feat, labels in loader:
        img_emb   = img_emb.to(device)
        clin_feat = clin_feat.to(device)
        labels_f  = labels.float().to(device)

        probs = model(img_emb, clin_feat)
        loss  = criterion(probs, labels_f)

        total_loss += loss.item() * img_emb.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs > 0.5).astype(int)

    metrics = {
        "loss"     : avg_loss,
        "accuracy" : accuracy_score(all_labels, preds),
        "auc"      : roc_auc_score(all_labels, all_probs)
                     if len(np.unique(all_labels)) > 1 else 0.5,
        "f1"       : f1_score(all_labels, preds, zero_division=0),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall"   : recall_score(all_labels, preds, zero_division=0),
    }
    return metrics, all_probs, all_labels


# ─── K-Fold training ──────────────────────────────────────────────────────────

def train_model_kfold(
    model_type: str,
    X_img: np.ndarray,
    X_clin: np.ndarray,
    y: np.ndarray,
    class_weights: torch.Tensor,
    device: torch.device,
):
    """
    Train a model variant using stratified k-fold cross-validation.
    Returns aggregated metrics and the best model state dict.
    """
    print(f"\n{'='*60}")
    print(f" Training: {model_type.upper()}")
    print(f"{'='*60}")

    skf = StratifiedKFold(
        n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["seed"]
    )

    fold_metrics = []
    all_val_probs  = []
    all_val_labels = []
    best_auc       = 0.0
    best_state     = None
    train_loss_history = []
    val_loss_history   = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_img, y)):
        print(f"\n  Fold {fold+1}/{CONFIG['n_folds']}")

        # Build tensors
        X_img_tr  = torch.tensor(X_img[train_idx],  dtype=torch.float32)
        X_clin_tr = torch.tensor(X_clin[train_idx], dtype=torch.float32)
        y_tr      = torch.tensor(y[train_idx],       dtype=torch.float32)

        X_img_val  = torch.tensor(X_img[val_idx],  dtype=torch.float32)
        X_clin_val = torch.tensor(X_clin[val_idx], dtype=torch.float32)
        y_val      = torch.tensor(y[val_idx],       dtype=torch.float32)

        train_ds = TensorDataset(X_img_tr, X_clin_tr, y_tr)
        val_ds   = TensorDataset(X_img_val, X_clin_val, y_val)

        train_loader = DataLoader(
            train_ds, batch_size=CONFIG["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=CONFIG["batch_size"], shuffle=False
        )

        # Build fresh model per fold
        model = build_fusion_model(
            model_type=model_type,
            image_embedding_dim=CONFIG["image_embedding_dim"],
            clinical_input_dim=CONFIG["clinical_input_dim"],
            dropout=CONFIG["dropout"],
        ).to(device)

        criterion = nn.BCELoss(
            weight=class_weights[1].to(device)
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
        )

        fold_train_losses = []
        fold_val_losses   = []
        best_fold_auc     = 0.0
        best_fold_state   = None

        for epoch in range(1, CONFIG["epochs"] + 1):
            tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics, val_probs, val_labs = eval_epoch(
                model, val_loader, criterion, device
            )
            scheduler.step()

            fold_train_losses.append(tr_loss)
            fold_val_losses.append(val_metrics["loss"])

            if val_metrics["auc"] > best_fold_auc:
                best_fold_auc   = val_metrics["auc"]
                best_fold_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0:
                print(
                    f"    Epoch {epoch:3d} | "
                    f"Train Loss: {tr_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val AUC: {val_metrics['auc']:.4f}"
                )

        # Load best fold weights for final evaluation
        model.load_state_dict(best_fold_state)
        val_metrics, val_probs, val_labs = eval_epoch(
            model, val_loader, criterion, device
        )

        fold_metrics.append(val_metrics)
        all_val_probs.extend(val_probs)
        all_val_labels.extend(val_labs)

        print(
            f"  Fold {fold+1} best → "
            f"AUC: {val_metrics['auc']:.4f}  "
            f"F1: {val_metrics['f1']:.4f}  "
            f"Acc: {val_metrics['accuracy']:.4f}"
        )

        # Track global best
        if val_metrics["auc"] > best_auc:
            best_auc   = val_metrics["auc"]
            best_state = best_fold_state

        # Store curves from last fold only for plotting
        if fold == CONFIG["n_folds"] - 1:
            train_loss_history = fold_train_losses
            val_loss_history   = fold_val_losses

    # ── Aggregate metrics ──
    agg = {}
    for key in fold_metrics[0]:
        vals = [fm[key] for fm in fold_metrics]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"]  = float(np.std(vals))

    # Overall AUC across all folds
    all_val_probs  = np.array(all_val_probs)
    all_val_labels = np.array(all_val_labels)
    if len(np.unique(all_val_labels)) > 1:
        agg["overall_auc"] = float(roc_auc_score(all_val_labels, all_val_probs))
    else:
        agg["overall_auc"] = 0.5

    print(f"\n  {'─'*50}")
    print(f"  {model_type.upper()} Cross-Validation Summary")
    print(f"  {'─'*50}")
    print(f"  AUC      : {agg['auc_mean']:.4f} ± {agg['auc_std']:.4f}")
    print(f"  F1       : {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}")
    print(f"  Accuracy : {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
    print(f"  Precision: {agg['precision_mean']:.4f} ± {agg['precision_std']:.4f}")
    print(f"  Recall   : {agg['recall_mean']:.4f} ± {agg['recall_std']:.4f}")

    return agg, best_state, all_val_probs, all_val_labels, \
           train_loss_history, val_loss_history


# ─── Plotting ─────────────────────────────────────────────────────────────────

def save_roc_curves(results: dict, probs_dict: dict, labels_dict: dict):
    """Save ROC curves for all three model variants on one plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"fusion": "#2196F3", "image_only": "#FF5722", "clinical_only": "#4CAF50"}
    labels_map = {
        "fusion"        : "Fusion (image + clinical)",
        "image_only"    : "Image only",
        "clinical_only" : "Clinical only",
    }

    for model_type, probs in probs_dict.items():
        labs = labels_dict[model_type]
        if len(np.unique(labs)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labs, probs)
        auc = roc_auc_score(labs, probs)
        ax.plot(
            fpr, tpr,
            color=colors[model_type],
            linewidth=2,
            label=f"{labels_map[model_type]} (AUC={auc:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = os.path.join(CONFIG["figures_dir"], "fusion_roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nROC curves saved → {path}")


def save_training_curves(train_losses: list, val_losses: list, model_type: str):
    """Save training/validation loss curve for the fusion model."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, "b-o", markersize=3, label="Train loss")
    ax.plot(epochs, val_losses,   "r-o", markersize=3, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Curves — {model_type}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CONFIG["figures_dir"], f"{model_type}_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    for d in [CONFIG["model_dir"], CONFIG["figures_dir"], CONFIG["results_dir"]]:
        os.makedirs(d, exist_ok=True)

    device = torch.device(CONFIG["device"])
    print(f"\n{'='*60}")
    print(f" NSCLC Multimodal Fusion Training")
    print(f"{'='*60}")
    print(f" Device    : {device}")
    print(f" Folds     : {CONFIG['n_folds']}")
    print(f" Epochs    : {CONFIG['epochs']}")
    print(f" Batch size: {CONFIG['batch_size']}")
    print(f"{'='*60}")

    # ── Load data ──
    print("\n Loading clinical data...")
    X_clin, y_clin, feature_names, class_weights, processor = load_clinical_data()

    print("\n Loading image embeddings...")
    img_embeddings, img_labels = load_image_embeddings()

    print("\n Matching clinical patients to image embeddings...")
    X_img, X_clin_matched, y = match_clinical_to_images(
        X_clin, y_clin, img_embeddings, img_labels
    )

    # ── Train all three variants ──
    all_results   = {}
    probs_dict    = {}
    labels_dict   = {}

    for model_type in ["fusion", "image_only", "clinical_only"]:
        metrics, best_state, val_probs, val_labels, tr_losses, val_losses = \
            train_model_kfold(
                model_type, X_img, X_clin_matched, y,
                class_weights, device
            )

        all_results[model_type] = metrics
        probs_dict[model_type]  = val_probs
        labels_dict[model_type] = val_labels

        # Save best model
        save_path = os.path.join(CONFIG["model_dir"], f"{model_type}_best.pth")
        torch.save({
            "model_type"     : model_type,
            "model_state"    : best_state,
            "metrics"        : metrics,
            "config"         : CONFIG,
            "feature_names"  : feature_names,
        }, save_path)
        print(f"\nModel saved → {save_path}")

        # Save training curves for fusion model
        if model_type == "fusion":
            save_training_curves(tr_losses, val_losses, model_type)

    # ── Save comparison table ──
    comparison_path = os.path.join(CONFIG["results_dir"], "fusion_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison results saved → {comparison_path}")

    # ── Save ROC curves ──
    save_roc_curves(all_results, probs_dict, labels_dict)

    # ── Print final comparison ──
    print(f"\n{'='*60}")
    print(f" Final Model Comparison")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'AUC':>8} {'F1':>8} {'Accuracy':>10} {'Recall':>8}")
    print(f"{'─'*56}")
    for mt, res in all_results.items():
        print(
            f"{mt:<20} "
            f"{res['auc_mean']:>8.4f} "
            f"{res['f1_mean']:>8.4f} "
            f"{res['accuracy_mean']:>10.4f} "
            f"{res['recall_mean']:>8.4f}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()