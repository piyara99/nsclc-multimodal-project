"""
Improved Fusion Model Training — Weighted Average Fusion.

Changes from v1:
    - WeightedFusionModel: learnable alpha blends image/clinical logits
    - Focal loss replaces BCE for better handling of class imbalance
    - Epochs 60 -> 120, LR 5e-4 -> 1e-4, Dropout 0.3 -> 0.4
    - CosineAnnealingWarmRestarts scheduler
    - Gradient clipping for stability

Usage:
    python src/train_fusion.py
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
    precision_score, recall_score, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.clinical_processor import ClinicalProcessor
from src.models.fusion_model import build_fusion_model

# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG = {
    "clinical_csv"       : "data/metadata/tcga_clinical_master_deduped.csv",
    "embeddings_path"    : "data/features/image_embeddings.npy",
    "labels_path"        : "data/features/image_labels.npy",
    "model_dir"          : "outputs/models",
    "figures_dir"        : "outputs/figures",
    "results_dir"        : "outputs/results",

    "clinical_input_dim" : 5,
    "image_embedding_dim": 256,
    "dropout"            : 0.4,

    "epochs"             : 120,
    "batch_size"         : 16,
    "learning_rate"      : 1e-4,
    "weight_decay"       : 1e-3,
    "n_folds"            : 5,
    "T_0"                : 30,

    "seed"               : 42,
    "device"             : "cuda" if torch.cuda.is_available() else "cpu",
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ─── Weighted Average Fusion Model ────────────────────────────────────────────

class WeightedFusionModel(nn.Module):
    """
    Weighted average fusion — each modality produces independent logits,
    combined via a learnable scalar alpha (clamped to [0.1, 0.9]).

    More principled than fixed concatenation: the model learns how much
    to trust each modality for this specific dataset.
    """

    def __init__(
        self,
        image_embedding_dim: int = 256,
        clinical_input_dim: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.image_branch = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Learnable fusion weight, initialised at 0.5 (equal weighting)
        self.alpha   = nn.Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_emb: torch.Tensor, clinical_feat: torch.Tensor) -> torch.Tensor:
        img_logit  = self.image_branch(image_emb).squeeze(1)
        clin_logit = self.clinical_branch(clinical_feat).squeeze(1)
        alpha      = torch.clamp(self.alpha, 0.1, 0.9)
        return self.sigmoid(alpha * img_logit + (1 - alpha) * clin_logit)

    def get_modality_weights(self):
        a = float(torch.clamp(self.alpha, 0.1, 0.9).item())
        return {"image_weight": round(a, 4), "clinical_weight": round(1 - a, 4)}


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary focal loss — downweights easy examples, focuses on hard ones."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce   = nn.functional.binary_cross_entropy(probs, targets, reduction="none")
        pt    = torch.where(targets == 1, probs, 1 - probs)
        alpha = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1 - self.alpha),
        )
        return (alpha * (1 - pt) ** self.gamma * bce).mean()


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_and_match_data():
    processor = ClinicalProcessor(CONFIG["clinical_csv"])
    X_clin, y_clin, feature_names, *_ = processor.get_features_and_labels(fit_scaler=True)

    img_embeddings = np.load(CONFIG["embeddings_path"]).astype(np.float32)
    img_labels     = np.load(CONFIG["labels_path"])

    np.random.seed(CONFIG["seed"])
    X_img_matched = []
    SUBTYPE_IDX   = 2

    for i in range(len(y_clin)):
        sv          = int(round(float(X_clin[i][SUBTYPE_IDX])))
        img_subtype = 0 if sv <= 0 else 1
        pool        = np.where(img_labels == img_subtype)[0]
        X_img_matched.append(img_embeddings[np.random.choice(pool)])

    X_img_matched = np.array(X_img_matched, dtype=np.float32)

    print(f"\n[Data] Patients  : {len(y_clin)}")
    print(f"[Data] Balance   : {dict(zip(*np.unique(y_clin, return_counts=True)))}")
    print(f"[Data] Features  : {feature_names}")

    return X_img_matched, X_clin, y_clin, feature_names, processor


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for img_emb, clin_feat, labels in loader:
        img_emb, clin_feat, labels = (
            img_emb.to(device), clin_feat.to(device), labels.float().to(device)
        )
        optimizer.zero_grad()
        probs = model(img_emb, clin_feat)
        loss  = criterion(probs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * img_emb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for img_emb, clin_feat, labels in loader:
        img_emb, clin_feat = img_emb.to(device), clin_feat.to(device)
        probs = model(img_emb, clin_feat)
        loss  = criterion(probs, labels.float().to(device))
        total_loss += loss.item() * img_emb.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs > 0.5).astype(int)

    return {
        "loss"      : total_loss / len(loader.dataset),
        "accuracy"  : accuracy_score(all_labels, preds),
        "auc"       : roc_auc_score(all_labels, all_probs)
                      if len(np.unique(all_labels)) > 1 else 0.5,
        "f1"        : f1_score(all_labels, preds, zero_division=0),
        "precision" : precision_score(all_labels, preds, zero_division=0),
        "recall"    : recall_score(all_labels, preds, zero_division=0),
    }, all_probs, all_labels


# ─── K-Fold ───────────────────────────────────────────────────────────────────

def train_kfold(model_type, X_img, X_clin, y, device):
    print(f"\n{'='*60}\n Training: {model_type.upper()}\n{'='*60}")

    skf            = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True,
                                     random_state=CONFIG["seed"])
    fold_metrics   = []
    all_val_probs  = []
    all_val_labels = []
    best_auc       = 0.0
    best_state     = None
    tr_l_last, vl_l_last = [], []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_img, y)):
        print(f"\n  Fold {fold+1}/{CONFIG['n_folds']}")

        tr_ds = TensorDataset(
            torch.tensor(X_img[tr_idx],  dtype=torch.float32),
            torch.tensor(X_clin[tr_idx], dtype=torch.float32),
            torch.tensor(y[tr_idx],      dtype=torch.float32),
        )
        vl_ds = TensorDataset(
            torch.tensor(X_img[vl_idx],  dtype=torch.float32),
            torch.tensor(X_clin[vl_idx], dtype=torch.float32),
            torch.tensor(y[vl_idx],      dtype=torch.float32),
        )
        tr_loader = DataLoader(tr_ds, batch_size=CONFIG["batch_size"], shuffle=True)
        vl_loader = DataLoader(vl_ds, batch_size=CONFIG["batch_size"], shuffle=False)

        # Build model
        if model_type == "fusion":
            model = WeightedFusionModel(
                image_embedding_dim=CONFIG["image_embedding_dim"],
                clinical_input_dim=CONFIG["clinical_input_dim"],
                dropout=CONFIG["dropout"],
            ).to(device)
            n_pos    = int(y[tr_idx].sum())
            n_neg    = len(tr_idx) - n_pos
            alpha_fl = n_neg / (n_pos + n_neg) if n_pos > 0 else 0.75
            criterion = FocalLoss(alpha=alpha_fl, gamma=2.0)
        elif model_type == "concat_fusion":
            model = build_fusion_model(
                model_type=model_type,
                image_embedding_dim=CONFIG["image_embedding_dim"],
                clinical_input_dim=CONFIG["clinical_input_dim"],
                dropout=CONFIG["dropout"],
            ).to(device)
            n_total  = len(tr_idx)
            n_pos    = int(y[tr_idx].sum())
            n_neg    = n_total - n_pos
            w1       = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
            criterion = nn.BCELoss(
                weight=torch.tensor(w1, dtype=torch.float32).to(device)
            )
        else:
            model = build_fusion_model(
                model_type=model_type,
                image_embedding_dim=CONFIG["image_embedding_dim"],
                clinical_input_dim=CONFIG["clinical_input_dim"],
                dropout=CONFIG["dropout"],
            ).to(device)
            n_total  = len(tr_idx)
            n_pos    = int(y[tr_idx].sum())
            n_neg    = n_total - n_pos
            w1       = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
            criterion = nn.BCELoss(
                weight=torch.tensor(w1, dtype=torch.float32).to(device)
            )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=CONFIG["T_0"], T_mult=1, eta_min=1e-6
        )

        best_fold_auc, best_fold_state = 0.0, None
        fold_tr_l, fold_vl_l = [], []

        for epoch in range(1, CONFIG["epochs"] + 1):
            tr_loss = train_epoch(model, tr_loader, criterion, optimizer, device)
            val_met, val_probs, val_labs = eval_epoch(
                model, vl_loader, criterion, device
            )
            scheduler.step(epoch)
            fold_tr_l.append(tr_loss)
            fold_vl_l.append(val_met["loss"])

            if val_met["auc"] > best_fold_auc:
                best_fold_auc   = val_met["auc"]
                best_fold_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 20 == 0:
                print(f"    Epoch {epoch:3d} | Loss: {tr_loss:.4f} | "
                      f"AUC: {val_met['auc']:.4f} | F1: {val_met['f1']:.4f}")

        model.load_state_dict(best_fold_state)
        val_met, val_probs, val_labs = eval_epoch(
            model, vl_loader, criterion, device
        )

        if model_type == "fusion" and hasattr(model, "get_modality_weights"):
            w = model.get_modality_weights()
            print(f"  Learned weights → Image: {w['image_weight']}  "
                  f"Clinical: {w['clinical_weight']}")

        fold_metrics.append(val_met)
        all_val_probs.extend(val_probs)
        all_val_labels.extend(val_labs)
        print(f"  Fold {fold+1} → AUC: {val_met['auc']:.4f}  "
              f"F1: {val_met['f1']:.4f}  Acc: {val_met['accuracy']:.4f}")

        if val_met["auc"] > best_auc:
            best_auc   = val_met["auc"]
            best_state = best_fold_state

        if fold == CONFIG["n_folds"] - 1:
            tr_l_last, vl_l_last = fold_tr_l, fold_vl_l

    agg = {}
    for key in fold_metrics[0]:
        vals = [fm[key] for fm in fold_metrics]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"]  = float(np.std(vals))

    all_val_probs  = np.array(all_val_probs)
    all_val_labels = np.array(all_val_labels)
    agg["overall_auc"] = (
        float(roc_auc_score(all_val_labels, all_val_probs))
        if len(np.unique(all_val_labels)) > 1 else 0.5
    )

    print(f"\n  {model_type.upper()} Summary")
    print(f"  AUC: {agg['auc_mean']:.4f} +/- {agg['auc_std']:.4f}  |  "
          f"F1: {agg['f1_mean']:.4f}  |  Acc: {agg['accuracy_mean']:.4f}  |  "
          f"Recall: {agg['recall_mean']:.4f}")

    return agg, best_state, all_val_probs, all_val_labels, tr_l_last, vl_l_last


# ─── Plotting ─────────────────────────────────────────────────────────────────

def save_roc_curves(probs_dict, labels_dict):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = {
        "fusion": "#1565C0",
        "concat_fusion": "#6a1b9a",
        "image_only": "#c62828",
        "clinical_only": "#2e7d32",
    }
    names   = {
        "fusion":         "Weighted Fusion (image + clinical)",
        "concat_fusion":  "Concat Fusion (image + clinical)",
        "image_only":     "Image only",
        "clinical_only":  "Clinical only",
    }
    for mt, probs in probs_dict.items():
        labs = labels_dict[mt]
        if len(np.unique(labs)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labs, probs)
        auc = roc_auc_score(labs, probs)
        ax.plot(fpr, tpr, color=colors[mt], linewidth=2,
                label=f"{names[mt]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CONFIG["figures_dir"], "fusion_roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ROC curves saved → {path}")


def save_training_curves(tr_l, vl_l):
    epochs = range(1, len(tr_l) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, tr_l, color="#1565C0", linewidth=1.5, label="Train loss")
    ax.plot(epochs, vl_l, color="#c62828", linewidth=1.5, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Fusion Model Training Curves (Weighted Average Fusion)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CONFIG["figures_dir"], "fusion_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    for d in [CONFIG["model_dir"], CONFIG["figures_dir"], CONFIG["results_dir"]]:
        os.makedirs(d, exist_ok=True)

    device = torch.device(CONFIG["device"])

    print(f"\n{'='*60}")
    print(f" NSCLC Improved Fusion Training (v2)")
    print(f"{'='*60}")
    print(f" Device  : {device}")
    print(f" Epochs  : {CONFIG['epochs']}  LR: {CONFIG['learning_rate']}")
    print(f" Dropout : {CONFIG['dropout']}  Fusion: weighted average")
    print(f"{'='*60}")

    X_img, X_clin, y, feature_names, processor = load_and_match_data()

    all_results = {}
    probs_dict  = {}
    labels_dict = {}

    for model_type in ["fusion", "concat_fusion", "image_only", "clinical_only"]:
        metrics, best_state, val_probs, val_labels, tr_l, vl_l = train_kfold(
            model_type, X_img, X_clin, y, device
        )
        all_results[model_type] = metrics
        probs_dict[model_type]  = val_probs
        labels_dict[model_type] = val_labels

        save_path = os.path.join(CONFIG["model_dir"], f"{model_type}_best.pth")
        torch.save({
            "model_type"    : model_type,
            "model_state"   : best_state,
            "metrics"       : metrics,
            "config"        : CONFIG,
            "feature_names" : feature_names,
        }, save_path)
        print(f"Saved → {save_path}")

        if model_type == "fusion":
            save_training_curves(tr_l, vl_l)

    results_path = os.path.join(CONFIG["results_dir"], "fusion_comparison.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    save_roc_curves(probs_dict, labels_dict)

    print(f"\n{'='*60}")
    print(f" Final Comparison")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'AUC':>8} {'F1':>8} {'Accuracy':>10} {'Recall':>8}")
    print(f"{'─'*56}")
    for mt, res in all_results.items():
        print(f"{mt:<20} {res['auc_mean']:>8.4f} {res['f1_mean']:>8.4f} "
              f"{res['accuracy_mean']:>10.4f} {res['recall_mean']:>8.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()