"""
SHAP Explainability for Clinical Features in the Fusion Model.

Uses SHAP KernelExplainer to compute feature importance scores
for each clinical variable's contribution to recurrence prediction.

Generates:
    - Summary bar plot (mean absolute SHAP values per feature)
    - Beeswarm plot (distribution of SHAP values across patients)
    - Per-patient waterfall plots (individual explanations)

Usage:
    python src/explainability/shap_explainer.py
"""

import os
import sys
import json
import numpy as np
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocess.clinical_processor import ClinicalProcessor
from src.models.fusion_model import build_fusion_model
from src.models.resnet_encoder import build_model as build_resnet


# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "clinical_csv"      : "data/metadata/tcga_clinical_master_deduped.csv",
    "embeddings_path"   : "data/features/image_embeddings.npy",
    "labels_path"       : "data/features/image_labels.npy",
    "fusion_checkpoint" : "outputs/models/fusion_best.pth",
    "figures_dir"       : "outputs/figures",
    "results_dir"       : "outputs/results",
    "clinical_input_dim": 5,
    "image_embedding_dim": 256,
    "n_background"      : 20,    # number of background samples for KernelExplainer
    "n_explain"         : 30,    # number of patients to explain
    "seed"              : 42,
    "device"            : "cuda" if torch.cuda.is_available() else "cpu",
}

np.random.seed(CONFIG["seed"])


# ─── Prediction wrapper ───────────────────────────────────────────────────────

class ClinicalPredictionWrapper:
    """
    Wraps the fusion model so SHAP can perturb only clinical features
    while holding image embeddings fixed at their mean.

    This isolates the clinical feature contribution to recurrence prediction.
    """

    def __init__(self, model, mean_image_embedding: np.ndarray, device):
        self.model              = model
        self.mean_image_embedding = mean_image_embedding   # (256,)
        self.device             = device
        self.model.eval()

    def predict(self, clinical_features: np.ndarray) -> np.ndarray:
        """
        Args:
            clinical_features: (N, n_clinical_features) numpy array

        Returns:
            probs: (N,) numpy array of recurrence probabilities
        """
        N = clinical_features.shape[0]

        # Repeat mean image embedding for all N samples
        img_emb = np.tile(self.mean_image_embedding, (N, 1)).astype(np.float32)

        img_tensor  = torch.tensor(img_emb,             dtype=torch.float32).to(self.device)
        clin_tensor = torch.tensor(clinical_features,   dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probs = self.model(img_tensor, clin_tensor).cpu().numpy()

        return probs  # (N,)


# ─── SHAP computation ─────────────────────────────────────────────────────────

def compute_shap_values(
    wrapper: ClinicalPredictionWrapper,
    X_clin: np.ndarray,
    feature_names: list,
):
    """
    Compute SHAP values using KernelExplainer.

    Args:
        wrapper: ClinicalPredictionWrapper instance.
        X_clin: (N, n_features) scaled clinical features.
        feature_names: List of feature name strings.

    Returns:
        shap_values: (N, n_features) SHAP values array.
        explainer: fitted KernelExplainer.
    """
    print(f"\nComputing SHAP values for {CONFIG['n_explain']} patients...")
    print(f"Background samples: {CONFIG['n_background']}")

    # Background dataset — use random subset as reference
    bg_idx = np.random.choice(
        len(X_clin),
        size=min(CONFIG["n_background"], len(X_clin)),
        replace=False
    )
    background = X_clin[bg_idx]

    # Patients to explain
    explain_idx = np.random.choice(
        len(X_clin),
        size=min(CONFIG["n_explain"], len(X_clin)),
        replace=False
    )
    X_explain = X_clin[explain_idx]

    # KernelExplainer — model-agnostic, works with any prediction function
    explainer = shap.KernelExplainer(wrapper.predict, background)

    # Compute SHAP values (suppress verbose output)
    shap_values = explainer.shap_values(X_explain, nsamples=100, silent=True)

    print(f"SHAP values computed. Shape: {shap_values.shape}")

    return shap_values, explainer, X_explain, explain_idx


# ─── Plotting ─────────────────────────────────────────────────────────────────

def save_shap_summary_bar(shap_values, feature_names, save_path):
    """
    Bar chart of mean absolute SHAP values per feature.
    Shows overall feature importance.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # (n_features,)
    sorted_idx    = np.argsort(mean_abs_shap)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if i == 0 else "#90CAF9" for i in range(len(feature_names))]

    bars = ax.barh(
        [feature_names[i] for i in sorted_idx[::-1]],
        mean_abs_shap[sorted_idx[::-1]],
        color=colors[::-1],
        edgecolor="white",
        height=0.6,
    )

    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(
        "Clinical Feature Importance (SHAP)\nMean Absolute Contribution to Recurrence Prediction",
        fontsize=11
    )
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary bar chart saved → {save_path}")


def save_shap_beeswarm(shap_values, X_explain, feature_names, save_path):
    """
    Beeswarm plot showing distribution of SHAP values across all explained patients.
    Each dot is one patient; colour shows feature value (high=red, low=blue).
    """
    # Create SHAP Explanation object
    explanation = shap.Explanation(
        values=shap_values,
        data=X_explain,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.beeswarm(explanation, show=False, max_display=5)
    plt.title(
        "SHAP Beeswarm Plot — Clinical Feature Contributions",
        fontsize=11, pad=12
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP beeswarm plot saved → {save_path}")


def save_shap_waterfall(shap_values, X_explain, feature_names,
                        explainer, save_path, patient_idx=0):
    """
    Waterfall plot for a single patient showing step-by-step
    how each feature pushed the prediction above/below the base rate.
    """
    explanation = shap.Explanation(
        values=shap_values[patient_idx],
        base_values=float(explainer.expected_value),
        data=X_explain[patient_idx],
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"SHAP Waterfall — Patient {patient_idx}", fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP waterfall plot saved → {save_path}")


def save_shap_results_json(shap_values, feature_names, save_path):
    """Save SHAP importance scores as JSON for reporting."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = {
        feature_names[i]: round(float(mean_abs[i]), 6)
        for i in range(len(feature_names))
    }
    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    with open(save_path, "w") as f:
        json.dump({"feature_importance_shap": importance}, f, indent=2)
    print(f"SHAP importance scores saved → {save_path}")
    print(f"\nFeature importance ranking:")
    for feat, score in importance.items():
        print(f"  {feat:<20} : {score:.6f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CONFIG["figures_dir"], exist_ok=True)
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    device = torch.device(CONFIG["device"])
    print(f"\n{'='*60}")
    print(f" SHAP Clinical Feature Explainability")
    print(f"{'='*60}")
    print(f" Device : {device}")

    # ── Load clinical data ──
    processor = ClinicalProcessor(CONFIG["clinical_csv"])
    X_clin, y, feature_names, *_ = processor.get_features_and_labels(fit_scaler=True)
    print(f"\n Clinical features : {feature_names}")
    print(f" Patients          : {len(X_clin)}")

    # ── Load image embeddings to compute mean ──
    img_embeddings = np.load(CONFIG["embeddings_path"]).astype(np.float32)
    mean_img_emb   = img_embeddings.mean(axis=0)   # (256,)
    print(f" Mean image embedding computed from {len(img_embeddings)} patches")

    # ── Load fusion model ──
    model = build_fusion_model(
        model_type="fusion",
        image_embedding_dim=CONFIG["image_embedding_dim"],
        clinical_input_dim=CONFIG["clinical_input_dim"],
    ).to(device)

    checkpoint = torch.load(CONFIG["fusion_checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f" Fusion model loaded")

    # ── Wrap model for SHAP ──
    wrapper = ClinicalPredictionWrapper(model, mean_img_emb, device)

    # ── Compute SHAP values ──
    shap_values, explainer, X_explain, explain_idx = compute_shap_values(
        wrapper, X_clin, feature_names
    )

    # ── Generate and save plots ──
    save_shap_summary_bar(
        shap_values, feature_names,
        os.path.join(CONFIG["figures_dir"], "shap_summary_bar.png")
    )

    save_shap_beeswarm(
        shap_values, X_explain, feature_names,
        os.path.join(CONFIG["figures_dir"], "shap_beeswarm.png")
    )

    save_shap_waterfall(
        shap_values, X_explain, feature_names, explainer,
        os.path.join(CONFIG["figures_dir"], "shap_waterfall_patient0.png"),
        patient_idx=0
    )

    save_shap_results_json(
        shap_values, feature_names,
        os.path.join(CONFIG["results_dir"], "shap_importance.json")
    )

    print(f"\n{'='*60}")
    print(f" SHAP explainability complete.")
    print(f" Figures saved to: {CONFIG['figures_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()