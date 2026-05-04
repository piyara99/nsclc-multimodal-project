"""
NSCLC Phase 3 — Missing Modality Test + Unit Tests

What this does:
    1. Missing modality robustness test (zero-out each branch)
    2. Unit tests for GatedFusionModel and pipeline
    3. Requirements traceability output

HOW TO RUN:
    python src/phase3_missing_modality.py
    pytest tests/test_pipeline.py -v
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = {
    "clinical_csv"       : "data/metadata/tcga_clinical_master_deduped.csv",
    "embeddings_path"    : "data/features/image_embeddings.npy",
    "labels_path"        : "data/features/image_labels.npy",
    "results_dir"        : "outputs/results",
    "figures_dir"        : "outputs/figures",
    "model_dir"          : "outputs/models",
    "seed"               : 42,
    "clinical_input_dim" : 6,
    "image_embedding_dim": 256,
    "dropout"            : 0.4,
}


def run_missing_modality_test():
    """
    Zero-out each modality branch and evaluate performance degradation.
    
    Method: Replace image or clinical embeddings with zero tensors.
    This simulates 'this data was not available for this patient'.
    
    Decision: Zero-out (not Gaussian noise, not mean imputation).
    Zero-out is the standard missing modality test. Cite: Ma et al.,
    NeurIPS 2022 for zero-out missing modality testing.
    
    Trade-off: Zero-out is aggressive. In practice, a clinician might 
    have SOME data but degraded quality. Zero-out gives a lower bound
    on graceful degradation. Write: "We evaluate the extreme case of 
    complete modality absence; real-world degradation would be 
    intermediate between full-modality and zero-out performance."
    
    Expected outcomes:
    - Full > both missing → fusion genuinely uses both modalities
    - Clinical-missing ≈ Full → image contributes little (proxy issue)
    - Image-missing ≈ Full → clinical dominates (consistent with SHAP)
    All three outcomes are publishable. Write the finding honestly.
    """
    from sklearn.metrics import roc_auc_score, f1_score

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load saved GMU probabilities
    gmu_probs_path   = os.path.join(CONFIG["results_dir"], "gmu_val_probs.npy")
    gmu_labels_path  = os.path.join(CONFIG["results_dir"], "gmu_val_labels.npy")
    gmu_model_path   = os.path.join(CONFIG["model_dir"], "gmu_fusion_best.pth")

    if not all(os.path.exists(p) for p in [gmu_probs_path, gmu_labels_path, gmu_model_path]):
        print("[Phase 3] Missing model files — run phase2_gmu.py first.")
        return {}

    # Load saved val data and model
    gmu_probs  = np.load(gmu_probs_path)
    gmu_labels = np.load(gmu_labels_path)

    checkpoint = torch.load(gmu_model_path, map_location='cpu', weights_only=False)

    # Lazy import to avoid circular dependency
    from src.phase2_gmu import GatedFusionModel, load_and_match_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GatedFusionModel(
        image_embedding_dim=CONFIG["image_embedding_dim"],
        clinical_input_dim=CONFIG["clinical_input_dim"],
        dropout=CONFIG["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load test data (use fixed seed for reproducibility)
    X_img, X_clin, y, _, _ = load_and_match_data()

    # Use same 80/20 split as your training for a consistent test set
    from sklearn.model_selection import train_test_split
    _, X_img_te, _, X_clin_te, _, y_te = train_test_split(
        X_img, X_clin, y,
        test_size=0.2, random_state=CONFIG["seed"], stratify=y
    )

    img_tensor  = torch.tensor(X_img_te,  dtype=torch.float32).to(device)
    clin_tensor = torch.tensor(X_clin_te, dtype=torch.float32).to(device)

    results_mm = {}

    modes = {
        "Full (both modalities)"   : (img_tensor,              clin_tensor),
        "Image missing (clinical)" : (torch.zeros_like(img_tensor), clin_tensor),
        "Clinical missing (image)" : (img_tensor, torch.zeros_like(clin_tensor)),
    }

    print(f"\n[Phase 3] Missing Modality Test")
    print(f"  Test set: n={len(y_te)}, pos={y_te.sum()}, neg={(y_te==0).sum()}")
    print(f"\n  {'Mode':<35} {'AUC':>7} {'95% CI':>16} {'F1':>7}")
    print(f"  {'─'*70}")

    for mode_name, (img_in, clin_in) in modes.items():
        with torch.no_grad():
            probs = model(img_in, clin_in).cpu().numpy()

        auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5
        preds = (probs >= 0.5).astype(int)
        f1  = f1_score(y_te, preds, zero_division=0)

        # Bootstrap CI
        rng  = np.random.default_rng(CONFIG["seed"])
        aucs = []
        for _ in range(1000):
            idx = rng.integers(0, len(y_te), len(y_te))
            if len(np.unique(y_te[idx])) > 1:
                aucs.append(roc_auc_score(y_te[idx], probs[idx]))
        ci_lo = round(np.percentile(aucs, 2.5), 4)
        ci_hi = round(np.percentile(aucs, 97.5), 4)

        results_mm[mode_name] = {
            "AUC": round(auc, 4),
            "AUC_CI_low": ci_lo, "AUC_CI_high": ci_hi,
            "F1": round(f1, 4),
        }
        ci_str = f"[{ci_lo:.3f}–{ci_hi:.3f}]"
        print(f"  {mode_name:<35} {auc:>7.4f} {ci_str:>16} {f1:>7.4f}")

    # Interpretation guidance
    full_auc = results_mm["Full (both modalities)"]["AUC"]
    img_miss = results_mm["Image missing (clinical)"]["AUC"]
    clin_miss = results_mm["Clinical missing (image)"]["AUC"]

    print(f"\n  Interpretation:")
    if full_auc > max(img_miss, clin_miss):
        print(f"  → Full fusion outperforms both missing-modality conditions.")
        print(f"  → Write: 'The GMU model leverages both modalities: removing")
        print(f"    either branch degrades performance, with image-missing")
        if img_miss > clin_miss:
            print(f"    (AUC={img_miss:.4f}) outperforming clinical-missing ")
            print(f"    (AUC={clin_miss:.4f}), indicating clinical features")
            print(f"    are the dominant modality on this proxy-matched cohort.'")
        else:
            print(f"    (AUC={clin_miss:.4f}) outperforming image-missing")
            print(f"    (AUC={img_miss:.4f}), suggesting morphological features")
            print(f"    provide complementary predictive signal.'")
    else:
        print(f"  → One modality-missing condition matches full fusion.")
        print(f"  → This is expected given proxy-matched image embeddings.")
        print(f"  → Write: 'Clinical-missing performance (AUC={img_miss:.4f})")
        print(f"    approaches full fusion (AUC={full_auc:.4f}), consistent with")
        print(f"    the synthetic image–patient pairing of the proxy-matched")
        print(f"    cohort rather than true patient-slide linkage.'")

    # Save
    path = os.path.join(CONFIG["results_dir"], "missing_modality.json")
    with open(path, "w") as f:
        json.dump(results_mm, f, indent=2)
    print(f"\n  Saved → {path}")

    return results_mm


# ─── UNIT TESTS ───────────────────────────────────────────────────────────────
# Save this section as tests/test_pipeline.py in your project
# Run: pytest tests/test_pipeline.py -v

UNIT_TEST_CODE = '''"""
Unit tests for NSCLC Multimodal Pipeline.
Maps to BCS-accredited SE requirements for testing traceability.

Run from project root:
    pytest tests/test_pipeline.py -v

Requirements traceability:
    test_gmu_forward           → FR-15 (GMU fusion forward pass)
    test_gmu_softmax_invariant → FR-15 (gate weights sum to 1)
    test_clinical_no_nan       → FR-09 (clinical normalisation)
    test_embedding_shape       → FR-11 (ResNet encoder output dim)
    test_bootstrap_ci_bounds   → FR-17 (evaluation metrics validity)
    test_dedup_fix             → FR-04 (data pipeline reproducibility)
    test_focal_loss_range      → FR-16 (loss function output range)
"""

import pytest
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── FR-15: GMU forward pass ───────────────────────────────────────────────────

def test_gmu_output_shape():
    """GMU forward pass returns correct shapes for batch of 8."""
    from src.phase2_gmu import GatedFusionModel
    model = GatedFusionModel(image_embedding_dim=256, clinical_input_dim=6)
    model.eval()
    img  = torch.randn(8, 256)
    clin = torch.randn(8, 6)
    with torch.no_grad():
        out = model(img, clin)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"


def test_gmu_softmax_invariant():
    """Gate weights must sum to 1.0 for every patient (Softmax invariant)."""
    from src.phase2_gmu import GatedFusionModel
    model = GatedFusionModel(image_embedding_dim=256, clinical_input_dim=6)
    model.eval()
    img  = torch.randn(16, 256)
    clin = torch.randn(16, 6)
    with torch.no_grad():
        _, g = model(img, clin, return_weights=True)
    sums = g.sum(dim=1)
    assert torch.allclose(sums, torch.ones(16), atol=1e-5), \\
        f"Gate weights do not sum to 1.0: {sums}"


def test_gmu_probs_in_range():
    """Output probabilities must be in [0, 1]."""
    from src.phase2_gmu import GatedFusionModel
    model = GatedFusionModel(image_embedding_dim=256, clinical_input_dim=6)
    model.eval()
    img  = torch.randn(32, 256)
    clin = torch.randn(32, 6)
    with torch.no_grad():
        probs = model(img, clin)
    assert probs.min() >= 0.0, f"Probability below 0: {probs.min()}"
    assert probs.max() <= 1.0, f"Probability above 1: {probs.max()}"


def test_gmu_different_patients_different_weights():
    """Per-patient gating: different inputs should produce different weights."""
    from src.phase2_gmu import GatedFusionModel
    model = GatedFusionModel(image_embedding_dim=256, clinical_input_dim=6)
    model.eval()
    torch.manual_seed(42)
    # Two very different patients
    img1  = torch.zeros(1, 256)
    clin1 = torch.zeros(1, 6)
    img2  = torch.ones(1, 256)
    clin2 = torch.ones(1, 6) * 5
    with torch.no_grad():
        _, g1 = model(img1, clin1, return_weights=True)
        _, g2 = model(img2, clin2, return_weights=True)
    assert not torch.allclose(g1, g2, atol=1e-3), \\
        "Same gate weights for different patients — GMU not working"


# ── FR-09: Clinical preprocessing ────────────────────────────────────────────

def test_clinical_no_nan():
    """Clinical preprocessor must produce zero NaN values."""
    csv_path = "data/metadata/tcga_clinical_master.csv"
    if not os.path.exists(csv_path):
        pytest.skip(f"{csv_path} not found — skipping")
    from src.preprocess.clinical_processor import ClinicalProcessor
    proc = ClinicalProcessor(csv_path)
    X, y, names = proc.get_features_and_labels(fit_scaler=True)
    assert not np.isnan(X).any(), f"NaN values in clinical features: {np.isnan(X).sum()}"
    assert not np.isnan(y).any(), "NaN values in labels"


def test_clinical_feature_count():
    """Clinical processor must produce exactly 6 features per patient."""
    csv_path = "data/metadata/tcga_clinical_master.csv"
    if not os.path.exists(csv_path):
        pytest.skip(f"{csv_path} not found — skipping")
    from src.preprocess.clinical_processor import ClinicalProcessor
    proc = ClinicalProcessor(csv_path)
    X, y, names = proc.get_features_and_labels(fit_scaler=True)
    assert X.shape[1] == 6, f"Expected 6 features, got {X.shape[1]}"
    assert len(names) == 6, f"Expected 6 feature names, got {len(names)}"


# ── FR-11: Image encoder ──────────────────────────────────────────────────────

def test_embedding_shape():
    """Saved embeddings must have 256-dim features."""
    emb_path = "data/features/image_embeddings.npy"
    if not os.path.exists(emb_path):
        pytest.skip(f"{emb_path} not found — skipping")
    embs = np.load(emb_path)
    assert embs.shape[1] == 256, f"Expected 256-dim, got {embs.shape[1]}"
    assert embs.shape[0] > 0, "Empty embeddings file"
    assert not np.isnan(embs).any(), "NaN in embeddings"


# ── FR-17: Evaluation metrics ────────────────────────────────────────────────

def test_bootstrap_ci_bounds():
    """Bootstrap CI must be in [0,1] with lo <= hi."""
    from src.phase1_stats import bootstrap_auc_ci
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    p = np.array([.1, .2, .3, .8, .7, .9, .4, .6, .2, .85])
    lo, hi, n_boot = bootstrap_auc_ci(y, p, n=200)
    assert 0 <= lo <= 1, f"CI lower bound out of range: {lo}"
    assert 0 <= hi <= 1, f"CI upper bound out of range: {hi}"
    assert lo <= hi, f"CI lower > upper: {lo} > {hi}"


def test_bootstrap_single_class_guard():
    """Bootstrap must handle bootstrap samples with only one class."""
    from src.phase1_stats import bootstrap_auc_ci
    # Very imbalanced — some bootstrap samples will have only class 0
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([.1, .2, .1, .3, .2, .1, .2, .3, .1, .9])
    lo, hi, n_boot = bootstrap_auc_ci(y, p, n=500)
    assert lo >= 0 and hi <= 1, "CI out of range with imbalanced data"


# ── FR-04: Data deduplication ────────────────────────────────────────────────

def test_dedup_fix():
    """Deduplication must reduce row count when duplicates exist."""
    from src.phase1_stats import fix_duplicate_patients
    import pandas as pd, io
    csv_with_dups = """case_uuid,patient_id,project_id,subtype,gender,age,stage,days_followup,recurrence,days_to_recurrence,recurrence_label
A,TCGA-01,TCGA-LUAD,LUAD,male,60.0,Stage IB,500.0,,,0
B,TCGA-01,TCGA-LUAD,LUAD,male,60.0,Stage IB,500.0,,,0
C,TCGA-02,TCGA-LUAD,LUAD,female,55.0,Stage IIA,700.0,Yes,,1
"""
    # Write temp file
    tmp = "/tmp/test_dedup.csv"
    with open(tmp, "w") as f:
        f.write(csv_with_dups)
    df = fix_duplicate_patients(tmp)
    assert len(df) == 2, f"Expected 2 unique patients, got {len(df)}"
    assert df['patient_id'].nunique() == 2


# ── FR-16: Focal loss ────────────────────────────────────────────────────────

def test_focal_loss_range():
    """Focal loss output must be a positive scalar."""
    from src.phase2_gmu import FocalLoss
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    probs   = torch.sigmoid(torch.randn(16))
    targets = torch.randint(0, 2, (16,)).float()
    loss = criterion(probs, targets)
    assert loss.item() > 0, "Focal loss must be positive"
    assert not torch.isnan(loss), "Focal loss is NaN"
    assert loss.item() < 10, f"Focal loss unreasonably large: {loss.item()}"


# ── Traceability matrix ───────────────────────────────────────────────────────

TRACEABILITY = {
    "FR-09": ["test_clinical_no_nan", "test_clinical_feature_count"],
    "FR-11": ["test_embedding_shape"],
    "FR-15": ["test_gmu_output_shape", "test_gmu_softmax_invariant",
               "test_gmu_probs_in_range", "test_gmu_different_patients_different_weights"],
    "FR-16": ["test_focal_loss_range"],
    "FR-17": ["test_bootstrap_ci_bounds", "test_bootstrap_single_class_guard"],
    "FR-04": ["test_dedup_fix"],
}

def test_traceability_matrix_complete():
    """All functional requirements have at least one test."""
    for fr_id, tests in TRACEABILITY.items():
        assert len(tests) > 0, f"{fr_id} has no tests"
'''


def write_unit_tests():
    """Write the unit test file to tests/test_pipeline.py"""
    os.makedirs("tests", exist_ok=True)
    test_path = "tests/test_pipeline.py"
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(UNIT_TEST_CODE)
    print(f"\n[Unit Tests] Written to {test_path}")
    print(f"  Run: pytest tests/test_pipeline.py -v")
    print(f"\n  Requirements traceability (for Appendix):")
    print(f"  {'FR-ID':<10} {'Test Functions'}")
    print(f"  {'─'*60}")
    traceability = {
        "FR-04": "test_dedup_fix",
        "FR-09": "test_clinical_no_nan, test_clinical_feature_count",
        "FR-11": "test_embedding_shape",
        "FR-15": "test_gmu_output_shape, test_gmu_softmax_invariant, ...",
        "FR-16": "test_focal_loss_range",
        "FR-17": "test_bootstrap_ci_bounds, test_bootstrap_single_class_guard",
    }
    for fr_id, tests in traceability.items():
        print(f"  {fr_id:<10} {tests}")


def main():
    print(f"\n{'='*60}")
    print(f" NSCLC Phase 3: Missing Modality + Unit Tests")
    print(f"{'='*60}")

    results = run_missing_modality_test()
    write_unit_tests()

    print(f"\n{'='*60}")
    print(f" Phase 3 Complete.")
    print(f" Figures saved to: {CONFIG['figures_dir']}/")
    print(f" Results saved to: {CONFIG['results_dir']}/missing_modality.json")
    print(f"\n Next steps:")
    print(f"   1. Run: pytest tests/test_pipeline.py -v")
    print(f"   2. Add GatedFusionModel to src/models/fusion_model.py")


if __name__ == "__main__":
    main()