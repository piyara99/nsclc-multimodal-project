"""
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
    assert torch.allclose(sums, torch.ones(16), atol=1e-5), \
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
    assert not torch.allclose(g1, g2, atol=1e-3), \
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
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "test_dedup.csv")
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
