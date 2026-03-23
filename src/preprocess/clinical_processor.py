"""
Clinical Data Preprocessor for NSCLC Multimodal Fusion.

Loads TCGA clinical master CSV, cleans, encodes, and normalises
all clinical features for input into the MLP clinical encoder.

Features used:
    - age             (continuous, normalised)
    - gender          (binary encoded: male=0, female=1)
    - subtype         (binary encoded: LUAD=0, LUSC=1)
    - stage_numeric   (ordinal: IA=1 ... IV=4, derived from stage string)
    - days_followup   (continuous, normalised)

Target:
    - recurrence_label (binary: 0=no recurrence, 1=recurrence)

Usage:
    from src.preprocess.clinical_processor import ClinicalProcessor
    processor = ClinicalProcessor('data/metadata/tcga_clinical_master.csv')
    X, y, feature_names = processor.get_features_and_labels()
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class ClinicalProcessor:
    """
    Preprocesses TCGA clinical data for use in the MLP encoder.

    Args:
        csv_path (str): Path to tcga_clinical_master.csv
        scaler_save_path (str): Where to save the fitted scaler for inference.
    """

    FEATURE_COLUMNS = [
        "age",
        "gender",
        "subtype",
        "stage_numeric",
        "days_followup",
    ]

    STAGE_MAP = {
        # Stage I variants
        "stage i"   : 1, "stage ia"  : 1, "stage ib"  : 1,
        "stage is"  : 1,
        # Stage II variants
        "stage ii"  : 2, "stage iia" : 2, "stage iib" : 2,
        # Stage III variants
        "stage iii" : 3, "stage iiia": 3, "stage iiib": 3,
        "stage iiic": 3,
        # Stage IV variants
        "stage iv"  : 4, "stage iva" : 4, "stage ivb" : 4,
    }

    def __init__(
        self,
        csv_path: str,
        scaler_save_path: str = "data/features/clinical_scaler.json",
    ):
        self.csv_path = csv_path
        self.scaler_save_path = scaler_save_path
        self.scaler = StandardScaler()
        self.feature_names = self.FEATURE_COLUMNS.copy()
        self._fitted = False

        self.df = self._load_and_clean()

    # ── Loading & cleaning ────────────────────────────────────────────────────

    def _load_and_clean(self) -> pd.DataFrame:
        """Load CSV and apply all cleaning and encoding steps."""
        df = pd.read_csv(self.csv_path)
        print(f"[ClinicalProcessor] Loaded {len(df)} patients from {self.csv_path}")

        # Drop rows with missing recurrence label — these cannot be used
        before = len(df)
        df = df.dropna(subset=["recurrence_label"])
        after = len(df)
        if before != after:
            print(f"  Dropped {before - after} rows with missing recurrence_label")

        # Gender encoding: male=0, female=1
        df["gender"] = df["gender"].str.lower().map(
            {"male": 0, "female": 1}
        ).fillna(0).astype(int)

        # Subtype encoding: LUAD=0, LUSC=1
        df["subtype"] = df["subtype"].str.upper().map(
            {"LUAD": 0, "LUSC": 1}
        ).fillna(0).astype(int)

        # Stage: convert string → ordinal integer
        df["stage_numeric"] = (
            df["stage"]
            .str.lower()
            .str.strip()
            .map(self.STAGE_MAP)
        )
        # Fill unknown stages with median
        stage_median = df["stage_numeric"].median()
        df["stage_numeric"] = df["stage_numeric"].fillna(stage_median)

        # Age: fill missing with median
        df["age"] = df["age"].fillna(df["age"].median())

        # Days followup: fill missing with median
        df["days_followup"] = df["days_followup"].fillna(
            df["days_followup"].median()
        )

        # Ensure label is integer
        df["recurrence_label"] = df["recurrence_label"].astype(int)

        print(f"  Recurrence distribution: "
              f"{dict(df['recurrence_label'].value_counts().sort_index())}")
        print(f"  Features: {self.FEATURE_COLUMNS}")

        return df

    # ── Feature extraction ────────────────────────────────────────────────────

    def get_features_and_labels(self, fit_scaler: bool = True):
        """
        Returns scaled feature matrix and label array.

        Args:
            fit_scaler: If True, fit the scaler on this data (training).
                        If False, use the previously fitted scaler (inference).

        Returns:
            X (np.ndarray): shape (N, num_features), float32
            y (np.ndarray): shape (N,), int
            feature_names (list): list of feature column names
        """
        X_raw = self.df[self.FEATURE_COLUMNS].values.astype(np.float32)
        y     = self.df["recurrence_label"].values.astype(np.int64)

        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_raw).astype(np.float32)
            self._fitted = True
            self._save_scaler(X_raw)
        else:
            if not self._fitted:
                raise RuntimeError(
                    "Scaler not fitted. Call get_features_and_labels(fit_scaler=True) first."
                )
            X_scaled = self.scaler.transform(X_raw).astype(np.float32)

        print(f"  Feature matrix shape : {X_scaled.shape}")
        print(f"  Label distribution   : {dict(zip(*np.unique(y, return_counts=True)))}")

        return X_scaled, y, self.FEATURE_COLUMNS

    def get_patient_ids(self) -> list:
        """Return list of patient_id strings in the same order as features."""
        return self.df["patient_id"].tolist()

    def get_dataframe(self) -> pd.DataFrame:
        """Return the cleaned dataframe."""
        return self.df.copy()

    # ── Scaler persistence ────────────────────────────────────────────────────

    def _save_scaler(self, X_raw: np.ndarray):
        """Save scaler parameters as JSON for reproducibility."""
        os.makedirs(os.path.dirname(self.scaler_save_path), exist_ok=True)
        scaler_params = {
            "feature_names" : self.FEATURE_COLUMNS,
            "mean_"         : self.scaler.mean_.tolist(),
            "scale_"        : self.scaler.scale_.tolist(),
            "n_samples_seen": int(self.scaler.n_samples_seen_),
        }
        with open(self.scaler_save_path, "w") as f:
            json.dump(scaler_params, f, indent=2)
        print(f"  Scaler saved → {self.scaler_save_path}")

    def get_class_weights_tensor(self):
        """
        Compute inverse-frequency class weights for imbalanced binary labels.
        Returns a torch.Tensor of shape [2].
        """
        import torch
        y = self.df["recurrence_label"].values
        n_total = len(y)
        n_pos   = y.sum()
        n_neg   = n_total - n_pos
        # weight for class 0 (no recurrence), class 1 (recurrence)
        w0 = n_total / (2.0 * n_neg)
        w1 = n_total / (2.0 * n_pos)
        return torch.tensor([w0, w1], dtype=torch.float32)


if __name__ == "__main__":
    processor = ClinicalProcessor("data/metadata/tcga_clinical_master.csv")
    X, y, names = processor.get_features_and_labels(fit_scaler=True)

    print(f"\nFeature names : {names}")
    print(f"X shape       : {X.shape}")
    print(f"y shape       : {y.shape}")
    print(f"Class weights : {processor.get_class_weights_tensor()}")
    print(f"\nFirst 3 rows of X:\n{X[:3]}")
    print(f"First 3 labels : {y[:3]}")