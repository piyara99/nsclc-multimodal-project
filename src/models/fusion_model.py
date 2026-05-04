"""
Multimodal Late Fusion Model for NSCLC Recurrence Prediction.

Architecture:
    Image branch  : pre-extracted ResNet-50 embeddings (256-dim)
                    → Linear(256, 128) → BN → ReLU → Dropout
    Clinical branch: raw clinical features (5-dim)
                    → Linear(5, 32) → BN → ReLU → Dropout
                    → Linear(32, 64) → BN → ReLU → Dropout
    Fusion         : Concat(128 + 64 = 192)
                    → Linear(192, 64) → ReLU → Dropout
                    → Linear(64, 1) → Sigmoid

Model classes:
    WeightedFusionModel : scalar-weighted late fusion (matches fusion_best.pth)
    LateFusionModel     : concat fusion with named sub-modules
    ImageOnlyModel      : ablation baseline (image embeddings only)
    ClinicalOnlyModel   : ablation baseline (clinical features only)
    ConcatFusionModel   : explicit concat fusion baseline (for ablation)
"""

import torch
import torch.nn as nn


# ─── WeightedFusionModel ──────────────────────────────────────────────────────
# This class matches the keys saved in outputs/models/fusion_best.pth:
#   image_branch.*, clinical_branch.*, alpha
# Used by the dashboard for live prediction.

class WeightedFusionModel(nn.Module):
    """
    Scalar-weighted late fusion model for binary recurrence prediction.

    Combines image and clinical branches with a learnable scalar weight
    (alpha) that blends the two branch outputs before classification.

    State dict keys: image_branch.*, clinical_branch.*, alpha
    Checkpoint: outputs/models/fusion_best.pth

    Args:
        image_embedding_dim (int): Dimension of input image embeddings (256).
        clinical_input_dim  (int): Number of clinical input features (5).
        hidden_dim          (int): Hidden dimension for both branches.
        dropout             (float): Dropout rate.
    """

    def __init__(
        self,
        image_embedding_dim: int = 256,
        clinical_input_dim: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super(WeightedFusionModel, self).__init__()

        # Image branch: 256 → 128 → BN → ReLU → Dropout → 64 → output
        self.image_branch = nn.Sequential(
            nn.Linear(image_embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Clinical branch: 5 → 32 → BN → ReLU → Dropout → 64 → output
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Learnable scalar weight: blends image vs clinical logits
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        image_emb: torch.Tensor,
        clinical_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_emb     : (B, 256) pre-extracted image embeddings
            clinical_feat : (B, 5)   scaled clinical features
        Returns:
            probs : (B,) recurrence probability in [0, 1]
        """
        img_logit  = self.image_branch(image_emb).squeeze(1)     # (B,)
        clin_logit = self.clinical_branch(clinical_feat).squeeze(1)  # (B,)
        alpha      = torch.sigmoid(self.alpha)                    # clamp to (0,1)
        fused      = alpha * img_logit + (1 - alpha) * clin_logit
        return torch.sigmoid(fused)


# ─── Clinical MLP Encoder ────────────────────────────────────────────────────

class ClinicalMLP(nn.Module):
    """
    Multi-Layer Perceptron encoder for structured clinical features.

    Args:
        input_dim    (int):  Number of clinical input features (default 5).
        hidden_dims  (list): Sizes of hidden layers.
        output_dim   (int):  Dimension of output clinical embedding.
        dropout      (float): Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: list = None,
        output_dim: int = 64,
        dropout: float = 0.3,
    ):
        super(ClinicalMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32]

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        layers += [
            nn.Linear(in_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        ]

        self.encoder = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ─── Image Embedding Projector ───────────────────────────────────────────────

class ImageProjector(nn.Module):
    """
    Projects pre-extracted image embeddings into a lower-dimensional space.

    Args:
        input_dim  (int): Dimension of incoming image embeddings (256).
        output_dim (int): Projected dimension.
        dropout    (float): Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.3,
    ):
        super(ImageProjector, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


# ─── Late Fusion Model (concat) ───────────────────────────────────────────────

class LateFusionModel(nn.Module):
    """
    Multimodal late fusion model using feature concatenation.

    State dict keys: image_projector.*, clinical_encoder.*, fusion_head.*

    Args:
        image_embedding_dim  (int):  Dimension of input image embeddings.
        clinical_input_dim   (int):  Number of clinical input features.
        image_proj_dim       (int):  Output dim of image projector branch.
        clinical_hidden_dims (list): Hidden layer sizes for clinical MLP.
        clinical_output_dim  (int):  Output dim of clinical MLP branch.
        fusion_hidden_dim    (int):  Hidden dim of fusion classification head.
        dropout              (float): Dropout rate across all components.
    """

    def __init__(
        self,
        image_embedding_dim: int = 256,
        clinical_input_dim: int = 5,
        image_proj_dim: int = 128,
        clinical_hidden_dims: list = None,
        clinical_output_dim: int = 64,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super(LateFusionModel, self).__init__()

        if clinical_hidden_dims is None:
            clinical_hidden_dims = [32]

        self.image_projector = ImageProjector(
            input_dim=image_embedding_dim,
            output_dim=image_proj_dim,
            dropout=dropout,
        )

        self.clinical_encoder = ClinicalMLP(
            input_dim=clinical_input_dim,
            hidden_dims=clinical_hidden_dims,
            output_dim=clinical_output_dim,
            dropout=dropout,
        )

        fusion_input_dim = image_proj_dim + clinical_output_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        image_emb: torch.Tensor,
        clinical_feat: torch.Tensor,
    ) -> torch.Tensor:
        img_out  = self.image_projector(image_emb)
        clin_out = self.clinical_encoder(clinical_feat)
        fused    = torch.cat([img_out, clin_out], dim=1)
        logits   = self.fusion_head(fused).squeeze(1)
        return self.sigmoid(logits)

    def get_embeddings(
        self,
        image_emb: torch.Tensor,
        clinical_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused embedding before classification head (for SHAP)."""
        img_out  = self.image_projector(image_emb)
        clin_out = self.clinical_encoder(clinical_feat)
        return torch.cat([img_out, clin_out], dim=1)


# ─── ConcatFusionModel ────────────────────────────────────────────────────────

class ConcatFusionModel(nn.Module):
    """
    Explicit concatenation fusion baseline (ablation study).

    Identical to LateFusionModel in structure — exists as a named
    class so train_fusion.py can train and save it under 'concat_fusion'
    with a distinct checkpoint name.

    State dict keys: image_projector.*, clinical_encoder.*, fusion_head.*
    Checkpoint: outputs/models/concat_fusion_best.pth
    """

    def __init__(
        self,
        image_embedding_dim: int = 256,
        clinical_input_dim: int = 5,
        image_proj_dim: int = 128,
        clinical_output_dim: int = 64,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super(ConcatFusionModel, self).__init__()

        self.image_projector = ImageProjector(
            input_dim=image_embedding_dim,
            output_dim=image_proj_dim,
            dropout=dropout,
        )

        self.clinical_encoder = ClinicalMLP(
            input_dim=clinical_input_dim,
            hidden_dims=[32],
            output_dim=clinical_output_dim,
            dropout=dropout,
        )

        fusion_input_dim = image_proj_dim + clinical_output_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        image_emb: torch.Tensor,
        clinical_feat: torch.Tensor,
    ) -> torch.Tensor:
        img_out  = self.image_projector(image_emb)
        clin_out = self.clinical_encoder(clinical_feat)
        fused    = torch.cat([img_out, clin_out], dim=1)
        return self.sigmoid(self.fusion_head(fused).squeeze(1))


# ─── Ablation Baselines ───────────────────────────────────────────────────────

class ImageOnlyModel(nn.Module):
    """Ablation baseline: image embeddings only."""

    def __init__(
        self,
        image_embedding_dim: int = 256,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super(ImageOnlyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, image_emb: torch.Tensor, clinical_feat=None) -> torch.Tensor:
        return self.net(image_emb).squeeze(1)


class ClinicalOnlyModel(nn.Module):
    """Ablation baseline: clinical features only."""

    def __init__(
        self,
        clinical_input_dim: int = 5,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super(ClinicalOnlyModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64]

        layers = []
        in_dim = clinical_input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, image_emb=None, clinical_feat: torch.Tensor = None) -> torch.Tensor:
        return self.net(clinical_feat).squeeze(1)


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_fusion_model(
    model_type: str = "fusion",
    image_embedding_dim: int = 256,
    clinical_input_dim: int = 5,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Factory function to build any model variant.

    Args:
        model_type: One of 'weighted_fusion', 'fusion', 'concat_fusion',
                    'image_only', 'clinical_only'

    Notes:
        'weighted_fusion' → WeightedFusionModel (matches fusion_best.pth)
        'fusion'          → LateFusionModel (concat, named sub-modules)
        'concat_fusion'   → ConcatFusionModel (explicit concat ablation)
        'image_only'      → ImageOnlyModel
        'clinical_only'   → ClinicalOnlyModel
    """
    if model_type == "weighted_fusion":
        return WeightedFusionModel(
            image_embedding_dim=image_embedding_dim,
            clinical_input_dim=clinical_input_dim,
            dropout=dropout,
        )
    elif model_type in ("fusion", "concat_fusion"):
        if model_type == "concat_fusion":
            return ConcatFusionModel(
                image_embedding_dim=image_embedding_dim,
                clinical_input_dim=clinical_input_dim,
                dropout=dropout,
            )
        return LateFusionModel(
            image_embedding_dim=image_embedding_dim,
            clinical_input_dim=clinical_input_dim,
            dropout=dropout,
        )
    elif model_type == "image_only":
        return ImageOnlyModel(
            image_embedding_dim=image_embedding_dim,
            dropout=dropout,
        )
    elif model_type == "clinical_only":
        return ClinicalOnlyModel(
            clinical_input_dim=clinical_input_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from: weighted_fusion, fusion, concat_fusion, image_only, clinical_only"
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B = 8
    dummy_img  = torch.randn(B, 256).to(device)
    dummy_clin = torch.randn(B, 5).to(device)

    for mt in ("weighted_fusion", "fusion", "concat_fusion", "image_only", "clinical_only"):
        model = build_fusion_model(model_type=mt).to(device)
        out   = model(dummy_img, dummy_clin)
        n     = sum(p.numel() for p in model.parameters())
        print(f"{mt:20s} → output: {out.shape}  params: {n:,}  sample: {out[0].item():.4f}")