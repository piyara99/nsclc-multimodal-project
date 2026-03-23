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

Also supports:
    - ImageOnlyModel   : ablation baseline using image embeddings only
    - ClinicalOnlyModel: ablation baseline using clinical features only
"""

import torch
import torch.nn as nn


# ─── Clinical MLP Encoder ────────────────────────────────────────────────────

class ClinicalMLP(nn.Module):
    """
    Multi-Layer Perceptron encoder for structured clinical features.

    Args:
        input_dim (int): Number of clinical input features (default 5).
        hidden_dims (list): Sizes of hidden layers.
        output_dim (int): Dimension of output clinical embedding.
        dropout (float): Dropout rate.
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
        input_dim (int): Dimension of incoming image embeddings (256).
        output_dim (int): Projected dimension.
        dropout (float): Dropout rate.
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


# ─── Late Fusion Model ────────────────────────────────────────────────────────

class LateFusionModel(nn.Module):
    """
    Multimodal late fusion model for binary recurrence prediction.

    Combines image embeddings (from ResNet-50) with clinical features
    (from MLP encoder) via feature concatenation followed by a
    classification head.

    Args:
        image_embedding_dim (int): Dimension of input image embeddings.
        clinical_input_dim (int): Number of clinical input features.
        image_proj_dim (int): Output dim of image projector branch.
        clinical_hidden_dims (list): Hidden layer sizes for clinical MLP.
        clinical_output_dim (int): Output dim of clinical MLP branch.
        fusion_hidden_dim (int): Hidden dim of fusion classification head.
        dropout (float): Dropout rate across all components.
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

        # Image branch
        self.image_projector = ImageProjector(
            input_dim=image_embedding_dim,
            output_dim=image_proj_dim,
            dropout=dropout,
        )

        # Clinical branch
        self.clinical_encoder = ClinicalMLP(
            input_dim=clinical_input_dim,
            hidden_dims=clinical_hidden_dims,
            output_dim=clinical_output_dim,
            dropout=dropout,
        )

        # Fusion head
        fusion_input_dim = image_proj_dim + clinical_output_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

        # Output activation
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        image_emb: torch.Tensor,
        clinical_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image_emb     : (B, image_embedding_dim) — pre-extracted embeddings
            clinical_feat : (B, clinical_input_dim)  — scaled clinical features

        Returns:
            probs : (B,) — recurrence probability in [0, 1]
        """
        img_out      = self.image_projector(image_emb)       # (B, 128)
        clin_out     = self.clinical_encoder(clinical_feat)  # (B, 64)
        fused        = torch.cat([img_out, clin_out], dim=1) # (B, 192)
        logits       = self.fusion_head(fused).squeeze(1)    # (B,)
        probs        = self.sigmoid(logits)
        return probs

    def get_embeddings(
        self,
        image_emb: torch.Tensor,
        clinical_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused embedding before classification head (for SHAP)."""
        img_out  = self.image_projector(image_emb)
        clin_out = self.clinical_encoder(clinical_feat)
        return torch.cat([img_out, clin_out], dim=1)


# ─── Ablation Baselines ───────────────────────────────────────────────────────

class ImageOnlyModel(nn.Module):
    """Ablation baseline: uses only image embeddings for recurrence prediction."""

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
    """Ablation baseline: uses only clinical features for recurrence prediction."""

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
        model_type: One of 'fusion', 'image_only', 'clinical_only'
    """
    assert model_type in ("fusion", "image_only", "clinical_only"), \
        f"model_type must be 'fusion', 'image_only', or 'clinical_only'"

    if model_type == "fusion":
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
    else:
        return ClinicalOnlyModel(
            clinical_input_dim=clinical_input_dim,
            dropout=dropout,
        )


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B = 8  # batch size

    dummy_img  = torch.randn(B, 256).to(device)
    dummy_clin = torch.randn(B, 5).to(device)

    for model_type in ("fusion", "image_only", "clinical_only"):
        model = build_fusion_model(model_type=model_type).to(device)
        out = model(dummy_img, dummy_clin)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{model_type:15s} → output: {out.shape}  "
              f"params: {n_params:,}  "
              f"sample prob: {out[0].item():.4f}")