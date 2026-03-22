"""
ResNet-50 based encoder for NSCLC histopathology image analysis.

Two modes:
  - classifier mode : used during subtype classification training (LUAD vs LUSC)
  - encoder mode    : used during offline feature extraction (returns embedding vector)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """
    ResNet-50 with a configurable head.

    Args:
        num_classes (int): Number of output classes for classification head.
            Set to 2 for LUAD vs LUSC subtype classification.
        pretrained (bool): Use ImageNet pretrained weights.
        embedding_dim (int): Dimension of the feature embedding layer.
        mode (str): 'classifier' returns class logits;
                    'encoder' returns the embedding vector only.
        dropout (float): Dropout rate before the final classification layer.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        embedding_dim: int = 256,
        mode: str = "classifier",
        dropout: float = 0.4,
    ):
        super(ResNetEncoder, self).__init__()

        assert mode in ("classifier", "encoder"), \
            f"mode must be 'classifier' or 'encoder', got '{mode}'"

        self.mode = mode
        self.embedding_dim = embedding_dim

        # Load pretrained ResNet-50 backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the original fully connected layer
        # backbone.fc outputs 2048-dim features
        in_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Projection head: 2048 -> embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Classification head: embedding_dim -> num_classes
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            In 'classifier' mode: logits of shape (batch_size, num_classes).
            In 'encoder' mode: embedding of shape (batch_size, embedding_dim).
        """
        features = self.backbone(x)           # (B, 2048)
        embedding = self.embedding_layer(features)  # (B, embedding_dim)

        if self.mode == "encoder":
            return embedding

        logits = self.classifier(embedding)   # (B, num_classes)
        return logits

    def set_mode(self, mode: str):
        """Switch between 'classifier' and 'encoder' mode at inference time."""
        assert mode in ("classifier", "encoder")
        self.mode = mode

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method — always returns embedding regardless of mode."""
        features = self.backbone(x)
        return self.embedding_layer(features)

    def freeze_backbone(self):
        """Freeze all backbone parameters — fine-tune head layers only."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    embedding_dim: int = 256,
    mode: str = "classifier",
    dropout: float = 0.4,
) -> ResNetEncoder:
    """Factory function for clean model instantiation."""
    model = ResNetEncoder(
        num_classes=num_classes,
        pretrained=pretrained,
        embedding_dim=embedding_dim,
        mode=mode,
        dropout=dropout,
    )
    return model


if __name__ == "__main__":
    # Quick sanity check — run from project root:
    # python src/models/resnet_encoder.py
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(num_classes=2, pretrained=True).to(device)

    dummy_input = torch.randn(4, 3, 224, 224).to(device)

    # Test classifier mode
    model.set_mode("classifier")
    logits = model(dummy_input)
    print(f"Classifier output shape : {logits.shape}")   # (4, 2)

    # Test encoder mode
    model.set_mode("encoder")
    embeddings = model(dummy_input)
    print(f"Encoder output shape    : {embeddings.shape}")  # (4, 256)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters        : {total_params:,}")
    print(f"Trainable parameters    : {trainable:,}")