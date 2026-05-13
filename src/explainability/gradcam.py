"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet-50.

Generates heatmaps highlighting which regions of a histopathology image
patch most influenced the model's subtype classification prediction.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization."

Usage:
    python src/explainability/gradcam.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torchvision.transforms as transforms
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.resnet_encoder import build_model


class GradCAM:
    """
    Grad-CAM implementation for ResNet-50.

    Hooks into the final convolutional layer (layer4) to capture
    activations and gradients during a forward-backward pass.

    Args:
        model: ResNetEncoder instance in 'classifier' mode.
        target_layer: Name of the layer to hook (default: 'layer4').
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = "layer4"):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._hooks      = []

        # Register hooks on the target layer of the backbone
        target = dict(model.backbone.named_children())[target_layer]
        self._hooks.append(
            target.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            target.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            image_tensor: Shape (1, 3, 224, 224) — normalised image tensor.
            class_idx: Target class index. If None, uses predicted class.

        Returns:
            heatmap: np.ndarray of shape (224, 224), values in [0, 1].
        """
        self.model.eval()
        self.model.set_mode("classifier")

        image_tensor = image_tensor.to(next(self.model.parameters()).device)
        image_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)            # (1, num_classes)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Pool gradients across spatial dimensions → (C,)
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])  # (C,)

        # Weight activations by pooled gradients → (C, H, W)
        activations = self.activations[0]                  # (C, H, W)
        for i, w in enumerate(pooled_grads):
            activations[i] *= w

        # Average over channels → (H, W)
        heatmap = activations.mean(dim=0).cpu().numpy()

        # ReLU — keep only positive influence
        heatmap = np.maximum(heatmap, 0)

        # Normalise to [0, 1]
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # Resize to input image size
        heatmap = cv2.resize(heatmap, (224, 224))

        return heatmap, class_idx

    def remove_hooks(self):
        """Clean up registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap=cm.jet,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_image: RGB image as np.ndarray, shape (H, W, 3), values [0, 255].
        heatmap: Grad-CAM heatmap, shape (H, W), values [0, 1].
        alpha: Transparency of the heatmap overlay.
        colormap: Matplotlib colormap to use.

    Returns:
        overlaid: np.ndarray of shape (H, W, 3), values [0, 255].
    """
    heatmap_colored = colormap(heatmap)[:, :, :3]           # (H, W, 3) float
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    original_image  = original_image.astype(np.uint8)
    overlaid = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid


def load_image_for_gradcam(image_path: str) -> tuple:
    """
    Load and preprocess an image for Grad-CAM.

    Returns:
        tensor: (1, 3, 224, 224) normalised tensor for model input.
        original: (224, 224, 3) uint8 array for visualisation.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    pil_img  = Image.open(image_path).convert("RGB")
    pil_img  = pil_img.resize((224, 224), Image.BILINEAR)
    original = np.array(pil_img)                  # (224, 224, 3)
    tensor   = transform(pil_img).unsqueeze(0)    # (1, 3, 224, 224)

    return tensor, original


def generate_gradcam_figure(
    model,
    image_paths: list,
    class_names: list,
    save_path: str,
    n_samples: int = 6,
):
    """
    Generate a multi-panel Grad-CAM figure showing original images
    alongside their heatmap overlays.

    Args:
        model: Trained ResNetEncoder.
        image_paths: List of image file paths to visualise.
        class_names: List of class name strings ['LUAD', 'LUSC'].
        save_path: Output PNG path.
        n_samples: Number of images to include (must be even).
    """
    device = next(model.parameters()).device
    gradcam = GradCAM(model, target_layer="layer4")

    # Sample n_samples images
    import random
    random.seed(42)
    selected_paths = random.sample(image_paths, min(n_samples, len(image_paths)))

    n_cols = 3
    n_rows = len(selected_paths) * 2 // n_cols
    if n_rows == 0:
        n_rows = 2

    fig, axes = plt.subplots(
        len(selected_paths), 2,
        figsize=(8, len(selected_paths) * 3)
    )
    if len(selected_paths) == 1:
        axes = [axes]

    for i, img_path in enumerate(selected_paths):
        tensor, original = load_image_for_gradcam(img_path)
        tensor = tensor.to(device)

        heatmap, pred_class = gradcam.generate(tensor)
        overlaid = overlay_heatmap(original, heatmap, alpha=0.45)

        # Determine true class from path
        true_class = "LUAD" if "LUAD" in img_path else "LUSC"
        pred_label = class_names[pred_class]

        axes[i][0].imshow(original)
        axes[i][0].set_title(f"Original\nTrue: {true_class}", fontsize=9)
        axes[i][0].axis("off")

        axes[i][1].imshow(overlaid)
        axes[i][1].set_title(f"Grad-CAM\nPred: {pred_label}", fontsize=9)
        axes[i][1].axis("off")

    plt.suptitle(
        "Grad-CAM: Tissue Regions Influencing Subtype Classification",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    gradcam.remove_hooks()
    print(f"Grad-CAM figure saved → {save_path}")


if __name__ == "__main__":
    import glob

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load trained model
    model = build_model(num_classes=3, pretrained=False, embedding_dim=256).to(device)
    checkpoint = torch.load("outputs/models/resnet_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.set_mode("classifier")
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Collect image paths
    luad_paths = glob.glob("data/raw/LUAD/*.jpg")[:50]
    lusc_paths = glob.glob("data/raw/LUSC/*.jpg")[:50]
    all_paths  = luad_paths + lusc_paths

    os.makedirs("outputs/figures", exist_ok=True)

    generate_gradcam_figure(
        model=model,
        image_paths=all_paths,
        class_names=["LUAD", "LUSC"],
        save_path="outputs/figures/gradcam_examples.png",
        n_samples=6,
    )

    print("\nGrad-CAM generation complete.")