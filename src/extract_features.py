"""
Offline Feature Extraction Script.

Loads the trained ResNet-50 encoder and extracts 256-dim embeddings
from all Kaggle LC25000 image patches. Saves embeddings and labels
as compressed .npy files for efficient fusion model training.

This only needs to be run ONCE. The saved embeddings are reused
for all subsequent training experiments.

Usage (from project root, nsclc env active):
    python src/extract_features.py

Outputs:
    data/features/image_embeddings.npy   — shape (N, 256)
    data/features/image_labels.npy       — shape (N,)  int  0=LUAD 1=LUSC
    data/features/image_paths.npy        — shape (N,)  str  file paths
    data/features/extraction_info.json   — metadata about the extraction run
"""

import os
import sys
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.kaggle_dataset import KaggleNSCLCDataset
from src.models.resnet_encoder import build_model


# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "model_checkpoint" : "outputs/models/resnet_best.pth",
    "data_dir"         : "data/raw",
    "output_dir"       : "data/features",
    "batch_size"       : 64,      # larger batch is fine at inference — no gradients
    "num_workers"      : 0,
    "embedding_dim"    : 256,
    "seed"             : 42,
    "device"           : "cuda" if torch.cuda.is_available() else "cpu",
}


# ─── Dataset that exposes ALL splits together ─────────────────────────────────

class FullKaggleDataset(torch.utils.data.Dataset):
    """
    Loads all images from LUAD and LUSC folders regardless of split.
    Used only for offline feature extraction.
    """

    CLASS_NAMES = ["LUAD", "LUSC"]

    def __init__(self, root_dir: str):
        import torchvision.transforms as transforms
        from PIL import Image

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.samples = []   # (path, label_int)
        self.Image = Image

        for label_idx, class_name in enumerate(self.CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            files = sorted([
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            for fname in files:
                self.samples.append(
                    (os.path.join(class_dir, fname), label_idx)
                )

        print(f"[FullKaggleDataset] Total images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label, path


def collate_with_paths(batch):
    """Custom collate to handle (image, label, path) tuples."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    paths  = [item[2] for item in batch]
    return images, labels, paths


# ─── Main extraction loop ─────────────────────────────────────────────────────

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = torch.device(CONFIG["device"])

    print(f"\n{'='*60}")
    print(f" Offline Feature Extraction")
    print(f"{'='*60}")
    print(f" Device     : {device}")
    print(f" Checkpoint : {CONFIG['model_checkpoint']}")
    print(f" Output     : {CONFIG['output_dir']}")
    print(f"{'='*60}\n")

    # ── Load trained model ──
    model = build_model(
        num_classes=2,
        pretrained=False,       # weights come from checkpoint
        embedding_dim=CONFIG["embedding_dim"],
        mode="encoder",         # returns embeddings, not logits
    ).to(device)

    checkpoint = torch.load(
        CONFIG["model_checkpoint"], map_location=device
    )
    model.load_state_dict(checkpoint["model_state"])
    model.set_mode("encoder")
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val AUC: {checkpoint['val_auc']:.4f})\n")

    # ── Dataset and loader ──
    dataset = FullKaggleDataset(root_dir=CONFIG["data_dir"])
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_with_paths,
        pin_memory=True,
    )

    # ── Extract embeddings ──
    all_embeddings = []
    all_labels     = []
    all_paths      = []

    start = time.time()

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            embeddings = model(images)          # (B, 256)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    elapsed = time.time() - start

    # ── Stack and save ──
    embeddings_arr = np.vstack(all_embeddings)          # (N, 256)
    labels_arr     = np.array(all_labels, dtype=np.int64)  # (N,)
    paths_arr      = np.array(all_paths)                   # (N,)

    emb_path   = os.path.join(CONFIG["output_dir"], "image_embeddings.npy")
    lbl_path   = os.path.join(CONFIG["output_dir"], "image_labels.npy")
    paths_path = os.path.join(CONFIG["output_dir"], "image_paths.npy")

    np.save(emb_path,   embeddings_arr)
    np.save(lbl_path,   labels_arr)
    np.save(paths_path, paths_arr)

    # ── Save metadata ──
    info = {
        "total_images"    : int(len(labels_arr)),
        "luad_count"      : int(np.sum(labels_arr == 0)),
        "lusc_count"      : int(np.sum(labels_arr == 1)),
        "embedding_dim"   : int(embeddings_arr.shape[1]),
        "embeddings_file" : emb_path,
        "labels_file"     : lbl_path,
        "paths_file"      : paths_path,
        "extraction_time" : f"{elapsed:.1f}s",
        "checkpoint_used" : CONFIG["model_checkpoint"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_auc"  : float(checkpoint["val_auc"]),
    }
    info_path = os.path.join(CONFIG["output_dir"], "extraction_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f" Extraction complete in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f" Total images    : {info['total_images']}")
    print(f" LUAD            : {info['luad_count']}")
    print(f" LUSC            : {info['lusc_count']}")
    print(f" Embedding shape : {embeddings_arr.shape}")
    print(f" Embedding mean  : {embeddings_arr.mean():.4f}")
    print(f" Embedding std   : {embeddings_arr.std():.4f}")
    print(f"\n Saved:")
    print(f"   {emb_path}")
    print(f"   {lbl_path}")
    print(f"   {paths_path}")
    print(f"   {info_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()