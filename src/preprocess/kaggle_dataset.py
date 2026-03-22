"""
Kaggle LC25000 Dataset loader for NSCLC subtype classification.
Classes: LUAD (lung adenocarcinoma) and LUSC (lung squamous cell carcinoma).
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class KaggleNSCLCDataset(Dataset):
    """
    Dataset for loading Kaggle LC25000 histopathology image patches.

    Expected folder structure:
        data/raw/LUAD/*.jpg
        data/raw/LUSC/*.jpg

    Args:
        root_dir (str): Path to data/raw/ folder.
        split (str): One of 'train', 'val', or 'test'.
        train_ratio (float): Proportion of data used for training.
        val_ratio (float): Proportion of data used for validation.
        seed (int): Random seed for reproducibility.
        transform: Optional torchvision transform pipeline.
    """

    CLASS_NAMES = ["LUAD", "LUSC"]

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
        transform=None,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.root_dir = root_dir
        self.split = split
        self.transform = transform if transform else self._default_transform(split)

        self.samples = []   # list of (image_path, label)
        self.labels = []    # list of int labels (0 = LUAD, 1 = LUSC)

        import random
        random.seed(seed)

        for label_idx, class_name in enumerate(self.CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Expected folder not found: {class_dir}"
                )

            files = sorted([
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            random.shuffle(files)

            n = len(files)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            if split == "train":
                selected = files[:n_train]
            elif split == "val":
                selected = files[n_train: n_train + n_val]
            else:  # test
                selected = files[n_train + n_val:]

            for fname in selected:
                self.samples.append(
                    (os.path.join(class_dir, fname), label_idx)
                )
                self.labels.append(label_idx)

        print(
            f"[KaggleNSCLCDataset] {split}: {len(self.samples)} samples "
            f"(LUAD={self.labels.count(0)}, LUSC={self.labels.count(1)})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def _default_transform(split: str):
        """
        Standard ImageNet-normalised transforms.
        Training split includes augmentation; val/test use centre crop only.
        """
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if split == "train":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ])


def get_class_weights(dataset):
    """
    Compute inverse-frequency class weights for handling class imbalance.
    Returns a tensor of shape [num_classes].
    """
    import torch
    from collections import Counter

    counts = Counter(dataset.labels)
    total = len(dataset.labels)
    weights = [total / counts[i] for i in range(len(KaggleNSCLCDataset.CLASS_NAMES))]
    return torch.tensor(weights, dtype=torch.float)


if __name__ == "__main__":
    # Quick sanity check — run from project root:
    # python src/preprocess/kaggle_dataset.py
    dataset = KaggleNSCLCDataset(root_dir="data/raw", split="train")
    img, label = dataset[0]
    print(f"Sample image shape : {img.shape}")
    print(f"Sample label       : {label} ({KaggleNSCLCDataset.CLASS_NAMES[label]})")
    print(f"Class weights      : {get_class_weights(dataset)}")