"""
dataset.py
==========
Dataset classes for DSA-LoRA: Domain-Shift-Aware Parameter-Efficient
Fine-Tuning for Malaria Detection in Low-Resource Clinical Settings.

Two datasets are used:

    NIH Malaria Dataset (source domain — HIC):
        Clean, standardised 224x224 cell images from a controlled lab.
        13,152 training images. Balanced classes.
        kaggle: shahriar26s/malaria-detection

    Lacuna Malaria Detection Dataset (target domain — LMIC):
        Full field-of-view images (4160x3120) captured via smartphone
        over a microscope in Uganda and Ghana clinical settings.
        2,747 images. Annotation via bounding box CSV.
        kaggle: rajsahu2004/lacuna-malaria-detection-dataset

The domain shift between these two datasets — different equipment,
image quality, resolution, and disease presentation — is the core
challenge this work addresses.
"""

import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ── STANDARD TRANSFORMS ───────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform():
    """
    Training transforms with augmentation.
    Used for all fine-tuning experiments (Stages 1, 3, 4).
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform():
    """
    Validation/test transforms — no augmentation.
    Used for evaluation and feature extraction (Stage 2).
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── NIH MALARIA DATASET (SOURCE DOMAIN) ──────────────────────────────────────

class NIHMalariaDataset(Dataset):
    """
    NIH Malaria Cell Image Dataset — source domain (HIC conditions).

    Expects folder structure:
        root_dir/
            Parasitized/    → label 1
            Uninfected/     → label 0

    Handles any capitalisation of folder names.
    Validates that both classes are present before returning.

    Args:
        root_dir:  path to split folder (train/valid/test)
        transform: torchvision transforms to apply
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []

        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        for folder in self.root_dir.iterdir():
            if not folder.is_dir():
                continue
            name_lower = folder.name.lower()

            if 'parasit' in name_lower:
                label = 1
            elif 'uninfect' in name_lower or 'normal' in name_lower:
                label = 0
            else:
                print(f"  WARNING: unrecognised folder '{folder.name}' — skipping")
                continue

            count = 0
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in valid_ext:
                    self.samples.append((str(img_path), label))
                    count += 1

            print(f"  '{folder.name}' → label {label}: {count} images")

        # Validate both classes are present
        labels_present = set(s[1] for s in self.samples)
        if labels_present != {0, 1}:
            raise ValueError(
                f"Dataset at {root_dir} only has classes {labels_present}. "
                f"Expected both 0 and 1. "
                f"Folders found: {[d.name for d in self.root_dir.iterdir() if d.is_dir()]}"
            )

        pos = sum(1 for s in self.samples if s[1] == 1)
        neg = sum(1 for s in self.samples if s[1] == 0)
        print(f"  Total: {len(self.samples)} | Parasitized: {pos} | Uninfected: {neg}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class NIHProbeDataset(Dataset):
    """
    Balanced probe subset of the NIH dataset for MMD feature extraction.

    Randomly samples n_samples images (balanced across classes) from
    the NIH training split. Used in Stage 2 to compute source domain
    feature representations at each ViT layer.

    Args:
        root_dir:  path to NIH training split folder
        transform: torchvision transforms (use get_val_transform())
        n_samples: total probe set size (n_samples/2 per class)
        seed:      random seed for reproducibility
    """

    def __init__(self, root_dir, transform=None, n_samples=500, seed=42):
        base    = NIHMalariaDataset(root_dir, transform=None)
        pos     = [s for s in base.samples if s[1] == 1]
        neg     = [s for s in base.samples if s[1] == 0]
        n_each  = n_samples // 2

        random.seed(seed)
        self.samples   = (random.sample(pos, min(n_each, len(pos))) +
                          random.sample(neg, min(n_each, len(neg))))
        self.transform = transform

        print(f"NIH probe set: {len(self.samples)} images "
              f"({sum(1 for s in self.samples if s[1]==1)} pos, "
              f"{sum(1 for s in self.samples if s[1]==0)} neg)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ── LACUNA MALARIA DATASET (TARGET DOMAIN — LMIC) ────────────────────────────

class LacunaDataset(Dataset):
    """
    Lacuna Malaria Detection Challenge Dataset — target domain (LMIC).

    Images are full field-of-view captures (4160x3120 pixels) taken by
    placing a smartphone over a microscope eyepiece in Uganda and Ghana.
    This represents real-world LMIC clinical imaging conditions.

    Annotation format: bounding box CSV with one row per detected object.
    Converted to image-level binary labels:
        - Any image with a Trophozoite annotation → Parasitized (1)
        - Images with only NEG or WBC annotations → Uninfected (0)

    Args:
        csv_path:  path to Train.csv
        img_dir:   path to images/ folder
        transform: torchvision transforms
    """

    def __init__(self, csv_path, img_dir, transform=None):
        self.transform = transform
        self.img_dir   = Path(img_dir)

        df = pd.read_csv(csv_path)

        # Convert to image-level binary labels
        image_labels = df.groupby('Image_ID')['class'].apply(
            lambda x: 1 if (x == 'Trophozoite').any() else 0
        ).reset_index()
        image_labels.columns = ['Image_ID', 'label']

        # Keep only images that exist in the folder
        existing     = set(os.listdir(self.img_dir))
        image_labels = image_labels[image_labels['Image_ID'].isin(existing)]
        self.samples = list(zip(image_labels['Image_ID'], image_labels['label']))

        pos = sum(1 for _, l in self.samples if l == 1)
        neg = sum(1 for _, l in self.samples if l == 0)
        print(f"Lacuna dataset: {len(self.samples)} images "
              f"| Parasitized: {pos} | Uninfected: {neg}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        image = Image.open(self.img_dir / img_id).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_balanced_indices(self, n, seed=42):
        """
        Return indices for a balanced subset of n samples (n/2 per class).
        Used to simulate few-shot LMIC data scarcity in Stage 3 and 4.

        Args:
            n:    total number of samples (must be even)
            seed: random seed

        Returns:
            list of n indices, balanced across classes
        """
        random.seed(seed)
        pos_idx = [i for i, (_, l) in enumerate(self.samples) if l == 1]
        neg_idx = [i for i, (_, l) in enumerate(self.samples) if l == 0]
        n_each  = n // 2

        if len(neg_idx) < n_each:
            print(f"  WARNING: only {len(neg_idx)} negative samples available, "
                  f"requested {n_each}. Using all negatives.")
            n_each = min(n_each, len(neg_idx))

        return (random.sample(pos_idx, min(n_each, len(pos_idx))) +
                random.sample(neg_idx, n_each))


class LacunaProbeDataset(Dataset):
    """
    Balanced probe subset of the Lacuna dataset for MMD feature extraction.

    Used in Stage 2 to compute target domain feature representations
    at each ViT layer for comparison with NIH source domain features.

    Args:
        csv_path:  path to Train.csv
        img_dir:   path to images/ folder
        transform: torchvision transforms (use get_val_transform())
        n_samples: total probe set size
        seed:      random seed
    """

    def __init__(self, csv_path, img_dir, transform=None,
                 n_samples=500, seed=42):
        base    = LacunaDataset(csv_path, img_dir, transform=None)
        indices = base.get_balanced_indices(n_samples, seed=seed)

        self.samples   = [base.samples[i] for i in indices]
        self.img_dir   = base.img_dir
        self.transform = transform

        pos = sum(1 for _, l in self.samples if l == 1)
        neg = sum(1 for _, l in self.samples if l == 0)
        print(f"Lacuna probe set: {len(self.samples)} images "
              f"({pos} pos, {neg} neg)")
        print(f"  Note: images are full FOV 4160x3120 → resized to 224x224")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        image = Image.open(self.img_dir / img_id).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
