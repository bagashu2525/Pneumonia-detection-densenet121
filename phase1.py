# ============================================
# PHASE 1 : DATA PREPROCESSING & LOADING
# phase1.py
# ============================================

import torch
import numpy as np

from torchvision import datasets, transforms

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)

# ============================================
# TRAIN TRANSFORMS
# ============================================

train_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05)
    ),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================
# VALIDATION / TEST TRANSFORMS
# ============================================

val_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================
# DATASETS
# ============================================

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    "dataset/test",
    transform=val_transform
)

# ============================================
# CLASS LABELS
# ============================================

print("\nClass Mapping:")

print(train_dataset.class_to_idx)

# ============================================
# BALANCED SAMPLER
# ============================================

targets = train_dataset.targets

class_counts = np.bincount(targets)

print("\nClass Counts:")

print(class_counts)

class_weights = 1. / class_counts

sample_weights = [
    class_weights[t]
    for t in targets
]

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ============================================
# DATALOADERS
# ============================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=sampler,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)

print("\n[OK] Phase 1 Loaded Successfully")