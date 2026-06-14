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
# CONFIGURATION
# ============================================

BATCH_SIZE = 16
NUM_WORKERS = 2

# ============================================
# TRAIN TRANSFORMS
# ============================================

train_transform = transforms.Compose([

    # ----------------------------------------
    # Convert X-Ray to 3 channels
    # ----------------------------------------

    transforms.Grayscale(
        num_output_channels=3
    ),

    # ----------------------------------------
    # Resize for DenseNet121
    # ----------------------------------------

    transforms.Resize(
        (224, 224)
    ),

    # ----------------------------------------
    # Data Augmentation
    # ----------------------------------------

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.RandomAffine(

        degrees=0,

        translate=(0.05, 0.05)
    ),

    transforms.RandomPerspective(

        distortion_scale=0.1,

        p=0.3
    ),

    transforms.RandomApply(

        [
            transforms.ColorJitter(

                brightness=0.05,

                contrast=0.05
            )
        ],

        p=0.3
    ),

    transforms.RandomApply(

        [
            transforms.GaussianBlur(
                kernel_size=3
            )
        ],

        p=0.2
    ),

    # ----------------------------------------
    # Tensor Conversion
    # ----------------------------------------

    transforms.ToTensor(),

    # ----------------------------------------
    # ImageNet Normalization
    # Required for DenseNet121
    # ----------------------------------------

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

# ============================================
# VALIDATION / TEST TRANSFORMS
# ============================================

val_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

# ============================================
# DATASETS
# ============================================

print("\nLoading Datasets...")

train_dataset = datasets.ImageFolder(

    root="dataset/train",

    transform=train_transform
)

val_dataset = datasets.ImageFolder(

    root="dataset/val",

    transform=val_transform
)

test_dataset = datasets.ImageFolder(

    root="dataset/test",

    transform=val_transform
)

# ============================================
# CLASS INFORMATION
# ============================================

CLASS_NAMES = list(
    train_dataset.class_to_idx.keys()
)

print("\n===================================")
print("CLASS MAPPING")
print("===================================")

print(train_dataset.class_to_idx)

print("\nClass Names:")

print(CLASS_NAMES)

# ============================================
# CLASS DISTRIBUTION
# ============================================

targets = train_dataset.targets

class_counts = np.bincount(
    targets
)

print("\n===================================")
print("TRAIN DATA DISTRIBUTION")
print("===================================")

for idx, count in enumerate(class_counts):

    print(
        f"{CLASS_NAMES[idx]} : {count}"
    )

# ============================================
# WEIGHTED RANDOM SAMPLER
# ============================================

class_weights = (
    1.0 / class_counts
)

sample_weights = [

    class_weights[target]

    for target in targets
]

sampler = WeightedRandomSampler(

    weights=sample_weights,

    num_samples=len(sample_weights),

    replacement=True
)

print("\nWeightedRandomSampler Enabled")

# ============================================
# CUDA SETTINGS
# ============================================

PIN_MEMORY = torch.cuda.is_available()

# ============================================
# TRAIN LOADER
# ============================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=sampler,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    persistent_workers=(
        NUM_WORKERS > 0
    )
)

# ============================================
# VALIDATION LOADER
# ============================================

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    persistent_workers=(
        NUM_WORKERS > 0
    )
)

# ============================================
# TEST LOADER
# ============================================

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    persistent_workers=(
        NUM_WORKERS > 0
    )
)

# ============================================
# DATASET SUMMARY
# ============================================

print("\n===================================")
print("DATASET SUMMARY")
print("===================================")

print(
    f"Training Images   : "
    f"{len(train_dataset)}"
)

print(
    f"Validation Images : "
    f"{len(val_dataset)}"
)

print(
    f"Testing Images    : "
    f"{len(test_dataset)}"
)

print(
    f"Batch Size        : "
    f"{BATCH_SIZE}"
)

print(
    f"Workers           : "
    f"{NUM_WORKERS}"
)

print(
    f"Pin Memory        : "
    f"{PIN_MEMORY}"
)

print("\n[OK] Phase 1 Loaded Successfully")