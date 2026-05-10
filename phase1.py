# ============================================
# PHASE 1 : DATA PREPROCESSING & LOADING
# ============================================

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================
# TRANSFORMS
# ============================================

train_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),

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
# DATALOADERS
# ============================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

print("✅ Phase 1 Loaded")