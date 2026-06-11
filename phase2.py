# ============================================
# PHASE 2 : MODEL ARCHITECTURE
# phase2.py
# ============================================

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from torchvision.models import DenseNet121_Weights

# ============================================
# DEVICE
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing Device:", device)

# ============================================
# LOAD PRETRAINED DENSENET121
# ============================================

model = models.densenet121(
    weights=DenseNet121_Weights.DEFAULT
)

print("[INFO] DenseNet121 Loaded")

# ============================================
# FREEZE ALL PARAMETERS
# ============================================

for param in model.parameters():
    param.requires_grad = False

print("[INFO] All Layers Frozen")

# ============================================
# UNFREEZE DENSEBLOCK 3
# ============================================

for param in model.features.denseblock3.parameters():
    param.requires_grad = True

print("[INFO] DenseBlock3 Unfrozen")

# ============================================
# UNFREEZE DENSEBLOCK 4
# ============================================

for param in model.features.denseblock4.parameters():
    param.requires_grad = True

print("[INFO] DenseBlock4 Unfrozen")

# ============================================
# CUSTOM CLASSIFIER
# ============================================

model.classifier = nn.Sequential(

    nn.Linear(
        in_features=1024,
        out_features=512
    ),

    nn.BatchNorm1d(512),

    nn.ReLU(inplace=True),

    nn.Dropout(0.5),

    nn.Linear(
        in_features=512,
        out_features=1
    )
)

# ============================================
# UNFREEZE CLASSIFIER
# ============================================

for param in model.classifier.parameters():
    param.requires_grad = True

print("[INFO] Custom Classifier Added")

# ============================================
# MOVE MODEL TO DEVICE
# ============================================

model = model.to(device)

# ============================================
# LOSS FUNCTION
# ============================================

# Binary Classification:
# 0 = Normal
# 1 = Pneumonia

criterion = nn.BCEWithLogitsLoss()

# ============================================
# OPTIMIZER
# ============================================

optimizer = optim.Adam(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=1e-5,

    weight_decay=1e-4
)

# ============================================
# LEARNING RATE SCHEDULER
# ============================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode='min',

    factor=0.5,

    patience=2,

    min_lr=1e-7
)

# ============================================
# TRAINABLE PARAMETER COUNT
# ============================================

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("\n===================================")
print("MODEL SUMMARY")
print("===================================")
print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")
print("===================================")

print("\n[OK] Phase 2 Loaded Successfully")