# ============================================
# PHASE 2 : MODEL ARCHITECTURE
# phase2.py
# ============================================

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

# ============================================
# DEVICE
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing Device:", device)

# ============================================
# LOAD DENSENET121
# ============================================

model = models.densenet121(
    pretrained=True
)

# ============================================
# FREEZE ALL PARAMETERS
# ============================================

for param in model.parameters():

    param.requires_grad = False

# ============================================
# UNFREEZE LAST DENSE BLOCK
# ============================================

for param in model.features.denseblock4.parameters():

    param.requires_grad = True

# ============================================
# CUSTOM CLASSIFIER
# ============================================

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(1024,512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512,3)
)

# ============================================
# UNFREEZE CLASSIFIER
# ============================================

for param in model.classifier.parameters():

    param.requires_grad = True

# ============================================
# MOVE TO DEVICE
# ============================================

model = model.to(device)

# ============================================
# LOSS FUNCTION
# ============================================

# Balance class weight to penalize False Positives more heavily
# Ratio of Normal (1341) to Pneumonia (3875) = 1341/3875 ≈ 0.35
pos_weight = torch.tensor([0.35]).to(device)
#criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion = nn.CrossEntropyLoss(pos_weight=pos_weight)

# ============================================
# OPTIMIZER
# ============================================

optimizer = optim.Adam(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=1e-5
)

print("[OK] Phase 2 Loaded Successfully")