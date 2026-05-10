# ============================================
# PHASE 2 : MODEL ARCHITECTURE
# ============================================

import torch
import torch.nn as nn
import torchvision.models as models

# ============================================
# DEVICE
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================
# LOAD DENSENET121
# ============================================

model = models.densenet121(pretrained=True)

# ============================================
# FREEZE FEATURES
# ============================================

for param in model.features.parameters():

    param.requires_grad = False

# ============================================
# CUSTOM CLASSIFIER
# ============================================

model.classifier = nn.Sequential(

    nn.Dropout(0.3),

    nn.Linear(1024, 1),

    nn.Sigmoid()
)

model = model.to(device)

# ============================================
# LOSS FUNCTION
# ============================================

criterion = nn.BCELoss()

# ============================================
# OPTIMIZER
# ============================================

import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

print("✅ Phase 2 Loaded")