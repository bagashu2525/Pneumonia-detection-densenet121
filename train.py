# ============================================
# MAIN TRAINING PIPELINE
# train.py
# ============================================

# ============================================
# IMPORT ALL PHASES
# ============================================

from phase1 import (
    train_loader,
    val_loader,
    test_loader
)

from phase2 import (
    model,
    criterion,
    optimizer,
    device
)

from phase3 import train_model

from phase4 import evaluate_model

# ============================================
# TRAIN MODEL
# ============================================

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=10
)

# ============================================
# LOAD BEST MODEL
# ============================================

import torch

model.load_state_dict(
    torch.load(
        "best_pneumonia_model.pth",
        map_location=device
    )
)

# ============================================
# EVALUATE MODEL
# ============================================

evaluate_model(
    model=model,
    test_loader=test_loader,
    device=device
)

print("\n✅ COMPLETE PIPELINE FINISHED")