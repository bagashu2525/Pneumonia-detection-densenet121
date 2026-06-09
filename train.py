# ============================================
# MAIN TRAINING PIPELINE
# train.py
# ============================================

# ============================================
# IMPORTS
# ============================================

import time
import torch

# ============================================
# IMPORT DATA LOADERS
# ============================================

from phase1 import (
    train_loader,
    val_loader,
    test_loader
)

# ============================================
# IMPORT MODEL COMPONENTS
# ============================================

from phase2 import (
    model,
    criterion,
    optimizer,
    device
)

# ============================================
# IMPORT TRAINING
# ============================================

from phase3 import train_model

# ============================================
# IMPORT EVALUATION
# ============================================

from phase4 import evaluate_model

# ============================================
# OPTIONAL GRAPH FUNCTIONS
# ============================================

try:
    from utils import (
        plot_training_curves,
        save_confusion_matrix
    )
except:
    plot_training_curves = None
    save_confusion_matrix = None

# ============================================
# START TIMER
# ============================================

start_time = time.time()

# ============================================
# TRAIN MODEL
# ============================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=10
)

# ============================================
# GENERATE TRAINING GRAPHS
# ============================================

if plot_training_curves is not None:

    print("\nGenerating Training Graphs...")

    plot_training_curves(history)

# ============================================
# LOAD BEST MODEL
# ============================================

print("\nLoading Best Model...")

model.load_state_dict(
    torch.load(
        "best_multiclass_model.pth",
        map_location=device
    )
)

# ============================================
# EVALUATE MODEL
# ============================================

print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

results = evaluate_model(
    model=model,
    test_loader=test_loader,
    device=device
)

# ============================================
# SAVE CONFUSION MATRIX
# ============================================

if (
    save_confusion_matrix is not None
    and "confusion_matrix" in results
):

    save_confusion_matrix(
        results["confusion_matrix"]
    )

# ============================================
# SAVE CLASSIFICATION REPORT
# ============================================

if "classification_report" in results:

    with open(
        "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            results["classification_report"]
        )

# ============================================
# PRINT FINAL RESULTS
# ============================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

if "accuracy" in results:
    print(
        f"Accuracy  : {results['accuracy']:.4f}"
    )

if "precision" in results:
    print(
        f"Precision : {results['precision']:.4f}"
    )

if "recall" in results:
    print(
        f"Recall    : {results['recall']:.4f}"
    )

if "f1" in results:
    print(
        f"F1 Score  : {results['f1']:.4f}"
    )

# ============================================
# PRINT CLASS-WISE METRICS
# ============================================

if "classification_report" in results:

    print("\nClassification Report:\n")

    print(
        results["classification_report"]
    )

# ============================================
# TOTAL EXECUTION TIME
# ============================================

total_time = time.time() - start_time

hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

print("\n" + "=" * 60)
print("EXECUTION TIME")
print("=" * 60)

print(
    f"{hours:02d}:{minutes:02d}:{seconds:02d}"
)

# ============================================
# GENERATED FILES
# ============================================

print("\nGenerated Files:")

generated_files = [
    "training_loss.png",
    "validation_loss.png",
    "accuracy_curve.png",
    "confusion_matrix.png",
    "classification_report.txt",
    "best_multiclass_model.pth"
]

for file in generated_files:
    print(f"✓ {file}")

# ============================================
# SUCCESS
# ============================================

print("\n" + "=" * 60)
print("MULTICLASS TB + PNEUMONIA + NORMAL")
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)