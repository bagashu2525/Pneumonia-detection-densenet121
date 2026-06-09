# ============================================
# utils.py
# ============================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# CREATE RESULTS DIRECTORY
# ============================================

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# ============================================
# TRAINING CURVES
# ============================================

def plot_training_curves(history):

    train_losses = history["train_loss"]
    val_losses = history["val_loss"]

    train_accs = history["train_acc"]
    val_accs = history["val_acc"]

    epochs = range(
        1,
        len(train_losses) + 1
    )

    # ----------------------------------------
    # Training Loss
    # ----------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        marker='o',
        label="Training Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "training_loss.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    # ----------------------------------------
    # Validation Loss
    # ----------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        val_losses,
        marker='o',
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss vs Epoch")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "validation_loss.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    # ----------------------------------------
    # Accuracy Graph
    # ----------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accs,
        marker='o',
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        val_accs,
        marker='o',
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epoch")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "accuracy_curve.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    print(
        "[INFO] Training graphs saved."
    )

# ============================================
# CONFUSION MATRIX
# ============================================

def save_confusion_matrix(cm):

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Normal",
            "Pneumonia",
            "Tuberculosis"
        ],
        yticklabels=[
            "Normal",
            "Pneumonia",
            "Tuberculosis"
        ]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.title("Confusion Matrix")

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "confusion_matrix.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    print(
        "[INFO] Confusion matrix saved."
    )

# ============================================
# K-FOLD ACCURACY GRAPH
# ============================================

def plot_kfold_results(
    fold_scores
):

    folds = np.arange(
        1,
        len(fold_scores) + 1
    )

    plt.figure(figsize=(8,5))

    plt.bar(
        folds,
        fold_scores
    )

    plt.xlabel("Fold")

    plt.ylabel("Accuracy (%)")

    plt.title(
        "K-Fold Cross Validation Accuracy"
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "kfold_accuracy.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    print(
        "[INFO] K-Fold graph saved."
    )

# ============================================
# TRAINING TIME GRAPH
# ============================================

def plot_epoch_times(
    epoch_times
):

    epochs = np.arange(
        1,
        len(epoch_times) + 1
    )

    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        epoch_times,
        marker='o'
    )

    plt.xlabel("Epoch")

    plt.ylabel(
        "Time (seconds)"
    )

    plt.title(
        "Training Time Per Epoch"
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "epoch_time.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    print(
        "[INFO] Epoch time graph saved."
    )

# ============================================
# CLASS DISTRIBUTION GRAPH
# ============================================

def plot_class_distribution(
    class_counts
):

    labels = list(
        class_counts.keys()
    )

    counts = list(
        class_counts.values()
    )

    plt.figure(figsize=(8,5))

    plt.bar(
        labels,
        counts
    )

    plt.xlabel("Class")

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Dataset Class Distribution"
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "class_distribution.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    print(
        "[INFO] Class distribution graph saved."
    )

# ============================================
# CONFIDENCE SCORE HISTOGRAM
# ============================================

def plot_confidence_distribution(
    confidence_scores
):

    plt.figure(figsize=(8,5))

    plt.hist(
        confidence_scores,
        bins=20
    )

    plt.xlabel(
        "Confidence Score (%)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Prediction Confidence Distribution"
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "confidence_distribution.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    print(
        "[INFO] Confidence histogram saved."
    )