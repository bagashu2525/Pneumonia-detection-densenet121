# ============================================
# PHASE 4 : EVALUATION
# ============================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(
    model,
    test_loader,
    device
):

    model.eval()

    y_true = []
    y_pred = []

    confidence_scores = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            # ====================================
            # SOFTMAX PROBABILITIES
            # ====================================

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            preds = torch.argmax(
                probabilities,
                dim=1
            )

            confidences = torch.max(
                probabilities,
                dim=1
            )[0]

            confidence_scores.extend(
                confidences.cpu().numpy()
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    # ========================================
    # NUMPY CONVERSION
    # ========================================

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ========================================
    # METRICS
    # ========================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # ========================================
    # CONFUSION MATRIX
    # ========================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # ========================================
    # PRINT RESULTS
    # ========================================

    print("\n========== TEST RESULTS ==========\n")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:\n")
    print(cm)

    # ========================================
    # CLASSIFICATION REPORT
    # ========================================

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Normal",
            "Pneumonia",
            "Tuberculosis"
        ]
    )

    print("\n========== CLASSIFICATION REPORT ==========\n")

    print(report)

    # Save report

    with open(
        "/content/classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    # ========================================
    # CONFUSION MATRIX GRAPH
    # ========================================

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Normal",
            "Pneumonia",
            "TB"
        ],
        yticklabels=[
            "Normal",
            "Pneumonia",
            "TB"
        ]
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "/content/confusion_matrix.png"
    )

    plt.close()

    # ========================================
    # CONFIDENCE SCORE STATISTICS
    # ========================================

    avg_confidence = (
        np.mean(confidence_scores) * 100
    )

    max_confidence = (
        np.max(confidence_scores) * 100
    )

    min_confidence = (
        np.min(confidence_scores) * 100
    )

    print("\n========== CONFIDENCE SCORES ==========\n")

    print(
        f"Average Confidence : "
        f"{avg_confidence:.2f}%"
    )

    print(
        f"Maximum Confidence : "
        f"{max_confidence:.2f}%"
    )

    print(
        f"Minimum Confidence : "
        f"{min_confidence:.2f}%"
    )

    # ========================================
    # CONFIDENCE HISTOGRAM
    # ========================================

    plt.figure(figsize=(8,5))

    plt.hist(
        np.array(confidence_scores) * 100,
        bins=20
    )

    plt.xlabel("Confidence (%)")

    plt.ylabel("Number of Predictions")

    plt.title(
        "Prediction Confidence Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        "/content/confidence_distribution.png"
    )

    plt.close()

    print("\n[OK] Saved Files:")

    print("   confusion_matrix.png")

    print("   confidence_distribution.png")

    print("   classification_report.txt")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_confidence": avg_confidence
    }