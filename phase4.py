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
    classification_report,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize


def evaluate_model(
    model,
    test_loader,
    device
):

    model.eval()

    y_true = []
    y_pred = []

    confidence_scores = []
    all_probabilities = []

    # ========================================
    # INFERENCE
    # ========================================

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

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

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    # ========================================
    # CONVERT TO NUMPY
    # ========================================

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    all_probabilities = np.array(
        all_probabilities
    )

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
    # MULTICLASS ROC-AUC
    # ========================================

    try:

        y_true_bin = label_binarize(
            y_true,
            classes=[0, 1, 2]
        )

        roc_auc = roc_auc_score(
            y_true_bin,
            all_probabilities,
            multi_class="ovr"
        )

    except:

        roc_auc = 0.0

    # ========================================
    # CONFUSION MATRIX
    # ========================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

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

    # ========================================
    # SAVE REPORT
    # ========================================

    with open(
        "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # ========================================
    # CONFUSION MATRIX PLOT
    # ========================================

    plt.figure(figsize=(8, 6))

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
        "confusion_matrix.png"
    )

    plt.close()

    # ========================================
    # CONFIDENCE STATISTICS
    # ========================================

    confidence_scores = np.array(
        confidence_scores
    )

    avg_confidence = (
        np.mean(confidence_scores) * 100
    )

    std_confidence = (
        np.std(confidence_scores) * 100
    )

    max_confidence = (
        np.max(confidence_scores) * 100
    )

    min_confidence = (
        np.min(confidence_scores) * 100
    )

    # ========================================
    # CONFIDENCE HISTOGRAM
    # ========================================

    plt.figure(figsize=(8, 5))

    plt.hist(
        confidence_scores * 100,
        bins=20
    )

    plt.xlabel(
        "Confidence (%)"
    )

    plt.ylabel(
        "Number of Predictions"
    )

    plt.title(
        "Prediction Confidence Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        "confidence_distribution.png"
    )

    plt.close()

    # ========================================
    # PRINT RESULTS
    # ========================================

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print("\n" + "=" * 60)
    print("CONFIDENCE ANALYSIS")
    print("=" * 60)

    print(
        f"Average Confidence : "
        f"{avg_confidence:.2f}%"
    )

    print(
        f"Std Confidence     : "
        f"{std_confidence:.2f}%"
    )

    print(
        f"Maximum Confidence : "
        f"{max_confidence:.2f}%"
    )

    print(
        f"Minimum Confidence : "
        f"{min_confidence:.2f}%"
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(report)

    print("\nGenerated Files:")

    print(
        "✓ confusion_matrix.png"
    )

    print(
        "✓ confidence_distribution.png"
    )

    print(
        "✓ classification_report.txt"
    )

    # ========================================
    # RETURN RESULTS
    # ========================================

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc,

        "avg_confidence":
            avg_confidence,

        "std_confidence":
            std_confidence,

        "confusion_matrix":
            cm,

        "classification_report":
            report
    }