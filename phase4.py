# ============================================
# PHASE 4 : EVALUATION
# phase4.py
# ============================================

import time
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
    roc_curve,
    roc_auc_score
)

# ============================================
# EVALUATION FUNCTION
# ============================================

def evaluate_model(
    model,
    test_loader,
    device
):

    model.eval()

    probabilities = []
    labels_list = []

    # ========================================
    # INFERENCE TIME
    # ========================================

    start_time = time.time()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.sigmoid(outputs)

            probabilities.extend(
                probs.cpu().numpy()
            )

            labels_list.extend(
                labels.numpy()
            )

    end_time = time.time()

    total_inference_time = (
        end_time - start_time
    )

    # ========================================
    # CONVERT TO NUMPY
    # ========================================

    probabilities = np.array(
        probabilities
    ).flatten()

    labels_list = np.array(
        labels_list
    ).flatten()

    # ========================================
    # DEFAULT THRESHOLD
    # ========================================

    preds_50 = (
        probabilities > 0.5
    ).astype(int)

    accuracy_50 = accuracy_score(
        labels_list,
        preds_50
    )

    precision_50 = precision_score(
        labels_list,
        preds_50,
        zero_division=0
    )

    recall_50 = recall_score(
        labels_list,
        preds_50,
        zero_division=0
    )

    f1_50 = f1_score(
        labels_list,
        preds_50,
        zero_division=0
    )

    cm_50 = confusion_matrix(
        labels_list,
        preds_50
    )

    # ========================================
    # ROC + AUC
    # ========================================

    fpr, tpr, thresholds = roc_curve(
        labels_list,
        probabilities
    )

    auc_score = roc_auc_score(
        labels_list,
        probabilities
    )

    # ========================================
    # OPTIMAL THRESHOLD
    # ========================================

    j_scores = tpr - fpr

    best_idx = np.argmax(
        j_scores
    )

    optimal_threshold = float(
        np.clip(
            thresholds[best_idx],
            0.1,
            0.9
        )
    )

    preds_opt = (
        probabilities > optimal_threshold
    ).astype(int)

    accuracy_opt = accuracy_score(
        labels_list,
        preds_opt
    )

    precision_opt = precision_score(
        labels_list,
        preds_opt,
        zero_division=0
    )

    recall_opt = recall_score(
        labels_list,
        preds_opt,
        zero_division=0
    )

    f1_opt = f1_score(
        labels_list,
        preds_opt,
        zero_division=0
    )

    cm_opt = confusion_matrix(
        labels_list,
        preds_opt
    )

    # ========================================
    # CONFIDENCE SCORE STATISTICS
    # ========================================

    avg_confidence = (
        np.mean(probabilities)
        * 100
    )

    max_confidence = (
        np.max(probabilities)
        * 100
    )

    min_confidence = (
        np.min(probabilities)
        * 100
    )

    # ========================================
    # PRINT RESULTS
    # ========================================

    print("\n====================================")
    print("THRESHOLD = 0.50")
    print("====================================")

    print(f"Accuracy  : {accuracy_50:.4f}")
    print(f"Precision : {precision_50:.4f}")
    print(f"Recall    : {recall_50:.4f}")
    print(f"F1 Score  : {f1_50:.4f}")
    print(f"AUC Score : {auc_score:.4f}")

    print("\nConfusion Matrix:\n")
    print(cm_50)

    print("\nClassification Report:\n")

    print(
        classification_report(
            labels_list,
            preds_50,
            target_names=[
                "Normal",
                "Pneumonia"
            ]
        )
    )

    print("\n====================================")
    print("OPTIMAL THRESHOLD")
    print("====================================")

    print(
        f"Optimal Threshold : "
        f"{optimal_threshold:.4f}"
    )

    print(
        f"Accuracy          : "
        f"{accuracy_opt:.4f}"
    )

    print(
        f"Precision         : "
        f"{precision_opt:.4f}"
    )

    print(
        f"Recall            : "
        f"{recall_opt:.4f}"
    )

    print(
        f"F1 Score          : "
        f"{f1_opt:.4f}"
    )

    # ========================================
    # CONFIDENCE STATS
    # ========================================

    print("\n====================================")
    print("CONFIDENCE SCORES")
    print("====================================")

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
    # TIME COMPLEXITY
    # ========================================

    avg_time_per_image = (

        total_inference_time

        / len(labels_list)

    )

    print("\n====================================")
    print("INFERENCE TIME")
    print("====================================")

    print(
        f"Total Test Time : "
        f"{total_inference_time:.4f} sec"
    )

    print(
        f"Time Per Image  : "
        f"{avg_time_per_image*1000:.4f} ms"
    )

    # ========================================
    # CONFUSION MATRIX HEATMAP
    # ========================================

    plt.figure(figsize=(6,5))

    sns.heatmap(

        cm_opt,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=[
            "Normal",
            "Pneumonia"
        ],

        yticklabels=[
            "Normal",
            "Pneumonia"
        ]
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(
        "project_outputs/confusion_matrix.png",
        dpi=300
    )



    plt.show()

    # ========================================
    # ROC CURVE
    # ========================================

    plt.figure(figsize=(6,5))

    plt.plot(

        fpr,

        tpr,

        label=f"AUC = {auc_score:.4f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "project_outputs/roc_curve.png",
        dpi=300
    )

    plt.show()

    # ========================================
    # RETURN RESULTS
    # ========================================

    results = {

        "accuracy": accuracy_opt,

        "precision": precision_opt,

        "recall": recall_opt,

        "f1_score": f1_opt,

        "auc_score": auc_score,

        "optimal_threshold":
            optimal_threshold,

        "confusion_matrix":
            cm_opt,

        "avg_confidence":
            avg_confidence,

        "inference_time":
            total_inference_time
    }

    return results