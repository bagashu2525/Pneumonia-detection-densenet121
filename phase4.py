# ============================================
# PHASE 4 : EVALUATION
# ============================================

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)

def evaluate_model(
    model,
    test_loader,
    device
):

    model.eval()

    probabilities = []
    labels_list = []

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

    probabilities = np.array(probabilities).flatten()
    labels_list = np.array(labels_list).flatten()

    # ========================================
    # STANDARD METRICS (THRESHOLD = 0.5)
    # ========================================

    preds_50 = (probabilities > 0.5).astype(float)

    accuracy_50 = accuracy_score(labels_list, preds_50)
    precision_50 = precision_score(labels_list, preds_50, zero_division=0)
    recall_50 = recall_score(labels_list, preds_50, zero_division=0)
    f1_50 = f1_score(labels_list, preds_50, zero_division=0)
    cm_50 = confusion_matrix(labels_list, preds_50)

    # ========================================
    # OPTIMAL THRESHOLD METRICS (ROC YOUDEN'S J)
    # ========================================

    fpr, tpr, thresholds = roc_curve(labels_list, probabilities)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = float(np.clip(thresholds[best_idx], 0.1, 0.9))

    preds_opt = (probabilities > optimal_threshold).astype(float)

    accuracy_opt = accuracy_score(labels_list, preds_opt)
    precision_opt = precision_score(labels_list, preds_opt, zero_division=0)
    recall_opt = recall_score(labels_list, preds_opt, zero_division=0)
    f1_opt = f1_score(labels_list, preds_opt, zero_division=0)
    cm_opt = confusion_matrix(labels_list, preds_opt)

    print("\n========== TEST RESULTS (THRESHOLD = 0.50) ==========\n")
    print(f"Accuracy  : {accuracy_50:.4f}")
    print(f"Precision : {precision_50:.4f}")
    print(f"Recall    : {recall_50:.4f}")
    print(f"F1 Score  : {f1_50:.4f}")
    print("\nConfusion Matrix (0.50):\n", cm_50)

    print(f"\n========== TEST RESULTS (OPTIMIZED THRESHOLD = {optimal_threshold:.4f}) ==========\n")
    print(f"Accuracy  : {accuracy_opt:.4f}")
    print(f"Precision : {precision_opt:.4f}")
    print(f"Recall    : {recall_opt:.4f}")
    print(f"F1 Score  : {f1_opt:.4f}")
    print("\nConfusion Matrix (Optimized):\n", cm_opt)