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
    confusion_matrix
)

def evaluate_model(
    model,
    test_loader,
    device
):

    model.eval()

    predictions = []
    labels_list = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            preds = (outputs > 0.5).float()

            predictions.extend(
                preds.cpu().numpy()
            )

            labels_list.extend(
                labels.numpy()
            )

    predictions = np.array(predictions).flatten()

    labels_list = np.array(labels_list).flatten()

    # ========================================
    # METRICS
    # ========================================

    accuracy = accuracy_score(
        labels_list,
        predictions
    )

    precision = precision_score(
        labels_list,
        predictions
    )

    recall = recall_score(
        labels_list,
        predictions
    )

    f1 = f1_score(
        labels_list,
        predictions
    )

    cm = confusion_matrix(
        labels_list,
        predictions
    )

    print("\n========== TEST RESULTS ==========\n")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:\n")

    print(cm)