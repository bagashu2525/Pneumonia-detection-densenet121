# ============================================
# MAIN TRAINING PIPELINE
# train.py
# ============================================

# ============================================
# IMPORTS
# ============================================

import os
import shutil
import torch
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files

# ============================================
# OUTPUT DIRECTORY
# ============================================

OUTPUT_DIR = "project_outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================
# PHASE 1
# ============================================

from phase1 import (
    train_loader,
    val_loader,
    test_loader
)

# ============================================
# PHASE 2
# ============================================

from phase2 import (
    model,
    criterion,
    optimizer,
    scheduler,
    device
)

# ============================================
# PHASE 3
# ============================================

from phase3 import train_model

# ============================================
# PHASE 4
# ============================================

from phase4 import evaluate_model

# ============================================
# TRAIN MODEL
# ============================================

print("\n===================================")
print("STARTING TRAINING")
print("===================================")

history = train_model(

    model=model,

    train_loader=train_loader,

    val_loader=val_loader,

    criterion=criterion,

    optimizer=optimizer,

    scheduler=scheduler,

    device=device,

    epochs=10,

    patience=5,

    model_path=f"{OUTPUT_DIR}/best_pneumonia_model.pth"
)

# ============================================
# LOSS CURVE
# ============================================

plt.figure(figsize=(8,5))

plt.plot(
    history["train_losses"],
    label="Training Loss"
)

plt.plot(
    history["val_losses"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/loss_curve.png",
    dpi=300
)

plt.close()

# ============================================
# ACCURACY CURVE
# ============================================

plt.figure(figsize=(8,5))

plt.plot(
    history["train_accuracies"],
    label="Training Accuracy"
)

plt.plot(
    history["val_accuracies"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/accuracy_curve.png",
    dpi=300
)

plt.close()

# ============================================
# SAVE TRAINING HISTORY CSV
# ============================================

history_df = pd.DataFrame({

    "Train Loss":
        history["train_losses"],

    "Validation Loss":
        history["val_losses"],

    "Train Accuracy":
        history["train_accuracies"],

    "Validation Accuracy":
        history["val_accuracies"]
})

history_df.to_csv(

    f"{OUTPUT_DIR}/training_history.csv",

    index=False
)

print(
    "\n[OK] Training History CSV Saved"
)

# ============================================
# OVERFITTING ANALYSIS
# ============================================

final_train_acc = history[
    "train_accuracies"
][-1]

final_val_acc = history[
    "val_accuracies"
][-1]

generalization_gap = (

    final_train_acc

    - final_val_acc

)

print("\n===================================")
print("OVERFITTING ANALYSIS")
print("===================================")

print(
    f"Training Accuracy : "
    f"{final_train_acc:.4f}"
)

print(
    f"Validation Accuracy : "
    f"{final_val_acc:.4f}"
)

print(
    f"Generalization Gap : "
    f"{generalization_gap:.4f}"
)

if generalization_gap > 0.10:

    print(
        "[WARNING] Possible Overfitting"
    )

else:

    print(
        "[OK] Good Generalization"
    )

# ============================================
# LOAD BEST MODEL
# ============================================

print("\nLoading Best Model...")

model.load_state_dict(

    torch.load(

        f"{OUTPUT_DIR}/best_pneumonia_model.pth",

        map_location=device
    )
)

print(
    "[OK] Best Model Loaded"
)

# ============================================
# EVALUATE MODEL
# ============================================

results = evaluate_model(

    model=model,

    test_loader=test_loader,

    device=device
)

# ============================================
# SAVE RESULTS TXT
# ============================================

with open(

    f"{OUTPUT_DIR}/results.txt",

    "w"

) as file:

    file.write(
        "PNEUMONIA DETECTION RESULTS\n"
    )

    file.write(
        "===========================\n\n"
    )

    for key, value in results.items():

        file.write(
            f"{key}: {value}\n"
        )

    file.write(
        f"\nTraining Time: "
        f"{history['training_time']:.2f} sec\n"
    )

    file.write(
        f"Generalization Gap: "
        f"{generalization_gap:.4f}\n"
    )

print(
    "[OK] Results Saved"
)

# ============================================
# FINAL SUMMARY
# ============================================

print("\n===================================")
print("FINAL MODEL PERFORMANCE")
print("===================================")

print(
    f"Accuracy           : "
    f"{results['accuracy']:.4f}"
)

print(
    f"Precision          : "
    f"{results['precision']:.4f}"
)

print(
    f"Recall             : "
    f"{results['recall']:.4f}"
)

print(
    f"F1 Score           : "
    f"{results['f1_score']:.4f}"
)

print(
    f"AUC Score          : "
    f"{results['auc_score']:.4f}"
)

print(
    f"Optimal Threshold  : "
    f"{results['optimal_threshold']:.4f}"
)

print(
    f"Average Confidence : "
    f"{results['avg_confidence']:.2f}%"
)

print(
    f"Inference Time     : "
    f"{results['inference_time']:.4f} sec"
)

print(
    f"Training Time      : "
    f"{history['training_time']:.2f} sec"
)

# ============================================
# GENERATED FILES
# ============================================

print("\n===================================")
print("GENERATED FILES")
print("===================================")

generated_files = [

    "best_pneumonia_model.pth",

    "loss_curve.png",

    "accuracy_curve.png",

    "confusion_matrix.png",

    "roc_curve.png",

    "training_history.csv",

    "results.txt"
]

for file in generated_files:

    print(file)

# ============================================
# CREATE ZIP
# ============================================

print("\nCreating ZIP File...")

shutil.make_archive(

    "Pneumonia_Project_Output",

    "zip",

    OUTPUT_DIR
)

print(
    "[OK] ZIP File Created"
)

# ============================================
# DOWNLOAD ZIP
# ============================================

files.download(
    "Pneumonia_Project_Output.zip"
)

print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("===================================")