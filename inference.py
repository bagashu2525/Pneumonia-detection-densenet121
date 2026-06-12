# ============================================
# inference.py
# ============================================

import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision.models import DenseNet121_Weights

from PIL import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt

from gradcam import (
    GradCAM,
    overlay_heatmap,
    save_gradcam
)

# ============================================
# PUBLICATION QUALITY GRAD-CAM REPORT
# ============================================

from matplotlib.cm import ScalarMappable

def create_gradcam_report(
    original_image,
    cam,
    overlay,
    prediction,
    confidence,
    true_label=None,
    save_path="gradcam_report.png"
):

    # ========================================
    # RESIZE CAM FOR VISUALIZATION
    # ========================================

    cam_display = cv2.resize(
        cam,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    # ========================================
    # FIGURE
    # ========================================

    fig = plt.figure(
        figsize=(16, 9),
        facecolor="white"
    )

    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[12, 1, 1]
    )

    # ========================================
    # ORIGINAL IMAGE
    # ========================================

    ax1 = fig.add_subplot(gs[0, 0])

    ax1.imshow(
        original_image,
        cmap="gray"
    )

    ax1.set_title(
        "Original Chest X-ray",
        fontsize=14,
        fontweight="bold"
    )

    ax1.axis("off")

    # ========================================
    # HEATMAP
    # ========================================

    ax2 = fig.add_subplot(gs[0, 1])

    heat = ax2.imshow(
        cam_display,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    ax2.set_title(
        "Grad-CAM Heatmap",
        fontsize=14,
        fontweight="bold"
    )

    ax2.axis("off")

    # ========================================
    # OVERLAY
    # ========================================

    ax3 = fig.add_subplot(gs[0, 2])

    ax3.imshow(
        overlay
    )

    ax3.set_title(
        "Grad-CAM Overlay",
        fontsize=14,
        fontweight="bold"
    )

    ax3.axis("off")

    # ========================================
    # PREDICTION PANEL
    # ========================================

    ax4 = fig.add_subplot(gs[0, 3])

    ax4.axis("off")

    pred_color = (
        "green"
        if prediction == "NORMAL"
        else "red"
    )

    info = (
        f"Predicted: {prediction}\n\n"
        f"Confidence: {confidence*100:.2f}%"
    )

    if true_label is not None:

        info += (
            f"\n\nTrue Label: {true_label}"
        )

    ax4.text(
        0.05,
        0.80,
        info,
        fontsize=14,
        color=pred_color,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="black",
            alpha=0.95
        )
    )

    # ========================================
    # COLORBAR
    # ========================================

    cax = fig.add_subplot(gs[1, :])

    sm = ScalarMappable(
        cmap="jet"
    )

    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        cax=cax,
        orientation="horizontal"
    )

    cbar.set_label(
        "Grad-CAM Activation Strength",
        fontsize=12,
        fontweight="bold"
    )

    # ========================================
    # EXPLANATION PANEL
    # ========================================

    ax_exp = fig.add_subplot(gs[2, :])

    ax_exp.axis("off")

    ax_exp.text(
        0.5,
        0.70,
        "Grad-CAM highlights image regions that most influenced the model prediction.",
        ha="center",
        fontsize=10
    )

    ax_exp.text(
        0.5,
        0.25,
        "Blue = Low Importance   |   Green = Moderate Importance   |   Yellow = High Importance   |   Red = Very High Importance",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

    # ========================================
    # SAVE
    # ========================================

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"[OK] Saved Report: {save_path}"
    )

    plt.show()

    plt.close()
# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = (
    "best_pneumonia_model (2).pth"
)

IMAGE_PATH = (
    "images\person100_bacteria_480.jpeg"
)

OUTPUT_DIR = (
    "inference_outputs"
)

THRESHOLD = 0.50

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================
# DEVICE
# ============================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

print("\nUsing Device:", device)

# ============================================
# LOAD MODEL
# ============================================

model = models.densenet121(

    weights=None
)

# ============================================
# UPDATED CLASSIFIER
# MUST MATCH PHASE 2
# ============================================

model.classifier = nn.Sequential(

    nn.Linear(
        1024,
        512
    ),

    nn.BatchNorm1d(
        512
    ),

    nn.ReLU(
        inplace=True
    ),

    nn.Dropout(
        0.5
    ),

    nn.Linear(
        512,
        1
    )
)

# ============================================
# LOAD TRAINED WEIGHTS
# ============================================

model.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=device
    )
)
# Disable inplace ReLU
for module in model.modules():

    if isinstance(module, nn.ReLU):

        module.inplace = False
model = model.to(device)

model.eval()

print(
    "[OK] Model Loaded Successfully"
)

# ============================================
# TRANSFORM
# ============================================

transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

# ============================================
# LOAD IMAGE
# ============================================

original_image = Image.open(
    IMAGE_PATH
).convert("RGB")

# ============================================
# PREPROCESS
# ============================================

input_tensor = transform(
    original_image
)

input_tensor = input_tensor.unsqueeze(
    0
)

input_tensor = input_tensor.to(
    device
)

# ============================================
# INFERENCE TIMER
# ============================================

start_time = time.time()

with torch.no_grad():

    output = model(
        input_tensor
    )

    probability = torch.sigmoid(
        output
    ).item()

end_time = time.time()

inference_time = (
    end_time - start_time
)

# ============================================
# PREDICTION
# ============================================

prediction = (

    "PNEUMONIA"

    if probability > THRESHOLD

    else "NORMAL"
)

# ============================================
# PRINT RESULTS
# ============================================

print("\n===================================")
print("PREDICTION RESULT")
print("===================================")

print(
    f"Prediction : {prediction}"
)

print(
    f"Confidence : "
    f"{probability*100:.2f}%"
)

print(
    f"Inference Time : "
    f"{inference_time*1000:.2f} ms"
)

# ============================================
# GRAD-CAM
# ============================================

target_layer = (
    model.features.denseblock4.denselayer16.conv2
)


gradcam = GradCAM(

    model,

    target_layer
)

cam, _, _ = gradcam.generate_cam(
    input_tensor
)

# ============================================
# IMAGE FOR DISPLAY
# ============================================

image_np = np.array(

    original_image.resize(
        (224,224)
    )

).astype(
    np.float32
) / 255.0

# ============================================
# OVERLAY
# ============================================

overlay = overlay_heatmap(
    cam,
    image_np,
    alpha=0.30
)

# ============================================
# SAVE HEATMAP
# ============================================

gradcam_path = (

    f"{OUTPUT_DIR}/"

    f"gradcam_result.png"
)

save_gradcam(

    overlay,

    gradcam_path
)

print(
    f"[OK] GradCAM Saved: "
    f"{gradcam_path}"
)

# ============================================
# GENERATE REPORT
# ============================================

create_gradcam_report(

    original_image=np.array(
        original_image.resize(
            (224, 224)
        )
    ),

    cam=cam,

    overlay=overlay,

    prediction=prediction,

    confidence=probability,

    true_label=None,   # optional

    save_path="gradcam_report.png"
)


# ============================================
# SAVE REPORT
# ============================================

report_path = (

    f"{OUTPUT_DIR}/"

    f"inference_report.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(
        "PNEUMONIA DETECTION REPORT\n\n"
    )

    f.write(
        f"Prediction: {prediction}\n"
    )

    f.write(
        f"Confidence: "
        f"{probability*100:.2f}%\n"
    )

    f.write(
        f"Inference Time: "
        f"{inference_time*1000:.2f} ms\n"
    )

    f.write(
        f"Threshold Used: "
        f"{THRESHOLD}\n"
    )

print(
    f"[OK] Report Saved: "
    f"{report_path}"
)

# ============================================
# CLEANUP
# ============================================

gradcam.remove_hooks()

print("\n===================================")
print("INFERENCE COMPLETED")
print("===================================")