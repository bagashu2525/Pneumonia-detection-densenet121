# ============================================
# inference.py
# ============================================

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

from gradcam import (
    GradCAM,
    overlay_heatmap
)

# ============================================
# DEVICE
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing Device:", device)

# ============================================
# LOAD MODEL
# ============================================

model = models.densenet121(
    weights=None
)

# ============================================
# CLASSIFIER
# IMPORTANT:
# NO SIGMOID HERE
# ============================================

model.classifier = nn.Sequential(

    nn.Dropout(0.3),

    nn.Linear(1024, 1)
)

# ============================================
# LOAD WEIGHTS
# ============================================

model.load_state_dict(
    torch.load(
        "best_pneumonia_model (1).pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("[OK] Model Loaded Successfully")

# ============================================
# IMAGE TRANSFORM
# ============================================

transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================
# LOAD IMAGE
# ============================================

image_path = "images\\person101_bacteria_484.jpeg"

original_image = Image.open(
    image_path
).convert("RGB")

# ============================================
# PREPROCESS IMAGE
# ============================================

input_tensor = transform(
    original_image
)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)

# ============================================
# PREDICTION
# ============================================

with torch.no_grad():

    output = model(input_tensor)

    probability = torch.sigmoid(
        output
    ).item()

# ============================================
# THRESHOLD
# ============================================

# Tuned to 0.70 to balance precision and reduce False Positives on normal adult scans
threshold = 0.70

prediction = (
    "PNEUMONIA"
    if probability > threshold
    else "NORMAL"
)

# ============================================
# PRINT RESULTS
# ============================================

print("\n========== RESULT ==========\n")

print(f"Prediction : {prediction}")

print(f"Confidence : {probability:.4f}")

# ============================================
# GRAD-CAM
# ============================================

target_layer = model.features.denseblock4

gradcam = GradCAM(
    model,
    target_layer
)

cam = gradcam.generate_cam(
    input_tensor
)

# ============================================
# IMAGE FOR DISPLAY
# ============================================

image_np = np.array(

    original_image.resize((224, 224))

) / 255.0

# ============================================
# OVERLAY
# ============================================

overlay = overlay_heatmap(
    cam,
    image_np
)

# ============================================
# VISUALIZATION
# ============================================

plt.figure(figsize=(12, 5))

# --------------------------------------------
# ORIGINAL IMAGE
# --------------------------------------------

plt.subplot(1, 2, 1)

plt.imshow(image_np)

plt.title("Original X-ray")

plt.axis("off")

# --------------------------------------------
# GRAD-CAM RESULT
# --------------------------------------------

plt.subplot(1, 2, 2)

plt.imshow(overlay)

plt.title(
    f"{prediction} ({probability:.2f})"
)

plt.axis("off")

plt.tight_layout()

# Save the visualization to a file to prevent GUI blocking in non-interactive/headless environments
output_path = "gradcam_result.png"
plt.savefig(output_path, dpi=300)
print(f"[OK] Grad-CAM Visualization saved to: {output_path}")