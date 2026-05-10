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

# ============================================
# LOAD MODEL
# ============================================

model = models.densenet121(
    pretrained=False
)

# ============================================
# CUSTOM CLASSIFIER
# ============================================

model.classifier = nn.Sequential(

    nn.Dropout(0.3),

    nn.Linear(1024, 1),

    nn.Sigmoid()
)

# ============================================
# LOAD TRAINED WEIGHTS
# ============================================

model.load_state_dict(
    torch.load(
        "best_pneumonia_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("✅ Model Loaded Successfully")

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
# LOAD TEST IMAGE
# ============================================

image_path = "test_image.jpeg"

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

probability = output.item()

prediction = (
    "PNEUMONIA"
    if probability > 0.5
    else "NORMAL"
)

# ============================================
# DISPLAY PREDICTION
# ============================================

print("\n========== RESULT ==========\n")

print(f"Prediction  : {prediction}")

print(f"Confidence  : {probability:.4f}")

# ============================================
# INITIALIZE GRAD-CAM
# ============================================

target_layer = model.features[-1]

gradcam = GradCAM(
    model,
    target_layer
)

# ============================================
# GENERATE CAM
# ============================================

cam = gradcam.generate_cam(
    input_tensor
)

# ============================================
# PREPARE IMAGE
# ============================================

image_np = np.array(
    original_image.resize((224, 224))
) / 255.0

# ============================================
# CREATE OVERLAY
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
# GRAD-CAM
# --------------------------------------------

plt.subplot(1, 2, 2)

plt.imshow(overlay)

plt.title(
    f"{prediction} ({probability:.2f})"
)

plt.axis("off")

plt.tight_layout()

plt.show()