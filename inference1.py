# ============================================
# inference.py
# ============================================

import io

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

from gradcam import GradCAM, overlay_heatmap

# ============================================
# DEVICE
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# MODEL & TRANSFORM (shared by script + web app)
# ============================================

model = models.densenet121(pretrained=False)

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1024, 1),
    nn.Sigmoid(),
)

model.load_state_dict(
    torch.load("best_pneumonia_model (2).pth", map_location=device)
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

PREDICTION_THRESHOLD = 0.5


def predict_from_image(image):
    """Run the same pipeline as this script's CLI flow on a PIL image."""

    original_image = image.convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        probability = model(input_tensor).item()

    prediction = (
        "PNEUMONIA" if probability > PREDICTION_THRESHOLD else "NORMAL"
    )
    confidence = (
        probability if prediction == "PNEUMONIA" else 1.0 - probability
    )

    cam = gradcam.generate_cam(input_tensor)
    image_np = np.array(original_image.resize((224, 224))) / 255.0
    overlay = overlay_heatmap(cam, image_np)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "pneumonia_probability": round(probability, 4),
        "overlay": overlay,
        "preview": image_np,
    }


def predict_from_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return predict_from_image(image)


def run_cli(image_path="images\IM-0003-0001.jpeg"):
    print("[OK] Model Loaded Successfully")

    original_image = Image.open(image_path).convert("RGB")
    result = predict_from_image(original_image)

    print("\n========== RESULT ==========\n")
    print(f"Prediction  : {result['prediction']}")
    print(f"Confidence  : {result['confidence']:.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(result["preview"])
    plt.title("Original X-ray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(result["overlay"])
    plt.title(
        f"{result['prediction']} ({result['pneumonia_probability']:.2f})"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_cli()