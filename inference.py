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

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ============================================
# LOAD MODEL
# ============================================

model = models.densenet121(weights=None)

# ============================================
# SAME ARCHITECTURE AS PHASE 2
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
        inplace=False
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
# LOAD TRAINED MODEL
# ============================================

model.load_state_dict(

    torch.load(
        "best_pneumonia_model (2).pth",
        map_location=device
    )
)

# ============================================
# DISABLE INPLACE RELU
# FOR GRADCAM
# ============================================

for module in model.modules():

    if isinstance(
        module,
        nn.ReLU
    ):

        module.inplace = False

model = model.to(device)

model.eval()

# ============================================
# IMAGE TRANSFORM
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
# GRADCAM TARGET LAYER
# ============================================

target_layer = (
    model.features.denseblock4.denselayer16.conv2
)

gradcam = GradCAM(
    model,
    target_layer
)

# ============================================
# THRESHOLD
# ============================================

PREDICTION_THRESHOLD = 0.50

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_from_image(image):

    original_image = image.convert(
        "RGB"
    )

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(device)

    # ========================================
    # INFERENCE
    # ========================================

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probability = torch.sigmoid(
            logits
        ).item()

    # ========================================
    # CLASS PREDICTION
    # ========================================

    prediction = (

        "PNEUMONIA"

        if probability > PREDICTION_THRESHOLD

        else "NORMAL"
    )

    confidence = (

        probability

        if prediction == "PNEUMONIA"

        else (1.0 - probability)
    )

    # ========================================
    # GRADCAM
    # ========================================

    cam = gradcam.generate_cam(
        input_tensor
    )

    image_np = np.array(

        original_image.resize(
            (224, 224)
        )

    ) / 255.0

    overlay = overlay_heatmap(

        cam,

        image_np
    )

    # ========================================
    # RETURN RESULTS
    # ========================================

    return {

        "prediction":
        prediction,

        "confidence":
        round(
            confidence,
            4
        ),

        "pneumonia_probability":
        round(
            probability,
            4
        ),

        "overlay":
        overlay,

        "preview":
        image_np,

        "cam":
        cam
    }

# ============================================
# BYTE INPUT SUPPORT
# ============================================

def predict_from_bytes(
    image_bytes
):

    image = Image.open(

        io.BytesIO(
            image_bytes
        )
    )

    return predict_from_image(
        image
    )

# ============================================
# CLI TEST
# ============================================

def run_cli(

    image_path=
    "00000002_000.png"
):

    print(
        "[OK] Model Loaded Successfully"
    )

    original_image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    result = predict_from_image(
        original_image
    )

    print(
        "\n========== RESULT ==========\n"
    )

    print(
        f"Prediction : "
        f"{result['prediction']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence']:.4f}"
    )

    print(
        f"Pneumonia Probability : "
        f"{result['pneumonia_probability']:.4f}"
    )

    # ========================================
    # VISUALIZATION
    # ========================================

    plt.figure(
        figsize=(15, 5)
    )

    # ORIGINAL

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        result["preview"]
    )

    plt.title(
        "Original X-ray"
    )

    plt.axis(
        "off"
    )

    # HEATMAP

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        result["cam"],
        cmap="jet"
    )

    plt.title(
        "Grad-CAM Heatmap"
    )

    plt.axis(
        "off"
    )

    # OVERLAY

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        result["overlay"]
    )

    plt.title(

        f"{result['prediction']} "

        f"({result['pneumonia_probability']:.2f})"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.show()

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    run_cli()