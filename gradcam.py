# ============================================
# gradcam.py
# ============================================

import torch
import cv2
import numpy as np


# ============================================
# GRAD-CAM CLASS
# ============================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(
            self.forward_hook
        )

        self.target_layer.register_full_backward_hook(
            self.backward_hook
        )

    # ========================================
    # FORWARD HOOK
    # ========================================

    def forward_hook(
        self,
        module,
        input,
        output
    ):
        self.activations = output

    # ========================================
    # BACKWARD HOOK
    # ========================================

    def backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):
        self.gradients = grad_output[0]

    # ========================================
    # GENERATE CAM
    # ========================================

    def generate_cam(
        self,
        input_tensor
    ):

        output = self.model(input_tensor)

        self.model.zero_grad()

        output.backward(
            torch.ones_like(output)
        )

        gradients = self.gradients[0]

        activations = self.activations[0]

        weights = torch.mean(
            gradients,
            dim=(1, 2)
        )

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        ).to(input_tensor.device)

        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()

        # Normalize
        cam = np.maximum(cam, 0)

        cam -= np.min(cam)

        if np.max(cam) > 0:
            cam /= np.max(cam)

        return cam


# ============================================
# CREATE HEATMAP
# ============================================

def create_heatmap(
    cam,
    output_size=(224, 224)
):

    cam = cv2.resize(
        cam,
        output_size,
        interpolation=cv2.INTER_CUBIC
    )

    # Blue → Green → Yellow → Red
    heatmap = cv2.applyColorMap(
        np.uint8(cam * 255),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    heatmap = (
        heatmap.astype(np.float32)
        / 255.0
    )

    return heatmap


# ============================================
# OVERLAY HEATMAP
# ============================================

def overlay_heatmap(
    cam,
    image,
    alpha=0.45
):

    heatmap = create_heatmap(
        cam,
        (
            image.shape[1],
            image.shape[0]
        )
    )

    image = image.astype(
        np.float32
    )

    if image.max() > 1:
        image /= 255.0

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    return overlay


# ============================================
# CREATE HEATMAP + OVERLAY
# ============================================

def create_heatmap_and_overlay(
    cam,
    image,
    alpha=0.45
):

    heatmap = create_heatmap(
        cam,
        (
            image.shape[1],
            image.shape[0]
        )
    )

    image = image.astype(
        np.float32
    )

    if image.max() > 1:
        image /= 255.0

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    return heatmap, overlay


# ============================================
# SAVE IMAGE
# ============================================

def save_image(
    image,
    save_path
):

    image = np.uint8(
        image * 255
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        save_path,
        image
    )

    print(
        f"[OK] Saved: {save_path}"
    )


# ============================================
# SAVE HEATMAP
# ============================================

def save_heatmap(
    heatmap,
    save_path
):

    save_image(
        heatmap,
        save_path
    )


# ============================================
# SAVE OVERLAY
# ============================================

def save_gradcam(
    overlay,
    save_path
):

    save_image(
        overlay,
        save_path
    )