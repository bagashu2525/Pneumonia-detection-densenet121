# ============================================
# gradcam.py
# ============================================

import torch
import cv2
import numpy as np

# ============================================
# GRAD-CAM
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

        # ------------------------------------
        # HOOKS
        # ------------------------------------

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self.forward_hook
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self.backward_hook
            )
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

        self.activations = (
            output.clone().detach()
        )

    # ========================================
    # BACKWARD HOOK
    # ========================================

    def backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = (
            grad_output[0].clone().detach()
        )

    # ========================================
    # GENERATE CAM
    # ========================================

    def generate_cam(
        self,
        input_tensor
    ):

        self.model.eval()

        self.gradients = None
        self.activations = None

        # ------------------------------------
        # FORWARD
        # ------------------------------------

        output = self.model(
            input_tensor
        )

        confidence = torch.sigmoid(
            output
        ).item()

        prediction = int(
            confidence > 0.5
        )

        # ------------------------------------
        # BACKWARD
        # ------------------------------------

        self.model.zero_grad()

        score = output[:, 0]

        score.backward(
            torch.ones_like(score)
        )

        # ------------------------------------
        # CHECK HOOKS
        # ------------------------------------

        if self.gradients is None:

            raise RuntimeError(
                "GradCAM gradients are None. "
                "Check target layer."
            )

        if self.activations is None:

            raise RuntimeError(
                "GradCAM activations are None. "
                "Check target layer."
            )

        # ------------------------------------
        # FEATURES
        # ------------------------------------

        gradients = self.gradients[0]

        activations = self.activations[0]

        # ------------------------------------
        # GLOBAL AVG POOLING
        # ------------------------------------

        weights = torch.mean(

            gradients,

            dim=(1, 2)
        )

        # ------------------------------------
        # CAM
        # ------------------------------------

        cam = torch.zeros(

            activations.shape[1:],

            dtype=torch.float32,

            device=activations.device
        )

        for i, weight in enumerate(weights):

            cam += (

                weight *

                activations[i]
            )

        # ------------------------------------
        # RELU
        # ------------------------------------

        cam = torch.relu(
            cam
        )

        # ------------------------------------
        # NORMALIZATION
        # ------------------------------------

        cam_min = cam.min()

        cam_max = cam.max()

        if (cam_max - cam_min) > 1e-8:

            cam = (

                cam - cam_min

            ) / (

                cam_max - cam_min
            )

        else:

            cam = torch.zeros_like(
                cam
            )

        cam = cam.cpu().numpy()

        return (

            cam,

            prediction,

            confidence
        )

    # ========================================
    # REMOVE HOOKS
    # ========================================

    def remove_hooks(self):

        if self.forward_handle:

            self.forward_handle.remove()

        if self.backward_handle:

            self.backward_handle.remove()


# ============================================
# OVERLAY HEATMAP
# ============================================

def overlay_heatmap(

    cam,

    image,

    alpha=0.55

):

    cam = cv2.resize(

        cam,

        (
            image.shape[1],
            image.shape[0]
        )
    )

    
    heatmap = cv2.applyColorMap(
    np.uint8(255 * cam),
    cv2.COLORMAP_JET
    )

    heatmap = (
        heatmap.astype(
            np.float32
        )
        / 255.0
    )

    image = image.astype(
        np.float32
    )

    if image.max() > 1:

        image /= 255.0

    overlay = (

        alpha * heatmap +

        (1 - alpha) * image
    )

    overlay = np.clip(

        overlay,

        0,

        1
    )

    return overlay


# ============================================
# SAVE GRADCAM
# ============================================

def save_gradcam(

    overlay,

    save_path

):

    cv2.imwrite(

        save_path,

        np.uint8(
            overlay * 255
        )
    )

    print(
        f"[OK] Saved GradCAM: "
        f"{save_path}"
    )