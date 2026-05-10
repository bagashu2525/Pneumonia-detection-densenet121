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

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # ------------------------------------
        # REGISTER HOOKS
        # ------------------------------------

        self.target_layer.register_forward_hook(
            self.forward_hook
        )

        self.target_layer.register_backward_hook(
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

    def generate_cam(self, input_tensor):

        # ------------------------------------
        # FORWARD PASS
        # ------------------------------------

        output = self.model(input_tensor)

        # ------------------------------------
        # CLEAR OLD GRADIENTS
        # ------------------------------------

        self.model.zero_grad()

        # ------------------------------------
        # BACKWARD PASS
        # ------------------------------------

        output.backward()

        # ====================================
        # GET GRADIENTS & ACTIVATIONS
        # ====================================

        gradients = self.gradients[0]

        activations = self.activations[0]

        # ====================================
        # GLOBAL AVERAGE POOLING
        # ====================================

        weights = torch.mean(
            gradients,
            dim=[1, 2]
        )

        # ====================================
        # CREATE EMPTY CAM
        # ====================================

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        ).to(input_tensor.device)

        # ====================================
        # WEIGHTED COMBINATION
        # ====================================

        for i, w in enumerate(weights):

            cam += w * activations[i]

        # ====================================
        # APPLY ReLU
        # ====================================

        cam = torch.relu(cam)

        # ====================================
        # NORMALIZE
        # ====================================

        cam -= cam.min()

        cam /= cam.max()

        cam = cam.detach().cpu().numpy()

        return cam

# ============================================
# OVERLAY HEATMAP FUNCTION
# ============================================

def overlay_heatmap(
    cam,
    image,
    alpha=0.4
):

    # ========================================
    # RESIZE CAM
    # ========================================

    cam = cv2.resize(
        cam,
        (image.shape[1], image.shape[0])
    )

    # ========================================
    # CREATE HEATMAP
    # ========================================

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = np.float32(heatmap) / 255

    # ========================================
    # COMBINE WITH ORIGINAL IMAGE
    # ========================================

    overlay = heatmap * alpha + image

    overlay = overlay / overlay.max()

    return overlay