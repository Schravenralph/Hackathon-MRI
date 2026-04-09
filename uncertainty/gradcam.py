import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from cancer_detection.model import BrainTumorClassifier


def generate_gradcam(
    model: BrainTumorClassifier,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_class: int | None = None,
) -> np.ndarray:
    """Generate Grad-CAM heatmap overlay.

    Args:
        model: The classifier model.
        input_tensor: Preprocessed tensor (1, 3, 224, 224).
        original_image: Original image as numpy array (H, W, 3) in [0, 1].
        target_class: Class index to explain. None = predicted class.

    Returns:
        Overlay image as numpy array (H, W, 3) in [0, 1].
    """
    target_layer = model.get_features_layer()
    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    cam_h, cam_w = grayscale_cam.shape
    if original_image.shape[0] != cam_h or original_image.shape[1] != cam_w:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray((original_image * 255).astype(np.uint8))
        pil_img = pil_img.resize((cam_w, cam_h))
        original_image = np.array(pil_img) / 255.0

    overlay = show_cam_on_image(
        original_image.astype(np.float32), grayscale_cam, use_rgb=True
    )
    return overlay / 255.0
