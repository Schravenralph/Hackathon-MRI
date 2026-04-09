import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cancer_detection.model import BrainTumorClassifier
from cancer_detection.preprocessing import crop_black_margins, preprocess_single

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}


class InferenceService:
    def __init__(self, weights_path: str | Path, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = BrainTumorClassifier(num_classes=4).to(self.device)
        checkpoint = torch.load(
            weights_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, image: Image.Image) -> dict:
        start = time.time()
        image_rgb = image.convert("RGB")
        input_tensor = preprocess_single(image_rgb).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        pred_idx = probs.argmax().item()
        return {
            "prediction_class": CLASS_NAMES[pred_idx],
            "display_name": DISPLAY_NAMES[CLASS_NAMES[pred_idx]],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)
            },
            "inference_time_ms": int((time.time() - start) * 1000),
        }

    def predict_with_gradcam(self, image: Image.Image) -> dict:
        from uncertainty.gradcam import generate_gradcam

        result = self.predict(image)

        image_rgb = image.convert("RGB")
        input_tensor = preprocess_single(image_rgb).to(self.device)
        cropped = crop_black_margins(image_rgb)
        original_np = np.array(cropped.resize((224, 224))) / 255.0

        overlay = generate_gradcam(
            self.model, input_tensor, original_np,
            target_class=CLASS_NAMES.index(result["prediction_class"]),
        )
        result["gradcam_overlay"] = overlay
        return result
