import numpy as np
import torch
from PIL import Image


def test_gradcam_output(sample_image):
    from cancer_detection.model import BrainTumorClassifier
    from cancer_detection.preprocessing import preprocess_single
    from uncertainty.gradcam import generate_gradcam

    model = BrainTumorClassifier(num_classes=4)
    model.eval()

    input_tensor = preprocess_single(sample_image)
    original_np = np.array(sample_image.resize((224, 224))) / 255.0

    overlay = generate_gradcam(model, input_tensor, original_np)
    assert overlay.shape == (224, 224, 3)
    assert overlay.min() >= 0.0
    assert overlay.max() <= 1.0


def test_inference_service_predict(sample_image, tmp_path):
    from cancer_detection.inference import InferenceService
    from cancer_detection.model import BrainTumorClassifier

    model = BrainTumorClassifier(num_classes=4)
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "test.pth")

    service = InferenceService(tmp_path / "test.pth")
    result = service.predict(sample_image)

    assert "prediction_class" in result
    assert result["prediction_class"] in ["glioma", "meningioma", "notumor", "pituitary"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert len(result["probabilities"]) == 4
    assert "inference_time_ms" in result
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01


def test_inference_service_predict_with_gradcam(sample_image, tmp_path):
    from cancer_detection.inference import InferenceService
    from cancer_detection.model import BrainTumorClassifier

    model = BrainTumorClassifier(num_classes=4)
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "test.pth")

    service = InferenceService(tmp_path / "test.pth")
    result = service.predict_with_gradcam(sample_image)

    assert "gradcam_overlay" in result
    assert result["gradcam_overlay"].shape == (224, 224, 3)
    assert "prediction_class" in result
