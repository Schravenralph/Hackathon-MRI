"""End-to-end workflow test: upload -> harmonize -> predict -> view."""
import io
import numpy as np
import torch
from PIL import Image as PILImage


def test_full_workflow(tmp_path):
    from app.config import settings

    settings.database_url = f"sqlite:///{tmp_path}/test.db"
    settings.upload_dir = tmp_path / "uploads"
    settings.upload_dir.mkdir()
    settings.harmonized_dir = tmp_path / "harmonized"
    settings.harmonized_dir.mkdir()
    settings.weights_dir = tmp_path / "weights"
    settings.weights_dir.mkdir()
    settings.mc_dropout_passes = 5

    from cancer_detection.model import BrainTumorClassifier
    from cancer_detection.inference import InferenceService

    model = BrainTumorClassifier(num_classes=4)
    weights_path = settings.weights_dir / "best_model.pth"
    torch.save({"model_state_dict": model.state_dict()}, weights_path)
    settings.model_checkpoint = "best_model.pth"

    from fastapi.testclient import TestClient
    from app.main import app

    app.state.inference_service = InferenceService(weights_path)

    with TestClient(app) as client:
        # 1. Create patient
        response = client.post(
            "/patients/",
            data={"name": "Demo Patient", "age": "60", "notes": "Full workflow test"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        patient_url = response.headers["location"]
        patient_id = patient_url.split("/")[-1]

        # 2. Upload scan
        img = PILImage.fromarray(np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/scans/",
            data={"patient_id": patient_id, "scanner_vendor": "Philips", "modality": "T1"},
            files={"file": ("brain_scan.jpg", buf.getvalue(), "image/jpeg")},
            follow_redirects=False,
        )
        assert response.status_code == 303
        scan_url = response.headers["location"]
        scan_id = scan_url.split("/")[-1]

        # 3. View scan
        response = client.get(scan_url)
        assert response.status_code == 200
        assert "Philips" in response.text

        # 4. Harmonize
        response = client.post(f"/scans/{scan_id}/harmonize", follow_redirects=False)
        assert response.status_code == 303

        # 5. Predict with uncertainty
        response = client.post(
            f"/scans/{scan_id}/predict?with_uncertainty=true",
            follow_redirects=False,
        )
        assert response.status_code == 303
        prediction_url = response.headers["location"]

        # 6. View prediction
        response = client.get(prediction_url)
        assert response.status_code == 200

        # 7. Serve Grad-CAM
        pred_id = prediction_url.split("/")[-1]
        response = client.get(f"/predictions/{pred_id}/gradcam")
        assert response.status_code == 200

        # 8. Health check
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

        # 9. Viewer page
        response = client.get(f"/viewer/{scan_id}")
        assert response.status_code == 200
