import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlmodel import Session, SQLModel, create_engine


@pytest.fixture
def client(tmp_path):
    from app.config import settings

    settings.database_url = f"sqlite:///{tmp_path}/test.db"
    settings.upload_dir = tmp_path / "uploads"
    settings.upload_dir.mkdir()
    settings.harmonized_dir = tmp_path / "harmonized"
    settings.harmonized_dir.mkdir()
    settings.weights_dir = tmp_path / "weights"
    settings.weights_dir.mkdir()

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "MRI Platform" in response.text


def test_patient_list_empty(client):
    response = client.get("/patients/")
    assert response.status_code == 200
    assert "Patients" in response.text


def test_create_and_view_patient(client):
    response = client.post(
        "/patients/",
        data={"name": "John Doe", "age": "55", "notes": "Test patient"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    redirect_url = response.headers["location"]
    response = client.get(redirect_url)
    assert response.status_code == 200
    assert "John Doe" in response.text
    assert "55" in response.text


def test_update_patient(client):
    response = client.post(
        "/patients/",
        data={"name": "Jane Doe", "age": "30"},
        follow_redirects=False,
    )
    redirect_url = response.headers["location"]
    patient_id = redirect_url.split("/")[-1]

    response = client.post(
        f"/patients/{patient_id}/edit",
        data={"name": "Jane Smith", "age": "31"},
        follow_redirects=False,
    )
    assert response.status_code == 303

    response = client.get(f"/patients/{patient_id}")
    assert "Jane Smith" in response.text


def test_delete_patient(client):
    response = client.post(
        "/patients/",
        data={"name": "Delete Me"},
        follow_redirects=False,
    )
    redirect_url = response.headers["location"]
    patient_id = redirect_url.split("/")[-1]

    response = client.post(
        f"/patients/{patient_id}/delete", follow_redirects=False
    )
    assert response.status_code == 303

    response = client.get(f"/patients/{patient_id}")
    assert response.status_code == 404


def _create_test_patient(client) -> str:
    response = client.post(
        "/patients/", data={"name": "Test Patient"}, follow_redirects=False
    )
    return response.headers["location"].split("/")[-1]


def _create_test_image_bytes() -> bytes:
    img = PILImage.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


def test_upload_scan(client):
    patient_id = _create_test_patient(client)
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/scans/",
        data={"patient_id": patient_id, "scanner_vendor": "Philips", "modality": "T1"},
        files={"file": ("scan.jpg", img_bytes, "image/jpeg")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    redirect_url = response.headers["location"]
    response = client.get(redirect_url)
    assert response.status_code == 200
    assert "Philips" in response.text
    assert "T1" in response.text


def test_delete_scan(client):
    patient_id = _create_test_patient(client)
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/scans/",
        data={"patient_id": patient_id, "scanner_vendor": "GE", "modality": "T2"},
        files={"file": ("scan.jpg", img_bytes, "image/jpeg")},
        follow_redirects=False,
    )
    scan_url = response.headers["location"]
    scan_id = scan_url.split("/")[-1]
    response = client.post(f"/scans/{scan_id}/delete", follow_redirects=False)
    assert response.status_code == 303


def _upload_scan_for_patient(client, patient_id: str) -> str:
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/scans/",
        data={"patient_id": patient_id, "scanner_vendor": "Siemens", "modality": "T1"},
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        follow_redirects=False,
    )
    return response.headers["location"].split("/")[-1]


def test_trigger_prediction(client, tmp_path):
    import torch
    from cancer_detection.model import BrainTumorClassifier
    from cancer_detection.inference import InferenceService

    model = BrainTumorClassifier(num_classes=4)
    weights_path = tmp_path / "weights" / "best_model.pth"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, weights_path)
    client.app.state.inference_service = InferenceService(weights_path)

    patient_id = _create_test_patient(client)
    scan_id = _upload_scan_for_patient(client, patient_id)

    response = client.post(
        f"/scans/{scan_id}/predict", follow_redirects=False
    )
    assert response.status_code == 303
    redirect_url = response.headers["location"]
    response = client.get(redirect_url)
    assert response.status_code == 200
