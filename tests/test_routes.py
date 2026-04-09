import pytest
from fastapi.testclient import TestClient
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
