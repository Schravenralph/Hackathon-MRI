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
