import pytest
import numpy as np
from PIL import Image
from sqlmodel import SQLModel, Session, create_engine


@pytest.fixture
def db_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    SQLModel.metadata.create_all(engine)
    yield engine


@pytest.fixture
def db_session(db_engine):
    with Session(db_engine) as session:
        yield session


@pytest.fixture
def sample_image():
    """Create a synthetic brain-like MRI image for testing."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    # Black background with bright oval center (simulates brain)
    rr, cc = np.ogrid[:256, :256]
    mask = ((rr - 128) ** 2 / 80**2 + (cc - 128) ** 2 / 60**2) < 1
    arr[mask] = np.random.randint(100, 200, (mask.sum(), 3), dtype=np.uint8)
    # Add a bright spot (simulates tumor)
    tumor = ((rr - 150) ** 2 + (cc - 140) ** 2) < 20**2
    arr[tumor] = np.random.randint(200, 255, (tumor.sum(), 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_image_path(sample_image, tmp_path):
    path = tmp_path / "test_scan.jpg"
    sample_image.save(path)
    return path
