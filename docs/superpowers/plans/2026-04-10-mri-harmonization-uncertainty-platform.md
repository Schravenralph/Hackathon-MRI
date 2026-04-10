# MRI Harmonization & Uncertainty Visualization Platform — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-stack MRI platform with scanner-agnostic harmonization and uncertainty visualization, demonstrating workflow innovation over model accuracy for Dutch radiologists.

**Architecture:** FastAPI + HTMX web platform backed by SQLite/SQLModel, integrating three ML pipelines: (1) EfficientNet-B0 brain tumor classification with Grad-CAM, (2) classical harmonization preprocessing (N4ITK + histogram matching + z-score normalization), and (3) Monte Carlo dropout uncertainty quantification. All ML runs in-process (model loaded at startup) for hackathon simplicity.

**Tech Stack:** Python 3.12+, uv, FastAPI, HTMX, Jinja2, Pico CSS, PyTorch, SimpleITK, scikit-image, SQLModel/SQLite, pytorch-grad-cam, matplotlib

**Specs:**
- `docs/superpowers/specs/2026-04-09-mri-harmonization-uncertainty-platform-design.md`
- `docs/superpowers/specs/2026-04-09-cancer-detection-subdomain-design.md`

**Prerequisites:**
- Python 3.12+ installed
- `uv` package manager installed
- Kaggle CLI configured (`kaggle` command) OR manual dataset download

---

## File Structure

```
Hackathon-MRI/
  pyproject.toml
  .gitignore
  app/
    __init__.py
    main.py                          # FastAPI app + lifespan
    config.py                        # Pydantic BaseSettings
    database.py                      # SQLModel engine + session
    deps.py                          # Shared dependencies (templates)
    models/
      __init__.py                    # Re-exports all models
      patient.py                     # Patient SQLModel
      scan.py                        # Scan SQLModel
      prediction.py                  # Prediction SQLModel
    routes/
      __init__.py
      patients.py                    # Patient CRUD routes
      scans.py                       # Scan CRUD + upload
      predictions.py                 # Prediction trigger + results
      harmonize.py                   # Harmonization trigger
      viewer.py                      # NiiVue viewer (stretch)
    templates/
      base.html                      # Base layout + Pico CSS + HTMX
      index.html                     # Home/dashboard
      patients/
        list.html                    # Patient list table
        detail.html                  # Patient detail + scan list
        _form.html                   # Create/edit form partial
      scans/
        detail.html                  # Scan detail + actions
        _upload.html                 # Upload form partial
      predictions/
        detail.html                  # Prediction results + overlays
    static/
      app.css                        # Custom styles
  harmonization/
    __init__.py
    pipeline.py                      # Orchestrates full pipeline
    bias_correction.py               # N4ITK wrapper
    intensity_norm.py                # Z-score normalization
    histogram_match.py               # Reference-based matching
  uncertainty/
    __init__.py
    mc_dropout.py                    # MC Dropout engine
    gradcam.py                       # Grad-CAM wrapper
    heatmap.py                       # Uncertainty visualization
  cancer_detection/
    __init__.py
    model.py                         # EfficientNet-B0 classifier
    preprocessing.py                 # Image preprocessing
    inference.py                     # Inference service
    dataset.py                       # PyTorch Dataset + DataLoader
    training/
      __init__.py
      train.py                       # Training loop
      evaluate.py                    # Evaluation metrics
      config.py                      # Hyperparameters
  data/                              # Gitignored
    raw/                             # Downloaded datasets
    harmonized/                      # Harmonization outputs
    uploads/                         # User uploads
    weights/                         # Model checkpoints
  tests/
    __init__.py
    conftest.py                      # Shared fixtures
    test_preprocessing.py
    test_model.py
    test_inference.py
    test_harmonization.py
    test_uncertainty.py
    test_crud.py
    test_routes.py
```

---

## Phase 1: Project Foundation

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `app/__init__.py`, `app/config.py`, `app/deps.py`
- Create: `app/models/__init__.py`, `app/routes/__init__.py`
- Create: `harmonization/__init__.py`
- Create: `uncertainty/__init__.py`
- Create: `cancer_detection/__init__.py`, `cancer_detection/training/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `data/.gitkeep`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "hackathon-mri"
version = "0.1.0"
description = "MRI Harmonization & Uncertainty Visualization Platform"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "sqlmodel>=0.0.22",
    "jinja2>=3.1.4",
    "python-multipart>=0.0.12",
    "pydantic-settings>=2.6.0",
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "pillow>=11.0.0",
    "numpy>=2.1.0",
    "scikit-image>=0.24.0",
    "scikit-learn>=1.5.0",
    "simpleitk>=2.4.0",
    "nibabel>=5.3.0",
    "matplotlib>=3.9.0",
    "pytorch-grad-cam>=1.5.4",
    "aiofiles>=24.1.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "httpx>=0.27.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["app", "cancer_detection", "harmonization", "uncertainty"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
.venv/
.env
data/raw/
data/uploads/
data/harmonized/
data/weights/
*.db
.DS_Store
```

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p app/models app/routes app/templates/patients app/templates/scans app/templates/predictions app/templates/viewer app/static
mkdir -p harmonization uncertainty cancer_detection/training
mkdir -p data/raw data/uploads data/harmonized data/weights tests
touch app/__init__.py app/models/__init__.py app/routes/__init__.py
touch harmonization/__init__.py uncertainty/__init__.py
touch cancer_detection/__init__.py cancer_detection/training/__init__.py
touch tests/__init__.py
touch data/.gitkeep
```

- [ ] **Step 4: Create `app/config.py`**

```python
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    database_url: str = "sqlite:///data/mri_platform.db"
    upload_dir: Path = Path("data/uploads")
    harmonized_dir: Path = Path("data/harmonized")
    weights_dir: Path = Path("data/weights")
    raw_data_dir: Path = Path("data/raw")
    model_checkpoint: str = "best_model.pth"
    mc_dropout_passes: int = 30
    class_names: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]
    display_names: dict[str, str] = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "notumor": "No Tumor",
        "pituitary": "Pituitary",
    }

    model_config = {"env_file": ".env"}


settings = Settings()
```

- [ ] **Step 5: Create `app/deps.py`**

```python
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")
```

- [ ] **Step 6: Create `tests/conftest.py`**

```python
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
```

- [ ] **Step 7: Install dependencies**

Run: `uv sync`
Expected: Dependencies resolve and install successfully.

- [ ] **Step 8: Verify test infrastructure**

Run: `uv run pytest tests/ -v`
Expected: `no tests ran` or `0 items collected` — confirms pytest discovers the tests directory.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore app/ harmonization/ uncertainty/ cancer_detection/ tests/ data/.gitkeep
git commit -m "feat: project setup with dependencies and directory structure"
```

---

### Task 2: Database Layer + Models

**Files:**
- Create: `app/database.py`
- Create: `app/models/patient.py`
- Create: `app/models/scan.py`
- Create: `app/models/prediction.py`
- Modify: `app/models/__init__.py`
- Create: `tests/test_crud.py`

- [ ] **Step 1: Write failing test for Patient model**

Create `tests/test_crud.py`:

```python
from uuid import UUID
from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models.patient import Patient, PatientCreate


def test_create_patient(db_session: Session):
    patient = Patient(name="Test Patient", age=45, notes="Test notes")
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    assert isinstance(patient.id, UUID)
    assert patient.name == "Test Patient"
    assert patient.age == 45
    assert patient.notes == "Test notes"
    assert isinstance(patient.created_at, datetime)


def test_patient_create_schema():
    data = PatientCreate(name="Jane Doe", age=30)
    assert data.name == "Jane Doe"
    assert data.age == 30
    assert data.notes is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_crud.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.models.patient'`

- [ ] **Step 3: Create `app/database.py`**

```python
from sqlmodel import SQLModel, Session, create_engine

from app.config import settings

engine = create_engine(settings.database_url, echo=False)


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
```

- [ ] **Step 4: Create `app/models/patient.py`**

```python
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlmodel import Field, Relationship, SQLModel


class PatientBase(SQLModel):
    name: str = Field(index=True)
    age: int | None = None
    notes: str | None = None


class Patient(PatientBase, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    scans: list["Scan"] = Relationship(
        back_populates="patient",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class PatientCreate(PatientBase):
    pass


class PatientUpdate(SQLModel):
    name: str | None = None
    age: int | None = None
    notes: str | None = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_crud.py -v`
Expected: 2 passed

- [ ] **Step 6: Add Scan model test to `tests/test_crud.py`**

Append to `tests/test_crud.py`:

```python
from app.models.scan import Scan, ScanCreate


def test_create_scan(db_session: Session):
    patient = Patient(name="Scan Patient")
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    scan = Scan(
        patient_id=patient.id,
        file_path="data/uploads/test.jpg",
        scanner_vendor="Philips",
        modality="T1",
        file_format="jpeg",
    )
    db_session.add(scan)
    db_session.commit()
    db_session.refresh(scan)

    assert isinstance(scan.id, UUID)
    assert scan.patient_id == patient.id
    assert scan.scanner_vendor == "Philips"
    assert scan.is_harmonized is False
    assert scan.harmonized_path is None


def test_scan_create_schema():
    from uuid import uuid4

    data = ScanCreate(patient_id=uuid4(), scanner_vendor="Siemens", modality="T2")
    assert data.scanner_vendor == "Siemens"
```

- [ ] **Step 7: Create `app/models/scan.py`**

```python
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlmodel import Field, Relationship, SQLModel


class ScanBase(SQLModel):
    scanner_vendor: str = Field(default="Unknown")
    modality: str = Field(default="T1")
    file_format: str = Field(default="jpeg")


class Scan(ScanBase, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    patient_id: UUID = Field(foreign_key="patient.id")
    file_path: str = ""
    is_harmonized: bool = Field(default=False)
    harmonized_path: str | None = None
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    patient: "Patient" = Relationship(back_populates="scans")
    predictions: list["Prediction"] = Relationship(
        back_populates="scan",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class ScanCreate(ScanBase):
    patient_id: UUID
```

- [ ] **Step 8: Run test**

Run: `uv run pytest tests/test_crud.py -v`
Expected: 4 passed

- [ ] **Step 9: Add Prediction model test to `tests/test_crud.py`**

Append to `tests/test_crud.py`:

```python
import json

from app.models.prediction import Prediction


def test_create_prediction(db_session: Session):
    patient = Patient(name="Predict Patient")
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    scan = Scan(patient_id=patient.id, file_path="test.jpg")
    db_session.add(scan)
    db_session.commit()
    db_session.refresh(scan)

    probs = {"glioma": 0.85, "meningioma": 0.10, "notumor": 0.03, "pituitary": 0.02}
    prediction = Prediction(
        scan_id=scan.id,
        prediction_class="glioma",
        confidence=0.85,
        probabilities_json=json.dumps(probs),
        inference_time_ms=120,
    )
    db_session.add(prediction)
    db_session.commit()
    db_session.refresh(prediction)

    assert prediction.prediction_class == "glioma"
    assert prediction.confidence == 0.85
    assert prediction.probabilities == probs
    assert prediction.inference_time_ms == 120
    assert prediction.ran_on_harmonized is False
```

- [ ] **Step 10: Create `app/models/prediction.py`**

```python
import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlmodel import Field, Relationship, SQLModel


class Prediction(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    scan_id: UUID = Field(foreign_key="scan.id")
    model_name: str = Field(default="efficientnet_b0")
    prediction_class: str
    confidence: float
    probabilities_json: str = Field(default="{}")
    uncertainty_map_path: str | None = None
    gradcam_path: str | None = None
    ran_on_harmonized: bool = Field(default=False)
    inference_time_ms: int = 0
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    scan: "Scan" = Relationship(back_populates="predictions")

    @property
    def probabilities(self) -> dict[str, float]:
        return json.loads(self.probabilities_json)
```

- [ ] **Step 11: Update `app/models/__init__.py`**

```python
from app.models.patient import Patient, PatientCreate, PatientUpdate
from app.models.prediction import Prediction
from app.models.scan import Scan, ScanCreate

__all__ = [
    "Patient",
    "PatientCreate",
    "PatientUpdate",
    "Scan",
    "ScanCreate",
    "Prediction",
]
```

- [ ] **Step 12: Run full test suite**

Run: `uv run pytest tests/test_crud.py -v`
Expected: 5 passed

- [ ] **Step 13: Add CRUD operations test**

Append to `tests/test_crud.py`:

```python
def test_patient_crud_operations(db_session: Session):
    # Create
    patient = Patient(name="CRUD Test", age=50)
    db_session.add(patient)
    db_session.commit()

    # Read
    fetched = db_session.get(Patient, patient.id)
    assert fetched is not None
    assert fetched.name == "CRUD Test"

    # Update
    fetched.name = "Updated Name"
    db_session.add(fetched)
    db_session.commit()
    db_session.refresh(fetched)
    assert fetched.name == "Updated Name"

    # Delete
    db_session.delete(fetched)
    db_session.commit()
    assert db_session.get(Patient, patient.id) is None


def test_cascade_delete_patient_deletes_scans(db_session: Session):
    patient = Patient(name="Cascade Test")
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    scan = Scan(patient_id=patient.id, file_path="test.jpg")
    db_session.add(scan)
    db_session.commit()

    db_session.delete(patient)
    db_session.commit()

    result = db_session.exec(select(Scan)).all()
    assert len(result) == 0
```

- [ ] **Step 14: Run tests**

Run: `uv run pytest tests/test_crud.py -v`
Expected: 7 passed

- [ ] **Step 15: Commit**

```bash
git add app/database.py app/models/ tests/test_crud.py
git commit -m "feat: database layer with Patient, Scan, Prediction models"
```

---

## Phase 2: Cancer Detection Pipeline

### Task 3: Image Preprocessing + Dataset

**Files:**
- Create: `cancer_detection/preprocessing.py`
- Create: `cancer_detection/dataset.py`
- Create: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing test for preprocessing**

Create `tests/test_preprocessing.py`:

```python
import numpy as np
import torch
from PIL import Image


def test_crop_black_margins(sample_image):
    from cancer_detection.preprocessing import crop_black_margins

    cropped = crop_black_margins(sample_image)
    # The synthetic image has black borders, so cropped should be smaller
    assert cropped.size[0] <= sample_image.size[0]
    assert cropped.size[1] <= sample_image.size[1]
    assert cropped.size[0] > 0
    assert cropped.size[1] > 0


def test_get_val_transforms(sample_image):
    from cancer_detection.preprocessing import get_val_transforms

    transform = get_val_transforms(224)
    tensor = transform(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_get_train_transforms(sample_image):
    from cancer_detection.preprocessing import get_train_transforms

    transform = get_train_transforms(224)
    tensor = transform(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_preprocess_single(sample_image):
    from cancer_detection.preprocessing import preprocess_single

    tensor = preprocess_single(sample_image)
    assert tensor.shape == (1, 3, 224, 224)  # Batch dim added
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cancer_detection.preprocessing'`

- [ ] **Step 3: Create `cancer_detection/preprocessing.py`**

```python
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def crop_black_margins(image: Image.Image, threshold: int = 10) -> Image.Image:
    """Crop black margins around the brain using simple thresholding."""
    gray = np.array(image.convert("L"))
    mask = gray > threshold
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image.crop((x0, y0, x1, y1))


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(crop_black_margins),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(crop_black_margins),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_single(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess a single image for inference. Returns tensor with batch dim."""
    transform = get_val_transforms(image_size)
    return transform(image).unsqueeze(0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: 4 passed

- [ ] **Step 5: Add Dataset test**

Append to `tests/test_preprocessing.py`:

```python
def test_brain_tumor_dataset(tmp_path):
    from cancer_detection.dataset import BrainTumorDataset

    # Create fake dataset structure
    for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(class_dir / f"img_{i}.jpg")

    dataset = BrainTumorDataset(tmp_path)
    assert len(dataset) == 12  # 4 classes * 3 images

    image, label = dataset[0]
    assert isinstance(image, Image.Image)
    assert isinstance(label, int)
    assert 0 <= label <= 3


def test_brain_tumor_dataset_with_transforms(tmp_path):
    from cancer_detection.dataset import BrainTumorDataset
    from cancer_detection.preprocessing import get_val_transforms

    for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(class_dir / "img.jpg")

    dataset = BrainTumorDataset(tmp_path, transform=get_val_transforms(224))
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
```

- [ ] **Step 6: Create `cancer_detection/dataset.py`**

```python
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BrainTumorDataset(Dataset):
    CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

    def __init__(
        self,
        root_dir: Path | str,
        transform: transforms.Compose | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
```

- [ ] **Step 7: Run all preprocessing tests**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: 6 passed

- [ ] **Step 8: Download dataset**

Run: `kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data/raw/ --unzip`

If Kaggle CLI is not configured, download manually from https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset and extract to `data/raw/brain-tumor-mri-dataset/`.

Verify structure:
```bash
ls data/raw/brain-tumor-mri-dataset/Training/
```
Expected: `glioma  meningioma  notumor  pituitary`

- [ ] **Step 9: Commit**

```bash
git add cancer_detection/preprocessing.py cancer_detection/dataset.py tests/test_preprocessing.py
git commit -m "feat: image preprocessing pipeline and dataset loader"
```

---

### Task 4: Model Architecture

**Files:**
- Create: `cancer_detection/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing test for model**

Create `tests/test_model.py`:

```python
import torch


def test_model_forward_pass():
    from cancer_detection.model import BrainTumorClassifier

    model = BrainTumorClassifier(num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.shape == (2, 4)


def test_model_features_layer():
    from cancer_detection.model import BrainTumorClassifier

    model = BrainTumorClassifier(num_classes=4)
    layer = model.get_features_layer()
    assert layer is not None
    # Should be a sequential or conv module, not a linear layer
    assert not isinstance(layer, torch.nn.Linear)


def test_model_has_dropout():
    from cancer_detection.model import BrainTumorClassifier

    model = BrainTumorClassifier(num_classes=4, dropout_rate=0.3)
    dropout_layers = [
        m for m in model.modules() if isinstance(m, torch.nn.Dropout)
    ]
    assert len(dropout_layers) >= 2  # At least 2 dropout layers in classifier
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cancer_detection.model'`

- [ ] **Step 3: Create `cancer_detection/model.py`**

```python
import torch
import torch.nn as nn
from torchvision import models


class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.67),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features_layer(self) -> nn.Module:
        """Return the last conv layer for Grad-CAM."""
        return self.backbone.features[-1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add cancer_detection/model.py tests/test_model.py
git commit -m "feat: EfficientNet-B0 brain tumor classifier architecture"
```

---

### Task 5: Training Pipeline

**Files:**
- Create: `cancer_detection/training/config.py`
- Create: `cancer_detection/training/train.py`
- Create: `cancer_detection/training/evaluate.py`

- [ ] **Step 1: Create `cancer_detection/training/config.py`**

```python
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    data_dir: str = "data/raw/brain-tumor-mri-dataset"
    output_dir: str = "data/weights"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    num_classes: int = 4
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    early_stopping_patience: int = 5
    freeze_epochs: int = 5
    val_split: float = 0.2
    seed: int = 42
```

- [ ] **Step 2: Write test for training one epoch**

Append a new test file `tests/test_training.py`:

```python
import torch
from pathlib import Path
from PIL import Image
import numpy as np


def _create_fake_dataset(tmp_path: Path, n_per_class: int = 5):
    """Create a minimal fake dataset for training tests."""
    for split in ["Training", "Testing"]:
        for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(n_per_class):
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(class_dir / f"img_{i}.jpg")


def test_training_one_epoch(tmp_path):
    from cancer_detection.training.config import TrainingConfig
    from cancer_detection.training.train import train

    _create_fake_dataset(tmp_path, n_per_class=4)

    config = TrainingConfig(
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "weights"),
        epochs=1,
        freeze_epochs=0,
        batch_size=4,
        num_workers=0,
    )
    best_f1 = train(config)
    assert best_f1 >= 0.0
    assert (tmp_path / "weights" / "best_model.pth").exists()


def test_evaluation(tmp_path):
    from cancer_detection.training.config import TrainingConfig
    from cancer_detection.training.train import train
    from cancer_detection.training.evaluate import evaluate

    _create_fake_dataset(tmp_path, n_per_class=4)

    config = TrainingConfig(
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "weights"),
        epochs=1,
        freeze_epochs=0,
        batch_size=4,
        num_workers=0,
    )
    train(config)

    results = evaluate(config)
    assert "accuracy" in results
    assert "weighted_f1" in results
    assert 0.0 <= results["accuracy"] <= 1.0
```

- [ ] **Step 3: Create `cancer_detection/training/train.py`**

```python
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from cancer_detection.dataset import BrainTumorDataset
from cancer_detection.model import BrainTumorClassifier
from cancer_detection.preprocessing import get_train_transforms, get_val_transforms
from cancer_detection.training.config import TrainingConfig


def compute_class_weights(dataset: BrainTumorDataset) -> torch.Tensor:
    counts = torch.zeros(len(dataset.CLASS_NAMES))
    for _, label in dataset.samples:
        counts[label] += 1
    weights = 1.0 / counts.clamp(min=1)
    return weights / weights.sum() * len(dataset.CLASS_NAMES)


def train(config: TrainingConfig | None = None) -> float:
    if config is None:
        config = TrainingConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Datasets
    train_dataset = BrainTumorDataset(
        Path(config.data_dir) / "Training",
        transform=get_train_transforms(config.image_size),
    )
    val_size = int(len(train_dataset) * config.val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    # Validation with val transforms
    val_dataset = BrainTumorDataset(
        Path(config.data_dir) / "Training",
        transform=get_val_transforms(config.image_size),
    )
    val_subset_proper = torch.utils.data.Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_subset_proper,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Model
    model = BrainTumorClassifier(num_classes=config.num_classes).to(device)
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Phase 1: Freeze backbone
    for param in model.backbone.features.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        model.backbone.classifier.parameters(),
        lr=config.head_lr,
        weight_decay=config.weight_decay,
    )

    best_val_f1 = 0.0
    patience_counter = 0
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        # Phase 2: Unfreeze backbone after freeze_epochs
        if epoch == config.freeze_epochs and config.freeze_epochs > 0:
            for param in model.backbone.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                [
                    {
                        "params": model.backbone.features.parameters(),
                        "lr": config.backbone_lr,
                    },
                    {
                        "params": model.backbone.classifier.parameters(),
                        "lr": config.head_lr,
                    },
                ],
                weight_decay=config.weight_decay,
            )
            print(f"  -> Backbone unfrozen at epoch {epoch + 1}")

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        train_acc = train_correct / max(train_total, 1)
        val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(
            len(all_labels), 1
        )
        val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                },
                output_dir / "best_model.pth",
            )
            print(f"  -> Saved best model (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"  -> Early stopping at epoch {epoch + 1}")
                break

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    return best_val_f1


if __name__ == "__main__":
    train()
```

- [ ] **Step 4: Create `cancer_detection/training/evaluate.py`**

```python
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from cancer_detection.dataset import BrainTumorDataset
from cancer_detection.model import BrainTumorClassifier
from cancer_detection.preprocessing import get_val_transforms
from cancer_detection.training.config import TrainingConfig


def evaluate(config: TrainingConfig | None = None) -> dict:
    if config is None:
        config = TrainingConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BrainTumorClassifier(num_classes=config.num_classes).to(device)
    checkpoint = torch.load(
        Path(config.output_dir) / "best_model.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = BrainTumorDataset(
        Path(config.data_dir) / "Testing",
        transform=get_val_transforms(config.image_size),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / max(
        len(all_labels), 1
    )
    weighted_f1 = f1_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=BrainTumorDataset.CLASS_NAMES,
            zero_division=0,
        )
    )
    print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")

    return {"accuracy": accuracy, "weighted_f1": weighted_f1}


if __name__ == "__main__":
    evaluate()
```

- [ ] **Step 5: Run training tests**

Run: `uv run pytest tests/test_training.py -v -x`
Expected: 2 passed (tests use fake data, ~30 sec each)

- [ ] **Step 6: Run full model training on real dataset**

Run: `uv run python -m cancer_detection.training.train`

Note: Takes ~30 minutes on CPU, ~5 minutes with GPU. Expected output: model saved to `data/weights/best_model.pth` with val F1 >= 0.95.

- [ ] **Step 7: Run evaluation on test set**

Run: `uv run python -m cancer_detection.training.evaluate`
Expected: Test accuracy >= 96%, weighted F1 >= 0.95.

- [ ] **Step 8: Commit**

```bash
git add cancer_detection/training/ tests/test_training.py
git commit -m "feat: training pipeline with 2-phase strategy and evaluation"
```

---

### Task 6: Grad-CAM + Inference Service

**Files:**
- Create: `uncertainty/gradcam.py`
- Create: `cancer_detection/inference.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write failing test for Grad-CAM and inference**

Create `tests/test_inference.py`:

```python
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

    # Save a model with random weights for testing
    model = BrainTumorClassifier(num_classes=4)
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "test.pth")

    service = InferenceService(tmp_path / "test.pth")
    result = service.predict(sample_image)

    assert "prediction_class" in result
    assert result["prediction_class"] in [
        "glioma", "meningioma", "notumor", "pituitary"
    ]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert len(result["probabilities"]) == 4
    assert "inference_time_ms" in result
    # Probabilities sum to ~1
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
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_inference.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'uncertainty.gradcam'`

- [ ] **Step 3: Create `uncertainty/gradcam.py`**

```python
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
    grayscale_cam = grayscale_cam[0, :]  # First image in batch

    # Ensure original_image matches cam spatial dims
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
```

- [ ] **Step 4: Create `cancer_detection/inference.py`**

```python
import json
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
            self.model,
            input_tensor,
            original_np,
            target_class=CLASS_NAMES.index(result["prediction_class"]),
        )
        result["gradcam_overlay"] = overlay
        return result
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_inference.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add uncertainty/gradcam.py cancer_detection/inference.py tests/test_inference.py
git commit -m "feat: Grad-CAM explainability and inference service"
```

---

## Phase 3: Web Platform

### Task 7: FastAPI Scaffold + Base Templates

**Files:**
- Create: `app/main.py`
- Create: `app/templates/base.html`
- Create: `app/templates/index.html`
- Create: `app/static/app.css`
- Create: `tests/test_routes.py`

- [ ] **Step 1: Write failing test for health endpoint**

Create `tests/test_routes.py`:

```python
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine


@pytest.fixture
def client(tmp_path):
    # Override settings before importing app
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
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_routes.py::test_health_endpoint -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Create `app/main.py`**

```python
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.deps import templates


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create directories
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.harmonized_dir.mkdir(parents=True, exist_ok=True)
    settings.weights_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(settings.database_url.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    init_db()

    # Load inference model if available
    weights_path = settings.weights_dir / settings.model_checkpoint
    if weights_path.exists():
        from cancer_detection.inference import InferenceService

        app.state.inference_service = InferenceService(weights_path)
    else:
        app.state.inference_service = None

    yield


app = FastAPI(
    title="MRI Harmonization & Uncertainty Platform",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": getattr(app.state, "inference_service", None) is not None,
    }
```

- [ ] **Step 4: Create `app/templates/base.html`**

```html
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}MRI Platform{% endblock %}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
    <link rel="stylesheet" href="/static/app.css">
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
</head>
<body>
    <nav class="container">
        <ul>
            <li><a href="/"><strong>MRI Platform</strong></a></li>
        </ul>
        <ul>
            <li><a href="/patients/">Patients</a></li>
            <li><a href="/api/health">Health</a></li>
        </ul>
    </nav>
    <main class="container">
        {% block content %}{% endblock %}
    </main>
    <footer class="container">
        <small>MRI Harmonization & Uncertainty Visualization Platform</small>
    </footer>
</body>
</html>
```

- [ ] **Step 5: Create `app/templates/index.html`**

```html
{% extends "base.html" %}
{% block title %}MRI Platform — Home{% endblock %}
{% block content %}
<hgroup>
    <h1>MRI Platform</h1>
    <p>Scanner-agnostic harmonization & uncertainty visualization</p>
</hgroup>
<div class="grid">
    <article>
        <header>Patients</header>
        <p>Manage patients and their MRI scans.</p>
        <a href="/patients/" role="button">View Patients</a>
    </article>
    <article>
        <header>Harmonization</header>
        <p>Normalize scans across Philips, Siemens, and GE scanners.</p>
    </article>
    <article>
        <header>Uncertainty</header>
        <p>See where the model is confident vs. where it needs human eyes.</p>
    </article>
</div>
{% endblock %}
```

- [ ] **Step 6: Create `app/static/app.css`**

```css
/* Custom overrides for Pico CSS */
footer {
    margin-top: 2rem;
    text-align: center;
    opacity: 0.6;
}

.scan-image {
    max-width: 100%;
    border-radius: var(--pico-border-radius);
}

.overlay-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}

.probability-bar {
    height: 1.5rem;
    border-radius: 4px;
    transition: width 0.3s ease;
}

.confidence-high { background-color: #4caf50; }
.confidence-medium { background-color: #ff9800; }
.confidence-low { background-color: #f44336; }

.htmx-indicator {
    display: none;
}
.htmx-request .htmx-indicator {
    display: inline;
}
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_routes.py -v`
Expected: 2 passed

- [ ] **Step 8: Commit**

```bash
git add app/main.py app/deps.py app/templates/ app/static/ tests/test_routes.py
git commit -m "feat: FastAPI scaffold with base templates and health endpoint"
```

---

### Task 8: Patient CRUD

**Files:**
- Create: `app/routes/patients.py`
- Create: `app/templates/patients/list.html`
- Create: `app/templates/patients/detail.html`
- Create: `app/templates/patients/_form.html`
- Modify: `app/main.py` (register router)

- [ ] **Step 1: Add Patient CRUD tests**

Append to `tests/test_routes.py`:

```python
def test_patient_list_empty(client):
    response = client.get("/patients/")
    assert response.status_code == 200
    assert "Patients" in response.text


def test_create_and_view_patient(client):
    # Create
    response = client.post(
        "/patients/",
        data={"name": "John Doe", "age": "55", "notes": "Test patient"},
        follow_redirects=False,
    )
    assert response.status_code == 303  # Redirect after create
    redirect_url = response.headers["location"]

    # View detail
    response = client.get(redirect_url)
    assert response.status_code == 200
    assert "John Doe" in response.text
    assert "55" in response.text


def test_update_patient(client):
    # Create first
    response = client.post(
        "/patients/",
        data={"name": "Jane Doe", "age": "30"},
        follow_redirects=False,
    )
    redirect_url = response.headers["location"]
    patient_id = redirect_url.split("/")[-1]

    # Update
    response = client.post(
        f"/patients/{patient_id}/edit",
        data={"name": "Jane Smith", "age": "31"},
        follow_redirects=False,
    )
    assert response.status_code == 303

    # Verify
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

    # Should be gone
    response = client.get(f"/patients/{patient_id}")
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_routes.py::test_patient_list_empty -v`
Expected: FAIL — 404 (route not registered)

- [ ] **Step 3: Create `app/routes/patients.py`**

```python
from uuid import UUID

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import Session, select

from app.database import get_session
from app.deps import templates
from app.models.patient import Patient

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("/", response_class=HTMLResponse)
def list_patients(request: Request, session: Session = Depends(get_session)):
    patients = session.exec(
        select(Patient).order_by(Patient.created_at.desc())
    ).all()
    return templates.TemplateResponse(
        "patients/list.html", {"request": request, "patients": patients}
    )


@router.get("/new", response_class=HTMLResponse)
def new_patient_form(request: Request):
    return templates.TemplateResponse(
        "patients/_form.html", {"request": request, "patient": None}
    )


@router.post("/")
def create_patient(
    name: str = Form(...),
    age: str = Form(""),
    notes: str = Form(""),
    session: Session = Depends(get_session),
):
    patient = Patient(
        name=name,
        age=int(age) if age.strip() else None,
        notes=notes if notes.strip() else None,
    )
    session.add(patient)
    session.commit()
    session.refresh(patient)
    return RedirectResponse(url=f"/patients/{patient.id}", status_code=303)


@router.get("/{patient_id}", response_class=HTMLResponse)
def patient_detail(
    request: Request,
    patient_id: UUID,
    session: Session = Depends(get_session),
):
    patient = session.get(Patient, patient_id)
    if not patient:
        return HTMLResponse("Patient not found", status_code=404)
    return templates.TemplateResponse(
        "patients/detail.html",
        {"request": request, "patient": patient, "scans": patient.scans},
    )


@router.get("/{patient_id}/edit", response_class=HTMLResponse)
def edit_patient_form(
    request: Request,
    patient_id: UUID,
    session: Session = Depends(get_session),
):
    patient = session.get(Patient, patient_id)
    if not patient:
        return HTMLResponse("Patient not found", status_code=404)
    return templates.TemplateResponse(
        "patients/_form.html", {"request": request, "patient": patient}
    )


@router.post("/{patient_id}/edit")
def update_patient(
    patient_id: UUID,
    name: str = Form(...),
    age: str = Form(""),
    notes: str = Form(""),
    session: Session = Depends(get_session),
):
    patient = session.get(Patient, patient_id)
    if not patient:
        return HTMLResponse("Patient not found", status_code=404)
    patient.name = name
    patient.age = int(age) if age.strip() else None
    patient.notes = notes if notes.strip() else None
    session.add(patient)
    session.commit()
    return RedirectResponse(url=f"/patients/{patient_id}", status_code=303)


@router.post("/{patient_id}/delete")
def delete_patient(
    patient_id: UUID, session: Session = Depends(get_session)
):
    patient = session.get(Patient, patient_id)
    if patient:
        session.delete(patient)
        session.commit()
    return RedirectResponse(url="/patients/", status_code=303)
```

- [ ] **Step 4: Create `app/templates/patients/list.html`**

```html
{% extends "base.html" %}
{% block title %}Patients{% endblock %}
{% block content %}
<h1>Patients</h1>
<a href="/patients/new" role="button">Add Patient</a>
{% if patients %}
<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Age</th>
            <th>Scans</th>
            <th>Created</th>
        </tr>
    </thead>
    <tbody>
        {% for patient in patients %}
        <tr>
            <td><a href="/patients/{{ patient.id }}">{{ patient.name }}</a></td>
            <td>{{ patient.age or '-' }}</td>
            <td>{{ patient.scans | length }}</td>
            <td>{{ patient.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p>No patients yet. Add one to get started.</p>
{% endif %}
{% endblock %}
```

- [ ] **Step 5: Create `app/templates/patients/detail.html`**

```html
{% extends "base.html" %}
{% block title %}{{ patient.name }}{% endblock %}
{% block content %}
<hgroup>
    <h1>{{ patient.name }}</h1>
    <p>Patient details</p>
</hgroup>

<article>
    <dl>
        <dt>Age</dt><dd>{{ patient.age or 'Not specified' }}</dd>
        <dt>Notes</dt><dd>{{ patient.notes or 'None' }}</dd>
        <dt>Created</dt><dd>{{ patient.created_at.strftime('%Y-%m-%d %H:%M') }}</dd>
    </dl>
    <footer>
        <a href="/patients/{{ patient.id }}/edit" role="button" class="outline">Edit</a>
        <form method="post" action="/patients/{{ patient.id }}/delete" style="display:inline">
            <button type="submit" class="secondary outline">Delete</button>
        </form>
    </footer>
</article>

<h2>Scans</h2>
<a href="#upload-form" role="button" class="outline">Upload Scan</a>

<div id="upload-form" style="display:none; margin-top:1rem;">
    <article>
        <form method="post" action="/scans/" enctype="multipart/form-data">
            <input type="hidden" name="patient_id" value="{{ patient.id }}">
            <label>MRI Image <input type="file" name="file" accept="image/*,.nii,.nii.gz" required></label>
            <div class="grid">
                <label>Scanner Vendor
                    <select name="scanner_vendor">
                        <option value="Unknown">Unknown</option>
                        <option value="Philips">Philips</option>
                        <option value="Siemens">Siemens</option>
                        <option value="GE">GE</option>
                    </select>
                </label>
                <label>Modality
                    <select name="modality">
                        <option value="T1">T1</option>
                        <option value="T2">T2</option>
                        <option value="FLAIR">FLAIR</option>
                        <option value="T1ce">T1ce</option>
                    </select>
                </label>
            </div>
            <button type="submit">Upload</button>
        </form>
    </article>
</div>

<script>
document.querySelector('a[href="#upload-form"]').addEventListener('click', function(e) {
    e.preventDefault();
    var form = document.getElementById('upload-form');
    form.style.display = form.style.display === 'none' ? 'block' : 'none';
});
</script>

{% if scans %}
<table>
    <thead>
        <tr>
            <th>Modality</th>
            <th>Scanner</th>
            <th>Harmonized</th>
            <th>Uploaded</th>
        </tr>
    </thead>
    <tbody>
        {% for scan in scans %}
        <tr>
            <td><a href="/scans/{{ scan.id }}">{{ scan.modality }}</a></td>
            <td>{{ scan.scanner_vendor }}</td>
            <td>{{ 'Yes' if scan.is_harmonized else 'No' }}</td>
            <td>{{ scan.uploaded_at.strftime('%Y-%m-%d %H:%M') }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p>No scans uploaded yet.</p>
{% endif %}
{% endblock %}
```

- [ ] **Step 6: Create `app/templates/patients/_form.html`**

```html
{% extends "base.html" %}
{% block title %}{{ 'Edit' if patient else 'New' }} Patient{% endblock %}
{% block content %}
<h1>{{ 'Edit' if patient else 'New' }} Patient</h1>
<form method="post" action="{{ '/patients/' ~ patient.id ~ '/edit' if patient else '/patients/' }}">
    <label>Name
        <input type="text" name="name" value="{{ patient.name if patient else '' }}" required>
    </label>
    <label>Age
        <input type="number" name="age" value="{{ patient.age if patient and patient.age else '' }}" min="0" max="150">
    </label>
    <label>Notes
        <textarea name="notes">{{ patient.notes if patient and patient.notes else '' }}</textarea>
    </label>
    <button type="submit">{{ 'Update' if patient else 'Create' }}</button>
    <a href="{{ '/patients/' ~ patient.id if patient else '/patients/' }}" role="button" class="outline">Cancel</a>
</form>
{% endblock %}
```

- [ ] **Step 7: Register router in `app/main.py`**

Add after the `app` definition, before the `@app.get("/")` route:

```python
from app.routes import patients

app.include_router(patients.router)
```

- [ ] **Step 8: Run tests**

Run: `uv run pytest tests/test_routes.py -v`
Expected: 6 passed (health, home, list, create+view, update, delete)

- [ ] **Step 9: Commit**

```bash
git add app/routes/patients.py app/templates/patients/ app/main.py tests/test_routes.py
git commit -m "feat: patient CRUD with HTMX forms and templates"
```

---

### Task 9: Scan CRUD + File Upload

**Files:**
- Create: `app/routes/scans.py`
- Create: `app/templates/scans/detail.html`
- Modify: `app/main.py` (register router)

- [ ] **Step 1: Add scan tests to `tests/test_routes.py`**

Append to `tests/test_routes.py`:

```python
import io
from PIL import Image as PILImage
import numpy as np


def _create_test_patient(client) -> str:
    """Helper: create a patient, return its ID."""
    response = client.post(
        "/patients/", data={"name": "Test Patient"}, follow_redirects=False
    )
    return response.headers["location"].split("/")[-1]


def _create_test_image_bytes() -> bytes:
    """Helper: create a JPEG image in memory."""
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
        data={
            "patient_id": patient_id,
            "scanner_vendor": "Philips",
            "modality": "T1",
        },
        files={"file": ("scan.jpg", img_bytes, "image/jpeg")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    redirect_url = response.headers["location"]

    # View scan detail
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
```

- [ ] **Step 2: Create `app/routes/scans.py`**

```python
import shutil
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sqlmodel import Session

from app.config import settings
from app.database import get_session
from app.deps import templates
from app.models.scan import Scan

router = APIRouter(prefix="/scans", tags=["scans"])


@router.post("/")
async def upload_scan(
    patient_id: UUID = Form(...),
    scanner_vendor: str = Form("Unknown"),
    modality: str = Form("T1"),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    # Determine format
    suffix = Path(file.filename or "scan.jpg").suffix.lower()
    file_format = "nifti" if suffix in (".nii", ".gz") else suffix.lstrip(".")

    # Create scan record first to get ID
    scan = Scan(
        patient_id=patient_id,
        scanner_vendor=scanner_vendor,
        modality=modality,
        file_format=file_format or "jpeg",
        file_path="",  # Will update after saving
    )
    session.add(scan)
    session.commit()
    session.refresh(scan)

    # Save file to data/uploads/{scan_id}/{filename}
    scan_dir = settings.upload_dir / str(scan.id)
    scan_dir.mkdir(parents=True, exist_ok=True)
    file_path = scan_dir / (file.filename or "scan.jpg")

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    scan.file_path = str(file_path)
    session.add(scan)
    session.commit()

    return RedirectResponse(url=f"/scans/{scan.id}", status_code=303)


@router.get("/{scan_id}", response_class=HTMLResponse)
def scan_detail(
    request: Request, scan_id: UUID, session: Session = Depends(get_session)
):
    scan = session.get(Scan, scan_id)
    if not scan:
        return HTMLResponse("Scan not found", status_code=404)
    return templates.TemplateResponse(
        "scans/detail.html",
        {
            "request": request,
            "scan": scan,
            "predictions": scan.predictions,
            "model_loaded": getattr(
                request.app.state, "inference_service", None
            )
            is not None,
        },
    )


@router.get("/{scan_id}/image")
def serve_scan_image(scan_id: UUID, session: Session = Depends(get_session)):
    scan = session.get(Scan, scan_id)
    if not scan or not scan.file_path:
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(scan.file_path)


@router.get("/{scan_id}/harmonized-image")
def serve_harmonized_image(
    scan_id: UUID, session: Session = Depends(get_session)
):
    scan = session.get(Scan, scan_id)
    if not scan or not scan.harmonized_path:
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(scan.harmonized_path)


@router.post("/{scan_id}/delete")
def delete_scan(scan_id: UUID, session: Session = Depends(get_session)):
    scan = session.get(Scan, scan_id)
    if scan:
        # Remove files from disk
        scan_dir = settings.upload_dir / str(scan.id)
        if scan_dir.exists():
            shutil.rmtree(scan_dir)
        patient_id = scan.patient_id
        session.delete(scan)
        session.commit()
        return RedirectResponse(
            url=f"/patients/{patient_id}", status_code=303
        )
    return RedirectResponse(url="/patients/", status_code=303)
```

- [ ] **Step 3: Create `app/templates/scans/detail.html`**

```html
{% extends "base.html" %}
{% block title %}Scan — {{ scan.modality }} ({{ scan.scanner_vendor }}){% endblock %}
{% block content %}
<nav aria-label="breadcrumb">
    <ul>
        <li><a href="/patients/">Patients</a></li>
        <li><a href="/patients/{{ scan.patient_id }}">Patient</a></li>
        <li>Scan</li>
    </ul>
</nav>

<h1>{{ scan.modality }} Scan ({{ scan.scanner_vendor }})</h1>

<div class="grid">
    <article>
        <header>Original Scan</header>
        <img src="/scans/{{ scan.id }}/image" alt="MRI Scan" class="scan-image">
    </article>
    {% if scan.is_harmonized %}
    <article>
        <header>Harmonized Scan</header>
        <img src="/scans/{{ scan.id }}/harmonized-image" alt="Harmonized Scan" class="scan-image">
    </article>
    {% endif %}
</div>

<article>
    <dl>
        <dt>Scanner Vendor</dt><dd>{{ scan.scanner_vendor }}</dd>
        <dt>Modality</dt><dd>{{ scan.modality }}</dd>
        <dt>Format</dt><dd>{{ scan.file_format }}</dd>
        <dt>Harmonized</dt><dd>{{ 'Yes' if scan.is_harmonized else 'No' }}</dd>
        <dt>Uploaded</dt><dd>{{ scan.uploaded_at.strftime('%Y-%m-%d %H:%M') }}</dd>
    </dl>
</article>

<div class="grid">
    {% if not scan.is_harmonized %}
    <form method="post" action="/scans/{{ scan.id }}/harmonize">
        <button type="submit" id="harmonize-btn">
            <span class="htmx-indicator" aria-busy="true"></span>
            Harmonize
        </button>
    </form>
    {% endif %}

    {% if model_loaded %}
    <form method="post" action="/scans/{{ scan.id }}/predict">
        <button type="submit">
            Run Prediction
        </button>
    </form>
    {% else %}
    <p><small>Model not loaded — train the model first to enable predictions.</small></p>
    {% endif %}

    <form method="post" action="/scans/{{ scan.id }}/delete">
        <button type="submit" class="secondary outline">Delete Scan</button>
    </form>
</div>

{% if predictions %}
<h2>Predictions</h2>
{% for pred in predictions %}
<article>
    <header>
        <strong>{{ pred.prediction_class | title }}</strong>
        — Confidence: {{ (pred.confidence * 100) | round(1) }}%
        {% if pred.ran_on_harmonized %}<mark>Harmonized</mark>{% endif %}
    </header>
    {% set probs = pred.probabilities %}
    {% for class_name, prob in probs.items() %}
    <div style="margin-bottom: 0.5rem;">
        <small>{{ class_name }}: {{ (prob * 100) | round(1) }}%</small>
        <div style="background: #e0e0e0; border-radius: 4px; overflow: hidden;">
            <div class="probability-bar {% if prob > 0.7 %}confidence-high{% elif prob > 0.3 %}confidence-medium{% else %}confidence-low{% endif %}"
                 style="width: {{ (prob * 100) | round(1) }}%;">&nbsp;</div>
        </div>
    </div>
    {% endfor %}
    <footer>
        <small>{{ pred.model_name }} | {{ pred.inference_time_ms }}ms |
               {{ pred.created_at.strftime('%Y-%m-%d %H:%M') }}</small>
        {% if pred.gradcam_path %}
        | <a href="/predictions/{{ pred.id }}">View Details + Grad-CAM</a>
        {% endif %}
    </footer>
</article>
{% endfor %}
{% endif %}
{% endblock %}
```

- [ ] **Step 4: Register router in `app/main.py`**

Add the import and include:

```python
from app.routes import patients, scans

app.include_router(patients.router)
app.include_router(scans.router)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_routes.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add app/routes/scans.py app/templates/scans/ app/main.py tests/test_routes.py
git commit -m "feat: scan CRUD with file upload and image serving"
```

---

### Task 10: Prediction Routes + Results

**Files:**
- Create: `app/routes/predictions.py`
- Create: `app/templates/predictions/detail.html`
- Modify: `app/main.py` (register router)

- [ ] **Step 1: Add prediction tests to `tests/test_routes.py`**

Append to `tests/test_routes.py`:

```python
import torch


def _upload_scan_for_patient(client, patient_id: str) -> str:
    """Helper: upload a scan, return scan ID."""
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/scans/",
        data={"patient_id": patient_id, "scanner_vendor": "Siemens", "modality": "T1"},
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        follow_redirects=False,
    )
    return response.headers["location"].split("/")[-1]


def test_trigger_prediction(client, tmp_path):
    """Test prediction trigger (requires mock model)."""
    # Create a fake model checkpoint so the service loads
    from cancer_detection.model import BrainTumorClassifier

    model = BrainTumorClassifier(num_classes=4)
    weights_path = tmp_path / "weights" / "best_model.pth"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, weights_path)

    # Reload app with model
    from app.config import settings

    settings.weights_dir = tmp_path / "weights"
    settings.model_checkpoint = "best_model.pth"

    from cancer_detection.inference import InferenceService

    client.app.state.inference_service = InferenceService(weights_path)

    patient_id = _create_test_patient(client)
    scan_id = _upload_scan_for_patient(client, patient_id)

    # Trigger prediction
    response = client.post(
        f"/scans/{scan_id}/predict", follow_redirects=False
    )
    assert response.status_code == 303
    redirect_url = response.headers["location"]

    # View prediction detail
    response = client.get(redirect_url)
    assert response.status_code == 200
    assert "Confidence" in response.text or "confidence" in response.text.lower()
```

- [ ] **Step 2: Create `app/routes/predictions.py`**

```python
import io
import json
from pathlib import Path
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from PIL import Image
from sqlmodel import Session

from app.config import settings
from app.database import get_session
from app.deps import templates
from app.models.prediction import Prediction
from app.models.scan import Scan

router = APIRouter(tags=["predictions"])


@router.post("/scans/{scan_id}/predict")
def trigger_prediction(
    request: Request,
    scan_id: UUID,
    session: Session = Depends(get_session),
):
    scan = session.get(Scan, scan_id)
    if not scan:
        return HTMLResponse("Scan not found", status_code=404)

    inference = getattr(request.app.state, "inference_service", None)
    if inference is None:
        return HTMLResponse("Model not loaded", status_code=503)

    # Load image
    image = Image.open(scan.file_path).convert("RGB")

    # Run inference with Grad-CAM
    result = inference.predict_with_gradcam(image)

    # Save Grad-CAM overlay
    gradcam_path = None
    if "gradcam_overlay" in result:
        overlay_img = Image.fromarray(
            (result["gradcam_overlay"] * 255).astype(np.uint8)
        )
        scan_dir = Path(scan.file_path).parent
        gradcam_file = scan_dir / "gradcam.png"
        overlay_img.save(gradcam_file)
        gradcam_path = str(gradcam_file)

    # Create prediction record
    prediction = Prediction(
        scan_id=scan.id,
        prediction_class=result["prediction_class"],
        confidence=result["confidence"],
        probabilities_json=json.dumps(result["probabilities"]),
        gradcam_path=gradcam_path,
        ran_on_harmonized=scan.is_harmonized,
        inference_time_ms=result["inference_time_ms"],
    )
    session.add(prediction)
    session.commit()
    session.refresh(prediction)

    return RedirectResponse(
        url=f"/predictions/{prediction.id}", status_code=303
    )


@router.get("/predictions/{prediction_id}", response_class=HTMLResponse)
def prediction_detail(
    request: Request,
    prediction_id: UUID,
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if not prediction:
        return HTMLResponse("Prediction not found", status_code=404)
    scan = session.get(Scan, prediction.scan_id)
    return templates.TemplateResponse(
        "predictions/detail.html",
        {"request": request, "prediction": prediction, "scan": scan},
    )


@router.get("/predictions/{prediction_id}/gradcam")
def serve_gradcam(
    prediction_id: UUID, session: Session = Depends(get_session)
):
    prediction = session.get(Prediction, prediction_id)
    if not prediction or not prediction.gradcam_path:
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(prediction.gradcam_path)


@router.get("/predictions/{prediction_id}/uncertainty-map")
def serve_uncertainty_map(
    prediction_id: UUID, session: Session = Depends(get_session)
):
    prediction = session.get(Prediction, prediction_id)
    if not prediction or not prediction.uncertainty_map_path:
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(prediction.uncertainty_map_path)
```

- [ ] **Step 3: Create `app/templates/predictions/detail.html`**

```html
{% extends "base.html" %}
{% block title %}Prediction — {{ prediction.prediction_class | title }}{% endblock %}
{% block content %}
<nav aria-label="breadcrumb">
    <ul>
        <li><a href="/patients/">Patients</a></li>
        <li><a href="/scans/{{ scan.id }}">Scan</a></li>
        <li>Prediction</li>
    </ul>
</nav>

<h1>Prediction: {{ prediction.prediction_class | replace('notumor', 'No Tumor') | title }}</h1>

<div class="grid">
    <article>
        <header>Original Scan</header>
        <img src="/scans/{{ scan.id }}/image" alt="MRI Scan" class="scan-image">
    </article>
    {% if prediction.gradcam_path %}
    <article>
        <header>Grad-CAM Overlay</header>
        <img src="/predictions/{{ prediction.id }}/gradcam" alt="Grad-CAM" class="scan-image">
        <footer><small>Regions that influenced the prediction</small></footer>
    </article>
    {% endif %}
</div>

<article>
    <header>Classification Result</header>
    <h2>{{ (prediction.confidence * 100) | round(1) }}% {{ prediction.prediction_class | replace('notumor', 'No Tumor') | title }}</h2>

    {% set probs = prediction.probabilities %}
    {% for class_name, prob in probs.items() %}
    <div style="margin-bottom: 0.5rem;">
        <div style="display: flex; justify-content: space-between;">
            <small>{{ class_name | replace('notumor', 'No Tumor') | title }}</small>
            <small>{{ (prob * 100) | round(1) }}%</small>
        </div>
        <progress value="{{ (prob * 100) | round(0) | int }}" max="100"></progress>
    </div>
    {% endfor %}

    <footer>
        <small>
            Model: {{ prediction.model_name }} |
            Inference: {{ prediction.inference_time_ms }}ms |
            {% if prediction.ran_on_harmonized %}Ran on harmonized scan{% else %}Ran on original scan{% endif %} |
            {{ prediction.created_at.strftime('%Y-%m-%d %H:%M') }}
        </small>
    </footer>
</article>

{% if prediction.uncertainty_map_path %}
<article>
    <header>Uncertainty Analysis</header>
    <img src="/predictions/{{ prediction.id }}/uncertainty-map" alt="Uncertainty" class="scan-image">
    <footer><small>Red = high uncertainty (model unsure), Green = low uncertainty (model confident)</small></footer>
</article>
{% endif %}

<a href="/scans/{{ scan.id }}" role="button" class="outline">Back to Scan</a>
{% endblock %}
```

- [ ] **Step 4: Register router in `app/main.py`**

```python
from app.routes import patients, scans, predictions

app.include_router(patients.router)
app.include_router(scans.router)
app.include_router(predictions.router)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_routes.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add app/routes/predictions.py app/templates/predictions/ app/main.py tests/test_routes.py
git commit -m "feat: prediction routes with Grad-CAM overlay display"
```

---

## Phase 4: Harmonization Pipeline

### Task 11: Core Harmonization Pipeline

**Files:**
- Create: `harmonization/bias_correction.py`
- Create: `harmonization/intensity_norm.py`
- Create: `harmonization/histogram_match.py`
- Create: `harmonization/pipeline.py`
- Create: `tests/test_harmonization.py`

- [ ] **Step 1: Write failing tests for harmonization modules**

Create `tests/test_harmonization.py`:

```python
import numpy as np
from PIL import Image


def _make_grayscale_array(size: int = 128, seed: int = 0) -> np.ndarray:
    """Create a synthetic MRI-like grayscale image."""
    rng = np.random.RandomState(seed)
    arr = np.zeros((size, size), dtype=np.float32)
    # Brain-like oval
    rr, cc = np.ogrid[:size, :size]
    mask = ((rr - size // 2) ** 2 / (size // 3) ** 2 + (cc - size // 2) ** 2 / (size // 4) ** 2) < 1
    arr[mask] = rng.uniform(80, 200, mask.sum()).astype(np.float32)
    return arr


def test_zscore_normalize():
    from harmonization.intensity_norm import zscore_normalize

    arr = _make_grayscale_array()
    result = zscore_normalize(arr)

    # Non-zero region should have mean ~0 and std ~1
    mask = arr > 10
    brain = result[mask]
    assert abs(brain.mean()) < 0.1
    assert abs(brain.std() - 1.0) < 0.1


def test_histogram_match():
    from harmonization.histogram_match import histogram_match

    source = _make_grayscale_array(seed=0)
    reference = _make_grayscale_array(seed=42) * 1.5 + 20  # Different distribution

    result = histogram_match(source, reference)
    assert result.shape == source.shape
    # Result histogram should be closer to reference than source was
    assert result.dtype == source.dtype


def test_bias_correction():
    from harmonization.bias_correction import apply_bias_correction_2d

    arr = _make_grayscale_array()
    # Add synthetic bias field (smooth gradient)
    bias = np.linspace(0.5, 1.5, arr.shape[1])
    biased = arr * bias[np.newaxis, :]

    result = apply_bias_correction_2d(biased)
    assert result.shape == arr.shape
    assert not np.allclose(result, biased)  # Something changed


def test_full_pipeline():
    from harmonization.pipeline import harmonize_2d

    source = _make_grayscale_array(seed=0)
    reference = _make_grayscale_array(seed=42)

    results = harmonize_2d(source, reference=reference)

    assert "original" in results
    assert "bias_corrected" in results
    assert "normalized" in results
    assert "harmonized" in results
    assert results["harmonized"].shape == source.shape


def test_pipeline_without_reference():
    from harmonization.pipeline import harmonize_2d

    source = _make_grayscale_array()
    results = harmonize_2d(source, reference=None)

    assert "harmonized" in results
    assert results["harmonized"].shape == source.shape


def test_generate_comparison_histogram():
    from harmonization.pipeline import generate_comparison_histogram

    original = _make_grayscale_array(seed=0)
    harmonized = _make_grayscale_array(seed=42)

    png_bytes = generate_comparison_histogram(original, harmonized)
    assert len(png_bytes) > 100  # Valid PNG has some size
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_harmonization.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create `harmonization/intensity_norm.py`**

```python
import numpy as np


def zscore_normalize(
    image: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Z-score normalization: zero mean, unit variance within brain mask."""
    if mask is None:
        mask = image > np.percentile(image[image > 0], 10) if image.max() > 0 else image > 0
    if not mask.any():
        return image.copy()
    brain_voxels = image[mask].astype(np.float64)
    mean = brain_voxels.mean()
    std = brain_voxels.std()
    if std < 1e-8:
        return image.copy()
    result = image.copy().astype(np.float64)
    result[mask] = (brain_voxels - mean) / std
    return result.astype(np.float32)
```

- [ ] **Step 4: Create `harmonization/histogram_match.py`**

```python
import numpy as np
from skimage.exposure import match_histograms


def histogram_match(
    source: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Match histogram of source image to reference template."""
    if source.ndim == 2 and reference.ndim == 2:
        matched = match_histograms(source, reference)
    else:
        channel_axis = -1 if source.ndim == 3 else None
        matched = match_histograms(
            source, reference, channel_axis=channel_axis
        )
    return matched.astype(source.dtype)
```

- [ ] **Step 5: Create `harmonization/bias_correction.py`**

```python
import numpy as np


def apply_bias_correction_2d(image: np.ndarray) -> np.ndarray:
    """Apply bias field correction for 2D MRI images.

    Uses SimpleITK N4 if available, falls back to log-domain
    low-frequency subtraction.
    """
    try:
        import SimpleITK as sitk

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
        mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
        corrected = corrector.Execute(sitk_image, mask)
        return sitk.GetArrayFromImage(corrected)
    except Exception:
        return _fallback_bias_correction(image)


def _fallback_bias_correction(image: np.ndarray) -> np.ndarray:
    """Fallback: homomorphic filtering via log-domain Gaussian blur."""
    from scipy.ndimage import gaussian_filter

    # Avoid log(0)
    safe_image = image.astype(np.float64)
    safe_image = np.clip(safe_image, 1.0, None)
    log_image = np.log(safe_image)

    # Estimate bias field as low-frequency component
    low_freq = gaussian_filter(log_image, sigma=30)
    corrected = log_image - low_freq + log_image.mean()

    return np.exp(corrected).astype(np.float32)
```

- [ ] **Step 6: Create `harmonization/pipeline.py`**

```python
import io

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harmonization.bias_correction import apply_bias_correction_2d
from harmonization.histogram_match import histogram_match
from harmonization.intensity_norm import zscore_normalize


def harmonize_2d(
    image: np.ndarray, reference: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Full harmonization pipeline for 2D images.

    Returns dict with intermediate results for visualization.
    """
    results = {"original": image.copy()}

    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    # Step 1: Bias field correction
    corrected = apply_bias_correction_2d(gray)
    results["bias_corrected"] = corrected

    # Step 2: Z-score normalization
    normalized = zscore_normalize(corrected)
    results["normalized"] = normalized

    # Step 3: Histogram matching (if reference provided)
    if reference is not None:
        if reference.ndim == 3:
            reference = np.mean(reference, axis=2).astype(np.float32)
        ref_normalized = zscore_normalize(reference)
        matched = histogram_match(normalized, ref_normalized)
        results["histogram_matched"] = matched
    else:
        matched = normalized

    results["harmonized"] = matched
    return results


def generate_comparison_histogram(
    original: np.ndarray,
    harmonized: np.ndarray,
    title: str = "Before/After Harmonization",
) -> bytes:
    """Generate a before/after histogram comparison as PNG bytes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    orig_flat = original.flatten()
    orig_flat = orig_flat[orig_flat > 0]  # Exclude background
    ax1.hist(orig_flat, bins=100, alpha=0.7, color="#e53935", label="Original")
    ax1.set_title("Before Harmonization")
    ax1.set_xlabel("Intensity")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    harm_flat = harmonized.flatten()
    harm_flat = harm_flat[np.isfinite(harm_flat)]
    if len(harm_flat) > 0:
        ax2.hist(harm_flat, bins=100, alpha=0.7, color="#43a047", label="Harmonized")
    ax2.set_title("After Harmonization")
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_harmonization.py -v`
Expected: 6 passed

- [ ] **Step 8: Commit**

```bash
git add harmonization/ tests/test_harmonization.py
git commit -m "feat: harmonization pipeline with bias correction, z-score, and histogram matching"
```

---

### Task 12: Harmonization Route + Before/After UI

**Files:**
- Create: `app/routes/harmonize.py`
- Modify: `app/main.py` (register router)

- [ ] **Step 1: Add harmonization route test to `tests/test_routes.py`**

Append to `tests/test_routes.py`:

```python
def test_harmonize_scan(client):
    patient_id = _create_test_patient(client)
    scan_id = _upload_scan_for_patient(client, patient_id)

    response = client.post(
        f"/scans/{scan_id}/harmonize", follow_redirects=False
    )
    assert response.status_code == 303

    # Verify scan is now harmonized
    response = client.get(f"/scans/{scan_id}")
    assert response.status_code == 200
    assert "Harmonized" in response.text
```

- [ ] **Step 2: Create `app/routes/harmonize.py`**

```python
from pathlib import Path
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from PIL import Image
from sqlmodel import Session

from app.config import settings
from app.database import get_session
from app.models.scan import Scan
from harmonization.pipeline import generate_comparison_histogram, harmonize_2d

router = APIRouter(tags=["harmonization"])


@router.post("/scans/{scan_id}/harmonize")
def harmonize_scan(
    scan_id: UUID, session: Session = Depends(get_session)
):
    scan = session.get(Scan, scan_id)
    if not scan:
        return HTMLResponse("Scan not found", status_code=404)

    # Load image
    image = np.array(Image.open(scan.file_path).convert("RGB"))

    # Run harmonization (no reference for now — uses z-score + bias correction)
    results = harmonize_2d(image, reference=None)
    harmonized = results["harmonized"]

    # Save harmonized image
    scan_dir = Path(scan.file_path).parent
    harmonized_path = scan_dir / "harmonized.png"

    # Rescale to 0-255 for saving
    h_min, h_max = harmonized.min(), harmonized.max()
    if h_max - h_min > 1e-8:
        harmonized_uint8 = (
            (harmonized - h_min) / (h_max - h_min) * 255
        ).astype(np.uint8)
    else:
        harmonized_uint8 = np.zeros_like(harmonized, dtype=np.uint8)

    Image.fromarray(harmonized_uint8).save(harmonized_path)

    # Save comparison histogram
    original_gray = np.mean(image, axis=2) if image.ndim == 3 else image
    histogram_png = generate_comparison_histogram(
        original_gray.astype(np.float32), harmonized
    )
    histogram_path = scan_dir / "histogram_comparison.png"
    histogram_path.write_bytes(histogram_png)

    # Update scan record
    scan.is_harmonized = True
    scan.harmonized_path = str(harmonized_path)
    session.add(scan)
    session.commit()

    return RedirectResponse(url=f"/scans/{scan_id}", status_code=303)


@router.get("/scans/{scan_id}/histogram")
def serve_histogram(scan_id: UUID, session: Session = Depends(get_session)):
    scan = session.get(Scan, scan_id)
    if not scan:
        return HTMLResponse("Not found", status_code=404)
    histogram_path = Path(scan.file_path).parent / "histogram_comparison.png"
    if not histogram_path.exists():
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(histogram_path)
```

- [ ] **Step 3: Register router in `app/main.py`**

```python
from app.routes import patients, scans, predictions, harmonize

app.include_router(patients.router)
app.include_router(scans.router)
app.include_router(predictions.router)
app.include_router(harmonize.router)
```

- [ ] **Step 4: Update `app/templates/scans/detail.html` to show histogram**

Add after the harmonized image article, inside the `{% if scan.is_harmonized %}` block:

```html
    <article>
        <header>Intensity Histogram Comparison</header>
        <img src="/scans/{{ scan.id }}/histogram" alt="Histogram comparison" class="scan-image">
        <footer><small>Before vs after harmonization — aligned histograms mean consistent intensities</small></footer>
    </article>
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_routes.py -v`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add app/routes/harmonize.py app/templates/scans/detail.html app/main.py tests/test_routes.py
git commit -m "feat: harmonization route with before/after histogram visualization"
```

---

## Phase 5: Uncertainty Engine

### Task 13: MC Dropout + Uncertainty Metrics

**Files:**
- Create: `uncertainty/mc_dropout.py`
- Create: `uncertainty/heatmap.py`
- Create: `tests/test_uncertainty.py`

- [ ] **Step 1: Write failing tests for uncertainty**

Create `tests/test_uncertainty.py`:

```python
import numpy as np
import torch

from cancer_detection.model import BrainTumorClassifier


def test_mc_dropout_produces_variance():
    from uncertainty.mc_dropout import mc_dropout_predict

    model = BrainTumorClassifier(num_classes=4)
    x = torch.randn(1, 3, 224, 224)

    result = mc_dropout_predict(model, x, n_passes=10)

    assert "mean_probs" in result
    assert result["mean_probs"].shape == (4,)
    assert "variance" in result
    assert result["variance"].shape == (4,)
    assert "predictive_entropy" in result
    assert "mutual_information" in result
    assert "all_probs" in result
    assert result["all_probs"].shape == (10, 4)

    # Variance should be > 0 (dropout causes variation)
    assert result["variance"].sum() > 0
    # Probabilities should sum to ~1
    assert abs(result["mean_probs"].sum() - 1.0) < 0.05
    # Entropy should be non-negative
    assert result["predictive_entropy"] >= 0


def test_mc_dropout_different_passes_give_different_results():
    from uncertainty.mc_dropout import mc_dropout_predict

    model = BrainTumorClassifier(num_classes=4)
    x = torch.randn(1, 3, 224, 224)

    result = mc_dropout_predict(model, x, n_passes=10)
    # Individual passes should differ
    diffs = np.diff(result["all_probs"], axis=0)
    assert np.abs(diffs).sum() > 0


def test_uncertainty_heatmap_generation():
    from uncertainty.heatmap import generate_uncertainty_bar_chart

    variance = np.array([0.01, 0.15, 0.02, 0.08])
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    png_bytes = generate_uncertainty_bar_chart(variance, class_names)
    assert len(png_bytes) > 100
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_mc_dropout_visualization():
    from uncertainty.heatmap import generate_mc_dropout_visualization

    all_probs = np.random.dirichlet([1, 1, 1, 1], size=30)
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    png_bytes = generate_mc_dropout_visualization(all_probs, class_names)
    assert len(png_bytes) > 100
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_uncertainty.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create `uncertainty/mc_dropout.py`**

```python
import numpy as np
import torch

from cancer_detection.model import BrainTumorClassifier


def enable_mc_dropout(model: BrainTumorClassifier):
    """Enable dropout layers during inference for MC Dropout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: BrainTumorClassifier,
    input_tensor: torch.Tensor,
    n_passes: int = 30,
    device: torch.device | None = None,
) -> dict:
    """Run N forward passes with dropout enabled.

    Returns dict with mean_probs, variance, predictive_entropy,
    mutual_information, and all_probs.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    enable_mc_dropout(model)

    all_probs = []
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        for _ in range(n_passes):
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            all_probs.append(probs.cpu().numpy())

    all_probs = np.array(all_probs)  # (n_passes, num_classes)
    mean_probs = all_probs.mean(axis=0)
    variance = all_probs.var(axis=0)

    # Predictive entropy: H[y|x]
    predictive_entropy = -np.sum(
        mean_probs * np.log(mean_probs + 1e-10)
    )

    # Expected entropy per pass
    per_pass_entropy = -np.sum(
        all_probs * np.log(all_probs + 1e-10), axis=1
    )
    expected_entropy = per_pass_entropy.mean()

    # Mutual information: epistemic uncertainty
    mutual_information = predictive_entropy - expected_entropy

    return {
        "mean_probs": mean_probs,
        "variance": variance,
        "predictive_entropy": float(predictive_entropy),
        "mutual_information": float(mutual_information),
        "all_probs": all_probs,
    }
```

- [ ] **Step 4: Create `uncertainty/heatmap.py`**

```python
import io

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_uncertainty_bar_chart(
    variance: np.ndarray,
    class_names: list[str],
) -> bytes:
    """Generate a horizontal bar chart of per-class uncertainty as PNG bytes."""
    fig, ax = plt.subplots(figsize=(6, 3))
    mean_var = np.mean(variance)
    colors = [
        "#d32f2f" if v > mean_var else "#4caf50" for v in variance
    ]
    ax.barh(class_names, variance, color=colors)
    ax.set_xlabel("Prediction Variance (Uncertainty)")
    ax.set_title("Per-Class Uncertainty (MC Dropout)")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_mc_dropout_visualization(
    all_probs: np.ndarray,
    class_names: list[str],
) -> bytes:
    """Generate a violin plot showing MC Dropout prediction spread as PNG bytes."""
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = list(range(len(class_names)))
    parts = ax.violinplot(
        [all_probs[:, i] for i in positions],
        positions=positions,
        showmeans=True,
        showmedians=True,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Softmax Probability")
    ax.set_title(f"MC Dropout Distribution ({all_probs.shape[0]} passes)")
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_uncertainty.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add uncertainty/mc_dropout.py uncertainty/heatmap.py tests/test_uncertainty.py
git commit -m "feat: MC Dropout uncertainty engine with metrics and visualization"
```

---

### Task 14: Uncertainty Visualization + UI Integration

**Files:**
- Modify: `app/routes/predictions.py` (add uncertainty to prediction flow)
- Modify: `app/templates/predictions/detail.html` (show uncertainty)

- [ ] **Step 1: Add uncertainty integration test to `tests/test_routes.py`**

Append to `tests/test_routes.py`:

```python
def test_prediction_with_uncertainty(client, tmp_path):
    """Test that predictions include uncertainty analysis."""
    from cancer_detection.model import BrainTumorClassifier
    from cancer_detection.inference import InferenceService

    model = BrainTumorClassifier(num_classes=4)
    weights_path = tmp_path / "weights" / "best_model.pth"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, weights_path)
    client.app.state.inference_service = InferenceService(weights_path)

    from app.config import settings
    settings.mc_dropout_passes = 5  # Fewer passes for testing speed

    patient_id = _create_test_patient(client)
    scan_id = _upload_scan_for_patient(client, patient_id)

    # Trigger prediction with uncertainty
    response = client.post(
        f"/scans/{scan_id}/predict?with_uncertainty=true",
        follow_redirects=False,
    )
    assert response.status_code == 303
    redirect_url = response.headers["location"]

    response = client.get(redirect_url)
    assert response.status_code == 200
    # Should show uncertainty metrics
    assert "uncertainty" in response.text.lower() or "entropy" in response.text.lower()
```

- [ ] **Step 2: Update `app/routes/predictions.py` to include uncertainty**

Replace the `trigger_prediction` function:

```python
@router.post("/scans/{scan_id}/predict")
def trigger_prediction(
    request: Request,
    scan_id: UUID,
    with_uncertainty: bool = False,
    session: Session = Depends(get_session),
):
    scan = session.get(Scan, scan_id)
    if not scan:
        return HTMLResponse("Scan not found", status_code=404)

    inference = getattr(request.app.state, "inference_service", None)
    if inference is None:
        return HTMLResponse("Model not loaded", status_code=503)

    # Load image
    image = Image.open(scan.file_path).convert("RGB")

    # Run inference with Grad-CAM
    result = inference.predict_with_gradcam(image)

    scan_dir = Path(scan.file_path).parent

    # Save Grad-CAM overlay
    gradcam_path = None
    if "gradcam_overlay" in result:
        overlay_img = Image.fromarray(
            (result["gradcam_overlay"] * 255).astype(np.uint8)
        )
        gradcam_file = scan_dir / "gradcam.png"
        overlay_img.save(gradcam_file)
        gradcam_path = str(gradcam_file)

    # Run uncertainty analysis if requested
    uncertainty_map_path = None
    if with_uncertainty:
        from cancer_detection.preprocessing import preprocess_single
        from uncertainty.mc_dropout import mc_dropout_predict
        from uncertainty.heatmap import (
            generate_uncertainty_bar_chart,
            generate_mc_dropout_visualization,
        )
        from app.config import settings

        input_tensor = preprocess_single(image).to(inference.device)
        uc_result = mc_dropout_predict(
            inference.model, input_tensor,
            n_passes=settings.mc_dropout_passes,
        )

        display_names = list(settings.display_names.values())

        # Save uncertainty bar chart
        bar_png = generate_uncertainty_bar_chart(
            uc_result["variance"], display_names
        )
        bar_path = scan_dir / "uncertainty_bar.png"
        bar_path.write_bytes(bar_png)

        # Save MC Dropout violin plot
        violin_png = generate_mc_dropout_visualization(
            uc_result["all_probs"], display_names
        )
        violin_path = scan_dir / "uncertainty_violin.png"
        violin_path.write_bytes(violin_png)

        uncertainty_map_path = str(bar_path)

        # Update result with uncertainty metrics
        result["predictive_entropy"] = uc_result["predictive_entropy"]
        result["mutual_information"] = uc_result["mutual_information"]
        result["variance"] = uc_result["variance"].tolist()

    # Create prediction record
    prediction = Prediction(
        scan_id=scan.id,
        prediction_class=result["prediction_class"],
        confidence=result["confidence"],
        probabilities_json=json.dumps(result["probabilities"]),
        gradcam_path=gradcam_path,
        uncertainty_map_path=uncertainty_map_path,
        ran_on_harmonized=scan.is_harmonized,
        inference_time_ms=result["inference_time_ms"],
    )
    session.add(prediction)
    session.commit()
    session.refresh(prediction)

    return RedirectResponse(
        url=f"/predictions/{prediction.id}", status_code=303
    )
```

- [ ] **Step 3: Add uncertainty violin plot route**

Append to `app/routes/predictions.py`:

```python
@router.get("/predictions/{prediction_id}/uncertainty-violin")
def serve_uncertainty_violin(
    prediction_id: UUID, session: Session = Depends(get_session)
):
    prediction = session.get(Prediction, prediction_id)
    if not prediction or not prediction.uncertainty_map_path:
        return HTMLResponse("Not found", status_code=404)
    violin_path = Path(prediction.uncertainty_map_path).parent / "uncertainty_violin.png"
    if not violin_path.exists():
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(violin_path)
```

- [ ] **Step 4: Update `app/templates/predictions/detail.html` uncertainty section**

Replace the existing uncertainty block at the bottom of the template with:

```html
{% if prediction.uncertainty_map_path %}
<h2>Uncertainty Analysis</h2>
<p>MC Dropout uncertainty quantification — shows where the model is confident vs. unsure.</p>
<div class="grid">
    <article>
        <header>Per-Class Uncertainty</header>
        <img src="/predictions/{{ prediction.id }}/uncertainty-map" alt="Uncertainty bar chart" class="scan-image">
        <footer><small>Red = high variance (model unsure), Green = low variance (model confident)</small></footer>
    </article>
    <article>
        <header>MC Dropout Distribution</header>
        <img src="/predictions/{{ prediction.id }}/uncertainty-violin" alt="MC Dropout violin plot" class="scan-image">
        <footer><small>Spread of predictions across {{ 30 }} forward passes with dropout</small></footer>
    </article>
</div>
{% endif %}
```

- [ ] **Step 5: Update scan detail template to offer uncertainty option**

In `app/templates/scans/detail.html`, update the prediction form to include the uncertainty toggle:

```html
    {% if model_loaded %}
    <form method="post" action="/scans/{{ scan.id }}/predict?with_uncertainty=true">
        <button type="submit">Run Prediction + Uncertainty</button>
    </form>
    <form method="post" action="/scans/{{ scan.id }}/predict">
        <button type="submit" class="outline">Quick Prediction (no uncertainty)</button>
    </form>
    {% else %}
    <p><small>Model not loaded — train the model first to enable predictions.</small></p>
    {% endif %}
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_routes.py -v`
Expected: 11 passed

- [ ] **Step 7: Commit**

```bash
git add app/routes/predictions.py app/templates/predictions/detail.html app/templates/scans/detail.html tests/test_routes.py
git commit -m "feat: uncertainty visualization with MC Dropout bar charts and violin plots"
```

---

## Phase 6: Integration & Validation

### Task 15: End-to-End Integration

**Files:**
- Create: `app/routes/viewer.py`
- Create: `app/templates/viewer/viewer.html`
- Modify: `app/main.py` (register viewer router)
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
"""End-to-end workflow test: upload -> harmonize -> predict -> view."""
import io

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from app.config import settings


@pytest.fixture
def full_client(tmp_path):
    """Client with a mock model loaded."""
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
        yield client


def test_full_workflow(full_client):
    client = full_client

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
    img = PILImage.fromarray(
        np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/scans/",
        data={
            "patient_id": patient_id,
            "scanner_vendor": "Philips",
            "modality": "T1",
        },
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
    response = client.post(
        f"/scans/{scan_id}/harmonize", follow_redirects=False
    )
    assert response.status_code == 303

    # Verify harmonized
    response = client.get(scan_url)
    assert "Harmonized" in response.text or "harmonized" in response.text.lower()

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
    # Should contain prediction class and Grad-CAM
    assert "Grad-CAM" in response.text or "gradcam" in response.text.lower()

    # 7. Serve Grad-CAM image
    pred_id = prediction_url.split("/")[-1]
    response = client.get(f"/predictions/{pred_id}/gradcam")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")

    # 8. Health check
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
```

- [ ] **Step 2: Create `app/routes/viewer.py`**

```python
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlmodel import Session

from app.database import get_session
from app.deps import templates
from app.models.scan import Scan

router = APIRouter(prefix="/viewer", tags=["viewer"])


@router.get("/{scan_id}", response_class=HTMLResponse)
def viewer(
    request: Request, scan_id: UUID, session: Session = Depends(get_session)
):
    scan = session.get(Scan, scan_id)
    if not scan:
        return HTMLResponse("Scan not found", status_code=404)
    return templates.TemplateResponse(
        "viewer/viewer.html",
        {
            "request": request,
            "scan": scan,
            "predictions": scan.predictions,
        },
    )
```

- [ ] **Step 3: Create `app/templates/viewer/viewer.html`**

```html
{% extends "base.html" %}
{% block title %}Viewer — {{ scan.modality }}{% endblock %}
{% block content %}
<h1>MRI Viewer</h1>
<p>{{ scan.modality }} scan from {{ scan.scanner_vendor }} scanner</p>

<div class="overlay-grid">
    <div>
        <h3>Original</h3>
        <img src="/scans/{{ scan.id }}/image" alt="Original scan" class="scan-image" id="original">
    </div>

    {% if scan.is_harmonized %}
    <div>
        <h3>Harmonized</h3>
        <img src="/scans/{{ scan.id }}/harmonized-image" alt="Harmonized scan" class="scan-image" id="harmonized">
    </div>
    {% endif %}

    {% for pred in predictions %}
    {% if pred.gradcam_path %}
    <div>
        <h3>Grad-CAM</h3>
        <img src="/predictions/{{ pred.id }}/gradcam" alt="Grad-CAM overlay" class="scan-image" id="gradcam">
        <small>{{ pred.prediction_class | replace('notumor', 'No Tumor') | title }}
               ({{ (pred.confidence * 100) | round(1) }}%)</small>
    </div>
    {% endif %}

    {% if pred.uncertainty_map_path %}
    <div>
        <h3>Uncertainty</h3>
        <img src="/predictions/{{ pred.id }}/uncertainty-map" alt="Uncertainty" class="scan-image" id="uncertainty">
    </div>
    {% endif %}
    {% endfor %}
</div>

{% if scan.is_harmonized %}
<article>
    <header>Histogram Comparison</header>
    <img src="/scans/{{ scan.id }}/histogram" alt="Histogram comparison" class="scan-image">
</article>
{% endif %}

{% for pred in predictions %}
{% if pred.uncertainty_map_path %}
<article>
    <header>MC Dropout Distribution</header>
    <img src="/predictions/{{ pred.id }}/uncertainty-violin" alt="MC Dropout distribution" class="scan-image">
</article>
{% endif %}
{% endfor %}

<a href="/scans/{{ scan.id }}" role="button" class="outline">Back to Scan</a>
{% endblock %}
```

- [ ] **Step 4: Register viewer router in `app/main.py`**

```python
from app.routes import patients, scans, predictions, harmonize, viewer

app.include_router(patients.router)
app.include_router(scans.router)
app.include_router(predictions.router)
app.include_router(harmonize.router)
app.include_router(viewer.router)
```

- [ ] **Step 5: Add viewer link to scan detail template**

In `app/templates/scans/detail.html`, add near the action buttons:

```html
    <a href="/viewer/{{ scan.id }}" role="button" class="outline">Open Viewer</a>
```

- [ ] **Step 6: Run integration tests**

Run: `uv run pytest tests/test_integration.py -v`
Expected: 1 passed (covers entire workflow)

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (25+ tests across all files)

- [ ] **Step 8: Commit**

```bash
git add app/routes/viewer.py app/templates/viewer/ app/templates/scans/detail.html app/main.py tests/test_integration.py
git commit -m "feat: end-to-end integration with viewer and full workflow test"
```

---

### Task 16: Validation Suite

**Files:**
- Create: `scripts/validate.py`

- [ ] **Step 1: Create `scripts/validate.py`**

```python
"""Validation suite: checks all success criteria from the spec."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def check_model_exists() -> bool:
    path = Path("data/weights/best_model.pth")
    exists = path.exists()
    print(f"[{'PASS' if exists else 'FAIL'}] Model checkpoint exists: {path}")
    return exists


def check_classification_accuracy() -> bool:
    from cancer_detection.training.config import TrainingConfig
    from cancer_detection.training.evaluate import evaluate

    config = TrainingConfig()
    if not Path(config.output_dir, "best_model.pth").exists():
        print("[SKIP] No trained model — cannot evaluate accuracy")
        return True
    if not Path(config.data_dir, "Testing").exists():
        print("[SKIP] No test dataset — cannot evaluate accuracy")
        return True

    results = evaluate(config)
    passed = results["accuracy"] >= 0.96
    print(
        f"[{'PASS' if passed else 'FAIL'}] Test accuracy: "
        f"{results['accuracy']:.4f} (target: >= 0.96)"
    )
    f1_passed = results["weighted_f1"] >= 0.95
    print(
        f"[{'PASS' if f1_passed else 'FAIL'}] Weighted F1: "
        f"{results['weighted_f1']:.4f} (target: >= 0.95)"
    )
    return passed and f1_passed


def check_inference_latency() -> bool:
    if not Path("data/weights/best_model.pth").exists():
        print("[SKIP] No trained model — cannot check latency")
        return True

    from cancer_detection.inference import InferenceService

    service = InferenceService("data/weights/best_model.pth")
    img = Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    )

    # Warm up
    service.predict(img)

    # Measure
    times = []
    for _ in range(10):
        start = time.time()
        service.predict(img)
        times.append((time.time() - start) * 1000)

    avg_ms = np.mean(times)
    passed = avg_ms < 500
    print(
        f"[{'PASS' if passed else 'FAIL'}] Inference latency: "
        f"{avg_ms:.0f}ms avg (target: < 500ms)"
    )
    return passed


def check_harmonization_pipeline() -> bool:
    from harmonization.pipeline import harmonize_2d

    arr = np.random.uniform(50, 200, (128, 128)).astype(np.float32)
    ref = np.random.uniform(80, 220, (128, 128)).astype(np.float32)

    results = harmonize_2d(arr, reference=ref)
    passed = all(
        k in results
        for k in ["original", "bias_corrected", "normalized", "harmonized"]
    )
    print(
        f"[{'PASS' if passed else 'FAIL'}] Harmonization pipeline produces "
        f"all intermediate results"
    )
    return passed


def check_uncertainty_engine() -> bool:
    from uncertainty.mc_dropout import mc_dropout_predict

    from cancer_detection.model import BrainTumorClassifier

    model = BrainTumorClassifier(num_classes=4)
    x = torch.randn(1, 3, 224, 224)
    result = mc_dropout_predict(model, x, n_passes=10)

    passed = result["variance"].sum() > 0
    print(
        f"[{'PASS' if passed else 'FAIL'}] MC Dropout produces non-zero "
        f"variance"
    )

    entropy_ok = result["predictive_entropy"] >= 0
    print(
        f"[{'PASS' if entropy_ok else 'FAIL'}] Predictive entropy "
        f"is non-negative: {result['predictive_entropy']:.4f}"
    )
    return passed and entropy_ok


def check_crud_server() -> bool:
    try:
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            health = client.get("/api/health")
            passed = health.status_code == 200
            print(
                f"[{'PASS' if passed else 'FAIL'}] Health endpoint "
                f"returns 200"
            )

            patients = client.get("/patients/")
            patients_ok = patients.status_code == 200
            print(
                f"[{'PASS' if patients_ok else 'FAIL'}] Patients "
                f"list returns 200"
            )
            return passed and patients_ok
    except Exception as e:
        print(f"[FAIL] Server check failed: {e}")
        return False


def main():
    print("=" * 60)
    print("MRI Platform Validation Suite")
    print("=" * 60)

    checks = [
        ("Model checkpoint exists", check_model_exists),
        ("Classification accuracy >= 96%", check_classification_accuracy),
        ("Inference latency < 500ms", check_inference_latency),
        ("Harmonization pipeline", check_harmonization_pipeline),
        ("Uncertainty engine", check_uncertainty_engine),
        ("CRUD server health", check_crud_server),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\n--- {name} ---")
        try:
            results.append(check_fn())
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} checks passed")
    print("=" * 60)

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run validation**

Run: `uv run python scripts/validate.py`
Expected: All checks pass (or SKIP for checks requiring trained model/dataset).

- [ ] **Step 3: Run full test suite one final time**

Run: `uv run pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/validate.py
git commit -m "feat: validation suite checking all success criteria"
```

---

## Summary

| Phase | Tasks | What It Delivers |
|-------|-------|-----------------|
| 1. Foundation | 1-2 | Project structure, database, models |
| 2. Cancer Detection | 3-6 | Trained EfficientNet-B0 with Grad-CAM + inference |
| 3. Web Platform | 7-10 | Full CRUD with HTMX, prediction workflow |
| 4. Harmonization | 11-12 | Bias correction, z-score, histogram matching + UI |
| 5. Uncertainty | 13-14 | MC Dropout + bar chart/violin plot visualization |
| 6. Integration | 15-16 | End-to-end workflow, viewer, validation |

**Total: 16 tasks, ~80 steps, 15 source files, 7 test files**
