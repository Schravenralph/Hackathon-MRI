import json
from uuid import UUID
from datetime import datetime, timezone
from sqlmodel import Session, select
from app.models.patient import Patient, PatientCreate, PatientUpdate
from app.models.scan import Scan, ScanCreate
from app.models.prediction import Prediction


# ---------------------------------------------------------------------------
# Patient tests
# ---------------------------------------------------------------------------

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


def test_patient_update_schema():
    update = PatientUpdate(name="Updated Name")
    assert update.name == "Updated Name"
    assert update.age is None
    assert update.notes is None


def test_read_patient(db_session: Session):
    patient = Patient(name="Read Test", age=50)
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    fetched = db_session.get(Patient, patient.id)
    assert fetched is not None
    assert fetched.id == patient.id
    assert fetched.name == "Read Test"
    assert fetched.age == 50


def test_update_patient(db_session: Session):
    patient = Patient(name="Original Name", age=40)
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    patient.name = "Updated Name"
    patient.age = 41
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    assert patient.name == "Updated Name"
    assert patient.age == 41


def test_delete_patient(db_session: Session):
    patient = Patient(name="To Delete", age=35)
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)
    patient_id = patient.id

    db_session.delete(patient)
    db_session.commit()

    fetched = db_session.get(Patient, patient_id)
    assert fetched is None


def test_list_patients(db_session: Session):
    db_session.add(Patient(name="Alice", age=30))
    db_session.add(Patient(name="Bob", age=25))
    db_session.commit()

    patients = db_session.exec(select(Patient)).all()
    assert len(patients) == 2
    names = {p.name for p in patients}
    assert "Alice" in names
    assert "Bob" in names


# ---------------------------------------------------------------------------
# Scan tests
# ---------------------------------------------------------------------------

def test_create_scan(db_session: Session):
    patient = Patient(name="Scan Owner", age=60)
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    scan = Scan(
        patient_id=patient.id,
        scanner_vendor="Siemens",
        modality="T2",
        file_format="dicom",
        file_path="/data/uploads/scan1.dcm",
    )
    db_session.add(scan)
    db_session.commit()
    db_session.refresh(scan)

    assert isinstance(scan.id, UUID)
    assert scan.patient_id == patient.id
    assert scan.scanner_vendor == "Siemens"
    assert scan.modality == "T2"
    assert scan.file_format == "dicom"
    assert scan.is_harmonized is False
    assert scan.harmonized_path is None
    assert isinstance(scan.uploaded_at, datetime)


def test_scan_create_schema():
    from uuid import uuid4
    patient_id = uuid4()
    data = ScanCreate(patient_id=patient_id, scanner_vendor="Philips", modality="T1", file_format="jpeg")
    assert data.patient_id == patient_id
    assert data.scanner_vendor == "Philips"


def test_scan_defaults(db_session: Session):
    patient = Patient(name="Default Scan Owner", age=55)
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    scan = Scan(patient_id=patient.id)
    db_session.add(scan)
    db_session.commit()
    db_session.refresh(scan)

    assert scan.scanner_vendor == "Unknown"
    assert scan.modality == "T1"
    assert scan.file_format == "jpeg"
    assert scan.file_path == ""
    assert scan.is_harmonized is False


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------

def test_create_prediction(db_session: Session):
    patient = Patient(name="Pred Owner", age=70)
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    scan = Scan(patient_id=patient.id, file_path="/data/uploads/scan2.dcm")
    db_session.add(scan)
    db_session.commit()
    db_session.refresh(scan)

    probs = {"glioma": 0.7, "meningioma": 0.1, "notumor": 0.1, "pituitary": 0.1}
    prediction = Prediction(
        scan_id=scan.id,
        model_name="efficientnet_b0",
        prediction_class="glioma",
        confidence=0.7,
        probabilities_json=json.dumps(probs),
        inference_time_ms=250,
    )
    db_session.add(prediction)
    db_session.commit()
    db_session.refresh(prediction)

    assert isinstance(prediction.id, UUID)
    assert prediction.scan_id == scan.id
    assert prediction.prediction_class == "glioma"
    assert prediction.confidence == 0.7
    assert prediction.inference_time_ms == 250
    assert isinstance(prediction.created_at, datetime)


def test_prediction_probabilities_property(db_session: Session):
    patient = Patient(name="Prob Owner", age=65)
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    scan = Scan(patient_id=patient.id, file_path="/data/uploads/scan3.dcm")
    db_session.add(scan)
    db_session.commit()
    db_session.refresh(scan)

    probs = {"glioma": 0.8, "meningioma": 0.05, "notumor": 0.1, "pituitary": 0.05}
    prediction = Prediction(
        scan_id=scan.id,
        prediction_class="glioma",
        confidence=0.8,
        probabilities_json=json.dumps(probs),
    )
    db_session.add(prediction)
    db_session.commit()
    db_session.refresh(prediction)

    result = prediction.probabilities
    assert isinstance(result, dict)
    assert result["glioma"] == 0.8
    assert result["meningioma"] == 0.05


def test_prediction_defaults(db_session: Session):
    patient = Patient(name="Default Pred Owner", age=48)
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    scan = Scan(patient_id=patient.id, file_path="/data/uploads/scan4.dcm")
    db_session.add(scan)
    db_session.commit()
    db_session.refresh(scan)

    prediction = Prediction(
        scan_id=scan.id,
        prediction_class="notumor",
        confidence=0.9,
    )
    db_session.add(prediction)
    db_session.commit()
    db_session.refresh(prediction)

    assert prediction.model_name == "efficientnet_b0"
    assert prediction.probabilities_json == "{}"
    assert prediction.uncertainty_map_path is None
    assert prediction.gradcam_path is None
    assert prediction.ran_on_harmonized is False
    assert prediction.inference_time_ms == 0


# ---------------------------------------------------------------------------
# Cascade delete tests
# ---------------------------------------------------------------------------

def test_cascade_delete_patient_deletes_scans(db_session: Session):
    patient = Patient(name="Cascade Test", age=55)
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    scan1 = Scan(patient_id=patient.id, file_path="/data/uploads/s1.dcm")
    scan2 = Scan(patient_id=patient.id, file_path="/data/uploads/s2.dcm")
    db_session.add(scan1)
    db_session.add(scan2)
    db_session.commit()
    db_session.refresh(scan1)
    db_session.refresh(scan2)
    scan1_id = scan1.id
    scan2_id = scan2.id

    db_session.delete(patient)
    db_session.commit()

    assert db_session.get(Scan, scan1_id) is None
    assert db_session.get(Scan, scan2_id) is None


def test_cascade_delete_scan_deletes_predictions(db_session: Session):
    patient = Patient(name="Cascade Pred Test", age=62)
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    scan = Scan(patient_id=patient.id, file_path="/data/uploads/scan5.dcm")
    db_session.add(scan)
    db_session.commit()
    db_session.refresh(scan)

    pred = Prediction(
        scan_id=scan.id,
        prediction_class="pituitary",
        confidence=0.65,
    )
    db_session.add(pred)
    db_session.commit()
    db_session.refresh(pred)
    pred_id = pred.id

    db_session.delete(scan)
    db_session.commit()

    assert db_session.get(Prediction, pred_id) is None


def test_full_cascade_delete(db_session: Session):
    """Deleting a Patient cascades to Scans then Predictions."""
    patient = Patient(name="Full Cascade", age=72)
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    scan = Scan(patient_id=patient.id, file_path="/data/uploads/scan6.dcm")
    db_session.add(scan)
    db_session.commit()
    db_session.refresh(scan)

    pred = Prediction(
        scan_id=scan.id,
        prediction_class="glioma",
        confidence=0.75,
    )
    db_session.add(pred)
    db_session.commit()
    db_session.refresh(pred)
    pred_id = pred.id
    scan_id = scan.id

    db_session.delete(patient)
    db_session.commit()

    assert db_session.get(Scan, scan_id) is None
    assert db_session.get(Prediction, pred_id) is None
