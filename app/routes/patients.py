from uuid import UUID

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, select

from app.database import get_session
from app.deps import templates
from app.models.patient import Patient

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("/")
def patient_list(request: Request, session: Session = Depends(get_session)):
    patients = session.exec(select(Patient)).all()
    return templates.TemplateResponse(request, "patients/list.html", {"patients": patients})


@router.get("/new")
def patient_new(request: Request):
    return templates.TemplateResponse(request, "patients/_form.html", {"patient": None})


@router.post("/")
def patient_create(
    request: Request,
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


@router.get("/{patient_id}")
def patient_detail(patient_id: UUID, request: Request, session: Session = Depends(get_session)):
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return templates.TemplateResponse(
        request,
        "patients/detail.html",
        {"patient": patient, "scans": patient.scans},
    )


@router.get("/{patient_id}/edit")
def patient_edit_form(patient_id: UUID, request: Request, session: Session = Depends(get_session)):
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return templates.TemplateResponse(request, "patients/_form.html", {"patient": patient})


@router.post("/{patient_id}/edit")
def patient_update(
    patient_id: UUID,
    request: Request,
    name: str = Form(...),
    age: str = Form(""),
    notes: str = Form(""),
    session: Session = Depends(get_session),
):
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient.name = name
    patient.age = int(age) if age.strip() else None
    patient.notes = notes if notes.strip() else None
    session.add(patient)
    session.commit()
    return RedirectResponse(url=f"/patients/{patient_id}", status_code=303)


@router.post("/{patient_id}/delete")
def patient_delete(patient_id: UUID, session: Session = Depends(get_session)):
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    session.delete(patient)
    session.commit()
    return RedirectResponse(url="/patients/", status_code=303)
