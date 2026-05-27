import shutil
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from sqlmodel import Session, select

from app.config import settings
from app.database import get_session
from app.deps import templates
from app.models.scan import Scan
from app.security import safe_db_path, safe_filename, safe_join

router = APIRouter(prefix="/scans", tags=["scans"])


@router.post("/")
async def scan_upload(
    request: Request,
    patient_id: UUID = Form(...),
    scanner_vendor: str = Form("Unknown"),
    modality: str = Form("T1"),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    # Strip any directory component the client put in `filename` — this
    # is the primary path-traversal sink (Aikido scans.py:46).
    filename = safe_filename(file.filename, default="scan")
    suffix = Path(filename).suffix.lstrip(".").lower() or "jpeg"

    # Create scan record first to get the ID
    scan = Scan(
        patient_id=patient_id,
        scanner_vendor=scanner_vendor,
        modality=modality,
        file_format=suffix,
    )
    session.add(scan)
    session.commit()
    session.refresh(scan)

    # Save file to upload_dir/{scan_id}/{filename}; `safe_join` is
    # belt-and-braces — `safe_filename` already removed separators.
    scan_dir = settings.upload_dir / str(scan.id)
    scan_dir.mkdir(parents=True, exist_ok=True)
    dest = safe_join(scan_dir, filename)
    contents = await file.read()
    dest.write_bytes(contents)

    # Update the scan with the file path
    scan.file_path = str(dest)
    session.add(scan)
    session.commit()

    return RedirectResponse(url=f"/scans/{scan.id}", status_code=303)


@router.get("/{scan_id}")
def scan_detail(scan_id: UUID, request: Request, session: Session = Depends(get_session)):
    scan = session.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    # Eagerly load patient for breadcrumb
    patient = scan.patient
    predictions = scan.predictions
    return templates.TemplateResponse(
        request,
        "scans/detail.html",
        {"scan": scan, "patient": patient, "predictions": predictions},
    )


@router.get("/{scan_id}/image")
def scan_image(scan_id: UUID, session: Session = Depends(get_session)):
    scan = session.get(Scan, scan_id)
    if not scan or not scan.file_path:
        raise HTTPException(status_code=404, detail="Scan image not found")
    path = safe_db_path(scan.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Scan image file missing")
    return FileResponse(path)


@router.get("/{scan_id}/harmonized-image")
def scan_harmonized_image(scan_id: UUID, session: Session = Depends(get_session)):
    scan = session.get(Scan, scan_id)
    if not scan or not scan.harmonized_path:
        raise HTTPException(status_code=404, detail="Harmonized image not found")
    path = safe_db_path(scan.harmonized_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Harmonized image file missing")
    return FileResponse(path)


@router.post("/{scan_id}/delete")
def scan_delete(scan_id: UUID, session: Session = Depends(get_session)):
    scan = session.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    patient_id = scan.patient_id

    # Remove files from disk
    scan_dir = settings.upload_dir / str(scan_id)
    if scan_dir.exists():
        shutil.rmtree(scan_dir)

    if scan.harmonized_path:
        harmonized = Path(scan.harmonized_path)
        if harmonized.exists():
            harmonized.unlink()

    session.delete(scan)
    session.commit()

    return RedirectResponse(url=f"/patients/{patient_id}", status_code=303)
