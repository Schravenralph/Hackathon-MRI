from pathlib import Path
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from PIL import Image
from sqlmodel import Session

from app.config import settings
from app.database import get_session
from app.models.scan import Scan
from app.security import safe_db_path
from harmonization.pipeline import generate_comparison_histogram, harmonize_2d

router = APIRouter(tags=["harmonization"])


@router.post("/scans/{scan_id}/harmonize")
def harmonize_scan(scan_id: UUID, session: Session = Depends(get_session)):
    scan = session.get(Scan, scan_id)
    if not scan:
        return HTMLResponse("Scan not found", status_code=404)

    scan_file = safe_db_path(scan.file_path)
    image = np.array(Image.open(scan_file).convert("RGB"))
    results = harmonize_2d(image, reference=None)
    harmonized = results["harmonized"]

    scan_dir = scan_file.parent
    harmonized_path = scan_dir / "harmonized.png"

    # Rescale to 0-255 for saving
    h_min, h_max = harmonized.min(), harmonized.max()
    if h_max - h_min > 1e-8:
        harmonized_uint8 = ((harmonized - h_min) / (h_max - h_min) * 255).astype(np.uint8)
    else:
        harmonized_uint8 = np.zeros_like(harmonized, dtype=np.uint8)

    Image.fromarray(harmonized_uint8).save(harmonized_path)

    # Save histogram
    original_gray = np.mean(image, axis=2) if image.ndim == 3 else image
    histogram_png = generate_comparison_histogram(original_gray.astype(np.float32), harmonized)
    histogram_path = scan_dir / "histogram_comparison.png"
    histogram_path.write_bytes(histogram_png)

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
    scan_file = safe_db_path(scan.file_path)
    histogram_path = scan_file.parent / "histogram_comparison.png"
    if not histogram_path.exists():
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(histogram_path)
