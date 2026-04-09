from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlmodel import Session

from app.database import get_session
from app.deps import templates
from app.models.scan import Scan

router = APIRouter(prefix="/viewer", tags=["viewer"])


@router.get("/{scan_id}", response_class=HTMLResponse)
def viewer(request: Request, scan_id: UUID, session: Session = Depends(get_session)):
    scan = session.get(Scan, scan_id)
    if not scan:
        return HTMLResponse("Scan not found", status_code=404)
    return templates.TemplateResponse(
        request,
        "viewer/viewer.html",
        {"scan": scan, "predictions": scan.predictions},
    )
