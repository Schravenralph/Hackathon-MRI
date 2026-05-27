import json
from pathlib import Path
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image
from sqlmodel import Session

from app.config import settings
from app.database import get_session
from app.deps import templates
from app.models.prediction import Prediction
from app.models.scan import Scan


def _safe_under_uploads(stored: str) -> Path:
    """Reject DB-stored paths that escape `settings.upload_dir`.

    Defence-in-depth for FileResponse / Image.open sinks: every artefact
    is written by trusted server code under `upload_dir`, so anything
    pointing outside is treated as a tampered row.
    """
    upload_root = settings.upload_dir.resolve()
    try:
        resolved = Path(stored).resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid stored path") from exc
    if not (resolved == upload_root or upload_root in resolved.parents):
        raise HTTPException(status_code=400, detail="Invalid stored path")
    return resolved

router = APIRouter(tags=["predictions"])


@router.post("/scans/{scan_id}/predict")
def trigger_prediction(
    scan_id: UUID,
    request: Request,
    with_uncertainty: bool = False,
    session: Session = Depends(get_session),
):
    scan = session.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    inference_service = request.app.state.inference_service
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service not available")

    raw_path = scan.harmonized_path if scan.is_harmonized and scan.harmonized_path else scan.file_path
    image_path = _safe_under_uploads(raw_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Scan image file not found")

    image = Image.open(image_path)
    result = inference_service.predict_with_gradcam(image)

    # Save Grad-CAM overlay PNG to scan dir
    scan_dir = settings.upload_dir / str(scan_id)
    scan_dir.mkdir(parents=True, exist_ok=True)
    gradcam_file = scan_dir / "gradcam.png"
    Image.fromarray((result["gradcam_overlay"] * 255).astype(np.uint8)).save(gradcam_file)

    uncertainty_map_path = None
    if with_uncertainty:
        from cancer_detection.preprocessing import preprocess_single
        from uncertainty.mc_dropout import mc_dropout_predict
        from uncertainty.heatmap import generate_uncertainty_bar_chart, generate_mc_dropout_visualization

        input_tensor = preprocess_single(image.convert("RGB")).to(inference_service.device)
        uc_result = mc_dropout_predict(
            inference_service.model, input_tensor,
            n_passes=settings.mc_dropout_passes,
        )
        display_names = list(settings.display_names.values())

        bar_png = generate_uncertainty_bar_chart(uc_result["variance"], display_names)
        bar_path = scan_dir / "uncertainty_bar.png"
        bar_path.write_bytes(bar_png)

        violin_png = generate_mc_dropout_visualization(uc_result["all_probs"], display_names)
        violin_path = scan_dir / "uncertainty_violin.png"
        violin_path.write_bytes(violin_png)

        uncertainty_map_path = str(bar_path)

    prediction = Prediction(
        scan_id=scan_id,
        prediction_class=result["prediction_class"],
        confidence=result["confidence"],
        probabilities_json=json.dumps(result["probabilities"]),
        gradcam_path=str(gradcam_file),
        ran_on_harmonized=scan.is_harmonized,
        inference_time_ms=result["inference_time_ms"],
        uncertainty_map_path=uncertainty_map_path,
    )
    session.add(prediction)
    session.commit()
    session.refresh(prediction)

    return RedirectResponse(url=f"/predictions/{prediction.id}", status_code=303)


@router.get("/predictions/{prediction_id}", response_class=None)
def prediction_detail(
    prediction_id: UUID,
    request: Request,
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    scan = prediction.scan
    return templates.TemplateResponse(
        request,
        "predictions/detail.html",
        {"prediction": prediction, "scan": scan},
    )


@router.get("/predictions/{prediction_id}/gradcam")
def prediction_gradcam(prediction_id: UUID, session: Session = Depends(get_session)):
    prediction = session.get(Prediction, prediction_id)
    if not prediction or not prediction.gradcam_path:
        raise HTTPException(status_code=404, detail="Grad-CAM image not found")
    path = _safe_under_uploads(prediction.gradcam_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Grad-CAM file missing")
    return FileResponse(path, media_type="image/png")


@router.get("/predictions/{prediction_id}/uncertainty-map")
def prediction_uncertainty_map(
    prediction_id: UUID, session: Session = Depends(get_session)
):
    prediction = session.get(Prediction, prediction_id)
    if not prediction or not prediction.uncertainty_map_path:
        raise HTTPException(status_code=404, detail="Uncertainty map not found")
    path = _safe_under_uploads(prediction.uncertainty_map_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Uncertainty map file missing")
    return FileResponse(path, media_type="image/png")


@router.get("/predictions/{prediction_id}/uncertainty-violin")
def serve_uncertainty_violin(prediction_id: UUID, session: Session = Depends(get_session)):
    from fastapi.responses import HTMLResponse

    prediction = session.get(Prediction, prediction_id)
    if not prediction or not prediction.uncertainty_map_path:
        return HTMLResponse("Not found", status_code=404)
    bar_path = _safe_under_uploads(prediction.uncertainty_map_path)
    violin_path = bar_path.parent / "uncertainty_violin.png"
    if not violin_path.exists():
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(violin_path)
