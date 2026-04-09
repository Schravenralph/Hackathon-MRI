from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.deps import templates
from app.routes import patients, scans


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.harmonized_dir.mkdir(parents=True, exist_ok=True)
    settings.weights_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(settings.database_url.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    init_db()

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
app.include_router(patients.router)
app.include_router(scans.router)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": getattr(app.state, "inference_service", None) is not None,
    }
