# MRI Harmonization & Uncertainty Visualization Platform

**Date:** 2026-04-09
**Status:** Draft
**Scope:** Full-stack hackathon project — scanner-agnostic harmonization + radiologist-in-the-loop uncertainty visualization
**Philosophy:** Don't compete on accuracy. Compete on workflow fit, robustness, interpretability, and domain-smartness.

---

## 1. Vision & Positioning

### What We're Building

A two-pillar MRI toolkit that solves real workflow problems for Dutch radiologists, not another accuracy-chasing classifier.

**Pillar 1 — Scanner-Agnostic Harmonization Layer:** A preprocessing pipeline that normalizes MRI scans across Philips/Siemens/GE vendors so any downstream AI model works reliably on any scanner's output. The demo: same pretrained model, dramatic improvement on out-of-distribution scans.

**Pillar 2 — Radiologist-in-the-Loop Uncertainty Viewer:** An interactive web viewer that overlays uncertainty heatmaps on MRI scans, showing *where the model is confident* vs. *where it's guessing*. Uses Monte Carlo dropout on a pretrained model. Not a model innovation — a workflow innovation.

### Why This Stands Out

- **Not competing on accuracy** — leaderboards are saturated and margins are noise
- **Solving the real blocker** — scanner harmonization is what the AIFI national infrastructure project is hitting
- **Workflow innovation over model innovation** — judges who understand the domain will recognize this
- **Affordable and explainable** — no expensive compute, radiologists can see *why*
- **Dutch/European context** — BraTS includes European sites, AIFI is Dutch, we reference the local scanner landscape

### Demo Narrative

> "A radiologist at Radboudumc receives scans from three different hospitals, all running different scanners — Philips, Siemens, GE. Our harmonization layer normalizes them so the AI works equally well on all three. When the model flags a region, our uncertainty viewer shows exactly where it's confident and where it needs human eyes. The radiologist trusts the result because they can *see* why."

### Rejected Ideas (Stored for Future)

| Idea | Reason for Deferral |
|------|---------------------|
| Missing Modality Synthesis / Graceful Degradation | Requires BraSyn dataset and conditional synthesis model — too complex for MVP |
| Automated PI-RADS Reporting Assistant | Prostate-specific, requires ProstateX/PI-CAI data pipeline |
| Smart Triage / Worklist Prioritization | Strong idea but needs real clinical volume data |
| Cross-Scanner QA / Drift Detection Dashboard | MLOps focus, better as Phase 2 extension |

---

## 2. Architecture

```
+-------------------------------------------------------------+
|                  Web App (FastAPI + HTMX)                     |
|  +------------+  +-------------+  +------------------------+ |
|  | Upload     |  | CRUD        |  | NiiVue Viewer          | |
|  | & Ingest   |  | (Scans,     |  | (uncertainty overlay,  | |
|  |            |  | Patients,   |  |  heatmap toggle,       | |
|  |            |  | Results)    |  |  threshold slider)     | |
|  +-----+------+  +------+------+  +-----------+------------+ |
|        |                |                      |              |
|  +-----v----------------v----------------------v-----------+  |
|  |                 FastAPI Backend                          |  |
|  |  /api/scans  /api/patients  /api/predictions            |  |
|  |  /api/harmonize  /api/predict  /api/health              |  |
|  +-----+-----------------------------------+---------------+  |
|        |                                   |                  |
|  +-----v-----------------+  +--------------v---------------+  |
|  | Harmonization         |  | Uncertainty Engine           |  |
|  | Pipeline              |  | (MC Dropout / TTA)           |  |
|  | +------------------+  |  | +-------------------------+  |  |
|  | | Bias Field       |  |  | | Pretrained model        |  |  |
|  | | Correction (N4)  |  |  | | (EfficientNet-B0 or     |  |  |
|  | +------------------+  |  | |  nnU-Net for segm.)     |  |  |
|  | | Intensity        |  |  | +-------------------------+  |  |
|  | | Normalization    |  |  | | N forward passes        |  |  |
|  | +------------------+  |  | | (dropout enabled)       |  |  |
|  | | Histogram        |  |  | +-------------------------+  |  |
|  | | Matching         |  |  | | Variance -> heatmap     |  |  |
|  | +------------------+  |  | | + mean prediction       |  |  |
|  | | WhiteStripe      |  |  | +-------------------------+  |  |
|  | | (optional)       |  |  | | Grad-CAM overlay        |  |  |
|  | +------------------+  |  | +-------------------------+  |  |
|  +-----------------------+  +------------------------------+  |
|                                                               |
|  +-----------------------------------------------------------+|
|  | SQLite (SQLModel) -- scans, patients, predictions          ||
|  | Filesystem -- NIfTI/DICOM files, heatmap outputs           ||
|  +-----------------------------------------------------------+|
+---------------------------------------------------------------+
```

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Viewer | NiiVue (WebGL2) | Supports NIfTI natively, embeds in plain HTML, no React needed, 30+ format support |
| Harmonization | Classical preprocessing stack | Proven, hackathon-scoped, no GAN training. N4ITK + histogram matching + z-score normalization |
| Uncertainty | Monte Carlo dropout (30 forward passes) | Simpler than ensembles, works with any model with dropout layers, established in literature |
| Storage | SQLite + filesystem | No Docker/Postgres overhead for hackathon. Paths in DB, images on disk |
| Frontend | HTMX + Jinja2 + Pico CSS | Zero JS build step, server-rendered HTML fragments, clean minimal styling |
| ML serving | In-process (model loaded at startup) | Single-process simplicity for hackathon. No microservice overhead |

---

## 3. CRUD Data Model

### Entities

```
Patient
  - id: UUID (primary key)
  - name: str
  - age: int (optional)
  - notes: str (optional)
  - created_at: datetime

Scan
  - id: UUID (primary key)
  - patient_id: FK -> Patient
  - file_path: str (NIfTI for 3D volumetric, JPEG/PNG for 2D slices)
  - file_format: str (nifti / jpeg / png) -- determines which pipeline branch to use
  - scanner_vendor: str (Philips / Siemens / GE / Unknown)
  - modality: str (T1 / T2 / FLAIR / T1ce)
  - is_harmonized: bool (default: false)
  - harmonized_path: str (optional, path to harmonized output)
  - uploaded_at: datetime

Prediction
  - id: UUID (primary key)
  - scan_id: FK -> Scan
  - model_name: str
  - prediction_class: str (Glioma / Meningioma / Pituitary / No Tumor)
  - confidence: float
  - probabilities: JSON (all class probabilities)
  - uncertainty_map_path: str (optional, NIfTI heatmap)
  - gradcam_path: str (optional, Grad-CAM overlay image)
  - ran_on_harmonized: bool
  - inference_time_ms: int
  - created_at: datetime
```

### CRUD Operations

| Entity | Create | Read | Update | Delete |
|--------|--------|------|--------|--------|
| Patient | Form: name, age, notes | List + detail view | Edit form | Cascade deletes scans + predictions |
| Scan | File upload + vendor/modality selection | List per patient + viewer | Edit metadata only | Removes file from disk |
| Prediction | Auto-created by running inference | View with uncertainty overlay | Not editable | Delete to re-run |

All CRUD via HTMX forms — server-rendered HTML fragments, no JSON API for the frontend.

---

## 4. Harmonization Pipeline

### Approach

Classical preprocessing stack — no GAN training required, fully hackathon-scoped.

### Pipeline Steps

```
Raw MRI Scan
    |
    v
0. Format detection:
   - NIfTI (.nii/.nii.gz): full 3D pipeline (steps 1-6)
   - JPEG/PNG (2D slices, e.g. Kaggle dataset): simplified pipeline
     (skip step 2, apply 2D histogram matching + intensity norm only)
    |
    v
1. Load (SimpleITK for NIfTI / PIL for 2D images)
    |
    v
2. Brain mask extraction (NIfTI only: bet2 or simple thresholding)
    |
    v
3. Bias field correction -- N4ITK (SimpleITK)
   Removes scanner-specific intensity inhomogeneity
    |
    v
4. Intensity normalization -- Z-score within brain mask
   (zero mean, unit variance)
    |
    v
5. Histogram matching -- match to reference template
   (scikit-image exposure.match_histograms or MNI atlas)
    |
    v
6. (Optional) WhiteStripe normalization
   Targets white matter intensity peak for structural MRI
    |
    v
Harmonized scan (saved as NIfTI alongside original)
```

### Demo Flow

1. Upload a scan, select scanner vendor
2. Click "Harmonize"
3. Show before/after intensity histograms side-by-side
4. Run the same pretrained model on original vs. harmonized
5. Show accuracy improvement — same model, dramatic difference

### Libraries

| Library | Purpose |
|---------|---------|
| SimpleITK | N4ITK bias field correction, image I/O |
| nibabel | NIfTI file handling |
| scikit-image | `exposure.match_histograms` |
| intensity-normalization | WhiteStripe, z-score normalization |
| matplotlib | Histogram visualization for before/after |

### Datasets for Harmonization Demo

**BraTS 2024/2025** — 1,778 cases from 8 international institutions, different scanners. Standardized preprocessing already applied (1mm isotropic, SRI-24 registration), but residual inter-site differences persist. Available via Synapse ID: syn53708249.

**Brain Tumor MRI Dataset (masoudnickparvar)** — 7,023 images from 3 merged sources (Figshare, SARTAJ, Br35H) with different acquisition protocols. The mixed provenance is a *feature* for demonstrating harmonization.

### Research References

- [Cross-scanner harmonization methods comparison (2023)](https://www.sciencedirect.com/science/article/pii/S1053811923000605) — histogram matching, ComBat, CycleGAN, NST compared
- [Style transfer GAN for multi-site harmonization](https://pmc.ncbi.nlm.nih.gov/articles/PMC9137427/) — treats harmonization as style transfer
- [dMRIharmonization](https://github.com/pnlbwh/dMRIharmonization) — open-source Python CLI for diffusion MRI harmonization

---

## 5. Uncertainty Engine

### Approach

Monte Carlo Dropout on a pretrained classification/segmentation model.

### Pipeline

```
Preprocessed scan (harmonized)
    |
    v
1. Load pretrained model (EfficientNet-B0 for classification)
    |
    v
2. Enable dropout at inference time (model.train() for dropout layers only)
    |
    v
3. Run N=30 forward passes
    |
    v
4. For classification:
   - Collect 30 softmax vectors
   - Mean = prediction, Variance = uncertainty per class
   - High variance = "model unsure between glioma and meningioma"
    |
   For segmentation (if using BraTS + nnU-Net):
   - Collect 30 per-voxel probability maps
   - Mean = segmentation, Variance = per-voxel uncertainty
   - Boundary regions = naturally higher uncertainty
    |
    v
5. Generate heatmap overlay
   - High uncertainty = red
   - Low uncertainty = blue/green
   - Save as NIfTI alongside prediction
    |
    v
6. Generate Grad-CAM overlay
   - What regions influenced the prediction
   - Complementary to uncertainty (attention vs confidence)
    |
    v
7. Store both overlays + prediction in database
```

### Uncertainty Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Predictive entropy | H[y|x] = -sum(p_mean * log(p_mean)) | Total uncertainty (aleatoric + epistemic) |
| Mutual information | I[y; theta|x] = H[y|x] - E[H[y|x, theta]] | Epistemic uncertainty (model's ignorance) |
| Variance | Var(softmax) across N passes | Simple, interpretable spread measure |

### Viewer Integration

NiiVue embedded in the HTMX page with toggle controls:

- **Layer 1:** Raw scan
- **Layer 2:** Harmonized scan (toggle)
- **Layer 3:** Prediction overlay — classification label + confidence
- **Layer 4:** Uncertainty heatmap (toggle, with threshold slider)
- **Layer 5:** Grad-CAM overlay (toggle)

Slider for uncertainty threshold: "Show only regions where uncertainty > X"

### Research References

- [MC Dropout for brain tumor segmentation (2025)](https://arxiv.org/html/2510.15541v1) — empirical study on uncertainty-error correlation
- [MC Dropout for quantitative MRI](https://arxiv.org/abs/2112.01587) — improving accuracy and uncertainty quantification
- [Improving repeatability with MC Dropout](https://www.nature.com/articles/s41746-022-00709-3) — npj Digital Medicine

---

## 6. Tech Stack

| Layer | Technology | Version/Notes |
|-------|-----------|---------------|
| Package manager | uv | Rust-based, 10-100x faster than pip |
| Language | Python | 3.12+ |
| Web framework | FastAPI + Uvicorn | Async, auto OpenAPI docs |
| Frontend rendering | HTMX + Jinja2 | Server-rendered HTML fragments, no JS build |
| CSS | Pico CSS | Classless, minimal, clean |
| MRI viewer | NiiVue | WebGL2, NIfTI native, embeds via `<script>` |
| ORM | SQLModel | Type-safe, by FastAPI's author |
| Database | SQLite | Dev/hackathon; Postgres upgrade path via Alembic |
| Migrations | Alembic | Auto-generate from SQLModel |
| ML framework | PyTorch + torchvision | EfficientNet-B0 pretrained |
| Medical imaging | SimpleITK, nibabel | N4ITK bias correction, NIfTI I/O |
| Image processing | scikit-image | Histogram matching, exposure normalization |
| Uncertainty | Custom MC Dropout | 30 forward passes, variance computation |
| Explainability | pytorch-grad-cam | Grad-CAM on final conv layer |
| Visualization | matplotlib | Before/after histograms |
| Testing | pytest | Unit + integration tests |

### Project File Structure

```
Hackathon-MRI/
  pyproject.toml                    # uv project config
  uv.lock                          # lockfile
  src/
    app/
      __init__.py
      main.py                      # FastAPI app entry point
      config.py                    # Settings (Pydantic BaseSettings)
      database.py                  # SQLModel engine + session
      models/                      # SQLModel data models
        __init__.py
        patient.py
        scan.py
        prediction.py
      routes/                      # FastAPI route handlers
        __init__.py
        patients.py
        scans.py
        predictions.py
        harmonize.py
        viewer.py
      templates/                   # Jinja2 templates
        base.html
        patients/
        scans/
        predictions/
        viewer.html
      static/                      # CSS, NiiVue JS, icons
    harmonization/
      __init__.py
      pipeline.py                  # Full harmonization pipeline
      bias_correction.py           # N4ITK wrapper
      intensity_norm.py            # Z-score normalization
      histogram_match.py           # Reference-based matching
    uncertainty/
      __init__.py
      mc_dropout.py                # Monte Carlo dropout engine
      gradcam.py                   # Grad-CAM wrapper
      heatmap.py                   # Uncertainty map generation
    cancer_detection/
      __init__.py
      model.py                     # EfficientNet-B0 wrapper
      preprocessing.py             # Image preprocessing
      inference.py                 # Single-image inference
      training/                    # Training scripts (optional)
        train.py
        evaluate.py
        config.py
  data/                            # Gitignored, local data
    raw/                           # Downloaded datasets
    harmonized/                    # Harmonization outputs
    uploads/                       # User-uploaded scans
    weights/                       # Model checkpoints
  tests/
    test_harmonization.py
    test_uncertainty.py
    test_crud.py
    test_inference.py
  alembic/                         # Database migrations
    versions/
  docs/
    00-onboarding/
      QUICK-START.md
    01-architecture/
      system-overview.md
    02-development/
      dev-guide.md
    09-research/
      datasets.md
      harmonization-approaches.md
      uncertainty-methods.md
    superpowers/
      specs/
        2026-04-09-mri-harmonization-uncertainty-platform-design.md
        2026-04-09-cancer-detection-subdomain-design.md
      plans/
```

---

## 7. Measurable Success Criteria

### Primary Metrics (MUST achieve)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Harmonization improvement | >5% accuracy gain on cross-site scans | Same model evaluated pre- vs post-harmonization on BraTS multi-site data |
| Uncertainty calibration | High-uncertainty regions correlate with actual errors | Overlap ratio between top-20% uncertainty voxels and misclassified regions |
| Upload-to-result latency | <30 seconds for a single 2D scan | End-to-end timer from upload to prediction + heatmap display |
| CRUD completeness | All 4 operations work for Patients, Scans, Predictions | Automated pytest + manual browser verification |
| Viewer interactivity | Toggle overlays, adjust threshold, zoom/pan | Manual verification in browser |

### Secondary Metrics (SHOULD achieve)

| Metric | Target | Notes |
|--------|--------|-------|
| Classification accuracy (baseline) | >= 96% on test set | EfficientNet-B0 on Brain Tumor MRI dataset |
| Harmonization visual quality | Histograms align across vendors | Side-by-side histogram comparison |
| Grad-CAM clinical alignment | Highlights tumor region, not background | Qualitative review of 20 samples |
| Model size | < 25MB | Deployable without GPU |
| Before/after demo clarity | Non-technical viewer understands the improvement | Narrative demo test |

### Anti-Metrics (Failure Indicators)

- Harmonization *reduces* accuracy on already-clean scans (overcorrection)
- Uncertainty maps are uniformly low or uniformly high (uninformative)
- Grad-CAM highlights skull/background instead of tumor region
- CRUD operations take >2 seconds to respond
- NiiVue viewer fails to load or render overlays

### Validation Plan

1. **Harmonization A/B test:** Run pretrained model on 100 BraTS scans from different sites, pre- and post-harmonization. Report per-site accuracy delta.
2. **Uncertainty-error correlation:** For each test scan, compute uncertainty score and check if high-uncertainty scans are more likely to be misclassified. Report Spearman rank correlation.
3. **End-to-end workflow test:** Upload scan -> harmonize -> predict -> view uncertainty. Time each step. Verify all CRUD operations.
4. **Cross-browser viewer test:** NiiVue rendering in Chrome, Firefox, Safari (WebGL2 support).

---

## Appendix A: Companion Spec

The cancer detection ML component is specified separately in:
`docs/superpowers/specs/2026-04-09-cancer-detection-subdomain-design.md`

That spec covers: EfficientNet-B0 architecture, training configuration, Grad-CAM implementation, dataset preprocessing, evaluation metrics, and API contract. It serves as the *baseline model* that Pillar 1 (harmonization) demonstrates improvement on, and that Pillar 2 (uncertainty) wraps with MC Dropout.

## Appendix B: Research Sources

### Harmonization
- [Cross-scanner harmonization comparison (2023)](https://www.sciencedirect.com/science/article/pii/S1053811923000605)
- [Style transfer GAN for multi-site harmonization](https://pmc.ncbi.nlm.nih.gov/articles/PMC9137427/)
- [dMRIharmonization (GitHub)](https://github.com/pnlbwh/dMRIharmonization)
- [Unpaired MRI harmonization with latent diffusion (2025)](https://link.springer.com/chapter/10.1007/978-3-032-04947-6_65)

### Uncertainty
- [MC Dropout for brain tumor segmentation (2025)](https://arxiv.org/html/2510.15541v1)
- [MC Dropout for quantitative MRI](https://arxiv.org/abs/2112.01587)
- [Improving repeatability with MC Dropout (npj Digital Medicine)](https://www.nature.com/articles/s41746-022-00709-3)
- [MC Frequency Dropout (2025)](https://arxiv.org/html/2501.11258v1)

### Brain Tumor Detection
- [Deep learning for brain tumor classification — systematic review (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092918/)
- [YOLOv7 for brain tumor detection (2025)](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2025.1508326/full)
- [Explainable AI + SHAP for brain tumor MRI (2025)](https://www.nature.com/articles/s41598-025-14901-4)

### Datasets
- [BraTS 2024 Challenge (Synapse)](https://www.synapse.org/brats2024)
- [BraTS 2024 on Hugging Face](https://huggingface.co/datasets/Spirit-26/BraTS-2024-Complete)
- [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [BRISC 2025 (Kaggle)](https://www.kaggle.com/datasets/briscdataset/brisc2025)

### Web Stack
- [FastAPI + HTMX tutorial (TestDriven.io)](https://testdriven.io/blog/fastapi-htmx/)
- [fastapi-sqlmodel-crud (GitHub)](https://github.com/amisadmin/fastapi-sqlmodel-crud)
- [NiiVue documentation](https://niivue.com/docs/)
- [Anomalib (GitHub)](https://github.com/open-edge-platform/anomalib)
