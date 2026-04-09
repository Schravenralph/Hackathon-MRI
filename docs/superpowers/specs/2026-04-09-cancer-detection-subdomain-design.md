# Cancer Detection Subdomain Design Spec

**Date:** 2026-04-09
**Subdomain:** MRI Brain Tumor Classification (Cancer Detection)
**Parent project:** Hackathon-MRI (general MRI anomaly scanning with CRUD frontend)
**Stack:** Python + uv, PyTorch, FastAPI

---

## 1. Executive Summary

This spec defines the cancer detection ML component of the Hackathon-MRI system. The component classifies brain MRI images into four categories -- Glioma, Meningioma, Pituitary tumor, and No Tumor -- using transfer learning on EfficientNet-B0, with Grad-CAM explainability overlays. It integrates with the parent CRUD application via a FastAPI inference endpoint.

---

## 2. State of the Art (Research Summary)

### 2.1 Current Benchmarks (2024-2026)

| Model / Method | Accuracy | F1-Score | Source |
|---|---|---|---|
| Cross ViT + Stochastic Depth | 99.24% | 99.23% | [PMC Systematic Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092918/) |
| MobileNetV3 | 99.75% | -- | [PMC Systematic Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092918/) |
| Stack Ensemble (VGG16+ResNet+InceptionV3) | 99.66% | -- | [Scientific Reports](https://www.nature.com/articles/s41598-023-50505-6) |
| EfficientNet-B3 (transfer learning) | 99.23% | -- | [GitHub: zacharyvunguyen](https://github.com/zacharyvunguyen/Brain-Tumor-MR-Image-Classification-using-Transfer-Learning-with-EfficientNet-) |
| EfficientNet-B2 | 99.06% | 98.79% | [Wiley: BrainNet](https://onlinelibrary.wiley.com/doi/10.1155/2024/3583612) |
| EfficientNet-B0 (fine-tuned) | 99.00% | -- | [DebuggerCafe](https://debuggercafe.com/brain-mri-classification-using-pytorch-efficientnetb0/) |
| Hybrid CNN + Grad-CAM + XGBoost | 99.77% | -- | [MDPI Applied Sciences](https://www.mdpi.com/2076-3417/15/10/5412) |
| Hierarchical Multi-Scale ViT | 98.70% | -- | [Scientific Reports](https://www.nature.com/articles/s41598-025-23100-0) |
| ResNet-50 (baseline) | 95.80% | -- | [Scientific Reports](https://www.nature.com/articles/s41598-025-23100-0) |
| Tiny-ViT-5M | 98.41% | -- | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169743925002941) |

### 2.2 Architecture Comparison

**EfficientNet** achieves the best accuracy-to-parameter ratio. EfficientNet-B0 (5.3M params) matches models 3-4x its size. It is the recommended backbone for a hackathon context: fast to train, well-documented, strong transfer learning from ImageNet, and proven on this exact dataset.

**Vision Transformers (ViT)** show superior performance when data is abundant (Tiny-ViT-5M: 98.41%, Hierarchical ViT: 98.7%), but require more compute and data. They excel at capturing long-range spatial dependencies but are harder to debug and explain.

**ResNet-50** is the proven baseline (95.8%) but consistently underperforms modern architectures on this task.

### 2.3 Known Failure Modes

1. **Scanner artifact sensitivity** -- Field inhomogeneity causes large performance drops. N4ITK bias field correction during preprocessing is essential. ([BME Frontiers](https://spj.science.org/doi/10.34133/2022/9807590))
2. **Cross-institution generalization** -- Models trained on one scanner/protocol fail on others. Differences in T1/T2/FLAIR sequences and hardware settings cause distribution shift. ([npj Precision Oncology](https://www.nature.com/articles/s41698-024-00789-2))
3. **Class imbalance bias** -- Even mild imbalance (1.21 ratio in this dataset) can bias toward majority class during training. Focal loss or class-weighted cross-entropy mitigates this. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10453020/))
4. **Overfitting to dataset-specific patterns** -- The Masoudnickparvar dataset combines 3 source datasets (Figshare, SARTAJ, Br35H) with different acquisition protocols, creating confounding correlations between source and label. ([Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset))
5. **Glioma mislabeling** -- The SARTAJ subset had incorrectly categorized glioma images; some were removed and replaced with Figshare images, but residual label noise may exist. ([Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset))

---

## 3. Dataset Strategy

### 3.1 Primary Dataset

**Brain Tumor MRI Dataset by masoudnickparvar** (Kaggle)
- 7,023 images, 4 classes
- Class distribution: Glioma 1,321 (23.1%), Meningioma 1,339 (23.4%), No Tumor 1,595 (28.5%), Pituitary 1,457 (25.0%)
- Imbalance ratio: 1.21 (mild -- manageable with weighted loss)
- Source: Combination of Figshare, SARTAJ, and Br35H datasets
- Pre-split into Training and Testing folders by the dataset author
- Variable image sizes (require resize during preprocessing)

### 3.2 Known Issues with This Dataset

1. **Glioma label noise** -- Original SARTAJ glioma labels had errors. Some images were deleted and replaced from Figshare, but residual mislabeling may persist.
2. **Variable image dimensions** -- No standard size; must resize after margin removal.
3. **Mixed acquisition protocols** -- Three source datasets mean inconsistent contrast, resolution, and orientation.
4. **No segmentation masks** -- Classification labels only; no pixel-level tumor boundaries.

### 3.3 Complementary Dataset (Optional Extension)

**BRISC 2025** (6,000 images, same 4 classes + segmentation masks)
- Expert-annotated by certified radiologists
- Includes axial, sagittal, and coronal planes
- Adds segmentation capability if the project extends beyond classification
- Available on Kaggle: [BRISC 2025](https://www.kaggle.com/datasets/briscdataset/brisc2025)

**Recommendation:** Start with Masoudnickparvar only. Add BRISC 2025 in a second phase if (a) segmentation is needed, or (b) cross-dataset validation is desired.

### 3.4 Preprocessing Pipeline

```
Raw MRI Image
    |
    v
1. Load image (PIL/OpenCV)
    |
    v
2. Convert to RGB (some images are grayscale)
    |
    v
3. Crop extra black margins (contour-based cropping)
    |
    v
4. Resize to 224x224 (EfficientNet-B0 native input)
    |
    v
5. Normalize (ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    |
    v
6. (Training only) Apply augmentations
    |
    v
Tensor ready for model
```

### 3.5 Augmentation Strategy

Based on research showing brightness augmentation and elastic deformation work best for brain MRI ([DIVA Portal](https://www.diva-portal.org/smash/get/diva2:1588376/FULLTEXT01.pdf), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6917660/)):

| Augmentation | Parameters | Rationale |
|---|---|---|
| Random Horizontal Flip | p=0.5 | Brain symmetry makes this safe |
| Random Rotation | +/- 15 degrees | Simulates head tilt variation |
| Random Affine (translate) | +/- 10% | Simulates positioning differences |
| Color Jitter (brightness) | factor=0.2 | Simulates scanner intensity variation |
| Random Resized Crop | scale=(0.85, 1.0) | Focus variation |
| Gaussian Blur | kernel=3, p=0.1 | Simulates slight defocus |

**NOT recommended:** Vertical flip (anatomically invalid), extreme elastic deformation (creates unrealistic morphology), color hue shifts (MRI is grayscale-origin).

---

## 4. Design Choices Table

| # | Question | Recommendation | Reasoning | Alternatives |
|---|----------|---------------|-----------|--------------|
| 1 | Classification vs segmentation vs both? | **Classification only** (Phase 1) | The Masoudnickparvar dataset has classification labels only. Classification achieves 99%+ accuracy and is sufficient for tumor type identification. Segmentation can be added in Phase 2 with BRISC 2025. | Both (requires BRISC 2025 dataset); Segmentation only (U-Net + BraTS data) |
| 2 | Model architecture? | **EfficientNet-B0** | Best accuracy-to-parameter ratio (5.3M params, 99% accuracy). Well-documented for this exact dataset. Fast training (critical for hackathon). PyTorch `torchvision.models` has it built-in. | EfficientNet-B3 (99.23% but 12M params, slower); ViT-Small (98.4%, needs more data); ResNet-50 (baseline, 95.8%) |
| 3 | Transfer learning base? | **ImageNet-1K pretrained weights** | Standard practice for medical imaging transfer learning. EfficientNet-B0 pretrained on ImageNet learns generalizable low-level features (edges, textures) that transfer well to MRI. Fine-tune all layers with differential learning rates. | RadImageNet (medical-specific pretraining, less accessible); Training from scratch (insufficient data at 7K images) |
| 4 | Data augmentation strategy? | **Moderate geometric + brightness** | Horizontal flip, rotation (+/-15), brightness jitter (0.2), random crop. Research shows brightness and mild geometric augmentation are most effective for brain MRI. Avoid anatomically invalid transforms. | Heavy augmentation with elastic deformation (risk of unrealistic images); GAN-based synthetic augmentation (complex, overkill for 7K balanced dataset); No augmentation (risks overfitting) |
| 5 | Train/val/test split strategy? | **Use dataset's pre-split test set; create 80/20 stratified train/val from training folder; validate with 5-fold stratified CV for model selection** | The dataset already provides a test partition. Stratified splitting preserves class proportions. 5-fold CV for hyperparameter tuning ensures robustness. Final evaluation on held-out test set only once. | Simple 70/15/15 random split (ignores existing partition); Leave-one-out CV (too expensive); No validation set (risks overfitting) |
| 6 | Evaluation metrics? | **Primary: Weighted F1-score. Secondary: Per-class precision/recall, confusion matrix, accuracy, AUC-ROC (one-vs-rest)** | F1-score balances precision and recall per class, critical for medical diagnosis where both false positives and false negatives have clinical consequences. Weighted F1 accounts for mild class imbalance. Confusion matrix reveals specific misclassification patterns. | Accuracy alone (misleading with any imbalance); Dice score (relevant for segmentation, not classification); Sensitivity/specificity only (misses multi-class nuance) |
| 7 | Inference pipeline design? | **Single-image async pipeline: upload -> preprocess -> inference -> Grad-CAM -> JSON response with class, confidence, heatmap** | Async allows non-blocking serving. Single-image matches clinical workflow (one scan at a time). Grad-CAM overlay provides immediate explainability. JSON response integrates cleanly with CRUD frontend. | Batch inference (not needed for clinical use case); ONNX Runtime (optimization premature for hackathon); TorchServe (heavy infrastructure overhead) |
| 8 | Model serving (API vs embedded)? | **FastAPI REST API with /predict endpoint** | FastAPI is async-native, auto-generates OpenAPI docs, has built-in validation with Pydantic, and outperforms Flask. Separate service allows independent scaling and updates. Research confirms FastAPI is becoming the standard for ML model serving in medical imaging. | Flask (slower, no async); gRPC (better throughput but harder frontend integration); Embedded in CRUD app (tight coupling, harder to update model) |
| 9 | Explainability (GradCAM, SHAP, etc.)? | **Grad-CAM on final convolutional layer + confidence scores** | Grad-CAM is the de facto standard for CNN explainability in brain tumor classification. It highlights which regions influenced the prediction, critical for clinician trust. Low computational overhead (single backward pass). Research shows Grad-CAM + ResNet/EfficientNet achieves 98%+ accuracy with interpretable outputs. | Grad-CAM++ (better multi-region localization but marginal improvement); LIME (perturbation-based, slower); SHAP (feature-level, not spatial -- less intuitive for imaging); Attention maps (only for ViT architectures) |
| 10 | Single dataset or combine multiple? | **Single dataset (Masoudnickparvar) for Phase 1; add BRISC 2025 for cross-validation in Phase 2** | Masoudnickparvar is sufficient (7K images, 4 classes, well-studied). Adding datasets introduces domain alignment complexity. For a hackathon, simplicity and fast iteration matter more. BRISC 2025 adds segmentation masks for future extension. | Combine immediately (risk of domain shift confounds); BraTS only (3D volumetric, different pipeline); LGG Segmentation (small, different task) |

---

## 5. Model Pipeline Design

### 5.1 Architecture

```
                          +------------------+
                          |  CRUD Frontend   |
                          |  (Lightweight UI)|
                          +--------+---------+
                                   |
                                   | HTTP POST /api/predict
                                   | (multipart/form-data: MRI image)
                                   v
                          +------------------+
                          |  FastAPI Gateway  |
                          |  /api/predict     |
                          |  /api/health      |
                          |  /api/model-info  |
                          +--------+---------+
                                   |
                      +------------+------------+
                      |                         |
                      v                         v
              +---------------+         +---------------+
              | Preprocessing |         |  Grad-CAM     |
              | Pipeline      |         |  Generator    |
              | - RGB convert |         |  - Heatmap    |
              | - Margin crop |         |  - Overlay    |
              | - Resize 224  |         |  - Base64 img |
              | - Normalize   |         +---------------+
              +-------+-------+                 ^
                      |                         |
                      v                         |
              +---------------+                 |
              | EfficientNet  |-----------------+
              | B0 (frozen    |
              |  + fine-tuned)|
              | 4-class head  |
              +-------+-------+
                      |
                      v
              +---------------+
              |  Response     |
              |  {            |
              |   class,      |
              |   confidence, |
              |   gradcam_img,|
              |   all_probs   |
              |  }            |
              +---------------+
```

### 5.2 Model Architecture Detail

```python
# Pseudocode for model definition
model = EfficientNet-B0(pretrained=ImageNet)
model.classifier = Sequential(
    Dropout(0.3),
    Linear(1280, 512),
    ReLU(),
    Dropout(0.2),
    Linear(512, 4)  # 4 classes: glioma, meningioma, pituitary, no_tumor
)
```

### 5.3 Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Better weight decay handling than Adam |
| Learning rate | 1e-4 (backbone), 1e-3 (classifier head) | Differential LR preserves pretrained features |
| LR scheduler | CosineAnnealingWarmRestarts | Smooth decay with periodic recovery |
| Loss function | CrossEntropyLoss with class weights | Handles mild imbalance (1.21 ratio) |
| Batch size | 32 | Fits in 8GB GPU; good gradient estimation |
| Epochs | 30 (early stopping patience=5) | Prevents overfitting; typical convergence at 15-20 epochs |
| Weight decay | 1e-4 | Regularization |
| Mixed precision | FP16 (torch.cuda.amp) | 2x training speedup, minimal accuracy loss |

### 5.4 Training Phases

1. **Phase 1 (epochs 1-5):** Freeze backbone, train classifier head only. LR=1e-3.
2. **Phase 2 (epochs 6-30):** Unfreeze all layers. Differential LR: backbone=1e-4, head=1e-3. Early stopping on validation F1.

---

## 6. Measurable Success Criteria

### 6.1 Primary Metrics (MUST achieve)

| Metric | Target | Baseline (ResNet-50) | SOTA Reference |
|---|---|---|---|
| Overall accuracy | >= 96% | 95.8% | 99.23% (EfficientNet-B3) |
| Weighted F1-score | >= 0.95 | -- | 0.9879 (EfficientNet-B2) |
| Per-class F1 (min) | >= 0.93 | -- | 0.94+ (Residual-Attention-UNet) |
| Inference latency | < 200ms per image | -- | -- |

### 6.2 Secondary Metrics (SHOULD achieve)

| Metric | Target | Notes |
|---|---|---|
| Overall accuracy | >= 98% | Stretch goal matching published results |
| Weighted F1-score | >= 0.97 | Stretch goal |
| AUC-ROC (per class) | >= 0.98 | One-vs-rest AUC |
| Grad-CAM clinical alignment | Qualitative review | Heatmap highlights tumor region, not background artifacts |
| Model size | < 25MB | Deployable without GPU in production |

### 6.3 Anti-Metrics (failure indicators)

- Any per-class recall < 0.90 (missing tumors is clinically dangerous)
- Grad-CAM highlighting non-brain regions (skull, background) for tumor classes
- Accuracy drop > 5% between validation and test set (overfitting signal)
- Inference latency > 500ms on CPU (unusable in practice)

---

## 7. Validation Plan

### 7.1 Model Selection (Development Phase)

1. Train with 5-fold stratified cross-validation on the training partition
2. Select the model configuration with the highest mean weighted F1 across folds
3. Track per-fold variance -- standard deviation > 2% indicates instability

### 7.2 Final Evaluation (One-Shot)

1. Retrain best configuration on full training partition
2. Evaluate once on the held-out test set (dataset's pre-split Testing folder)
3. Report full confusion matrix, per-class precision/recall/F1, and AUC-ROC curves
4. Generate Grad-CAM visualizations for 5 random correct and 5 random incorrect predictions per class

### 7.3 Robustness Checks

1. **Augmentation stress test:** Apply heavy augmentation (rotation +/-45, blur sigma=3) to test images and measure accuracy degradation
2. **Confidence calibration:** Plot reliability diagram; predictions should be well-calibrated (confident predictions should be correct at the stated rate)
3. **Misclassification analysis:** Manual review of all misclassified test images to identify patterns (e.g., specific tumor subtypes, image quality issues)

### 7.4 Comparison Method for Approaches

To compare alternative architectures or configurations:

| Step | Action |
|---|---|
| 1 | Define the experimental variable (e.g., architecture, augmentation, LR) |
| 2 | Hold all other variables constant |
| 3 | Run 5-fold stratified CV on training partition |
| 4 | Compare mean +/- std of weighted F1 across folds |
| 5 | Use paired t-test (p < 0.05) to determine statistical significance |
| 6 | If significant, select the higher-performing variant |
| 7 | Document in experiment log with hyperparameters, metrics, and training curves |

---

## 8. Integration Points with Parent CRUD App

### 8.1 API Contract

```
POST /api/predict
Content-Type: multipart/form-data
Body: file=<mri_image.jpg>

Response 200:
{
  "prediction": "glioma",
  "confidence": 0.9723,
  "probabilities": {
    "glioma": 0.9723,
    "meningioma": 0.0142,
    "pituitary": 0.0089,
    "no_tumor": 0.0046
  },
  "gradcam_image": "<base64-encoded-png>",
  "model_version": "v1.0.0",
  "inference_time_ms": 87
}

Response 422: Validation error (invalid file type, corrupt image)
Response 500: Internal model error
```

### 8.2 Health Check

```
GET /api/health

Response 200:
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v1.0.0",
  "gpu_available": false
}
```

### 8.3 Integration Architecture

The cancer detection service runs as a **separate FastAPI process** from the CRUD app:

- **CRUD app** handles: user auth, scan uploads, result storage, UI rendering, scan history
- **Cancer detection service** handles: inference only (stateless)
- **Communication:** CRUD app calls cancer detection service via HTTP on upload
- **Data flow:** CRUD app stores the prediction result + Grad-CAM image alongside the original scan record

### 8.4 Database Schema (CRUD side)

The CRUD app should store prediction results with fields like:

```
scan_predictions:
  id              UUID PRIMARY KEY
  scan_id         UUID REFERENCES scans(id)
  predicted_class VARCHAR(20)       -- glioma, meningioma, pituitary, no_tumor
  confidence      FLOAT
  probabilities   JSONB             -- all 4 class probabilities
  gradcam_path    VARCHAR(255)      -- path to stored Grad-CAM overlay image
  model_version   VARCHAR(20)
  inference_ms    INTEGER
  created_at      TIMESTAMP
```

---

## 9. Project File Structure

```
Hackathon-MRI/
  cancer_detection/
    __init__.py
    model/
      __init__.py
      architecture.py        # EfficientNet-B0 wrapper
      gradcam.py             # Grad-CAM implementation
      preprocessing.py       # Image preprocessing pipeline
    training/
      __init__.py
      train.py               # Training loop with CV
      evaluate.py            # Evaluation and metrics
      augmentation.py        # Augmentation transforms
      config.py              # Hyperparameters
    serving/
      __init__.py
      api.py                 # FastAPI app with /predict endpoint
      schemas.py             # Pydantic request/response models
    data/
      download.py            # Kaggle dataset download script
    weights/                 # Trained model checkpoints (gitignored)
    notebooks/
      01_eda.ipynb           # Exploratory data analysis
      02_training.ipynb      # Training experiments
      03_evaluation.ipynb    # Final evaluation + visualizations
  tests/
    test_preprocessing.py
    test_inference.py
    test_api.py
```

---

## 10. Implementation Phases

### Phase 1: MVP (Days 1-2)
- Download and explore dataset
- Implement preprocessing pipeline
- Train EfficientNet-B0 with transfer learning
- Achieve >= 96% accuracy on test set
- Basic FastAPI /predict endpoint

### Phase 2: Production Quality (Days 3-4)
- Add Grad-CAM explainability
- Implement 5-fold CV for model selection
- Add confidence calibration
- Integrate with CRUD app
- Add /health endpoint and error handling

### Phase 3: Polish (Day 5)
- Misclassification analysis
- Grad-CAM quality review
- API documentation (auto-generated by FastAPI)
- Performance optimization (ONNX export if needed)
- Final evaluation and metrics report

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Glioma label noise from SARTAJ subset | Medium | High (ceiling on accuracy) | Manual review of misclassified gliomas; cross-reference with Figshare labels |
| Overfitting to dataset-specific patterns | Medium | High | Augmentation, dropout, early stopping, cross-validation |
| GPU unavailability during hackathon | Low | High | EfficientNet-B0 trains in ~30 min on CPU; use Google Colab as backup |
| Model memorizes scanner artifacts | Medium | Medium | Crop black margins; apply intensity normalization; test with augmented images |
| FastAPI integration latency | Low | Medium | Async inference; model loaded at startup; warm-up request on init |
| Class imbalance affecting minority class (glioma) | Low | Medium | Class-weighted loss; monitor per-class recall; augment minority class if needed |

---

## 12. Open-Source Reference Implementations

These repositories were evaluated during research and can serve as implementation references:

1. **enrico310786/brain_tumor_classification** -- PyTorch, ResNet + EfficientNet + EfficientNet_V2 + CCT comparison on the same dataset. Good code structure for multi-architecture benchmarking. ([GitHub](https://github.com/enrico310786/brain_tumor_classification))

2. **zacharyvunguyen/Brain-Tumor-MR-Image-Classification-using-Transfer-Learning-with-EfficientNet** -- EfficientNet-B3, 99.23% accuracy, transfer learning pipeline with detailed preprocessing. ([GitHub](https://github.com/zacharyvunguyen/Brain-Tumor-MR-Image-Classification-using-Transfer-Learning-with-EfficientNet-))

3. **baotramduong/Brain-Tumor-Classification-with-Efficient-Net-and-Grad-CAM-Visualization** -- EfficientNet + Grad-CAM integration, closest to our target architecture. ([GitHub](https://github.com/baotramduong/Brain-Tumor-Classification-with-Efficient-Net-and-Grad-CAM-Visualization))

---

## 13. Key Research Sources

- [Frontiers: Explainable AI-driven MRI-based brain tumor classification (2025)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1700214/full)
- [Nature: Review of deep learning for brain tumor analysis in MRI (2024)](https://www.nature.com/articles/s41698-024-00789-2)
- [PMC: Deep Learning Approaches for Brain Tumor Detection -- Systematic Review 2020-2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092918/)
- [arXiv: Explainable Deep Learning for Brain Tumor Classification -- Comprehensive Benchmarking (2025)](https://arxiv.org/html/2511.17655v1)
- [Scientific Reports: Optimized deep learning with attention mechanisms and clinical explainability (2025)](https://www.nature.com/articles/s41598-025-04591-3)
- [BME Frontiers: Simulated MRI Artifacts -- Testing ML Failure Modes (2022)](https://spj.science.org/doi/10.34133/2022/9807590)
- [Frontiers: Scalable FastAPI-deployed CNN framework for brain tumor diagnosis (2026)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1772429/full)
- [PMC: Data Augmentation for Brain-Tumor Segmentation Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6917660/)
- [Springer: Medical image data augmentation techniques, comparisons and interpretations](https://link.springer.com/article/10.1007/s10462-023-10453-z)
- [Nature Scientific Data: BRISC 2025 Annotated Dataset](https://www.nature.com/articles/s41597-026-06753-y)
- [JMIR AI: Trade-Off Analysis of Classical ML and DL Models for Brain Tumor Detection (2025)](https://ai.jmir.org/2025/1/e76344)
- [ScienceDirect: Comparative evaluation of lightweight CNNs and ViTs for brain tumor classification (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0169743925002941)
- [MDPI: Novel Hybrid Deep Learning Model with Explainable AI for Brain Tumor Multi-Classification (2025)](https://www.mdpi.com/2076-3417/15/10/5412)
- [Kaggle: Brain Tumor MRI Dataset by masoudnickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
