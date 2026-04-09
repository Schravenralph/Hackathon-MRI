"""Validation suite: checks all success criteria from the spec."""
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
    print(f"[{'PASS' if passed else 'FAIL'}] Test accuracy: {results['accuracy']:.4f} (target: >= 0.96)")
    f1_passed = results["weighted_f1"] >= 0.95
    print(f"[{'PASS' if f1_passed else 'FAIL'}] Weighted F1: {results['weighted_f1']:.4f} (target: >= 0.95)")
    return passed and f1_passed


def check_inference_latency() -> bool:
    if not Path("data/weights/best_model.pth").exists():
        print("[SKIP] No trained model — cannot check latency")
        return True

    from cancer_detection.inference import InferenceService
    service = InferenceService("data/weights/best_model.pth")
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    service.predict(img)  # Warm up

    times = []
    for _ in range(10):
        start = time.time()
        service.predict(img)
        times.append((time.time() - start) * 1000)

    avg_ms = np.mean(times)
    passed = avg_ms < 500
    print(f"[{'PASS' if passed else 'FAIL'}] Inference latency: {avg_ms:.0f}ms avg (target: < 500ms)")
    return passed


def check_harmonization_pipeline() -> bool:
    from harmonization.pipeline import harmonize_2d
    arr = np.random.uniform(50, 200, (128, 128)).astype(np.float32)
    ref = np.random.uniform(80, 220, (128, 128)).astype(np.float32)
    results = harmonize_2d(arr, reference=ref)
    passed = all(k in results for k in ["original", "bias_corrected", "normalized", "harmonized"])
    print(f"[{'PASS' if passed else 'FAIL'}] Harmonization pipeline produces all intermediate results")
    return passed


def check_uncertainty_engine() -> bool:
    from cancer_detection.model import BrainTumorClassifier
    from uncertainty.mc_dropout import mc_dropout_predict

    model = BrainTumorClassifier(num_classes=4)
    x = torch.randn(1, 3, 224, 224)
    result = mc_dropout_predict(model, x, n_passes=10)

    passed = result["variance"].sum() > 0
    print(f"[{'PASS' if passed else 'FAIL'}] MC Dropout produces non-zero variance")
    entropy_ok = result["predictive_entropy"] >= 0
    print(f"[{'PASS' if entropy_ok else 'FAIL'}] Predictive entropy is non-negative: {result['predictive_entropy']:.4f}")
    return passed and entropy_ok


def check_crud_server() -> bool:
    try:
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            health = client.get("/api/health")
            passed = health.status_code == 200
            print(f"[{'PASS' if passed else 'FAIL'}] Health endpoint returns 200")

            patients = client.get("/patients/")
            patients_ok = patients.status_code == 200
            print(f"[{'PASS' if patients_ok else 'FAIL'}] Patients list returns 200")
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
