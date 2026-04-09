import numpy as np
import torch
from cancer_detection.model import BrainTumorClassifier


def test_mc_dropout_produces_variance():
    from uncertainty.mc_dropout import mc_dropout_predict
    model = BrainTumorClassifier(num_classes=4)
    x = torch.randn(1, 3, 224, 224)
    result = mc_dropout_predict(model, x, n_passes=10)

    assert "mean_probs" in result
    assert result["mean_probs"].shape == (4,)
    assert "variance" in result
    assert result["variance"].shape == (4,)
    assert "predictive_entropy" in result
    assert "mutual_information" in result
    assert "all_probs" in result
    assert result["all_probs"].shape == (10, 4)
    assert result["variance"].sum() > 0
    assert abs(result["mean_probs"].sum() - 1.0) < 0.05
    assert result["predictive_entropy"] >= 0


def test_mc_dropout_different_passes_give_different_results():
    from uncertainty.mc_dropout import mc_dropout_predict
    model = BrainTumorClassifier(num_classes=4)
    x = torch.randn(1, 3, 224, 224)
    result = mc_dropout_predict(model, x, n_passes=10)
    diffs = np.diff(result["all_probs"], axis=0)
    assert np.abs(diffs).sum() > 0


def test_uncertainty_heatmap_generation():
    from uncertainty.heatmap import generate_uncertainty_bar_chart
    variance = np.array([0.01, 0.15, 0.02, 0.08])
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    png_bytes = generate_uncertainty_bar_chart(variance, class_names)
    assert len(png_bytes) > 100
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_mc_dropout_visualization():
    from uncertainty.heatmap import generate_mc_dropout_visualization
    all_probs = np.random.dirichlet([1, 1, 1, 1], size=30)
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    png_bytes = generate_mc_dropout_visualization(all_probs, class_names)
    assert len(png_bytes) > 100
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
