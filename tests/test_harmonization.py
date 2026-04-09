import numpy as np
from PIL import Image


def _make_grayscale_array(size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    arr = np.zeros((size, size), dtype=np.float32)
    rr, cc = np.ogrid[:size, :size]
    mask = ((rr - size // 2) ** 2 / (size // 3) ** 2 + (cc - size // 2) ** 2 / (size // 4) ** 2) < 1
    arr[mask] = rng.uniform(80, 200, mask.sum()).astype(np.float32)
    return arr


def test_zscore_normalize():
    from harmonization.intensity_norm import zscore_normalize
    arr = _make_grayscale_array()
    result = zscore_normalize(arr)
    mask = arr > 10
    brain = result[mask]
    assert abs(brain.mean()) < 0.1
    assert abs(brain.std() - 1.0) < 0.1


def test_histogram_match():
    from harmonization.histogram_match import histogram_match
    source = _make_grayscale_array(seed=0)
    reference = _make_grayscale_array(seed=42) * 1.5 + 20
    result = histogram_match(source, reference)
    assert result.shape == source.shape
    assert result.dtype == source.dtype


def test_bias_correction():
    from harmonization.bias_correction import apply_bias_correction_2d
    arr = _make_grayscale_array()
    bias = np.linspace(0.5, 1.5, arr.shape[1])
    biased = arr * bias[np.newaxis, :]
    result = apply_bias_correction_2d(biased)
    assert result.shape == arr.shape
    assert not np.allclose(result, biased)


def test_full_pipeline():
    from harmonization.pipeline import harmonize_2d
    source = _make_grayscale_array(seed=0)
    reference = _make_grayscale_array(seed=42)
    results = harmonize_2d(source, reference=reference)
    assert "original" in results
    assert "bias_corrected" in results
    assert "normalized" in results
    assert "harmonized" in results
    assert results["harmonized"].shape == source.shape


def test_pipeline_without_reference():
    from harmonization.pipeline import harmonize_2d
    source = _make_grayscale_array()
    results = harmonize_2d(source, reference=None)
    assert "harmonized" in results
    assert results["harmonized"].shape == source.shape


def test_generate_comparison_histogram():
    from harmonization.pipeline import generate_comparison_histogram
    original = _make_grayscale_array(seed=0)
    harmonized = _make_grayscale_array(seed=42)
    png_bytes = generate_comparison_histogram(original, harmonized)
    assert len(png_bytes) > 100
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
