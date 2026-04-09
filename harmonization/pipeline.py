import io
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harmonization.bias_correction import apply_bias_correction_2d
from harmonization.histogram_match import histogram_match
from harmonization.intensity_norm import zscore_normalize


def harmonize_2d(image: np.ndarray, reference: np.ndarray | None = None) -> dict[str, np.ndarray]:
    results = {"original": image.copy()}
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    corrected = apply_bias_correction_2d(gray)
    results["bias_corrected"] = corrected

    normalized = zscore_normalize(corrected)
    results["normalized"] = normalized

    if reference is not None:
        if reference.ndim == 3:
            reference = np.mean(reference, axis=2).astype(np.float32)
        ref_normalized = zscore_normalize(reference)
        matched = histogram_match(normalized, ref_normalized)
        results["histogram_matched"] = matched
    else:
        matched = normalized

    results["harmonized"] = matched
    return results


def generate_comparison_histogram(original: np.ndarray, harmonized: np.ndarray, title: str = "Before/After Harmonization") -> bytes:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    orig_flat = original.flatten()
    orig_flat = orig_flat[orig_flat > 0]
    ax1.hist(orig_flat, bins=100, alpha=0.7, color="#e53935", label="Original")
    ax1.set_title("Before Harmonization")
    ax1.set_xlabel("Intensity")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    harm_flat = harmonized.flatten()
    harm_flat = harm_flat[np.isfinite(harm_flat)]
    if len(harm_flat) > 0:
        ax2.hist(harm_flat, bins=100, alpha=0.7, color="#43a047", label="Harmonized")
    ax2.set_title("After Harmonization")
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
