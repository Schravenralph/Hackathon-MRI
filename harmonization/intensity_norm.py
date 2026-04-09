import numpy as np


def zscore_normalize(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score normalization: zero mean, unit variance within brain mask."""
    if mask is None:
        mask = image > 0 if image.max() > 0 else np.ones(image.shape, dtype=bool)
    if not mask.any():
        return image.copy()
    brain_voxels = image[mask].astype(np.float64)
    mean = brain_voxels.mean()
    std = brain_voxels.std()
    if std < 1e-8:
        return image.copy()
    result = image.copy().astype(np.float64)
    result[mask] = (brain_voxels - mean) / std
    return result.astype(np.float32)
