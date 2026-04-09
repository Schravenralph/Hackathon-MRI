import numpy as np
from skimage.exposure import match_histograms


def histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if source.ndim == 2 and reference.ndim == 2:
        matched = match_histograms(source, reference)
    else:
        channel_axis = -1 if source.ndim == 3 else None
        matched = match_histograms(source, reference, channel_axis=channel_axis)
    return matched.astype(source.dtype)
