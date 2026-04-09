import numpy as np


def apply_bias_correction_2d(image: np.ndarray) -> np.ndarray:
    try:
        import SimpleITK as sitk
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
        mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
        corrected = corrector.Execute(sitk_image, mask)
        return sitk.GetArrayFromImage(corrected)
    except Exception:
        return _fallback_bias_correction(image)


def _fallback_bias_correction(image: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    safe_image = np.clip(image.astype(np.float64), 1.0, None)
    log_image = np.log(safe_image)
    low_freq = gaussian_filter(log_image, sigma=30)
    corrected = log_image - low_freq + log_image.mean()
    return np.exp(corrected).astype(np.float32)
