"""Measured image, segmentation and pore-mask metrics (no typed experiment results)."""

import numpy as np
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity

from porous_media.volumes import validate_volume


def reconstruction_metrics(reference, prediction, *, data_range: float = 2.0) -> dict:
    reference = validate_volume(reference, name="reference").astype(np.float64)
    prediction = validate_volume(prediction, name="prediction").astype(np.float64)
    if reference.shape != prediction.shape or not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("matching shapes and a positive fixed data_range are required")
    mse = float(np.mean((reference - prediction) ** 2))
    window = min(7, min(reference.shape))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        raise ValueError("SSIM requires at least three voxels in every dimension")
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(reference - prediction))),
        "psnr_db": None if mse == 0 else float(10.0 * np.log10(data_range**2 / mse)),
        "perfect_reconstruction": mse == 0,
        "ssim": float(
            structural_similarity(reference, prediction, data_range=data_range, win_size=window)
        ),
        "data_range": data_range,
    }


def pore_mask(volume, *, threshold: float | None = None, pores_are_dark: bool = True):
    volume = validate_volume(volume)
    if threshold is None:
        if volume.min() == volume.max():
            raise ValueError("constant images require an explicit pore threshold")
        threshold = float(threshold_otsu(volume))
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return volume < threshold if pores_are_dark else volume > threshold


def porosity(mask) -> float:
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.size == 0 or mask.dtype != np.bool_:
        raise ValueError("porosity requires a nonempty boolean 3D pore mask")
    return float(mask.mean())


def axial_two_point(mask, axis: int = 0, max_lag: int | None = None):
    """Probability that two voxels a given axial distance apart are both pores."""
    porosity(mask)
    mask = np.asarray(mask)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    limit = mask.shape[axis] - 1 if max_lag is None else max_lag
    if not 0 <= limit < mask.shape[axis]:
        raise ValueError("max_lag is outside the volume")
    values = []
    for lag in range(limit + 1):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, mask.shape[axis] - lag)
        right[axis] = slice(lag, mask.shape[axis])
        values.append(float(np.mean(mask[tuple(left)] & mask[tuple(right)])))
    return np.arange(limit + 1), np.asarray(values)


def segmentation_metrics(reference, prediction, num_classes: int) -> dict:
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.shape != prediction.shape or num_classes < 2 or reference.size == 0:
        raise ValueError("invalid segmentation inputs")
    for array in (reference, prediction):
        if (
            not np.isfinite(array).all()
            or np.any(array != np.floor(array))
            or np.any(array < 0)
            or np.any(array >= num_classes)
        ):
            raise ValueError("segmentation arrays must contain valid integer class IDs")
    matrix = np.bincount(
        (num_classes * reference.astype(int) + prediction.astype(int)).ravel(),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    diagonal = np.diag(matrix).astype(float)
    union = matrix.sum(0) + matrix.sum(1) - diagonal
    denominator = matrix.sum(0) + matrix.sum(1)
    iou = np.divide(diagonal, union, out=np.full(num_classes, np.nan), where=union > 0)
    f1 = np.divide(
        2 * diagonal, denominator, out=np.full(num_classes, np.nan), where=denominator > 0
    )
    return {
        "accuracy": float(diagonal.sum() / matrix.sum()),
        "macro_iou": float(np.nanmean(iou)),
        "macro_f1": float(np.nanmean(f1)),
        "confusion_matrix": matrix.tolist(),
        "absent_classes_excluded_from_macro": True,
    }
