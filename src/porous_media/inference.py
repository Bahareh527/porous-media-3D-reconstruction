"""Overlap-averaged volume prediction with edge coverage and explicit layouts."""

import numpy as np

from porous_media.volumes import iter_patches, validate_volume


def predict_volume(predictor, volume, patch_size: int, stride: int) -> np.ndarray:
    volume = validate_volume(volume)
    accumulated = np.zeros(volume.shape, dtype=np.float64)
    counts = np.zeros(volume.shape, dtype=np.uint32)
    for origin, patch in iter_patches(volume, patch_size, stride, cover_edges=True):
        prediction = np.asarray(predictor(patch[np.newaxis]))
        if prediction.shape != (1, patch_size, patch_size, patch_size, 1):
            raise ValueError("predictor must return (N, Z, Y, X, 1) patches")
        if not np.isfinite(prediction).all():
            raise ValueError("prediction contains nonfinite values")
        slices = tuple(slice(start, start + patch_size) for start in origin)
        accumulated[slices] += prediction[0, ..., 0]
        counts[slices] += 1
    if not np.all(counts):
        raise ValueError("patch grid did not cover every voxel")
    return (accumulated / counts).astype(np.float32)
