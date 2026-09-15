"""Volume validation, train-only scaling, spatial splits and aligned patches."""

from dataclasses import dataclass
from itertools import product

import numpy as np


def validate_volume(volume: np.ndarray, *, name: str = "volume") -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim != 3 or min(array.shape) == 0:
        raise ValueError(f"{name} must be a nonempty 3D array")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numerical values")
    return array


def remap_labels(labels: np.ndarray, mapping: dict[int, int], num_classes: int) -> np.ndarray:
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if any(type(value) is not int for value in mapping.values()):
        raise ValueError("mapping targets must be integer class IDs")
    original = validate_volume(labels, name="labels")
    if not np.equal(original, np.floor(original)).all():
        raise ValueError("class labels must be integers, not normalized intensities")
    values = set(np.unique(original).astype(int).tolist())
    if mapping:
        unknown = values - set(mapping)
        if unknown:
            raise ValueError(f"label mapping omits source classes: {sorted(unknown)}")
        result = np.empty(original.shape, dtype=np.int32)
        # Read from the original array so mappings cannot cascade (e.g. 0->1, 1->2).
        for source, target in mapping.items():
            result[original == source] = target
    else:
        result = original.astype(np.int32)
    if np.any(result < 0) or np.any(result >= num_classes):
        raise ValueError("mapped labels must lie in [0, num_classes)")
    return result


@dataclass(frozen=True)
class IntensityScaler:
    minimum: float
    maximum: float

    @classmethod
    def fit(cls, training_volume: np.ndarray) -> "IntensityScaler":
        training = validate_volume(training_volume, name="training volume")
        minimum, maximum = float(training.min()), float(training.max())
        if minimum == maximum:
            raise ValueError("training intensity volume is constant")
        return cls(minimum, maximum)

    def transform(self, volume: np.ndarray, *, clip: bool = True) -> np.ndarray:
        data = validate_volume(volume).astype(np.float32)
        if self.maximum <= self.minimum:
            raise ValueError("scaler maximum must exceed minimum")
        result = 2.0 * (data - self.minimum) / (self.maximum - self.minimum) - 1.0
        return np.clip(result, -1.0, 1.0) if clip else result


def split_regions(shape: tuple[int, ...], validation: int, test: int, gap: int = 0) -> dict:
    if len(shape) != 3 or validation < 1 or test < 1 or gap < 0:
        raise ValueError("invalid spatial split parameters")
    test_start = validation + gap
    train_start = test_start + test + gap
    if train_start >= shape[0]:
        raise ValueError("volume is too short for validation, test, gaps, and training")
    return {
        "validation": (0, validation),
        "test": (test_start, test_start + test),
        "train": (train_start, shape[0]),
    }


def patch_origins(
    shape: tuple[int, ...], patch_size: int, stride: int, *, cover_edges: bool = False
):
    if len(shape) != 3 or patch_size < 1 or not 1 <= stride <= patch_size:
        raise ValueError("invalid patch geometry")
    starts = []
    for size in shape:
        if size < patch_size:
            raise ValueError(f"dimension {size} is smaller than patch size {patch_size}")
        values = list(range(0, size - patch_size + 1, stride))
        if cover_edges and values[-1] != size - patch_size:
            values.append(size - patch_size)
        starts.append(values)
    yield from product(*starts)


def iter_patches(volume: np.ndarray, patch_size: int, stride: int, *, cover_edges: bool = False):
    array = validate_volume(volume)
    for origin in patch_origins(array.shape, patch_size, stride, cover_edges=cover_edges):
        slices = tuple(slice(start, start + patch_size) for start in origin)
        yield origin, array[slices][..., np.newaxis]


def encode_condition(
    labels: np.ndarray, num_classes: int, *, binary: bool = False, pore_label: int = 0
):
    labels = np.asarray(labels)
    if (
        not np.isfinite(labels).all()
        or np.any(labels != np.floor(labels))
        or np.any(labels < 0)
        or np.any(labels >= num_classes)
    ):
        raise ValueError("invalid categorical labels")
    if binary:
        return np.where(labels == pore_label, -1.0, 1.0).astype(np.float32)
    if num_classes < 2:
        raise ValueError("multimineral conditioning requires at least two classes")
    return (2.0 * labels / (num_classes - 1) - 1.0).astype(np.float32)
