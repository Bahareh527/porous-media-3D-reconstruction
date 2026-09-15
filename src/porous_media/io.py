"""Context-managed volume loading and paired patch manifests."""

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_netcdf(paths, variable: str) -> np.ndarray:
    try:
        from netCDF4 import Dataset
    except ImportError as error:
        raise ImportError("NetCDF inputs require pip install '.[netcdf]'") from error
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not paths:
        raise ValueError("at least one NetCDF block is required")
    blocks = []
    for path in paths:
        with Dataset(path, "r") as dataset:
            if variable not in dataset.variables:
                raise KeyError(f"variable {variable!r} not found in {Path(path).name}")
            values = dataset.variables[variable][:]
            if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
                raise ValueError("NetCDF contains masked voxels; resolve them explicitly")
            blocks.append(np.asarray(values))
    return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported patch manifest schema")
    if not manifest.get("patches"):
        raise ValueError("manifest contains no patches")
    return manifest


def patch_batches(
    manifest_path: str | Path, split: str, batch_size: int, *, shuffle: bool = False, seed: int = 42
):
    path = Path(manifest_path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    manifest = load_manifest(path)
    entries = [item for item in manifest["patches"] if item["split"] == split]
    if not entries:
        raise ValueError(f"manifest has no {split!r} patches")
    if shuffle:
        np.random.default_rng(seed).shuffle(entries)
    for start in range(0, len(entries), batch_size):
        images, labels = [], []
        for entry in entries[start : start + batch_size]:
            file = (path.parent / entry["file"]).resolve()
            if not file.is_relative_to(path.parent.resolve()):
                raise ValueError("patch path escapes the manifest directory")
            if sha256(file) != entry["sha256"]:
                raise ValueError(f"patch checksum mismatch: {file.name}")
            with np.load(file, allow_pickle=False) as patch:
                image, label = patch["image"], patch["labels"]
                if image.shape != label.shape or image.ndim != 4 or image.shape[-1] != 1:
                    raise ValueError("image and labels must be aligned channels-last 3D patches")
                if not np.isfinite(image).all() or not np.isfinite(label).all():
                    raise ValueError("patches must contain finite values")
                if (
                    np.any(label != np.floor(label))
                    or np.any(label < 0)
                    or np.any(label >= manifest["config"]["num_classes"])
                ):
                    raise ValueError("patch labels must be valid integer class IDs")
                images.append(image.astype(np.float32))
                labels.append(label.astype(np.int32))
        yield np.stack(images), np.stack(labels)
