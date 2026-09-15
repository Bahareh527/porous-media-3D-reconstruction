"""Streaming paired patch preparation with provenance and no spatial resampling."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from porous_media.config import ExperimentConfig
from porous_media.io import read_netcdf, sha256
from porous_media.volumes import (
    IntensityScaler,
    iter_patches,
    patch_origins,
    remap_labels,
    split_regions,
    validate_volume,
)


def synthetic_rock(shape=(48, 32, 32), seed: int = 42, num_classes: int = 4):
    """Make a synthetic texture/label pair; not a physical rock or thesis data."""
    rng = np.random.default_rng(seed)
    texture = gaussian_filter(rng.normal(size=shape), sigma=1.0)
    thresholds = np.quantile(texture, np.arange(1, num_classes) / num_classes)
    labels = np.digitize(texture, thresholds).astype(np.int32)
    images = (labels * 60.0 + 10.0 * rng.normal(size=shape)).astype(np.float32)
    return images, labels


def prepare(config: ExperimentConfig, project_root: str | Path, *, write: bool = False) -> dict:
    config.validate()
    root = Path(project_root).resolve()
    destination = config.destination(root)
    provenance = []
    if config.synthetic:
        images, labels = synthetic_rock(config.synthetic_shape, config.seed, config.num_classes)
    else:
        image_paths = [root / path for path in config.image_files]
        label_path = root / config.label_file
        images = read_netcdf(image_paths, config.image_variable)
        labels = read_netcdf(label_path, config.label_variable)
        provenance = [
            {"file": Path(path).name, "sha256": sha256(path)} for path in [*image_paths, label_path]
        ]
    images = validate_volume(images, name="images")
    if images.shape != np.asarray(labels).shape:
        raise ValueError("image and label volumes are not aligned")
    labels = remap_labels(
        labels, {int(k): v for k, v in config.label_mapping.items()}, config.num_classes
    )
    regions = split_regions(
        images.shape, config.validation_slices, config.test_slices, config.gap_slices
    )
    start, stop = regions["train"]
    scaler = IntensityScaler.fit(images[start:stop])
    metadata = {
        "schema_version": 1,
        "synthetic_only": config.synthetic,
        "config": config.to_dict(),
        "shape": list(images.shape),
        "normalization": asdict(scaler),
        "normalization_clipped": True,
        "label_values": np.unique(labels).tolist(),
        "regions": regions,
        "inputs": provenance,
        "patches": [],
    }
    # Validate every split before writing anything, including a partial previous run.
    for start, stop in regions.values():
        next(patch_origins((stop - start, *images.shape[1:]), config.patch_size, config.stride))
    if write and destination.exists() and any(destination.iterdir()):
        raise FileExistsError("run already exists; choose another run_name rather than overwrite")
    for split, (start, stop) in regions.items():
        scaled = scaler.transform(images[start:stop])
        split_labels = labels[start:stop]
        pairs = zip(
            iter_patches(scaled, config.patch_size, config.stride),
            iter_patches(split_labels, config.patch_size, config.stride),
            strict=True,
        )
        for index, ((origin, image_patch), (label_origin, label_patch)) in enumerate(pairs):
            if origin != label_origin:
                raise ValueError("image and label patch grids differ")
            relative = f"{split}/patch_{index:06d}.npz"
            entry = {"split": split, "file": relative, "origin": [origin[0] + start, *origin[1:]]}
            if write:
                path = destination / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, image=image_patch, labels=label_patch)
                entry["sha256"] = sha256(path)
            metadata["patches"].append(entry)
    if write:
        (destination / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
