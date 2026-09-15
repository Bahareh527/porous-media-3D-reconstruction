"""Explicit experiment configuration and safe output locations."""

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str = "synthetic-demo"
    seed: int = 42
    synthetic: bool = True
    image_files: list[str] = field(default_factory=list)
    label_file: str | None = None
    image_variable: str = "tomo"
    label_variable: str = "phase"
    synthetic_shape: list[int] = field(default_factory=lambda: [48, 32, 32])
    validation_slices: int = 16
    test_slices: int = 16
    gap_slices: int = 0
    patch_size: int = 16
    stride: int = 16
    num_classes: int = 4
    pore_label: int = 0
    label_mapping: dict[str, int] = field(default_factory=dict)
    voxel_size_m: float | None = None
    output_dir: str = "outputs"

    def validate(self) -> "ExperimentConfig":
        if not isinstance(self.run_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.run_name):
            raise ValueError("run_name must contain only letters, numbers, hyphens and underscores")
        for name in ("validation_slices", "test_slices", "patch_size", "stride", "num_classes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if type(self.seed) is not int or self.seed < 0 or type(self.synthetic) is not bool:
            raise ValueError("seed must be a nonnegative integer and synthetic a boolean")
        if (
            type(self.gap_slices) is not int
            or type(self.pore_label) is not int
            or self.gap_slices < 0
            or not 0 <= self.pore_label < self.num_classes
        ):
            raise ValueError("invalid split gap or pore label")
        if self.stride > self.patch_size:
            raise ValueError("stride cannot exceed patch_size (would leave uncovered voxels)")
        if len(self.synthetic_shape) != 3 or any(
            type(size) is not int or size < 1 for size in self.synthetic_shape
        ):
            raise ValueError("synthetic_shape must contain three positive sizes")
        if not self.synthetic and (not self.image_files or not self.label_file):
            raise ValueError("real-data mode requires image_files and label_file")
        if self.voxel_size_m is not None and (
            not math.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0
        ):
            raise ValueError("voxel_size_m must be positive and expressed in meters")
        return self

    def destination(self, project_root: str | Path) -> Path:
        root = Path(project_root).resolve()
        destination = (root / self.output_dir / self.run_name).resolve()
        if not destination.is_relative_to(root) or destination == root:
            raise ValueError("output location must remain inside the project root")
        return destination

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a JSON object")
    unknown = set(raw) - set(ExperimentConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    return ExperimentConfig(**raw).validate()
