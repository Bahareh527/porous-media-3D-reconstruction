"""Small actual NetCDF fixtures; explicitly optional in lightweight installations."""

from dataclasses import replace

import numpy as np
import pytest

from porous_media.config import ExperimentConfig
from porous_media.io import read_netcdf
from porous_media.preparation import prepare


def test_real_file_preparation(tmp_path):
    netcdf = pytest.importorskip("netCDF4")
    image = np.random.default_rng(42).normal(size=(48, 32, 32)).astype(np.float32)
    labels = (image > 0).astype(np.int32)
    for name, key, array in (("image.nc", "tomo", image), ("labels.nc", "phase", labels)):
        with netcdf.Dataset(tmp_path / name, "w") as dataset:
            for dimension, size in zip(("z", "y", "x"), array.shape, strict=True):
                dataset.createDimension(dimension, size)
            dataset.createVariable(key, array.dtype, ("z", "y", "x"))[:] = array
    config = replace(
        ExperimentConfig(),
        synthetic=False,
        image_files=["image.nc"],
        label_file="labels.nc",
        num_classes=2,
    )
    prepared = prepare(config, tmp_path)
    assert not prepared["synthetic_only"]
    assert prepared["normalization"]["minimum"] == float(image[32:].min())
    assert len(prepared["inputs"]) == 2
    with pytest.raises(KeyError):
        read_netcdf(tmp_path / "image.nc", "missing")


def test_masked_netcdf_rejected(tmp_path):
    netcdf = pytest.importorskip("netCDF4")
    with netcdf.Dataset(tmp_path / "masked.nc", "w") as dataset:
        for dimension in ("z", "y", "x"):
            dataset.createDimension(dimension, 3)
        variable = dataset.createVariable("tomo", "f4", ("z", "y", "x"), fill_value=-99)
        variable[:] = np.full((3, 3, 3), -99)
    with pytest.raises(ValueError, match="masked"):
        read_netcdf(tmp_path / "masked.nc", "tomo")
