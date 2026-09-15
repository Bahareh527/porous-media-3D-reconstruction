import json
from dataclasses import replace

import numpy as np
import pytest

from porous_media.config import ExperimentConfig, load_config
from porous_media.flow import (
    MILLIDARCY_M2,
    darcy_permeability,
    invasion_limits,
    occupancy_and_saturation,
)
from porous_media.inference import predict_volume
from porous_media.io import patch_batches
from porous_media.metrics import (
    axial_two_point,
    porosity,
    reconstruction_metrics,
    segmentation_metrics,
)
from porous_media.preparation import prepare
from porous_media.volumes import (
    IntensityScaler,
    encode_condition,
    iter_patches,
    remap_labels,
    split_regions,
)


@pytest.mark.parametrize(
    "changes",
    [
        {"patch_size": 0},
        {"stride": 17},
        {"run_name": "../bad"},
        {"voxel_size_m": float("nan")},
        {"patch_size": 1.5},
    ],
)
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(ExperimentConfig(), **changes).validate()


def test_config_unknown_fields_and_escape(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"typo": 5}')
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)
    with pytest.raises(ValueError, match="inside"):
        replace(ExperimentConfig(), output_dir="../outside").destination(tmp_path)


def test_label_mapping_is_not_cascading():
    labels = np.array([0, 1, 2]).reshape(3, 1, 1)
    assert remap_labels(labels, {0: 1, 1: 2, 2: 0}, 3).ravel().tolist() == [1, 2, 0]
    with pytest.raises(ValueError, match="omits"):
        remap_labels(labels, {0: 0}, 3)
    with pytest.raises(ValueError, match="integers"):
        remap_labels(labels + 0.5, {}, 3)
    with pytest.raises(ValueError):
        remap_labels(labels, {0: 0.5, 1: 1, 2: 2}, 3)


def test_scaler_uses_training_statistics():
    scaler = IntensityScaler.fit(np.arange(8).reshape(2, 2, 2))
    validation = np.full((2, 2, 2), 100)
    assert scaler.transform(validation).max() == 1
    assert scaler.transform(validation, clip=False).min() > 1
    assert (scaler.minimum, scaler.maximum) == (0, 7)
    with pytest.raises(ValueError, match="constant"):
        IntensityScaler.fit(validation)


def test_split_and_patch_grid():
    assert split_regions((30, 8, 8), 8, 8, 2) == {
        "validation": (0, 8),
        "test": (10, 18),
        "train": (20, 30),
    }
    volume = np.arange(10**3).reshape(10, 10, 10)
    for origin, patch in iter_patches(volume, 4, 3):
        slices = tuple(slice(start, start + 4) for start in origin)
        np.testing.assert_array_equal(patch[..., 0], volume[slices])


def test_preparation_and_paired_batches(tmp_path):
    config = ExperimentConfig()
    dry = prepare(config, tmp_path)
    assert not config.destination(tmp_path).exists()
    assert len(dry["patches"]) == 12
    saved = prepare(config, tmp_path, write=True)
    manifest = config.destination(tmp_path) / "manifest.json"
    batches = list(patch_batches(manifest, "train", 3))
    assert [len(images) for images, _ in batches] == [3, 1]
    assert all(images.shape == labels.shape for images, labels in batches)
    assert saved["synthetic_only"]
    with pytest.raises(FileExistsError):
        prepare(config, tmp_path, write=True)
    metadata = json.loads(manifest.read_text())
    metadata["patches"][0]["sha256"] = "invalid"
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum"):
        list(patch_batches(manifest, "validation", 1))


def test_preparation_preflights_all_splits(tmp_path):
    config = replace(ExperimentConfig(), validation_slices=1)
    with pytest.raises(ValueError, match="smaller"):
        prepare(config, tmp_path, write=True)
    assert not config.destination(tmp_path).exists()


def test_partial_run_is_not_overwritten(tmp_path):
    config = ExperimentConfig()
    config.destination(tmp_path).mkdir(parents=True)
    (config.destination(tmp_path) / "partial.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        prepare(config, tmp_path, write=True)


def test_overlap_inference_covers_edges():
    image = np.random.default_rng(42).normal(size=(11, 10, 9)).astype(np.float32)
    prediction = predict_volume(lambda batch: batch, image, patch_size=4, stride=3)
    np.testing.assert_allclose(prediction, image, atol=1e-6)


def test_metrics_measured_and_json_safe():
    reference = np.arange(125).reshape(5, 5, 5) / 125
    identical = reconstruction_metrics(reference, reference)
    assert identical["ssim"] == 1
    assert identical["psnr_db"] is None
    json.dumps(identical, allow_nan=False)
    measured = reconstruction_metrics(reference, reference + 0.1)
    assert measured["mse"] == pytest.approx(0.01)
    assert measured["psnr_db"] == pytest.approx(10 * np.log10(400))
    with pytest.raises(ValueError):
        reconstruction_metrics(reference, reference, data_range=float("nan"))


def test_pore_and_segmentation_metrics():
    mask = np.arange(27).reshape(3, 3, 3) % 2 == 0
    _, probability = axial_two_point(mask, max_lag=2)
    assert probability[0] == porosity(mask)
    with pytest.raises(ValueError):
        porosity(mask.astype(float))
    measured = segmentation_metrics([0, 0, 1, 1], [0, 1, 1, 1], 3)
    assert measured["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]
    assert measured["accuracy"] == 0.75
    assert measured["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_condition_encoding_and_nonfinite_rejection():
    np.testing.assert_array_equal(encode_condition(np.array([0, 1]), 2), [-1, 1])
    np.testing.assert_array_equal(encode_condition(np.array([0, 1]), 2, binary=True), [-1, 1])
    with pytest.raises(ValueError):
        encode_condition(np.array([np.nan]), 2)


def test_invasion_excludes_uninvaded_and_handles_small_network():
    pores, throats, saturation = occupancy_and_saturation([-1, 0, 2], [-1, 1], [1, 1, 1], [1, 1], 2)
    assert pores.tolist() == [False, True, False]
    assert throats.tolist() == [False, True]
    assert saturation == 0.4
    assert invasion_limits([0], [-1]).tolist() == [0, 1]


def test_darcy_si_conversion():
    result = darcy_permeability(1e-12, 1e-3, 1e-3, 1e-6, 1)
    assert result["permeability_m2"] == pytest.approx(1e-12, abs=1e-20)
    assert result["permeability_md"] == pytest.approx(1e-12 / MILLIDARCY_M2)
    with pytest.raises(ValueError):
        darcy_permeability(1, 0, 1, 1, 1)
