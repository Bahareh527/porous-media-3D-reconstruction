import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from porous_media.models import build_discriminator, build_hybrid, build_segmentation, build_vox2vox
from porous_media.training import GanTrainer


@pytest.fixture
def arrays():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    target = np.random.default_rng(42).uniform(-1, 1, (1, 16, 16, 16, 1)).astype(np.float32)
    labels = (target > 0).astype(np.int32)
    return target, labels


def trainer():
    shape = (16, 16, 16, 1)
    generator = build_vox2vox(shape, base_filters=2, depth=2, dropout=0)
    discriminator = build_discriminator(shape, base_filters=2, depth=2)
    return GanTrainer(generator, discriminator, class_weights=[1, 2])


@pytest.mark.parametrize("family", ["bce", "least_squares"])
def test_actual_gan_step_and_validation_does_not_update(arrays, family):
    target, labels = arrays
    learner = trainer()
    learner.loss_family = family
    result = learner.step(target, target, labels)
    assert np.isfinite(list(result.values())).all()
    before = [weight.numpy().copy() for weight in learner.generator.weights]
    learner.step(target, target, labels, training=False)
    for expected, weight in zip(before, learner.generator.weights, strict=True):
        np.testing.assert_array_equal(expected, weight)


@pytest.mark.parametrize("residual", [False, True])
def test_segmentation_step(arrays, residual):
    target, labels = arrays
    model = build_segmentation(
        (16, 16, 16, 1), num_classes=2, base_filters=2, depth=2, residual=residual
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    assert np.isfinite(model.train_on_batch(target, labels[..., 0]))
    assert model(target).shape == (1, 16, 16, 16, 2)


def test_hybrid_encoder_is_frozen(arrays):
    target, labels = arrays
    segmenter = build_segmentation((16, 16, 16, 1), base_filters=2, depth=2)
    generator = build_hybrid(segmenter, encoder_layer="encoder_features", decoder_filters=(4, 2))
    discriminator = build_discriminator((16, 16, 16, 1), conditional=False, base_filters=2, depth=2)
    before = [weight.numpy().copy() for weight in segmenter.weights]
    GanTrainer(generator, discriminator, conditional=False, l1_weight=0).step(
        target, target, labels
    )
    for expected, weight in zip(before, segmenter.weights, strict=True):
        np.testing.assert_array_equal(expected, weight)
    with pytest.raises(ValueError):
        build_hybrid(segmenter, encoder_layer="encoder_features", decoder_filters=(2,))


def test_serialization_and_checkpoint_resume(arrays, tmp_path):
    target, labels = arrays
    learner = trainer()

    def batches(split, epoch):
        yield target, target, labels

    history = learner.fit(batches, epochs=1, output_dir=tmp_path)
    assert history[0]["epoch"] == 1
    loaded = tf.keras.models.load_model(tmp_path / "generator.keras", compile=False)
    np.testing.assert_allclose(loaded(target), learner.generator(target), atol=1e-6)
    resumed = trainer()
    resumed.restore(tf.train.latest_checkpoint(str(tmp_path / "state")))
    assert int(resumed.epoch) == 1
    assert int(resumed.generator_optimizer.iterations) == 1
    resumed.fit(batches, epochs=1, output_dir=tmp_path)
    assert int(resumed.epoch) == 2


def test_model_geometry_rejected():
    with pytest.raises(ValueError):
        build_vox2vox((15, 16, 16, 1), base_filters=2, depth=2)
    with pytest.raises(ValueError, match="normalization"):
        build_discriminator((16, 16, 16, 1), base_filters=2, depth=3)


def test_cli_end_to_end(tmp_path):
    from porous_media.cli import main
    from porous_media.config import ExperimentConfig
    from porous_media.preparation import prepare

    config = ExperimentConfig()
    prepare(config, tmp_path, write=True)
    common = [
        "train",
        "--manifest",
        str(config.destination(tmp_path) / "manifest.json"),
        "--project-root",
        str(tmp_path),
        "--filters",
        "2",
        "--depth",
        "2",
        "--batch-size",
        "3",
    ]
    main([*common, "--mode", "segmentation", "--output", "outputs/segment"])
    main([*common, "--mode", "vox2vox", "--output", "outputs/vox"])
    main(
        [
            *common,
            "--mode",
            "hybrid",
            "--output",
            "outputs/hybrid",
            "--segmentation-checkpoint",
            str(tmp_path / "outputs/segment/segmentation.keras"),
            "--encoder-layer",
            "encoder_features",
            "--decoder-filters",
            "4",
            "2",
        ]
    )
    assert (tmp_path / "outputs/hybrid/generator.keras").is_file()
    with pytest.raises(FileExistsError):
        main([*common, "--mode", "vox2vox", "--output", "outputs/vox"])
