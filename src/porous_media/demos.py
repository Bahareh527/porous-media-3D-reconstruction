"""Small, self-contained smoke checks; never represent thesis results."""

import numpy as np

from porous_media.metrics import reconstruction_metrics, segmentation_metrics
from porous_media.preparation import synthetic_rock
from porous_media.volumes import IntensityScaler, encode_condition


def run_learning_demo(mode: str) -> dict:
    import tensorflow as tf

    from porous_media.models import (
        build_discriminator,
        build_hybrid,
        build_segmentation,
        build_vox2vox,
    )
    from porous_media.training import GanTrainer

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    images, labels = synthetic_rock((16, 16, 16), seed=42)
    images = IntensityScaler.fit(images).transform(images)[None, ..., None]
    labels = labels[None, ..., None]
    shape = (16, 16, 16, 1)
    if mode == "segmentation":
        models = {}
        for residual in (False, True):
            model = build_segmentation(
                shape, num_classes=4, base_filters=2, depth=2, residual=residual
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy"
            )
            loss = float(model.train_on_batch(images, labels[..., 0]))
            predicted = np.asarray(model(images, training=False)).argmax(-1)
            models[model.name] = {
                "loss": loss,
                **segmentation_metrics(labels[..., 0], predicted, 4),
            }
        return {"synthetic_only": True, "models": models}
    if mode == "vox2vox":
        generator = build_vox2vox(shape, base_filters=2, depth=2, dropout=0.0)
        condition = encode_condition(labels, 4)
        conditional, l1_weight = True, 100.0
    elif mode == "hybrid":
        segmenter = build_segmentation(shape, base_filters=2, depth=2)
        segmenter.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        segmenter.train_on_batch(images, labels[..., 0])
        generator = build_hybrid(
            segmenter, encoder_layer="encoder_features", decoder_filters=(4, 2)
        )
        condition = images
        conditional, l1_weight = False, 0.0
    else:
        raise ValueError("mode must be vox2vox, segmentation or hybrid")
    discriminator = build_discriminator(shape, conditional=conditional, base_filters=2, depth=2)
    trainer = GanTrainer(generator, discriminator, conditional=conditional, l1_weight=l1_weight)
    losses = trainer.step(condition, images, labels)
    predicted = np.asarray(generator(condition, training=False))
    return {
        "synthetic_only": True,
        "losses": losses,
        "metrics": reconstruction_metrics(images[0, ..., 0], predicted[0, ..., 0]),
    }
