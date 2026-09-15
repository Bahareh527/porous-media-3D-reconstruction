"""One coherent GAN objective, finite epoch iterators, and resumable state."""

import json
import math
from pathlib import Path

import tensorflow as tf


class GanTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        *,
        conditional=True,
        loss_family="least_squares",
        l1_weight=100.0,
        class_weights=None,
        learning_rate=2e-4,
    ):
        if (
            loss_family not in {"bce", "least_squares"}
            or not math.isfinite(l1_weight)
            or not math.isfinite(learning_rate)
            or l1_weight < 0
            or learning_rate <= 0
        ):
            raise ValueError("invalid GAN objective or learning rate")
        if class_weights is not None and (
            not class_weights
            or any(not math.isfinite(value) or value <= 0 for value in class_weights)
        ):
            raise ValueError("class weights must be positive")
        self.generator, self.discriminator = generator, discriminator
        self.conditional, self.loss_family, self.l1_weight = conditional, loss_family, l1_weight
        self.class_weights = class_weights
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.checkpoint = tf.train.Checkpoint(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            epoch=self.epoch,
        )

    def _adversarial(self, expected, logits):
        if self.loss_family == "least_squares":
            return tf.reduce_mean(tf.square(expected - logits))
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=expected, logits=logits)
        )

    def step(self, condition, target, labels=None, *, training=True):
        condition, target = (
            tf.convert_to_tensor(condition, tf.float32),
            tf.convert_to_tensor(target, tf.float32),
        )
        tf.debugging.assert_all_finite(condition, "nonfinite conditioning")
        tf.debugging.assert_all_finite(target, "nonfinite target")
        tf.debugging.assert_equal(tf.shape(condition)[:4], tf.shape(target)[:4])
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated = self.generator(condition, training=training)
            tf.debugging.assert_equal(tf.shape(generated), tf.shape(target))
            real_inputs = [condition, target] if self.conditional else target
            fake_inputs = [condition, generated] if self.conditional else generated
            real = self.discriminator(real_inputs, training=training)
            fake = self.discriminator(fake_inputs, training=training)
            discriminator_loss = 0.5 * (
                self._adversarial(tf.ones_like(real), real)
                + self._adversarial(tf.zeros_like(fake), fake)
            )
            adversarial_loss = self._adversarial(tf.ones_like(fake), fake)
            error = tf.abs(target - generated)
            if self.class_weights is not None:
                if labels is None:
                    raise ValueError("weighted reconstruction requires separate integer labels")
                labels = tf.convert_to_tensor(labels)
                tf.debugging.assert_equal(tf.shape(labels), tf.shape(target))
                tf.debugging.assert_equal(
                    tf.cast(labels, tf.float32), tf.floor(tf.cast(labels, tf.float32))
                )
                indices = tf.cast(labels, tf.int32)
                tf.debugging.assert_greater_equal(indices, 0)
                tf.debugging.assert_less(indices, len(self.class_weights))
                error *= tf.gather(tf.constant(self.class_weights, tf.float32), indices)
            l1_loss = tf.reduce_mean(error)
            generator_loss = adversarial_loss + self.l1_weight * l1_loss
        for loss in (generator_loss, discriminator_loss):
            tf.debugging.assert_all_finite(loss, "nonfinite GAN loss")
        if training:
            for tape, loss, model, optimizer in (
                (generator_tape, generator_loss, self.generator, self.generator_optimizer),
                (
                    discriminator_tape,
                    discriminator_loss,
                    self.discriminator,
                    self.discriminator_optimizer,
                ),
            ):
                gradients = tape.gradient(loss, model.trainable_variables)
                if any(gradient is None for gradient in gradients):
                    raise ValueError("a trainable variable is disconnected from its loss")
                for gradient in gradients:
                    tf.debugging.assert_all_finite(gradient, "nonfinite gradient")
                optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=True))
        return {
            "generator_loss": float(generator_loss),
            "discriminator_loss": float(discriminator_loss),
            "l1_loss": float(l1_loss),
            "adversarial_loss": float(adversarial_loss),
        }

    def fit(self, batch_factory, *, epochs: int, output_dir, metadata=None):
        if epochs < 1:
            raise ValueError("epochs must be positive")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        manager = tf.train.CheckpointManager(self.checkpoint, str(output / "state"), max_to_keep=3)
        history = []
        for _ in range(epochs):
            record = {"epoch": int(self.epoch) + 1}
            for split in ("train", "validation"):
                totals, count = {}, 0
                for condition, target, labels in batch_factory(split, int(self.epoch)):
                    result = self.step(condition, target, labels, training=split == "train")
                    size = len(target)
                    count += size
                    for name, value in result.items():
                        totals[name] = totals.get(name, 0.0) + size * value
                if not count:
                    raise ValueError(f"{split} iterator is empty")
                record[split] = {name: value / count for name, value in totals.items()}
            self.epoch.assign_add(1)
            manager.save()
            history.append(record)
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.generator.save(output / "generator.keras")
        self.discriminator.save(output / "discriminator.keras")
        (output / "run_metadata.json").write_text(
            json.dumps(
                {
                    "loss_family": self.loss_family,
                    "l1_weight": self.l1_weight,
                    "class_weights": self.class_weights,
                    "metadata": metadata or {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return history

    def restore(self, checkpoint_path):
        self.generator_optimizer.build(self.generator.trainable_variables)
        self.discriminator_optimizer.build(self.discriminator.trainable_variables)
        self.checkpoint.restore(str(checkpoint_path)).assert_existing_objects_matched()
