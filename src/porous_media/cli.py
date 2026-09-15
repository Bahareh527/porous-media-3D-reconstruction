"""Command-line preparation, training, evaluation and offline smoke checks."""

import argparse
import json
from pathlib import Path

import numpy as np

from porous_media.config import load_config
from porous_media.io import load_manifest, patch_batches, sha256
from porous_media.metrics import reconstruction_metrics
from porous_media.preparation import prepare
from porous_media.volumes import encode_condition


def train(args):
    import tensorflow as tf

    from porous_media.models import (
        build_discriminator,
        build_hybrid,
        build_segmentation,
        build_vox2vox,
    )
    from porous_media.training import GanTrainer

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    root = Path(args.project_root).resolve()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root / "outputs"):
        raise ValueError("training output must be inside project_root/outputs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("training output is nonempty; select a new run directory")
    tf.keras.utils.set_random_seed(args.seed)
    first_images, _ = next(patch_batches(manifest_path, "train", args.batch_size))
    next(patch_batches(manifest_path, "validation", args.batch_size))
    shape = tuple(first_images.shape[1:])
    classes = manifest["config"]["num_classes"]
    pore_label = manifest["config"]["pore_label"]
    if args.mode == "segmentation":
        model = build_segmentation(
            shape,
            num_classes=classes,
            base_filters=args.filters,
            depth=args.depth,
            residual=args.residual,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss="sparse_categorical_crossentropy",
        )
        output.mkdir(parents=True, exist_ok=True)
        history = []
        for epoch in range(args.epochs):
            model.reset_metrics()
            for images, labels in patch_batches(
                manifest_path, "train", args.batch_size, shuffle=True, seed=args.seed + epoch
            ):
                train_loss = float(model.train_on_batch(images, labels[..., 0]))
            model.reset_metrics()
            for images, labels in patch_batches(manifest_path, "validation", args.batch_size):
                validation_loss = float(model.test_on_batch(images, labels[..., 0]))
            if not np.isfinite([train_loss, validation_loss]).all():
                raise ValueError("segmentation training produced a nonfinite loss")
            history.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss}
            )
        model.save(output / "segmentation.keras")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    else:
        conditional = args.mode == "vox2vox"
        if conditional:
            generator = build_vox2vox(shape, base_filters=args.filters, depth=args.depth)
        else:
            if not args.segmentation_checkpoint or not args.encoder_layer:
                raise ValueError(
                    "hybrid training requires a segmentation checkpoint and explicit encoder layer"
                )
            segmenter = tf.keras.models.load_model(args.segmentation_checkpoint, compile=False)
            generator = build_hybrid(
                segmenter,
                encoder_layer=args.encoder_layer,
                decoder_filters=tuple(args.decoder_filters),
            )
        discriminator = build_discriminator(
            shape, conditional=conditional, base_filters=args.filters, depth=args.depth
        )
        l1_weight = (
            args.l1_weight if args.l1_weight is not None else (100.0 if conditional else 0.0)
        )
        trainer = GanTrainer(
            generator,
            discriminator,
            conditional=conditional,
            loss_family=args.loss_family,
            l1_weight=l1_weight,
            learning_rate=args.learning_rate,
        )

        def batches(split, epoch):
            for images, labels in patch_batches(
                manifest_path,
                split,
                args.batch_size,
                shuffle=split == "train",
                seed=args.seed + epoch,
            ):
                condition = (
                    encode_condition(labels, classes, binary=args.binary, pore_label=pore_label)
                    if conditional
                    else images
                )
                yield condition, images, labels

        trainer.fit(
            batches,
            epochs=args.epochs,
            output_dir=output,
            metadata={
                "manifest_sha256": sha256(manifest_path),
                "synthetic_only": manifest["synthetic_only"],
            },
        )
    (output / "training_config.json").write_text(
        json.dumps({key: value for key, value in vars(args).items() if key != "command"}, indent=2),
        encoding="utf-8",
    )
    return {"output": str(output), "synthetic_only": manifest["synthetic_only"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser(
        "prepare", help="validate and optionally write paired patches"
    )
    preparation.add_argument("--config", default="configs/demo.json")
    preparation.add_argument("--project-root", default=".")
    preparation.add_argument(
        "--write", action="store_true", help="write patches; disabled by default"
    )
    demo = commands.add_parser("demo", help="run one small synthetic training smoke check")
    demo.add_argument("--mode", choices=("vox2vox", "segmentation", "hybrid"), default="vox2vox")
    evaluation = commands.add_parser(
        "evaluate", help="measure reconstruction metrics from real arrays"
    )
    evaluation.add_argument("reference")
    evaluation.add_argument("prediction")
    evaluation.add_argument("--data-range", type=float, default=2.0)
    training = commands.add_parser("train", help="train from a paired patch manifest")
    training.add_argument("--manifest", required=True)
    training.add_argument("--mode", choices=("vox2vox", "segmentation", "hybrid"), required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--project-root", default=".")
    training.add_argument("--epochs", type=int, default=1)
    training.add_argument("--batch-size", type=int, default=1)
    training.add_argument("--filters", type=int, default=16)
    training.add_argument("--depth", type=int, default=3)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--learning-rate", type=float, default=2e-4)
    training.add_argument(
        "--loss-family", choices=("bce", "least_squares"), default="least_squares"
    )
    training.add_argument("--l1-weight", type=float)
    training.add_argument("--binary", action="store_true")
    training.add_argument("--residual", action="store_true")
    training.add_argument("--segmentation-checkpoint")
    training.add_argument("--encoder-layer")
    training.add_argument("--decoder-filters", nargs="+", type=int, default=[512, 256, 128, 64, 32])
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(load_config(args.config), args.project_root, write=args.write)
        result = {
            "synthetic_only": result["synthetic_only"],
            "patch_counts": {
                split: sum(item["split"] == split for item in result["patches"])
                for split in ("train", "validation", "test")
            },
            "write_enabled": args.write,
        }
    elif args.command == "demo":
        from porous_media.demos import run_learning_demo

        result = run_learning_demo(args.mode)
    elif args.command == "evaluate":
        result = reconstruction_metrics(
            np.load(args.reference, allow_pickle=False),
            np.load(args.prediction, allow_pickle=False),
            data_range=args.data_range,
        )
    else:
        if args.epochs < 1 or args.learning_rate <= 0:
            parser.error("epochs and learning_rate must be positive")
        result = train(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
