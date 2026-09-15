"""Named, configurable TensorFlow architectures derived from the thesis notebooks.

The Vox2Vox down/up topology is retained. Native U-Net/residual U-Net and the
shape-inferred hybrid decoder are reference implementations, not substitutes
for the historical ResNet18 checkpoint or an exact publication rerun.
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="porous_media")
class InstanceNormalization(layers.Layer):
    """Channels-last 3D instance normalization without TensorFlow Addons."""

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale", shape=(input_shape[-1],), initializer="ones")
        self.offset = self.add_weight(name="offset", shape=(input_shape[-1],), initializer="zeros")

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True)
        return (inputs - mean) * tf.math.rsqrt(variance + self.epsilon) * self.scale + self.offset

    def get_config(self):
        return {**super().get_config(), "epsilon": self.epsilon}


def _check_geometry(input_shape, depth, base_filters):
    if len(input_shape) != 4 or min(input_shape) < 1 or depth < 1 or base_filters < 1:
        raise ValueError("invalid model geometry")
    if any(size % (2**depth) for size in input_shape[:3]):
        raise ValueError("spatial dimensions must be divisible by 2**depth")


def _down(x, filters, kernel=4, *, normalize=True, dropout=0.2):
    x = layers.Conv3D(filters, kernel, strides=2, padding="same", kernel_initializer="he_normal")(x)
    if normalize:
        x = InstanceNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    return layers.Dropout(dropout)(x)


def build_vox2vox(
    input_shape=(128, 128, 128, 1), *, base_filters=64, depth=4, dropout=0.2, bottleneck_blocks=1
):
    _check_geometry(input_shape, depth, base_filters)
    if not 0 <= dropout < 1 or bottleneck_blocks < 0:
        raise ValueError("invalid dropout or bottleneck count")
    inputs = layers.Input(input_shape, name="conditioning")
    x, skips = inputs, []
    for level in range(depth - 1):
        x = _down(x, base_filters * 2**level, normalize=level > 0, dropout=dropout)
        skips.append(x)
    x = _down(x, base_filters * 2 ** (depth - 1), dropout=0.0)
    for _ in range(bottleneck_blocks):
        raw = layers.Conv3D(
            base_filters * 2 ** (depth - 1), 4, padding="same", kernel_initializer="he_normal"
        )(x)
        x = layers.LeakyReLU(negative_slope=0.2)(InstanceNormalization()(raw))
        x = layers.Concatenate()([x, raw])
    for level in range(depth - 2, -1, -1):
        x = layers.Conv3DTranspose(base_filters * 2**level, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(negative_slope=0.2)(InstanceNormalization()(x))
        x = layers.Dropout(dropout)(layers.Concatenate()([x, skips.pop()]))
    output = layers.Conv3DTranspose(
        1, 4, strides=2, padding="same", activation="tanh", name="reconstruction"
    )(x)
    return tf.keras.Model(inputs, output, name="vox2vox_generator")


def build_discriminator(input_shape, *, conditional=True, base_filters=64, depth=4):
    _check_geometry(input_shape, depth, base_filters)
    if min(input_shape[:3]) // 2**depth < 3:
        raise ValueError(
            "discriminator bottleneck requires at least three voxels per axis; "
            "reduce depth or increase patch size to avoid degenerate normalization"
        )
    target = layers.Input((*input_shape[:3], 1), name="target")
    if conditional:
        condition = layers.Input(input_shape, name="condition")
        x = layers.Concatenate()([condition, target])
        inputs = [condition, target]
    else:
        x, inputs = target, target
    for level in range(depth):
        x = _down(x, base_filters * 2**level, normalize=level > 0)
    x = layers.ZeroPadding3D()(x)
    x = layers.Conv3D(base_filters * 2**depth, 4)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(InstanceNormalization()(x))
    logits = layers.Conv3D(1, 4, name="patch_logits")(layers.ZeroPadding3D()(x))
    return tf.keras.Model(inputs, logits, name="patch_discriminator")


def _segmentation_block(x, filters, residual):
    shortcut = x
    x = layers.Conv3D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv3D(filters, 3, padding="same")(x)
    if residual:
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv3D(filters, 1, padding="same")(shortcut)
        x = layers.Add()([x, shortcut])
    return layers.ReLU()(x)


def build_segmentation(
    input_shape=(128, 128, 128, 1), *, num_classes=4, base_filters=16, depth=3, residual=False
):
    _check_geometry(input_shape, depth, base_filters)
    if num_classes < 2:
        raise ValueError("segmentation requires at least two classes")
    inputs = layers.Input(input_shape, name="grayscale_volume")
    x, skips = inputs, []
    for level in range(depth):
        x = _segmentation_block(x, base_filters * 2**level, residual)
        skips.append(x)
        x = layers.MaxPool3D()(x)
    x = _segmentation_block(x, base_filters * 2**depth, residual)
    x = layers.Activation("linear", name="encoder_features")(x)
    for level in range(depth - 1, -1, -1):
        x = layers.Conv3DTranspose(base_filters * 2**level, 2, strides=2, padding="same")(x)
        x = _segmentation_block(
            layers.Concatenate()([x, skips.pop()]), base_filters * 2**level, residual
        )
    outputs = layers.Conv3D(num_classes, 1, activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="residual_unet3d" if residual else "unet3d")


def build_hybrid(
    segmentation_model, *, encoder_layer, decoder_filters=(512, 256, 128, 64, 32), noise_std=0.1
):
    """Freeze an explicitly selected trained encoder; infer decoder spatial shape."""
    try:
        encoder = tf.keras.Model(
            segmentation_model.input,
            segmentation_model.get_layer(encoder_layer).output,
            name="frozen_encoder",
        )
    except ValueError as error:
        raise ValueError(
            f"encoder layer {encoder_layer!r} not found; inspect model.layers"
        ) from error
    encoder.trainable = False
    input_shape = tuple(segmentation_model.input_shape[1:])
    feature_shape = tuple(encoder.output_shape[1:])
    if len(feature_shape) != 4 or any(size is None for size in (*input_shape, *feature_shape)):
        raise ValueError("hybrid encoder must expose fixed channels-last 3D features")
    ratios = [
        size / feature for size, feature in zip(input_shape[:3], feature_shape[:3], strict=True)
    ]
    ratio = int(ratios[0])
    if len(set(ratios)) != 1 or ratio < 1 or ratio != ratios[0] or ratio & (ratio - 1):
        raise ValueError("encoder spatial downsampling must be an isotropic power of two")
    depth = ratio.bit_length() - 1
    if len(decoder_filters) != depth or min(decoder_filters, default=1) < 1 or noise_std < 0:
        raise ValueError(f"hybrid decoder requires exactly {depth} positive filter counts")
    inputs = layers.Input(input_shape, name="hybrid_input")
    x = encoder(inputs, training=False)
    x = layers.Conv3D(feature_shape[-1], 3, padding="same", activation="relu")(x)
    x = layers.GaussianNoise(noise_std)(x)
    for filters in decoder_filters:
        x = layers.Conv3DTranspose(filters, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(negative_slope=0.2)(layers.BatchNormalization()(x))
    output = layers.Conv3D(1, 3, padding="same", activation="tanh", name="hybrid_reconstruction")(x)
    return tf.keras.Model(inputs, output, name="segmentation_informed_hybrid")
