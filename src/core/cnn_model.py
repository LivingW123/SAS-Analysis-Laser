"""
cnn_model.py — shared CNN spectrum-regressor architecture and input reshape,
extracted out of train_cnn.py so every script that needs the same model
definition (chunked trainers, inference, comparisons, noise sweep) imports
it from one place instead of reaching into a training script.
"""

import numpy as np
import tensorflow as tf


def reshape_to_2d(X: np.ndarray) -> np.ndarray:
    """
    (n, 200) -> (n, 5, 48, 1)

    Rows 0-3: channels 0-191 reshaped to (4, 48).
    Row 4   : channels 192-199 (the 8 averaged bottom values), each value
              tiled across 6 consecutive columns.
    """
    n = X.shape[0]
    out = np.empty((n, 5, 48), dtype=np.float32)
    out[:, :4, :] = X[:, :192].reshape(n, 4, 48)
    out[:, 4, :] = np.repeat(X[:, 192:], 6, axis=1)  # (n, 8) -> (n, 48)
    return out[..., np.newaxis]


def build_model(n_bins: int) -> tf.keras.Model:
    """
    5x48x1 -> conv stack -> dense head -> n_bins (softmax).

    The image is only 5 rows tall, so pooling happens along the width axis
    only; 3x3 convs with 'same' padding keep the row dimension intact.
    """
    inp = tf.keras.Input(shape=(5, 48, 1), name="detector_image")

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x)   # 5x24

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x)   # 5x12

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    out = tf.keras.layers.Dense(n_bins, activation="softmax", name="spectrum")(x)
    return tf.keras.Model(inp, out, name=f"cnn_spectrum_regressor_n{n_bins}")
