"""
wave_module.py
--------------
Learnable damped wave positional modulation layer.

Formula: exp(-alpha * t) * sin(omega * t + phi)

Inserted AFTER CNN reshape, BEFORE transformer blocks.
Toggleable via USE_WAVE flag in config.

Research motivation (2025-2026):
  Wave-based representations encode periodic and decaying temporal
  structure that standard sinusoidal positional encodings lack.
  By making alpha, omega, phi learnable per-channel, the model can
  adapt its positional bias to the statistical structure of handwriting
  strokes — which are quasi-periodic and decay in energy over time.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class DampedWaveModulation(layers.Layer):
    """
    Applies learnable damped wave modulation to sequence features.

    Input shape:  (batch, time_steps, channels)
    Output shape: (batch, time_steps, channels)

    Each channel gets its own (alpha, omega, phi) triplet.
    The wave signal is ADDED to the input (residual style) so that
    if alpha → large, the wave decays to zero and the layer becomes
    an identity — making it safe to insert without breaking the model.
    """

    def __init__(self, num_channels, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.seq_len = seq_len

    def build(self, input_shape):
        # alpha: decay rate — initialized small so wave starts meaningful
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1, 1, self.num_channels),
            initializer=tf.keras.initializers.Constant(0.1),
            constraint=tf.keras.constraints.NonNeg(),  # must be >= 0 for decay
            trainable=True,
        )

        # omega: frequency — initialized to cover ~2 full cycles over seq_len
        self.omega = self.add_weight(
            name="omega",
            shape=(1, 1, self.num_channels),
            initializer=tf.keras.initializers.Constant(
                2.0 * np.pi * 2.0 / max(self.seq_len, 1)
            ),
            trainable=True,
        )

        # phi: phase offset — initialized to zero
        self.phi = self.add_weight(
            name="phi",
            shape=(1, 1, self.num_channels),
            initializer="zeros",
            trainable=True,
        )

        # scale: learned gate on the wave contribution
        # initialized near 0 so early training is stable
        self.scale = self.add_weight(
            name="wave_scale",
            shape=(1, 1, self.num_channels),
            initializer=tf.keras.initializers.Constant(0.01),
            trainable=True,
        )

        # time axis: shape (1, seq_len, 1) — broadcast over batch & channels
        t = np.linspace(0.0, 1.0, self.seq_len, dtype=np.float32)
        self.t = tf.constant(t[np.newaxis, :, np.newaxis], dtype=tf.float32)

        super().build(input_shape)

    def call(self, inputs, training=None):
        t = tf.cast(self.t, inputs.dtype)
        alpha = tf.cast(self.alpha, inputs.dtype)
        omega = tf.cast(self.omega, inputs.dtype)
        phi = tf.cast(self.phi, inputs.dtype)
        scale = tf.cast(self.scale, inputs.dtype)

        # Damped wave: exp(-alpha * t) * sin(omega * t + phi)
        # Shape: (1, seq_len, num_channels) — broadcasts over batch
        wave = tf.exp(-alpha * t) * tf.sin(omega * t + phi)

        # Residual addition with learned scale gate
        # If scale → 0, output ≈ input (safe fallback)
        return inputs + scale * wave

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_channels": self.num_channels,
            "seq_len": self.seq_len,
        })
        return config


def build_wave_module(num_channels, seq_len, name="damped_wave"):
    """
    Factory function — returns a DampedWaveModulation layer.
    Used in model.py so the wave can be toggled cleanly.
    """
    return DampedWaveModulation(
        num_channels=num_channels,
        seq_len=seq_len,
        name=name,
    )
