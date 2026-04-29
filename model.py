"""
model.py
--------
Builds the HTR model with optional wave module.

Architecture:
  Baseline : CNN → Positional Encoding → Transformer x4 → BiLSTM → Dense → CTC
  Research : CNN → [DampedWave] → Positional Encoding → Transformer x4 → BiLSTM → Dense → CTC

The wave module is inserted as a residual addition, so with scale≈0
at init the model starts as the baseline and gradually learns to use
the wave contribution — preventing any instability on first epoch.
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from config import IMG_HEIGHT, IMG_WIDTH, USE_WAVE
from wave_module import build_wave_module


# ─── POSITIONAL ENCODING ──────────────────────────────────────────────────────

class PositionalEncoding(layers.Layer):
    """Standard sinusoidal positional encoding (fixed, not learned)."""

    def build(self, input_shape):
        d_model = int(input_shape[-1])
        seq_len = int(input_shape[-2])
        positions = np.arange(seq_len)[:, np.newaxis]
        half_dim = max(d_model // 2, 1)
        div_term = np.exp(np.arange(half_dim) * -(math.log(10000.0) / half_dim))
        angles = positions * div_term[np.newaxis, :]
        pe = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
        pe = pe[:, :d_model].astype(np.float32)
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        return inputs + tf.cast(self.pe, inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


# ─── CNN RESIDUAL BLOCK ───────────────────────────────────────────────────────

def conv_block(x, filters, pool=True, dropout=0.0):
    """ResNet-style conv block with optional pooling and dropout."""
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("swish")(x)
    if pool:
        x = layers.MaxPooling2D((2, 2))(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x


# ─── TRANSFORMER BLOCK ────────────────────────────────────────────────────────

def transformer_block(x, d_model=256, num_heads=8, ff_dim=768, dropout=0.10):
    """Pre-norm transformer block (more stable than post-norm for CTC)."""
    # Self-attention branch
    attn_in = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
    )(attn_in, attn_in)
    x = layers.Add()([x, attn_out])

    # FFN branch
    ffn_in = layers.LayerNormalization(epsilon=1e-6)(x)
    ffn = layers.Dense(ff_dim, activation="gelu")(ffn_in)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    x = layers.Add()([x, ffn])
    return x


# ─── MAIN MODEL BUILDER ───────────────────────────────────────────────────────

def build_model(num_chars, max_label_len, use_wave=None):
    """
    Build HTR model.

    Parameters
    ----------
    num_chars    : vocabulary size (excluding blank)
    max_label_len: maximum label sequence length
    use_wave     : override config.USE_WAVE if provided (used in experiments)

    Returns
    -------
    Compiled Keras model ready for .fit()
    """
    if use_wave is None:
        use_wave = USE_WAVE

    SEQ_LEN = IMG_WIDTH // 4   # 64 time steps after 2× 2×2 pooling

    # ── Data Augmentation (active only during training) ──
    augment = tf.keras.Sequential([
        layers.RandomRotation(0.02),
        layers.RandomTranslation(0.03, 0.03),
        layers.RandomZoom(height_factor=0.05, width_factor=0.08),
        layers.RandomContrast(0.15),
    ], name="augment")

    # ── Inputs ──
    image_input    = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
    labels_input   = layers.Input(shape=(max_label_len,), dtype="int32", name="labels")
    input_len_inp  = layers.Input(shape=(1,), dtype="int32", name="input_length")
    label_len_inp  = layers.Input(shape=(1,), dtype="int32", name="label_length")

    # ── CNN ──
    x = augment(image_input)
    x = conv_block(x, 64,  pool=True,  dropout=0.05)
    x = conv_block(x, 128, pool=True,  dropout=0.08)
    x = conv_block(x, 256, pool=False, dropout=0.10)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.10)(x)

    # ── CNN → Sequence reshape ──
    # After 2×2 pooling ×2: height=8, width=64, channels=256
    x = layers.Permute((2, 1, 3))(x)           # (batch, width, height, channels)
    x = layers.Reshape((SEQ_LEN, 8 * 256))(x)  # (batch, 64, 2048)
    x = layers.Dense(256)(x)                   # (batch, 64, 256)

    # ── WAVE MODULE (research contribution) ──
    if use_wave:
        wave_layer = build_wave_module(
            num_channels=256,
            seq_len=SEQ_LEN,
            name="damped_wave_modulation",
        )
        x = wave_layer(x)

    # ── Positional Encoding ──
    x = PositionalEncoding(name="positional_encoding")(x)

    # ── Transformer Encoder ──
    for i in range(4):
        x = transformer_block(x, d_model=256, num_heads=8, ff_dim=768, dropout=0.10)

    # ── BiLSTM ──
    x = layers.Bidirectional(
        layers.LSTM(192, return_sequences=True, dropout=0.15)
    )(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(256, activation="gelu")(x)

    # ── CTC Output ──
    y_pred = layers.Dense(
        num_chars + 1,
        activation="softmax",
        dtype="float32",
        name="y_pred",
    )(x)

    # ── CTC Loss Layer ──
    def ctc_loss_fn(args):
        y_p, lbl, inp_len, lbl_len = args
        return tf.keras.backend.ctc_batch_cost(lbl, y_p, inp_len, lbl_len)

    loss = layers.Lambda(ctc_loss_fn, name="ctc")(
        [y_pred, labels_input, input_len_inp, label_len_inp]
    )

    def identity_loss(_y_true, y_pred_arg):
        return y_pred_arg

    model = models.Model(
        inputs=[image_input, labels_input, input_len_inp, label_len_inp],
        outputs=loss,
        name=f"htr_{'wave' if use_wave else 'baseline'}_v4",
    )
    model.identity_loss = identity_loss

    # Store inference sub-model reference for easy extraction later
    model._image_input = image_input
    model._y_pred      = y_pred

    tag = "WAVE" if use_wave else "BASELINE"
    print(f"\n{'='*50}")
    print(f"  Model built: {tag}")
    print(f"  Wave module: {'ENABLED ✓' if use_wave else 'DISABLED'}")
    print(f"  Params: {model.count_params():,}")
    print(f"{'='*50}\n")

    return model


def extract_inference_model(trained_model):
    """
    Extract a prediction-only sub-model from a trained model.
    Returns model that takes image → softmax probabilities.
    """
    image_input = trained_model._image_input
    y_pred      = trained_model._y_pred
    return models.Model(inputs=image_input, outputs=y_pred, name="infer_model")
