"""
train.py
--------
Main training script.

Usage in Kaggle:
    !python /kaggle/working/train.py

Toggle wave module in config.py:
    USE_WAVE = True   → trains research model
    USE_WAVE = False  → trains baseline model

Output files are tagged automatically (e.g. htr_ctc_wave_best.keras)
"""

import math
import os
import numpy as np
import tensorflow as tf

# ── local imports ──
from config import (
    SEED, IMG_WIDTH, BATCH_SIZE, EPOCHS, WARMUP_EPOCHS,
    BASE_LR, MIN_LR, WEIGHT_DECAY, MAX_SAMPLES,
    ENABLE_MIXED_PRECISION, USE_WAVE, get_paths,
)
from dataset  import load_raw, encode_labels, build_datasets
from model    import build_model, extract_inference_model
from decoder  import SamplePredictionCallback

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─── RUNTIME CONFIGURATION ────────────────────────────────────────────────────

def configure_runtime():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        if ENABLE_MIXED_PRECISION:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                print("[runtime] Mixed precision: ENABLED")
            except ValueError:
                print("[runtime] Mixed precision unavailable, using float32")
        else:
            tf.keras.mixed_precision.set_global_policy("float32")
            print("[runtime] Mixed precision: DISABLED (float32 for CTC stability)")
    else:
        print("[runtime] No GPU found — training on CPU")

    paths = get_paths()
    os.makedirs(os.path.dirname(paths["best"]), exist_ok=True)
    return paths


# ─── LEARNING RATE SCHEDULE ───────────────────────────────────────────────────

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine annealing."""

    def __init__(self, base_lr, warmup_steps, total_steps, min_lr):
        super().__init__()
        self.base_lr      = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps  = int(total_steps)
        self.min_lr       = float(min_lr)

    def __call__(self, step):
        step         = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps  = tf.cast(self.total_steps,  tf.float32)

        warmup_lr = self.base_lr * (step + 1.0) / tf.maximum(warmup_steps, 1.0)

        progress  = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        progress  = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = (self.min_lr
                     + 0.5 * (self.base_lr - self.min_lr)
                     * (1.0 + tf.cos(math.pi * progress)))

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr":      self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps":  self.total_steps,
            "min_lr":       self.min_lr,
        }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    paths = configure_runtime()

    # ── Load data ──
    images, texts, charset, char_to_idx, idx_to_char, blank_idx = load_raw(
        max_samples=MAX_SAMPLES
    )
    labels, label_lengths, max_label_len = encode_labels(texts, char_to_idx, blank_idx)

    # ── Build tf.data pipelines ──
    train_ds, val_ds, n_train, n_val = build_datasets(
        images, labels, label_lengths, BATCH_SIZE
    )

    # ── Build model ──
    model = build_model(len(charset), max_label_len, use_wave=USE_WAVE)

    # ── LR schedule ──
    steps_per_epoch = max(1, math.ceil(n_train / BATCH_SIZE))
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = steps_per_epoch * WARMUP_EPOCHS

    lr_schedule = WarmupCosineDecay(BASE_LR, warmup_steps, total_steps, MIN_LR)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        clipnorm=1.0,           # gradient clipping — prevents CTC explosion
    )

    model.compile(optimizer=optimizer, loss=model.identity_loss)
    model.summary()

    # ── Inference sub-model for sample callback ──
    infer_model = extract_inference_model(model)

    # ── Grab a few val images for the sample callback ──
    # Pull one batch from val_ds to get numpy arrays
    val_batch = next(iter(val_ds))
    val_imgs_cb  = val_batch[0]["image"].numpy()[:5]
    val_texts_cb = []
    val_lbls_cb  = val_batch[0]["labels"].numpy()[:5]
    val_llen_cb  = val_batch[0]["label_length"].numpy()[:5]
    for i in range(len(val_imgs_cb)):
        text = ""
        for idx in val_lbls_cb[i]:
            if idx == blank_idx:
                break
            text += idx_to_char.get(int(idx), "?")
        val_texts_cb.append(text)

    ctc_input_len = IMG_WIDTH // 4

    sample_cb = SamplePredictionCallback(
        val_images    = val_imgs_cb,
        val_texts     = val_texts_cb,
        infer_model   = infer_model,
        idx_to_char   = idx_to_char,
        blank_idx     = blank_idx,
        ctc_input_len = ctc_input_len,
        print_every   = 2,
        num_samples   = 5,
    )

    # ── Keras callbacks ──
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = paths["best"],
            monitor        = "val_loss",
            save_best_only = True,
            verbose        = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_loss",
            patience             = 12,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.CSVLogger(paths["history"]),
        sample_cb,
    ]

    # ── Train ──
    tag = "WAVE" if USE_WAVE else "BASELINE"
    print(f"\n{'='*50}")
    print(f"  Starting training: {tag} model")
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}")
    print(f"  Steps/epoch: {steps_per_epoch}")
    print(f"{'='*50}\n")

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS,
        callbacks       = callback_list,
    )

    model.save(paths["final"])

    print(f"\n✅ Training complete [{tag}]")
    print(f"   Best checkpoint : {paths['best']}")
    print(f"   Final model     : {paths['final']}")
    print(f"   History CSV     : {paths['history']}")

    return history


if __name__ == "__main__":
    main()
