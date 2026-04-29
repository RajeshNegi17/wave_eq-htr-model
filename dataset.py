"""
dataset.py
----------
Loads the .npz dataset, builds charset, and returns tf.data pipelines.
Works with your existing data2 (1).npz format.
"""

import os
import numpy as np
import tensorflow as tf

from config import DATA_PATH, IMG_HEIGHT, IMG_WIDTH, VAL_SPLIT, SEED


# ─── LOAD RAW DATA ────────────────────────────────────────────────────────────

def load_raw(data_path=None, max_samples=0):
    """
    Load images and text labels from .npz file.

    Returns
    -------
    images       : float32 array (N, H, W, 1), already normalized 0–1
    texts        : list of strings
    charset      : sorted list of unique characters
    char_to_idx  : dict char → int
    idx_to_char  : dict int → char
    blank_idx    : int (= len(charset), reserved for CTC blank)
    """
    path = data_path or DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"Check DATA_PATH in config.py"
        )

    data   = np.load(path, allow_pickle=True)
    images = data["images"].astype(np.float32)
    texts  = data["labels"].tolist()

    # Normalize to [0, 1] if not already
    if images.max() > 1.5:
        images = images / 255.0

    # Ensure channel dim exists: (N, H, W) → (N, H, W, 1)
    if images.ndim == 3:
        images = images[..., np.newaxis]

    print(f"[dataset] Loaded {len(images)} samples")
    print(f"[dataset] Image shape: {images.shape}")

    # Optional subset for quick testing
    if max_samples > 0 and max_samples < len(images):
        images = images[:max_samples]
        texts  = texts[:max_samples]
        print(f"[dataset] Using subset: {max_samples} samples")

    # Build charset
    charset     = sorted(set("".join(texts)))
    char_to_idx = {c: i for i, c in enumerate(charset)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    blank_idx   = len(charset)

    print(f"[dataset] Charset size: {len(charset)}  (blank_idx={blank_idx})")
    print(f"[dataset] Sample chars: {''.join(charset[:30])} ...")

    return images, texts, charset, char_to_idx, idx_to_char, blank_idx


# ─── ENCODE LABELS ────────────────────────────────────────────────────────────

def encode_labels(texts, char_to_idx, blank_idx):
    """
    Convert list of strings → padded int32 array + length array.

    Padding value = blank_idx (CTC blank token).
    """
    max_len = max(len(t) for t in texts)
    labels  = np.full((len(texts), max_len), blank_idx, dtype=np.int32)

    for i, text in enumerate(texts):
        for j, ch in enumerate(text):
            labels[i, j] = char_to_idx.get(ch, blank_idx)

    label_lengths = np.array([[len(t)] for t in texts], dtype=np.int32)

    return labels, label_lengths, max_len


# ─── BUILD TF.DATA PIPELINES ──────────────────────────────────────────────────

def build_datasets(images, labels, label_lengths, batch_size, val_split=None):
    """
    Split into train/val and return tf.data.Dataset objects.

    Input length for CTC = IMG_WIDTH // 4  (after 2× 2×2 MaxPool)
    """
    vs = val_split if val_split is not None else VAL_SPLIT
    n  = len(images)

    rng = np.random.default_rng(SEED)
    idx = np.arange(n)
    rng.shuffle(idx)

    val_n    = int(n * vs)
    val_idx  = idx[:val_n]
    train_idx = idx[val_n:]

    # Fixed input length for CTC (time steps after CNN)
    ctc_input_len = IMG_WIDTH // 4
    input_lengths = np.full((n, 1), ctc_input_len, dtype=np.int32)

    def make_ds(subset_idx, shuffle=False):
        imgs  = images[subset_idx]
        lbls  = labels[subset_idx]
        llen  = label_lengths[subset_idx]
        ilen  = input_lengths[subset_idx]
        dummy = np.zeros((len(subset_idx),), dtype=np.float32)

        ds = tf.data.Dataset.from_tensor_slices((
            {"image": imgs, "labels": lbls,
             "input_length": ilen, "label_length": llen},
            dummy,
        ))
        if shuffle:
            ds = ds.shuffle(min(len(subset_idx), 8192), seed=SEED,
                            reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(train_idx, shuffle=True)
    val_ds   = make_ds(val_idx,   shuffle=False)

    print(f"[dataset] Train: {len(train_idx)}  Val: {len(val_idx)}")
    return train_ds, val_ds, len(train_idx), len(val_idx)
