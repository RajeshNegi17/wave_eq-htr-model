"""
decoder.py
----------
CTC greedy decoder + blank collapse detection.
Used during training (sample prints) and evaluation.
"""

import numpy as np
import tensorflow as tf
import unicodedata


def ctc_greedy_decode(preds, input_lengths, idx_to_char, blank_idx):
    """
    Greedy CTC decode for a batch of predictions.

    Parameters
    ----------
    preds         : np.array (batch, time, num_classes)
    input_lengths : list/array of int, length per sample
    idx_to_char   : dict int → char
    blank_idx     : int, CTC blank token index

    Returns
    -------
    texts : list of decoded strings (one per sample)
    """
    texts = []
    for i in range(len(preds)):
        t = int(input_lengths[i]) if hasattr(input_lengths, '__len__') else input_lengths
        pred_i = preds[i:i+1, :t, :]  # (1, t, C)

        decoded, _ = tf.keras.backend.ctc_decode(
            pred_i,
            input_length=np.array([t]),
            greedy=True,
        )
        seq = decoded[0].numpy()[0]

        text = ""
        for idx in seq:
            if idx == -1 or idx == blank_idx:
                continue
            text += idx_to_char.get(int(idx), "?")

        texts.append(unicodedata.normalize("NFC", text))

    return texts


def decode_batch(infer_model, images, input_lengths, idx_to_char, blank_idx):
    """
    Run inference + decode for a batch of preprocessed images.
    images: np.array (batch, H, W, 1)
    """
    preds = infer_model.predict(images, verbose=0)
    return ctc_greedy_decode(preds, input_lengths, idx_to_char, blank_idx)


# ─── COLLAPSE DETECTION ───────────────────────────────────────────────────────

def detect_collapse(preds, blank_idx, threshold=0.95):
    """
    Warn if model is predicting blank > threshold of the time.
    This is a sign of CTC collapse (common early in training).

    preds: (batch, time, num_classes)
    Returns: True if collapse detected
    """
    blank_probs = preds[:, :, blank_idx]           # (batch, time)
    mean_blank  = float(np.mean(blank_probs))

    if mean_blank > threshold:
        print(f"\n  ⚠️  COLLAPSE WARNING: blank prob = {mean_blank:.3f} "
              f"(>{threshold:.0%}) — model may be stuck.")
        return True
    return False


# ─── SAMPLE PRINT CALLBACK ────────────────────────────────────────────────────

class SamplePredictionCallback(tf.keras.callbacks.Callback):
    """
    Prints decoded predictions on a small validation batch every N epochs.
    Helps detect blank collapse and verify learning progress.
    """

    def __init__(self, val_images, val_texts, infer_model,
                 idx_to_char, blank_idx,
                 ctc_input_len, print_every=2, num_samples=5):
        super().__init__()
        self.val_images    = val_images[:num_samples]   # (K, H, W, 1)
        self.val_texts     = val_texts[:num_samples]
        self.infer_model   = infer_model
        self.idx_to_char   = idx_to_char
        self.blank_idx     = blank_idx
        self.ctc_input_len = ctc_input_len
        self.print_every   = print_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return

        preds = self.infer_model.predict(self.val_images, verbose=0)

        # Collapse check
        detect_collapse(preds, self.blank_idx)

        decoded = ctc_greedy_decode(
            preds,
            [self.ctc_input_len] * len(self.val_images),
            self.idx_to_char,
            self.blank_idx,
        )

        print(f"\n  📝 Sample predictions (epoch {epoch+1}):")
        print(f"  {'GT':<30}  {'PRED':<30}")
        print(f"  {'-'*62}")
        for gt, pred in zip(self.val_texts, decoded):
            mark = "✓" if gt.strip() == pred.strip() else "✗"
            print(f"  {mark} {gt[:28]:<30}  {pred[:28]:<30}")
        print()
