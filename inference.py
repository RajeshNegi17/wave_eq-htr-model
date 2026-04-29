"""
inference.py
------------
Load a trained model checkpoint and predict text from:
  - A single word/line image
  - A paragraph image (via line + word segmentation)

Usage in Kaggle:
    !python /kaggle/working/inference.py

Set TEST_IMAGE_PATH below, or import predict_word / predict_paragraph
from other scripts.
"""

import cv2
import os
import numpy as np
import tensorflow as tf
import unicodedata

from config      import IMG_HEIGHT, IMG_WIDTH, OUTPUT_DIR, DATA_PATH
from decoder     import ctc_greedy_decode
from wave_module import DampedWaveModulation

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Which model to load for inference:
# Set to "wave" or "baseline" to pick the right checkpoint
MODEL_TAG = "wave"   # ← change to "baseline" to use baseline model

# Alternatively, point directly at a .keras file:
OVERRIDE_MODEL_PATH = None  # e.g. "/kaggle/working/htr_ctc_wave_best.keras"

# Test image path (change this to your actual image)
TEST_IMAGE_PATH = None  # e.g. "/kaggle/input/.../my_word.png"

# ─── LOAD CHARSET FROM DATA ───────────────────────────────────────────────────

def load_charset():
    data      = np.load(DATA_PATH, allow_pickle=True)
    texts     = data["labels"].tolist()
    charset   = sorted(set("".join(texts)))
    idx_to_char = {i: c for i, c in enumerate(charset)}
    blank_idx   = len(charset)
    print(f"[inference] Charset size: {len(charset)}  blank_idx={blank_idx}")
    return idx_to_char, blank_idx


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

def load_infer_model(model_tag=MODEL_TAG):
    if OVERRIDE_MODEL_PATH and os.path.exists(OVERRIDE_MODEL_PATH):
        model_path = OVERRIDE_MODEL_PATH
    else:
        model_path = os.path.join(OUTPUT_DIR, f"htr_ctc_{model_tag}_best.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run train.py with USE_WAVE={'True' if model_tag=='wave' else 'False'} first."
        )

    full = tf.keras.models.load_model(
        model_path,
        custom_objects={"DampedWaveModulation": DampedWaveModulation},
        compile=False,
    )

    image_input = full.inputs[0]
    y_pred      = full.get_layer("y_pred").output
    infer       = tf.keras.models.Model(inputs=image_input, outputs=y_pred)

    print(f"[inference] Model loaded: {os.path.basename(model_path)}")
    return infer


# ─── IMAGE PREPROCESSING ──────────────────────────────────────────────────────

def preprocess_word_array(img_gray):
    """
    Preprocess a grayscale numpy array (H, W) for model input.
    Returns: (1, IMG_HEIGHT, IMG_WIDTH, 1) tensor, ctc_input_len
    """
    img  = img_gray.astype(np.float32) / 255.0
    h, w = img.shape

    scale = IMG_HEIGHT / max(h, 1)
    new_w = max(1, min(int(w * scale), IMG_WIDTH))

    img = cv2.resize(img, (new_w, IMG_HEIGHT))

    padded = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    padded[:, :new_w] = img

    padded = padded[np.newaxis, :, :, np.newaxis]   # (1, H, W, 1)
    ctc_len = max(1, new_w // 4)
    return padded, ctc_len


def preprocess_word_file(img_path):
    """Load image from file path and preprocess."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    return preprocess_word_array(img)


# ─── SINGLE WORD / LINE PREDICTION ───────────────────────────────────────────

def predict_word(img_path, infer_model, idx_to_char, blank_idx):
    """Predict text from a single word/line image file."""
    img_prep, ctc_len = preprocess_word_file(img_path)
    preds   = infer_model.predict(img_prep, verbose=0)
    decoded = ctc_greedy_decode(preds, [ctc_len], idx_to_char, blank_idx)
    return decoded[0]


def predict_word_array(img_gray, infer_model, idx_to_char, blank_idx):
    """Predict text from a grayscale numpy array."""
    img_prep, ctc_len = preprocess_word_array(img_gray)
    preds   = infer_model.predict(img_prep, verbose=0)
    decoded = ctc_greedy_decode(preds, [ctc_len], idx_to_char, blank_idx)
    return decoded[0]


# ─── PARAGRAPH SEGMENTATION + PREDICTION ─────────────────────────────────────

def segment_lines(gray_img):
    """Segment a paragraph image into line images (top-to-bottom)."""
    _, thresh = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 6:
            lines.append((y, gray_img[y:y+h, x:x+w]))

    return [l[1] for l in sorted(lines, key=lambda z: z[0])]


def segment_words(line_img):
    """Segment a line image into word images (left-to-right)."""
    _, thresh = cv2.threshold(
        line_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    words = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:
            words.append((x, line_img[y:y+h, x:x+w]))

    return [w[1] for w in sorted(words, key=lambda z: z[0])]


def predict_paragraph(img_path, infer_model, idx_to_char, blank_idx):
    """
    Full pipeline: paragraph image → segmented lines → words → text.
    Returns multiline string.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read paragraph image: {img_path}")

    lines     = segment_lines(img)
    paragraph = []

    for line_img in lines:
        words     = segment_words(line_img)
        word_preds = [
            predict_word_array(w, infer_model, idx_to_char, blank_idx)
            for w in words
        ]
        paragraph.append(" ".join(word_preds))

    return "\n".join(paragraph)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    idx_to_char, blank_idx = load_charset()
    infer_model = load_infer_model(MODEL_TAG)

    if TEST_IMAGE_PATH and os.path.exists(TEST_IMAGE_PATH):
        print(f"\n[inference] Predicting: {TEST_IMAGE_PATH}")
        result = predict_word(TEST_IMAGE_PATH, infer_model, idx_to_char, blank_idx)
        print(f"  Prediction: {result}")
    else:
        # Demo: run on first 10 samples from the .npz val set
        print("\n[inference] No TEST_IMAGE_PATH set — running on .npz samples")
        from config  import BATCH_SIZE, MAX_SAMPLES
        from dataset import load_raw, encode_labels, build_datasets

        images, texts, charset, char_to_idx, idx_to_char2, blank_idx2 = load_raw(
            max_samples=MAX_SAMPLES
        )
        labels, label_lengths, _ = encode_labels(texts, char_to_idx, blank_idx2)
        _, val_ds, _, _ = build_datasets(images, labels, label_lengths, BATCH_SIZE)

        batch_inputs, _ = next(iter(val_ds))
        imgs  = batch_inputs["image"].numpy()[:10]
        lbls  = batch_inputs["labels"].numpy()[:10]

        gt_texts = []
        for i in range(len(imgs)):
            text = ""
            for idx in lbls[i]:
                if idx == blank_idx2:
                    break
                text += idx_to_char2.get(int(idx), "?")
            gt_texts.append(text)

        ctc_len = IMG_WIDTH // 4
        preds   = infer_model.predict(imgs, verbose=0)
        decoded = ctc_greedy_decode(preds, [ctc_len]*len(imgs), idx_to_char2, blank_idx2)

        print(f"\n  {'GT':<30}  {'PREDICTION':<30}")
        print(f"  {'─'*62}")
        for gt, pred in zip(gt_texts, decoded):
            mark = "✓" if gt.strip() == pred.strip() else "✗"
            print(f"  {mark} {gt[:28]:<30}  {pred[:28]:<30}")


if __name__ == "__main__":
    main()
