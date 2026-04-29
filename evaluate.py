"""
evaluate.py
-----------
Evaluates BOTH trained models (baseline + wave) and prints a
side-by-side CER / WER comparison table.

Usage in Kaggle:
    !python /kaggle/working/evaluate.py

Requirements:
    - htr_ctc_baseline_best.keras  (train with USE_WAVE=False)
    - htr_ctc_wave_best.keras      (train with USE_WAVE=True)
    Both saved to /kaggle/working/ by train.py
"""

import os
import numpy as np
import tensorflow as tf

from config      import (DATA_PATH, OUTPUT_DIR, IMG_WIDTH,
                         BATCH_SIZE, MAX_SAMPLES, SEED)
from dataset     import load_raw, encode_labels, build_datasets
from decoder     import ctc_greedy_decode
from wave_module import DampedWaveModulation


# ─── CER / WER ────────────────────────────────────────────────────────────────

def levenshtein(s1, s2):
    """Standard dynamic-programming edit distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j] + (c1 != c2), curr[-1] + 1, prev[j+1] + 1))
        prev = curr
    return prev[-1]


def compute_cer(gt_list, pred_list):
    """Character Error Rate = edit_distance(chars) / total_chars."""
    total_dist = 0
    total_len  = 0
    for gt, pred in zip(gt_list, pred_list):
        total_dist += levenshtein(list(gt), list(pred))
        total_len  += max(len(gt), 1)
    return total_dist / total_len


def compute_wer(gt_list, pred_list):
    """Word Error Rate = edit_distance(words) / total_words."""
    total_dist = 0
    total_len  = 0
    for gt, pred in zip(gt_list, pred_list):
        gt_words   = gt.split()
        pred_words = pred.split()
        total_dist += levenshtein(gt_words, pred_words)
        total_len  += max(len(gt_words), 1)
    return total_dist / total_len


# ─── LOAD MODEL + BUILD INFERENCE SUB-MODEL ───────────────────────────────────

def load_infer_model(keras_path):
    """Load a full CTC training model and extract the inference sub-model."""
    if not os.path.exists(keras_path):
        return None

    full = tf.keras.models.load_model(
        keras_path,
        custom_objects={"DampedWaveModulation": DampedWaveModulation},
        compile=False,
    )
    image_input = full.inputs[0]
    y_pred      = full.get_layer("y_pred").output
    infer       = tf.keras.models.Model(inputs=image_input, outputs=y_pred)
    print(f"  Loaded: {os.path.basename(keras_path)}")
    return infer


# ─── EVALUATE ONE MODEL ───────────────────────────────────────────────────────

def evaluate_model(infer_model, val_ds, idx_to_char, blank_idx, ctc_input_len):
    """Run full val set through model, return (gt_list, pred_list)."""
    all_gt   = []
    all_pred = []

    for batch_inputs, _ in val_ds:
        imgs      = batch_inputs["image"].numpy()
        lbls      = batch_inputs["labels"].numpy()
        llen      = batch_inputs["label_length"].numpy()
        batch_len = len(imgs)

        # Ground truth
        for i in range(batch_len):
            text = ""
            for idx in lbls[i]:
                if idx == blank_idx:
                    break
                text += idx_to_char.get(int(idx), "?")
            all_gt.append(text)

        # Predictions
        preds = infer_model.predict(imgs, verbose=0)
        decoded = ctc_greedy_decode(
            preds,
            [ctc_input_len] * batch_len,
            idx_to_char,
            blank_idx,
        )
        all_pred.extend(decoded)

    return all_gt, all_pred


# ─── PRINT COMPARISON TABLE ───────────────────────────────────────────────────

def print_results_table(results):
    """
    results: dict of {model_name: {"cer": float, "wer": float}}
    """
    line = "─" * 52
    print(f"\n{line}")
    print(f"  {'Model':<20}  {'CER':>8}  {'WER':>8}  {'CER%':>7}  {'WER%':>7}")
    print(f"{line}")
    for name, r in results.items():
        cer = r["cer"]
        wer = r["wer"]
        print(f"  {name:<20}  {cer:>8.4f}  {wer:>8.4f}  {cer*100:>6.2f}%  {wer*100:>6.2f}%")
    print(f"{line}")

    # Delta
    if len(results) == 2:
        names = list(results.keys())
        cer_delta = results[names[1]]["cer"] - results[names[0]]["cer"]
        wer_delta = results[names[1]]["wer"] - results[names[0]]["wer"]
        sign_cer  = "▼" if cer_delta < 0 else "▲"
        sign_wer  = "▼" if wer_delta < 0 else "▲"
        print(f"\n  Wave vs Baseline:")
        print(f"    CER change: {sign_cer} {abs(cer_delta)*100:.2f}%  "
              f"({'improved' if cer_delta < 0 else 'degraded'})")
        print(f"    WER change: {sign_wer} {abs(wer_delta)*100:.2f}%  "
              f"({'improved' if wer_delta < 0 else 'degraded'})")
    print()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Load data (same split as training for fair comparison)
    images, texts, charset, char_to_idx, idx_to_char, blank_idx = load_raw(
        max_samples=MAX_SAMPLES
    )
    labels, label_lengths, _ = encode_labels(texts, char_to_idx, blank_idx)
    _, val_ds, _, n_val = build_datasets(
        images, labels, label_lengths, BATCH_SIZE
    )

    ctc_input_len = IMG_WIDTH // 4

    baseline_path = os.path.join(OUTPUT_DIR, "htr_ctc_baseline_best.keras")
    wave_path     = os.path.join(OUTPUT_DIR, "htr_ctc_wave_best.keras")

    models_to_eval = {}
    if os.path.exists(baseline_path):
        models_to_eval["Baseline"] = baseline_path
    if os.path.exists(wave_path):
        models_to_eval["Wave"] = wave_path

    if not models_to_eval:
        print("❌ No trained models found in /kaggle/working/")
        print("   Run train.py with USE_WAVE=False then USE_WAVE=True first.")
        return

    results   = {}
    all_preds = {}

    for name, path in models_to_eval.items():
        print(f"\n[evaluate] Loading {name} model...")
        infer = load_infer_model(path)
        if infer is None:
            continue

        print(f"[evaluate] Running inference on {n_val} val samples...")
        gt_list, pred_list = evaluate_model(
            infer, val_ds, idx_to_char, blank_idx, ctc_input_len
        )

        cer = compute_cer(gt_list, pred_list)
        wer = compute_wer(gt_list, pred_list)

        results[name]   = {"cer": cer, "wer": wer}
        all_preds[name] = pred_list

        print(f"  {name}: CER={cer*100:.2f}%  WER={wer*100:.2f}%")

    # ── Print comparison table ──
    print_results_table(results)

    # ── Print sample predictions side by side ──
    if len(all_preds) == 2:
        names = list(all_preds.keys())
        gt_list = []
        for batch_inputs, _ in val_ds:
            lbls = batch_inputs["labels"].numpy()
            llen = batch_inputs["label_length"].numpy()
            for i in range(len(lbls)):
                text = ""
                for idx in lbls[i]:
                    if idx == blank_idx:
                        break
                    text += idx_to_char.get(int(idx), "?")
                gt_list.append(text)
            break  # just first batch for sample display

        print(f"\n  Sample predictions (first batch):")
        print(f"  {'GT':<25}  {names[0]:<25}  {names[1]:<25}")
        print(f"  {'─'*77}")
        for i in range(min(10, len(gt_list))):
            gt   = gt_list[i][:23]
            p0   = all_preds[names[0]][i][:23]
            p1   = all_preds[names[1]][i][:23]
            print(f"  {gt:<25}  {p0:<25}  {p1:<25}")

    # ── Save results ──
    results_file = os.path.join(OUTPUT_DIR, "evaluation_results.txt")
    with open(results_file, "w") as f:
        for name, r in results.items():
            f.write(f"{name}: CER={r['cer']*100:.4f}%  WER={r['wer']*100:.4f}%\n")
    print(f"\n  Results saved to: {results_file}")


if __name__ == "__main__":
    main()
