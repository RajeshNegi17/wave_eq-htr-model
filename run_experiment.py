"""
run_experiment.py
-----------------
ONE-CLICK script for Kaggle.

Runs the full research experiment:
  1. Train BASELINE model (USE_WAVE=False)
  2. Train WAVE model     (USE_WAVE=True)
  3. Evaluate both → print CER/WER comparison table

Usage in Kaggle notebook:
    !python /kaggle/working/run_experiment.py

To run only one model (e.g. wave), set:
    RUN_BASELINE = False
    RUN_WAVE     = True
"""

import os
import math
import numpy as np
import tensorflow as tf

# ── local imports ──
import config   # we'll patch config.USE_WAVE programmatically
from config  import (SEED, IMG_WIDTH, BATCH_SIZE, EPOCHS, WARMUP_EPOCHS,
                     BASE_LR, MIN_LR, WEIGHT_DECAY, MAX_SAMPLES,
                     ENABLE_MIXED_PRECISION, OUTPUT_DIR, get_paths)
from dataset import load_raw, encode_labels, build_datasets
from model   import build_model, extract_inference_model
from decoder import SamplePredictionCallback, ctc_greedy_decode
from evaluate import compute_cer, compute_wer, print_results_table
from train   import WarmupCosineDecay, configure_runtime

# ─── EXPERIMENT CONTROL ───────────────────────────────────────────────────────
RUN_BASELINE = True   # set False to skip baseline training
RUN_WAVE     = True   # set False to skip wave training

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─── TRAIN ONE MODEL ──────────────────────────────────────────────────────────

def train_one(use_wave, images, texts, labels, label_lengths,
              charset, char_to_idx, idx_to_char, blank_idx):
    """Train a single model configuration and return its best checkpoint path."""

    # Patch config flag so all downstream modules see it
    config.USE_WAVE = use_wave
    paths = get_paths()
    tag   = "WAVE" if use_wave else "BASELINE"

    # Skip if already trained
    if os.path.exists(paths["best"]):
        print(f"\n[experiment] {tag} checkpoint exists — skipping training.")
        return paths["best"]

    print(f"\n{'█'*55}")
    print(f"  TRAINING: {tag}")
    print(f"{'█'*55}")

    train_ds, val_ds, n_train, n_val = build_datasets(
        images, labels, label_lengths, BATCH_SIZE
    )

    model = build_model(len(charset), labels.shape[1], use_wave=use_wave)

    steps_per_epoch = max(1, math.ceil(n_train / BATCH_SIZE))
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = steps_per_epoch * WARMUP_EPOCHS

    lr_schedule = WarmupCosineDecay(BASE_LR, warmup_steps, total_steps, MIN_LR)
    optimizer   = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        clipnorm=1.0,
    )
    model.compile(optimizer=optimizer, loss=model.identity_loss)

    infer_model = extract_inference_model(model)

    # Sample callback setup
    val_batch       = next(iter(val_ds))
    val_imgs_cb     = val_batch[0]["image"].numpy()[:5]
    val_lbls_cb     = val_batch[0]["labels"].numpy()[:5]
    val_texts_cb    = []
    for i in range(len(val_imgs_cb)):
        text = ""
        for idx in val_lbls_cb[i]:
            if idx == blank_idx:
                break
            text += idx_to_char.get(int(idx), "?")
        val_texts_cb.append(text)

    sample_cb = SamplePredictionCallback(
        val_images=val_imgs_cb, val_texts=val_texts_cb,
        infer_model=infer_model, idx_to_char=idx_to_char,
        blank_idx=blank_idx, ctc_input_len=IMG_WIDTH // 4,
        print_every=2, num_samples=5,
    )

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            paths["best"], monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(paths["history"]),
        sample_cb,
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs)
    model.save(paths["final"])

    print(f"\n✅ {tag} training done → {paths['best']}")
    return paths["best"]


# ─── EVALUATE ONE MODEL ───────────────────────────────────────────────────────

def eval_one(checkpoint_path, val_ds, idx_to_char, blank_idx):
    """Load checkpoint and compute CER/WER on the validation set."""
    from wave_module import DampedWaveModulation

    full  = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={"DampedWaveModulation": DampedWaveModulation},
        compile=False,
    )
    infer = tf.keras.models.Model(
        inputs=full.inputs[0],
        outputs=full.get_layer("y_pred").output,
    )

    ctc_len  = IMG_WIDTH // 4
    all_gt   = []
    all_pred = []

    for batch_inputs, _ in val_ds:
        imgs = batch_inputs["image"].numpy()
        lbls = batch_inputs["labels"].numpy()
        n    = len(imgs)

        for i in range(n):
            text = ""
            for idx in lbls[i]:
                if idx == blank_idx:
                    break
                text += idx_to_char.get(int(idx), "?")
            all_gt.append(text)

        preds   = infer.predict(imgs, verbose=0)
        decoded = ctc_greedy_decode(preds, [ctc_len]*n, idx_to_char, blank_idx)
        all_pred.extend(decoded)

    cer = compute_cer(all_gt, all_pred)
    wer = compute_wer(all_gt, all_pred)
    return cer, wer, all_gt, all_pred


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    configure_runtime()

    # ── Load data once ──
    images, texts, charset, char_to_idx, idx_to_char, blank_idx = load_raw(
        max_samples=MAX_SAMPLES
    )
    labels, label_lengths, _ = encode_labels(texts, char_to_idx, blank_idx)

    # ── Training phase ──
    trained = {}

    if RUN_BASELINE:
        path = train_one(
            use_wave=False,
            images=images, texts=texts,
            labels=labels, label_lengths=label_lengths,
            charset=charset, char_to_idx=char_to_idx,
            idx_to_char=idx_to_char, blank_idx=blank_idx,
        )
        trained["Baseline"] = path

    if RUN_WAVE:
        path = train_one(
            use_wave=True,
            images=images, texts=texts,
            labels=labels, label_lengths=label_lengths,
            charset=charset, char_to_idx=char_to_idx,
            idx_to_char=idx_to_char, blank_idx=blank_idx,
        )
        trained["Wave"] = path

    # ── Evaluation phase ──
    if not trained:
        print("Nothing to evaluate.")
        return

    # Rebuild val_ds with fixed seed for fair comparison
    config.USE_WAVE = False  # irrelevant for data, but reset cleanly
    _, val_ds, _, n_val = build_datasets(images, labels, label_lengths, BATCH_SIZE)

    print(f"\n{'='*55}")
    print(f"  EVALUATION  ({n_val} validation samples)")
    print(f"{'='*55}")

    results   = {}
    all_preds = {}

    for name, path in trained.items():
        print(f"\n  Evaluating {name}...")
        cer, wer, gt_list, pred_list = eval_one(path, val_ds, idx_to_char, blank_idx)
        results[name]   = {"cer": cer, "wer": wer}
        all_preds[name] = pred_list
        print(f"  {name}: CER={cer*100:.2f}%  WER={wer*100:.2f}%")

    print_results_table(results)

    # ── Save summary ──
    summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("HTR Wave Module Research — Results\n")
        f.write("=" * 40 + "\n")
        for name, r in results.items():
            f.write(f"{name}: CER={r['cer']*100:.4f}%  WER={r['wer']*100:.4f}%\n")
        if len(results) == 2:
            names     = list(results.keys())
            cer_delta = results[names[1]]["cer"] - results[names[0]]["cer"]
            wer_delta = results[names[1]]["wer"] - results[names[0]]["wer"]
            f.write(f"\nWave vs Baseline:\n")
            f.write(f"  CER delta: {cer_delta*100:+.4f}%\n")
            f.write(f"  WER delta: {wer_delta*100:+.4f}%\n")

    print(f"\n  Summary saved → {summary_path}")
    print("\n✅ Experiment complete.\n")


if __name__ == "__main__":
    main()
