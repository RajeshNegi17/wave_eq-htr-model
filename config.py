"""
config.py
---------
Single source of truth for all hyperparameters.
Toggle USE_WAVE here to switch between baseline and research model.
"""

import os

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
SEED = 42

# ─── IMAGE ────────────────────────────────────────────────────────────────────
IMG_HEIGHT = 32
IMG_WIDTH  = 256

# ─── TRAINING ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 16
EPOCHS         = 80
WARMUP_EPOCHS  = 5
BASE_LR        = 3e-4
MIN_LR         = 1e-5
WEIGHT_DECAY   = 1e-4
VAL_SPLIT      = 0.1
MAX_SAMPLES    = 0          # 0 = use all; set e.g. 2000 for quick smoke test

ENABLE_MIXED_PRECISION = False

# ─── WAVE MODULE TOGGLE ───────────────────────────────────────────────────────
# Set True  → research model  (CNN → Wave → Transformer → BiLSTM → CTC)
# Set False → baseline model  (CNN → Transformer → BiLSTM → CTC)
USE_WAVE = True             # ← CHANGE THIS to compare

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATA_PATH  = "/kaggle/input/datasets/rajesh1717/htrdata/data2 (1).npz"
MODEL_PATH = "/kaggle/input/datasets/rajesh1717/htr-model/htr_ctc_words_multilang_best_v2.keras"
OUTPUT_DIR = "/kaggle/working"

def get_model_tag():
    """Returns 'wave' or 'baseline' for file naming."""
    return "wave" if USE_WAVE else "baseline"

def get_paths():
    tag = get_model_tag()
    return {
        "best":    os.path.join(OUTPUT_DIR, f"htr_ctc_{tag}_best.keras"),
        "final":   os.path.join(OUTPUT_DIR, f"htr_ctc_{tag}_final.keras"),
        "history": os.path.join(OUTPUT_DIR, f"htr_ctc_{tag}_history.csv"),
        "results": os.path.join(OUTPUT_DIR, f"htr_ctc_{tag}_results.txt"),
    }
