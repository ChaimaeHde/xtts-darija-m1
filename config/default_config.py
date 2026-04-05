"""
Configuration centralisée du projet XTTS-v2 Darija M1
"""

# ── Dataset ──────────────────────────────────────────
DATASET_NAME    = "atlasia/DODa-audio-dataset"
DATASET_SPLIT   = "train"
SPEAKER_INDICES = range(4000, 4650)   # M1 : 650 samples ≈ 35 min
TEXT_COL        = "darija_Arab_new"
DATA_DIR        = "doda_m1_35min"

# ── Modèle HuggingFace ───────────────────────────────
HF_REPO_ID      = "ChaimaeHde/xtts-darija-m1"
BASE_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# ── Training ─────────────────────────────────────────
EPOCHS               = 5
BATCH_SIZE           = 2
GRAD_ACCUM_STEPS     = 126      # effective batch = 252
LEARNING_RATE        = 5e-6
NUM_LOADER_WORKERS   = 0
MIXED_PRECISION      = True
SAVE_STEP            = 1000
SAVE_N_CHECKPOINTS   = 1
EVAL_SPLIT_SIZE      = 0.1

# ── Audio ─────────────────────────────────────────────
SAMPLE_RATE        = 22050
OUTPUT_SAMPLE_RATE = 24000

# ── GPTArgs ──────────────────────────────────────────
MAX_CONDITIONING_LENGTH = 132300
MIN_CONDITIONING_LENGTH = 66150
MAX_WAV_LENGTH          = 255995
MAX_TEXT_LENGTH         = 200
GPT_NUM_AUDIO_TOKENS    = 1026
GPT_START_AUDIO_TOKEN   = 1024
GPT_STOP_AUDIO_TOKEN    = 1025

# ── Chemins locaux (Colab) ───────────────────────────
import os
MODEL_DIR = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/"
OUT_PATH  = "/content/pfa_outputs_male/"
DRIVE_BEST = "/content/drive/MyDrive/xtts_darija_m1_best/"
