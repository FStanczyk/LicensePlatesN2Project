"""Ścieżki i stałe wspólne dla projektu."""

import multiprocessing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
COMBINED = DATA_DIR / "combined"
DOWNLOADS = DATA_DIR / "_downloads"
EXPERIMENT = DATA_DIR / "experiment"
POSE = DATA_DIR / "pose"
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
MODELS = RUNS / "modele"

KAGGLE_DATASET = "filipstanczyk/eu-license-plates-alpr"
KAGGLE_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
MODELS_TAG = "modele-2026-11-21"
MODELS_URL = "https://github.com/FStanczyk/LicensePlatesN2Project/releases/download"
KAGGLE_INPUT = Path("/kaggle/input/eu-license-plates-alpr")

SEED = 0
MP_START = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_dataset() -> Path:
    """Podpięty input w notebooku Kaggle, lokalnie data/combined."""
    return KAGGLE_INPUT if (KAGGLE_INPUT / "plates.csv").exists() else COMBINED
