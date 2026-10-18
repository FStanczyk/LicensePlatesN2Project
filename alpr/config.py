"""Ścieżki i stałe wspólne dla projektu."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
COMBINED = DATA_DIR / "combined"
DOWNLOADS = DATA_DIR / "_downloads"
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"

KAGGLE_DATASET = "filipstanczyk/eu-license-plates-alpr"
KAGGLE_INPUT = Path("/kaggle/input/eu-license-plates-alpr")

SEED = 0
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_dataset() -> Path:
    """Podpięty input w notebooku Kaggle, lokalnie data/combined."""
    return KAGGLE_INPUT if (KAGGLE_INPUT / "plates.csv").exists() else COMBINED
