"""Stałe wspólne dla całego projektu: ścieżki, ziarno losowości, zbiór na Kaggle."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
COMBINED = DATA_DIR / "combined"
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"

KAGGLE_DATASET = "filipstanczyk/eu-license-plates-alpr"
KAGGLE_INPUT = Path("/kaggle/input/eu-license-plates-alpr")  # dataset podpięty w notebooku przez "Add Input"

SEED = 0
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_dataset() -> Path:
    """Katalog zbioru (z plates.csv): w notebooku Kaggle podpięty input, lokalnie data/combined."""
    return KAGGLE_INPUT if (KAGGLE_INPUT / "plates.csv").exists() else COMBINED
