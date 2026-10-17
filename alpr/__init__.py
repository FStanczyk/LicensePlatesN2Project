"""Rozpoznawanie tablic rejestracyjnych (ALPR).

Moduły: config, dataset, geometry, download, preprocessing (Stańczyk), detection (Włodarczyk),
ocr (Golba), build (budowa zbioru, Stańczyk).
"""

from alpr.config import SEED
from alpr.dataset import crop_path, load_plates, normalize_text
from alpr.detection import PlateAligner, PlateDetector

__all__ = ["SEED", "PlateAligner", "PlateDetector", "crop_path", "load_plates", "normalize_text"]
