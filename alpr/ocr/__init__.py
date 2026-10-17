"""Odczyt znaków z wyciętej tablicy i metryki OCR (Golba, bloki 3-6).

Szkielet: ustalony interfejs, implementacje do uzupełnienia.

    from alpr.ocr import read_plate
    text, conf = read_plate(crop, engine="easyocr")

Wejście: wyprostowany wycinek BGR (kolumna crop w plates.csv albo detections.csv z detect_plates.py).
Wyjście: tekst znormalizowany jak kolumna text (alpr.dataset.normalize_text: A-Z, 0-9) i pewność 0-1.
Przy metrykach pomijamy tablice z blurred == 1 (load_plates(..., ocr_only=True)).
"""

from collections.abc import Callable

import numpy as np

from alpr.dataset import normalize_text

# silnik: (wycinek BGR) -> (surowy tekst, pewność 0-1)
Engine = Callable[[np.ndarray], tuple[str, float]]

ENGINES: dict[str, Engine] = {}  # do uzupełnienia: "easyocr", "tesseract", "lprnet", "crnn"


def read_plate(crop: np.ndarray, engine: str = "easyocr") -> tuple[str, float]:
    """Odczytuje tablicę wybranym silnikiem i normalizuje tekst."""
    if engine not in ENGINES:
        raise NotImplementedError(f"silnik {engine!r} nie jest jeszcze zaimplementowany (alpr/ocr)")
    raw, conf = ENGINES[engine](crop)
    return normalize_text(raw), conf


def character_accuracy(pred: str, true: str) -> float:
    """Character Accuracy Rate: 1 - odległość edycyjna / długość prawdziwego tekstu (README, "Metryki")."""
    raise NotImplementedError


def plate_accuracy(preds: list[str], trues: list[str]) -> float:
    """Odsetek tablic odczytanych bezbłędnie (Word/Plate Accuracy)."""
    raise NotImplementedError
