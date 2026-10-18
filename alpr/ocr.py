"""Odczyt znaków z wyprostowanego wycinka i metryki OCR (Golba). Szkielet do uzupełnienia.

Silnik: wycinek BGR -> (surowy tekst, pewność 0-1); wynik normalizowany jak kolumna text.
Metryki liczymy bez tablic z blurred == 1 (load_plates(..., ocr_only=True)).

    text, conf = read_plate(crop, engine="easyocr")
"""

from collections.abc import Callable

import numpy as np

from alpr.dataset import normalize_text

Engine = Callable[[np.ndarray], tuple[str, float]]

ENGINES: dict[str, Engine] = {}


def read_plate(crop: np.ndarray, engine: str = "easyocr") -> tuple[str, float]:
    if engine not in ENGINES:
        raise NotImplementedError(f"silnik {engine!r} nie jest jeszcze zaimplementowany (alpr/ocr.py)")
    raw, conf = ENGINES[engine](crop)
    return normalize_text(raw), conf


def character_accuracy(pred: str, true: str) -> float:
    """1 - odległość edycyjna / długość prawdziwego tekstu."""
    raise NotImplementedError


def plate_accuracy(preds: list[str], trues: list[str]) -> float:
    """Odsetek tablic odczytanych bezbłędnie."""
    raise NotImplementedError
