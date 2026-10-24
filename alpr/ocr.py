"""Odczyt znaków z wyprostowanego wycinka i metryki OCR (Golba).

Silnik: wycinek BGR -> (surowy tekst, pewność 0-1); wynik normalizowany jak kolumna text.
Metryki liczymy bez tablic z blurred == 1 (load_plates(..., ocr_only=True)).

    text, conf = read_plate(crop, engine="easyocr")
"""

from collections.abc import Callable
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from alpr.dataset import normalize_text

Engine = Callable[[np.ndarray], tuple[str, float]]

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


@cache
def easyocr_reader():
    import easyocr
    import torch

    return easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)


def reading_order(boxes: list[np.ndarray]) -> list[int]:
    """Indeksy fragmentów wierszami z góry na dół, w wierszu od lewej (tablice dwuwierszowe)."""
    centers = [(b[:, 0].mean(), b[:, 1].mean(), np.ptp(b[:, 1])) for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: centers[i][1])
    lines: list[list[int]] = []
    for i in order:
        if lines and abs(centers[i][1] - centers[lines[-1][0]][1]) < centers[i][2] / 2:
            lines[-1].append(i)
        else:
            lines.append([i])
    return [i for line in lines for i in sorted(line, key=lambda i: centers[i][0])]


def easyocr_engine(crop: np.ndarray) -> tuple[str, float]:
    """Fragmenty tekstu sklejane w kolejności czytania; pewność to średnia ważona długością."""
    found = easyocr_reader().readtext(crop, allowlist=ALLOWLIST, detail=1, paragraph=False)
    if not found:
        return "", 0.0
    boxes = [np.asarray(box, np.float32) for box, _, _ in found]
    parts = [found[i] for i in reading_order(boxes)]
    text = "".join(t for _, t, _ in parts)
    weights = [max(len(t), 1) for _, t, _ in parts]
    return text, float(np.average([c for _, _, c in parts], weights=weights))


ENGINES: dict[str, Engine] = {"easyocr": easyocr_engine}


def read_plate(crop: np.ndarray, engine: str = "easyocr") -> tuple[str, float]:
    if engine not in ENGINES:
        raise NotImplementedError(f"silnik {engine!r} nie jest jeszcze zaimplementowany (alpr/ocr.py)")
    raw, conf = ENGINES[engine](crop)
    return normalize_text(raw), conf


def read_crops(rows: pd.DataFrame, base: Path, engine: str = "easyocr") -> pd.DataFrame:
    """Dopisuje text i conf do wierszy z kolumną crop (względną do base); wiersz bez wycinka zostaje pusty."""
    texts, confs = [], []
    for i, crop in enumerate(rows.crop, 1):
        img = cv2.imread(str(base / crop)) if isinstance(crop, str) else None
        if img is None:
            texts.append(None)
            confs.append(None)
        else:
            text, conf = read_plate(img, engine)
            texts.append(text)
            confs.append(round(conf, 4))
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}")
    return rows.assign(text=texts, conf_ocr=confs)


def character_accuracy(pred: str, true: str) -> float:
    """1 - odległość edycyjna / długość prawdziwego tekstu."""
    raise NotImplementedError


def plate_accuracy(preds: list[str], trues: list[str]) -> float:
    """Odsetek tablic odczytanych bezbłędnie."""
    raise NotImplementedError
