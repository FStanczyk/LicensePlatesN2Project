"""Odczyt znaków z wyprostowanego wycinka i metryki OCR (Golba).

Silnik: wycinek BGR -> (surowy tekst, pewność 0-1); wynik normalizowany jak kolumna text.
Silnik wsadowy (BATCH_ENGINES): lista wycinków -> lista wyników, jeden przebieg modelu na wiele tablic.
Parametry silnika (np. decoder, mag_ratio, batch_size) trafiają do EasyOCR bez zmian.
Metryki liczymy bez tablic z blurred == 1 (load_plates(..., ocr_only=True)).

    text, conf = read_plate(crop, engine="easyocr")
    results = read_batch([crop1, crop2], engine="parseq")
"""

from collections.abc import Callable
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from alpr.dataset import normalize_text

Engine = Callable[[np.ndarray], tuple[str, float]]
BatchEngine = Callable[[list[np.ndarray]], list[tuple[str, float]]]

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MIN_HEIGHT = 0.6


@cache
def easyocr_reader():
    import easyocr
    import torch

    torch.backends.cudnn.benchmark = True
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


def plate_fragments(boxes: list[np.ndarray], min_height: float = MIN_HEIGHT) -> list[int]:
    """Fragmenty o wysokości >= min_height najwyższego; niższe to napisy z ramki albo "PL" z paska UE."""
    heights = [np.ptp(b[:, 1]) for b in boxes]
    return [i for i, h in enumerate(heights) if h >= min_height * max(heights)]


def easyocr_engine(crop: np.ndarray, min_height: float = MIN_HEIGHT, **params) -> tuple[str, float]:
    """Fragmenty tablicy (plate_fragments) sklejane w kolejności czytania; pewność to średnia ważona długością."""
    found = easyocr_reader().readtext(crop, allowlist=ALLOWLIST, detail=1, paragraph=False, **params)
    if not found:
        return "", 0.0
    found = [found[i] for i in plate_fragments([np.asarray(box, np.float32) for box, _, _ in found], min_height)]
    boxes = [np.asarray(box, np.float32) for box, _, _ in found]
    parts = [found[i] for i in reading_order(boxes)]
    text = "".join(t for _, t, _ in parts)
    weights = [max(len(t), 1) for _, t, _ in parts]
    return text, float(np.average([c for _, _, c in parts], weights=weights))


def tesseract_input(crop: np.ndarray, height: int = 64, binarize: bool = False, pad: int = 10) -> np.ndarray:
    """Szarość, wysokość height px, opcjonalnie Otsu, biała ramka (Tesseract słabo czyta tekst przy krawędzi)."""
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = height / grey.shape[0]
    grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    if binarize:
        grey = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.copyMakeBorder(
        grey, pad, pad, pad, pad, cv2.BORDER_REPLICATE if not binarize else cv2.BORDER_CONSTANT, value=255
    )


def tesseract_engine(
    crop: np.ndarray, psm: int = 7, oem: int = 1, whitelist: bool = True, **params
) -> tuple[str, float]:
    """Tesseract (LSTM, model eng); pewność to średnia pewność słów 0-1.

    whitelist: tylko A-Z i 0-9; Tesseract 5 zgłasza wtedy pewność 0, bez listy śmieci usuwa normalize_text.
    """
    import pytesseract

    config = f"--psm {psm} --oem {oem}" + (f" -c tessedit_char_whitelist={ALLOWLIST}" if whitelist else "")
    data = pytesseract.image_to_data(
        tesseract_input(crop, **params), config=config, output_type=pytesseract.Output.DICT
    )
    words = [(t, float(c)) for t, c in zip(data["text"], data["conf"], strict=True) if t.strip() and float(c) >= 0]
    if not words:
        return "", 0.0
    return "".join(t for t, _ in words), float(np.mean([c for _, c in words])) / 100


def tesseract_batch(crops: list[np.ndarray], workers: int = 16, **params) -> list[tuple[str, float]]:
    """Każde wywołanie to osobny proces tesseract, więc wycinki idą równolegle w wątkach."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda c: tesseract_engine(c, **params), crops))


@cache
def recognizer(weights: str):
    from alpr.recognizer import Recognizer

    return Recognizer(weights)


def network_batch(name: str) -> BatchEngine:
    """Silnik z wagami sieci name (scripts/train_ocr.py); domyślnie RUNS/ocr/<name>/best.pt."""

    def read(crops: list[np.ndarray], weights: str | None = None, batch: int = 1024) -> list[tuple[str, float]]:
        from alpr.config import RUNS

        return recognizer(str(weights or RUNS / "ocr" / name / "best.pt")).read(crops, batch)

    read.__name__ = f"{name}_batch"
    return read


ENGINES: dict[str, Engine] = {"easyocr": easyocr_engine}
BATCH_ENGINES: dict[str, BatchEngine] = {
    "tesseract": tesseract_batch,
    "crnn": network_batch("crnn"),
    "lprnet": network_batch("lprnet"),
    "parseq": network_batch("parseq"),
}


def engine_names() -> list[str]:
    return sorted(ENGINES | BATCH_ENGINES)


def parse_params(items: list[str]) -> dict:
    """["decoder=beamsearch", "mag_ratio=1.5"] -> parametry silnika z typami z YAML."""
    import yaml

    return {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in items)}


def read_batch(crops: list[np.ndarray], engine: str = "easyocr", **params) -> list[tuple[str, float]]:
    """Znormalizowane odczyty listy wycinków; silnik pojedynczy działa po kolei."""
    if engine in BATCH_ENGINES:
        found = BATCH_ENGINES[engine](crops, **params)
    elif engine in ENGINES:
        found = [ENGINES[engine](crop, **params) for crop in crops]
    else:
        raise NotImplementedError(f"silnik {engine!r} nie jest jeszcze zaimplementowany (alpr/ocr.py)")
    return [(normalize_text(raw), conf) for raw, conf in found]


def read_plate(crop: np.ndarray, engine: str = "easyocr", **params) -> tuple[str, float]:
    return read_batch([crop], engine, **params)[0]


def crop_images(rows: pd.DataFrame, base: Path, crops: dict[str, np.ndarray] | None = None) -> list:
    """Wycinki wierszy (z pamięci, inaczej z dysku względem base); None dla wiersza bez wycinka."""
    crops = crops or {}
    return [
        None if not isinstance(c, str) else crops[c] if c in crops else cv2.imread(str(base / c)) for c in rows.crop
    ]


def read_crops(
    rows: pd.DataFrame,
    base: Path,
    engine: str = "easyocr",
    crops: dict[str, np.ndarray] | None = None,
    chunk: int = 256,
    **params,
) -> pd.DataFrame:
    """Dopisuje text i conf do wierszy z kolumną crop (względną do base); wiersz bez wycinka zostaje pusty.

    crops: wycinki już w pamięci (klucz jak kolumna crop, np. z crop_plates), reszta czytana z dysku.
    Silnik dostaje po chunk wycinków naraz.
    """
    images = crop_images(rows, base, crops)
    present = [i for i, img in enumerate(images) if img is not None]
    texts, confs = [None] * len(images), [None] * len(images)
    for start in range(0, len(present), chunk):
        part = present[start : start + chunk]
        for i, (text, conf) in zip(part, read_batch([images[i] for i in part], engine, **params), strict=True):
            texts[i], confs[i] = text, round(conf, 4)
        print(f"  {min(start + chunk, len(present))}/{len(present)}")
    return rows.assign(text=texts, conf_ocr=confs)


def gate(readings: list[tuple[str, float]], margin: float = 0.1) -> tuple[str, float]:
    """Odczyt pierwszego widoku (wycinek bez zmian), chyba że inny widok jest pewniejszy o więcej niż margin."""
    best = max(range(len(readings)), key=lambda i: readings[i][1])
    return readings[best] if readings[best][1] > readings[0][1] + margin else readings[0]


def read_views(
    views: list[list[np.ndarray]], engine: str, margin: float = 0.1, chunk: int = 512, **params
) -> list[tuple[str, float]]:
    """Dla każdego wycinka lista widoków (pierwszy bez zmian); wszystkie czytane wsadami, wynik według gate."""
    flat = [v for vs in views for v in vs]
    found = []
    for start in range(0, len(flat), chunk):
        found += read_batch(flat[start : start + chunk], engine, **params)
    out, i = [], 0
    for vs in views:
        out.append(gate(found[i : i + len(vs)], margin))
        i += len(vs)
    return out


def read_deblurred(
    crops: list[np.ndarray],
    motion_angles: list[float],
    configs: list[dict],
    margin: float = 0.1,
    iters: int = 30,
    deblur: bool = True,
) -> list[tuple[str, float]]:
    """Odczyt z dekonwolucją (alpr.deblur.deconv_views) i kilkoma silnikami naraz: wygrywa pewniejszy odczyt.

    configs: np. [{"engine": "parseq", "weights": "a/best.pt"}, {"engine": "parseq", "weights": "b/best.pt"}].
    deblur=False: tylko wycinek bez zmian (łączenie silników bez dekonwolucji).
    """
    from alpr.deblur import deconv_views

    views = [deconv_views(c, a, iters) if deblur else [c] for c, a in zip(crops, motion_angles, strict=True)]
    results = [read_views(views, margin=margin, **cfg) for cfg in configs]
    return [max(options, key=lambda r: r[1]) for options in zip(*results, strict=True)]


def read_crops_deblurred(
    rows: pd.DataFrame,
    base: Path,
    motion_angles: dict[str, float],
    configs: list[dict],
    crops: dict[str, np.ndarray] | None = None,
    margin: float = 0.1,
    iters: int = 30,
    deblur: bool = True,
) -> pd.DataFrame:
    """Jak read_crops, ale przez read_deblurred; motion_angles: kąt ruchu (estimate_blur) dla kolumny image."""
    images = crop_images(rows, base, crops)
    present = [i for i, img in enumerate(images) if img is not None]
    angles = [motion_angles.get(rows.image.iloc[i], 0.0) for i in present]
    found = read_deblurred([images[i] for i in present], angles, configs, margin, iters, deblur)
    texts, confs = [None] * len(images), [None] * len(images)
    for i, (text, conf) in zip(present, found, strict=True):
        texts[i], confs[i] = text, round(conf, 4)
    return rows.assign(text=texts, conf_ocr=confs)


def edit_distance(a: str, b: str) -> int:
    """Odległość Levenshteina."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def character_accuracy(pred: str, true: str) -> float:
    """1 - odległość edycyjna / długość prawdziwego tekstu (nie mniej niż 0)."""
    return max(0.0, 1 - edit_distance(pred, true) / max(len(true), 1))


def plate_accuracy(preds: list[str], trues: list[str]) -> float:
    """Odsetek tablic odczytanych bezbłędnie."""
    return float(np.mean([p == t for p, t in zip(preds, trues, strict=True)]))
