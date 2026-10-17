"""Zbiór data/combined: wczytywanie i filtrowanie plates.csv, kolumny pochodne, format pól.

Przykład:
    from alpr.dataset import load_plates
    test = load_plates(sample_type="full_photo", split="test", originals_only=True)
    ocr = load_plates(split="test", ocr_only=True)        # bez UC3M-LP (zamazane znaki)

Ścieżki w kolumnach image i crop są względne do katalogu zbioru. Opis kolumn: README, sekcja "Dane".
"""

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from alpr.config import IMG_EXTS, default_dataset

FIELDS = [
    "plate_id",
    "sample_type",
    "split",
    "text",
    "text_raw",
    "blurred",
    "country",
    "lighting",
    "exposure",
    "lines",
    "artificial_corruption",
    "artificial_severity",
    "image",
    "x1",
    "y1",
    "x2",
    "y2",
    "polygon",
    "crop",
]
SPLITS = ["train", "val", "test"]


def normalize_text(text: str) -> str:
    """Wielkie litery, tylko A-Z, 0-9 i '*' (znak zamazany w data2). Tak zapisana jest kolumna text,
    więc tak samo trzeba normalizować odczyt OCR przed porównaniem."""
    return re.sub(r"[^A-Z0-9*]", "", text.upper())


def exposure(crop: np.ndarray) -> str:
    """Naświetlenie tablicy mierzone na wycinku: "under", gdy nawet jasne tło jest ciemne (95. percentyl
    jasności < 100), "over", gdy nawet znaki są jasne (5. percentyl > 100), inaczej "normal".
    Progi dobrane na data1 i data3 (ok. 3% under, 4% over wśród oryginałów)."""
    p5, p95 = np.percentile(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), [5, 95])
    return "under" if p95 < 100 else "over" if p5 > 100 else "normal"


def load_plates(
    dataset: str | Path | None = None,
    sample_type: str | None = None,
    split: str | None = None,
    originals_only: bool = False,
    ocr_only: bool = False,
) -> pd.DataFrame:
    """Wczytuje plates.csv z katalogu zbioru (domyślnie config.default_dataset()) i filtruje wiersze.

    sample_type: "full_photo" (detekcja) albo "plate_crop"; split: train / val / test;
    originals_only: bez kopii ze sztucznym zniekształceniem; ocr_only: bez tablic z zamazanymi znakami.
    Dodaje kolumny source (data1 / data2 / data3) i corrupted (original / corrupted).
    """
    df = pd.read_csv(Path(dataset or default_dataset()) / "plates.csv")
    if sample_type:
        df = df[df.sample_type == sample_type]
    if split:
        df = df[df.split == split]
    if originals_only:
        df = df[df.artificial_corruption == "none"]
    if ocr_only:
        df = df[df.blurred == 0]
    return add_derived(df.copy())


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Kolumny pochodne: source (prefiks plate_id) i corrupted (original / corrupted)."""
    return df.assign(
        source=df.plate_id.str.split("_").str[0],
        corrupted=np.where(df.artificial_corruption == "none", "original", "corrupted"),
    )


def parse_polygon(text: str) -> np.ndarray:
    """Kolumna polygon ("x1 y1 x2 y2 ...") -> narożniki 4x2."""
    return np.array(text.split(), dtype=np.float32).reshape(4, 2)


def format_polygon(poly: np.ndarray) -> str:
    """Narożniki 4x2 -> zapis jak w kolumnie polygon (liczby całkowite rozdzielone spacją)."""
    return " ".join(str(round(v)) for v in poly.flatten())


def list_images(folder: Path) -> list[Path]:
    """Wszystkie zdjęcia w folderze (rekurencyjnie), posortowane."""
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
