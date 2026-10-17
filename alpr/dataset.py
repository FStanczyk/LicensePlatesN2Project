"""Zbiór data/combined: wczytywanie plates.csv, format pól. Opis kolumn: README, sekcja "Dane"."""

import csv
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
    """Wielkie A-Z, 0-9 i '*' (znak zamazany); tak zapisana jest kolumna text."""
    return re.sub(r"[^A-Z0-9*]", "", text.upper())


def exposure(crop: np.ndarray) -> str:
    """under: 95. percentyl jasności < 100, over: 5. percentyl > 100, inaczej normal."""
    p5, p95 = np.percentile(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), [5, 95])
    return "under" if p95 < 100 else "over" if p5 > 100 else "normal"


def load_plates(
    dataset: str | Path | None = None,
    sample_type: str | None = None,
    split: str | None = None,
    originals_only: bool = False,
    ocr_only: bool = False,
) -> pd.DataFrame:
    """plates.csv po filtrach, z kolumnami source i corrupted."""
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
    return df.assign(
        source=df.plate_id.str.split("_").str[0],
        corrupted=np.where(df.artificial_corruption == "none", "original", "corrupted"),
    )


def parse_polygon(text: str) -> np.ndarray:
    return np.array(text.split(), dtype=np.float32).reshape(4, 2)


def format_polygon(poly: np.ndarray) -> str:
    return " ".join(str(round(v)) for v in poly.flatten())


def crop_path(dataset: str | Path, crop: str) -> Path:
    """Ścieżka wycinka; na Kaggle wycinki leżą w ocr/ocr/<split>/."""
    path = Path(dataset) / crop
    nested = Path(dataset) / "ocr" / crop
    return nested if not path.exists() and nested.exists() else path


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
