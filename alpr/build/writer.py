"""Zapis data/combined: zdjęcia i wycinki w dwóch wersjach (oryginał, sztucznie zniekształcona kopia)."""

import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from alpr.build.corruptions import add_corruption
from alpr.dataset import FIELDS, exposure, format_polygon, write_csv
from alpr.geometry import warp

MAX_SIDE = 1920
TEMPLATE = Path(__file__).parent / "template"


class CombinedWriter:
    def __init__(self, out: Path):
        self.out = out
        self.rows: list[dict] = []

    @classmethod
    def create(cls, out: Path) -> "CombinedWriter":
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(TEMPLATE, out)
        return cls(out)

    @staticmethod
    def _variants(img: np.ndarray, key: str) -> list[tuple[str, np.ndarray, dict]]:
        corrupted, name, severity = add_corruption(img, key)
        return [
            (key, img, {"artificial_corruption": "none", "artificial_severity": 0}),
            (f"{key}_{name}{severity}", corrupted, {"artificial_corruption": name, "artificial_severity": severity}),
        ]

    def _write(self, rel: str, img: np.ndarray, *params: int) -> None:
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img, list(params))

    def add_photo(self, key: str, split: str, img: np.ndarray, plates: list[dict]) -> None:
        """Całe zdjęcie z tablicami; plates: "poly" (4x2 w pikselach oryginału) i pola CSV."""
        h, w = img.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        if scale < 1:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        polys = [plate.pop("poly") * scale for plate in plates]

        for variant, image, fields in self._variants(img, key):
            image_rel = f"detection/images/{split}/{variant}.jpg"
            self._write(image_rel, image, cv2.IMWRITE_JPEG_QUALITY, 95)
            yolo = []
            for k, (poly, plate) in enumerate(zip(polys, plates, strict=True)):
                x1, y1 = poly.min(axis=0).clip(0)
                x2, y2 = np.minimum(poly.max(axis=0), [w, h])
                cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
                yolo.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                crop = warp(image, poly)
                crop_rel = f"ocr/{split}/{variant}_{k}.png"
                self._write(crop_rel, crop)
                self.rows.append(
                    {
                        "plate_id": f"{variant}_{k}",
                        "sample_type": "full_photo",
                        "split": split,
                        "image": image_rel,
                        "x1": round(x1),
                        "y1": round(y1),
                        "x2": round(x2),
                        "y2": round(y2),
                        "polygon": format_polygon(poly),
                        "crop": crop_rel,
                        "exposure": exposure(crop),
                        **fields,
                        **plate,
                    }
                )
            label = self.out / f"detection/labels/{split}/{variant}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(yolo) + "\n")

    def add_crop(self, key: str, split: str, img: np.ndarray, plate: dict) -> None:
        """Gotowy wycinek tablicy (bez zdjęcia)."""
        for variant, image, fields in self._variants(img, key):
            crop_rel = f"ocr/{split}/{variant}.png"
            self._write(crop_rel, image)
            self.rows.append(
                {
                    "plate_id": variant,
                    "sample_type": "plate_crop",
                    "split": split,
                    "crop": crop_rel,
                    "exposure": exposure(image),
                    **plate,
                    **fields,
                }
            )

    def save(self) -> Path:
        path = self.out / "plates.csv"
        write_csv(path, self.rows, FIELDS)
        return path

    def summary(self) -> Counter:
        """Liczba wierszy na (sample_type, split, czy kopia)."""
        return Counter((r["sample_type"], r["split"], r["artificial_corruption"] != "none") for r in self.rows)
