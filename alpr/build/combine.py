"""Składa data1, data2 i data3 w jeden znormalizowany zbiór (data/combined, trafia na Kaggle).

Każde zdjęcie (i każdy wycinek z data3) dostaje jedną kopię ze sztucznym zniekształceniem:
losowy typ z CORRUPTIONS i siła 1-5 (kolumny artificial_corruption, artificial_severity).
Kopia trafia do tego samego podziału co oryginał.

Wynik w katalogu out:
    plates.csv                               jedna linia na tablicę (tekst, kraj, oświetlenie, ramka, ...)
    detection/images/<split>/*.jpg           całe zdjęcia (data1, data2), dłuższy bok <= MAX_SIDE
    detection/labels/<split>/*.txt           etykiety YOLO (klasa 0 = tablica)
    detection/data.yaml
    ocr/<split>/*.png                        wycięte tablice ze wszystkich zbiorów
"""

import csv
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from alpr.build.corruptions import add_corruption
from alpr.config import SEED
from alpr.dataset import exposure, format_polygon, normalize_text
from alpr.geometry import four_corners, warp

MAX_SIDE = 1920
SOURCES = ("data1", "data2", "data3")


def split_for(rng: random.Random, val: float, test: float) -> str:
    r = rng.random()
    return "test" if r < test else "val" if r < test + val else "train"


def add_photo(out: Path, rows: list, source: str, stem: str, split: str, img: np.ndarray, plates: list[dict]) -> None:
    """Zapisuje zdjęcie i jego zniekształconą kopię: obraz do detekcji, etykiety YOLO i wycięte tablice.

    plates: "poly" (4x2, w pikselach oryginału) + pozostałe pola CSV. Wycinki są brane z zapisanego
    (pomniejszonego) zdjęcia, żeby oryginał i kopia miały tablice w tej samej rozdzielczości.
    """
    h, w = img.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    polys = [plate.pop("poly") * scale for plate in plates]
    name = f"{source}_{stem}"
    corrupted, corruption, severity = add_corruption(img, name)

    for variant, image, fields in (
        (name, img, {"artificial_corruption": "none", "artificial_severity": 0}),
        (f"{name}_{corruption}{severity}", corrupted,
         {"artificial_corruption": corruption, "artificial_severity": severity}),
    ):
        image_rel = f"detection/images/{split}/{variant}.jpg"
        (out / image_rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / image_rel), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        yolo = []
        for k, (poly, plate) in enumerate(zip(polys, plates, strict=True)):
            x1, y1 = poly.min(axis=0).clip(0)
            x2, y2 = np.minimum(poly.max(axis=0), [w, h])
            yolo.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            crop_rel = f"ocr/{split}/{variant}_{k}.png"
            (out / crop_rel).parent.mkdir(parents=True, exist_ok=True)
            crop = warp(image, poly)
            cv2.imwrite(str(out / crop_rel), crop)
            rows.append({
                "plate_id": f"{variant}_{k}", "sample_type": "full_photo", "split": split,
                "image": image_rel,
                "x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2),
                "polygon": format_polygon(poly), "crop": crop_rel,
                "exposure": exposure(crop),
                **fields, **plate,
            })
        label = out / f"detection/labels/{split}/{variant}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(yolo) + "\n")


def build_data1(src: Path, out: Path, rows: list) -> None:
    """data1: zdjęcia z Polski, adnotacje CVAT (prostokąt + numer), bez podziału, losujemy 70/15/15."""
    rng = random.Random(SEED)
    for image in ET.parse(src / "annotations.xml").getroot().iter("image"):
        img = cv2.imread(str(src / "photos" / image.get("name")))
        plates = []
        for box in image.iter("box"):
            x1, y1, x2, y2 = (float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            raw = box.findtext("attribute[@name='plate number']", "")
            plates.append({
                "poly": np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
                "text": normalize_text(raw), "text_raw": raw, "blurred": 0,
                "country": "pl", "lighting": "unknown", "lines": 1,
            })
        add_photo(out, rows, "data1", Path(image.get("name")).stem, split_for(rng, 0.15, 0.15), img, plates)


def build_data2(src: Path, out: Path, rows: list) -> None:
    """data2: UC3M-LP (Hiszpania), oficjalny podział train/test, z train wydzielamy 15% na val."""
    rng = random.Random(SEED)
    with open(src / "labels.csv", newline="") as f:
        photos = list(csv.DictReader(f))
    for photo in photos:
        stem = photo["file"][: -len(".jpg")]
        split = photo["split"] if photo["split"] == "test" else split_for(rng, 0.15, 0)
        img = cv2.imread(str(src / "photos" / photo["file"]))
        plates = []
        for lp in json.loads((src / "annotations" / f"{stem}.json").read_text())["lps"]:
            code, raw = lp["lp_id"].split("-", 1)  # np. "AN-61*7*DX": typ A, noc N
            plates.append({
                "poly": four_corners(np.array(lp["poly_coord"], dtype=float)),
                "text": normalize_text(raw), "text_raw": lp["lp_id"], "blurred": int("*" in raw),
                "country": "es", "lighting": {"D": "day", "N": "night"}[code[1]],
                "lines": 2 if code[0] == "B" else 1,
            })
        add_photo(out, rows, "data2", stem, split, img, plates)


def build_data3(src: Path, out: Path, rows: list) -> None:
    """data3: AUTO.RIA OCR EU, gotowe wycięte tablice z oficjalnym podziałem train/val/test."""
    with open(src / "labels.csv", newline="") as f:
        plates = list(csv.DictReader(f))
    # oryginalne nazwy plików mają spacje, cyrylicę, '&' itp., których Kaggle nie przyjmuje
    for i, plate in enumerate(plates):
        split = plate["file"].split("/")[0]
        stem = f"data3_{i:05d}"
        img = cv2.imread(str(src / "photos" / plate["file"]))  # PNG z kanałem alfa -> 3 kanały BGR
        corrupted, corruption, severity = add_corruption(img, stem)
        common = {
            "sample_type": "plate_crop", "split": split,
            "text": normalize_text(plate["text"]), "text_raw": plate["text"], "blurred": 0,
            "country": plate["country"], "lighting": "unknown", "lines": plate["lines"] or 1,
        }
        for plate_id, image, fields in (
            (stem, img, {"artificial_corruption": "none", "artificial_severity": 0}),
            (f"{stem}_{corruption}{severity}", corrupted,
             {"artificial_corruption": corruption, "artificial_severity": severity}),
        ):
            crop_rel = f"ocr/{split}/{plate_id}.png"
            (out / crop_rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / crop_rel), image)
            rows.append({"plate_id": plate_id, "crop": crop_rel, "exposure": exposure(image), **common, **fields})


BUILDERS = {"data1": build_data1, "data2": build_data2, "data3": build_data3}
