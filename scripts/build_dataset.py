#!/usr/bin/env python3
"""Składa data1, data2 i data3 w jeden znormalizowany zbiór w data/combined/ (do wrzucenia na Kaggle).

Użycie:
    pip install imagecorruptions==1.1.2 opencv-python
    python scripts/download_data.py          # najpierw pobierz zbiory źródłowe
    python scripts/build_dataset.py          # zbuduj data/combined/

Każde zdjęcie (i każdy wycinek z data3) dostaje jedną kopię ze sztucznym zniekształceniem:
losowy typ z CORRUPTIONS i siła 1-5 (kolumny artificial_corruption, artificial_severity).
Kopia trafia do tego samego podziału co oryginał.

Wynik:
    plates.csv                               jedna linia na tablicę (tekst, kraj, oświetlenie, ramka, ...)
    detection/images/<split>/*.jpg           całe zdjęcia (data1, data2), dłuższy bok <= MAX_SIDE
    detection/labels/<split>/*.txt           etykiety YOLO (klasa 0 = tablica)
    detection/data.yaml
    ocr/<split>/*.png                        wycięte tablice ze wszystkich zbiorów
"""

import csv
import json
import random
import re
import shutil
import sys
import warnings
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import cv2
import numpy as np
import skimage.filters

# imagecorruptions 1.1.2 nie działa z numpy 2 i scikit-image >= 0.19 (fog, gaussian_blur, glass_blur)
np.float_ = np.float64
from imagecorruptions import corrupt, corruptions  # noqa: E402

_gaussian = skimage.filters.gaussian
corruptions.gaussian = lambda img, sigma=1, multichannel=None, **kw: _gaussian(
    img, sigma=sigma, channel_axis=-1 if multichannel else None, **kw)
warnings.filterwarnings("ignore", module="imagecorruptions")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT = DATA_DIR / "combined"
MAX_SIDE = 1920
SEED = 0
FIELDS = ["plate_id", "sample_type", "split",
          "text", "text_raw", "blurred", "country", "lighting", "exposure", "lines",
          "artificial_corruption", "artificial_severity",
          "image", "x1", "y1", "x2", "y2", "polygon", "crop"]
# z imagecorruptions, plus "darken" (brak w pakiecie: brightness tylko rozjaśnia);
# pomijamy przekształcenia geometryczne (elastic_transform, zoom_blur), bo przesuwają tablicę
CORRUPTIONS = ["brightness", "darken", "contrast", "fog", "gaussian_noise", "shot_noise",
               "motion_blur", "defocus_blur", "jpeg_compression"]


def normalize(text: str) -> str:
    """Wielkie litery, tylko A-Z, 0-9 i '*' (znak zamazany w data2)."""
    return re.sub(r"[^A-Z0-9*]", "", text.upper())


def exposure(crop: np.ndarray) -> str:
    """Naświetlenie tablicy mierzone na wycinku: "under", gdy nawet jasne tło jest ciemne (95. percentyl
    jasności < 100), "over", gdy nawet znaki są jasne (5. percentyl > 100), inaczej "normal".
    Progi dobrane na data1 i data3 (ok. 3% under, 4% over wśród oryginałów)."""
    p5, p95 = np.percentile(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), [5, 95])
    return "under" if p95 < 100 else "over" if p5 > 100 else "normal"


def darken(img: np.ndarray, severity: int) -> np.ndarray:
    """Niedoświetlenie: korekcja gamma > 1 i obniżenie jasności."""
    gamma, gain = [(1.4, 0.85), (1.8, 0.7), (2.2, 0.55), (2.6, 0.42), (3.0, 0.3)][severity - 1]
    return (255 * gain * (img / 255.0) ** gamma).astype(np.uint8)


def add_corruption(img: np.ndarray, key: str) -> tuple[np.ndarray, str, int]:
    """Losuje (powtarzalnie, na podstawie key) zniekształcenie i siłę 1-5; img w BGR."""
    rng = random.Random(f"{SEED}:{key}")
    name, severity = rng.choice(CORRUPTIONS), rng.randint(1, 5)
    if name == "darken":
        return darken(img, severity), name, severity
    np.random.seed(zlib.crc32(key.encode()))  # imagecorruptions losuje przez np.random
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    out = corrupt(rgb, corruption_name=name, severity=severity)
    return np.ascontiguousarray(out[:, :, ::-1]), name, severity


def split_for(rng: random.Random, val: float, test: float) -> str:
    r = rng.random()
    return "test" if r < test else "val" if r < test + val else "train"


def four_corners(points: np.ndarray) -> np.ndarray:
    """Sprowadza wielokąt do 4 narożników (lewy górny, prawy górny, prawy dolny, lewy dolny).
    W data2 ok. 5% tablic ma 5-6 punktów (powtórzone punkty albo ucięty róg)."""
    if len(points) == 4:
        return points
    s, d = points.sum(axis=1), points[:, 0] - points[:, 1]
    return points[[s.argmin(), d.argmax(), s.argmax(), d.argmin()]]


def warp(img: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Prostuje tablicę z 4 narożników (lewy górny, prawy górny, prawy dolny, lewy dolny)."""
    tl, tr, br, bl = poly
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(poly.astype(np.float32), dst), (w, h))


def add_photo(rows: list, source: str, stem: str, split: str, img: np.ndarray, plates: list[dict]) -> None:
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
        (OUT / image_rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / image_rel), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        yolo = []
        for k, (poly, plate) in enumerate(zip(polys, plates)):
            x1, y1 = poly.min(axis=0).clip(0)
            x2, y2 = np.minimum(poly.max(axis=0), [w, h])
            yolo.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            crop_rel = f"ocr/{split}/{variant}_{k}.png"
            (OUT / crop_rel).parent.mkdir(parents=True, exist_ok=True)
            crop = warp(image, poly)
            cv2.imwrite(str(OUT / crop_rel), crop)
            rows.append({
                "plate_id": f"{variant}_{k}", "sample_type": "full_photo", "split": split,
                "image": image_rel,
                "x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2),
                "polygon": " ".join(str(round(v)) for v in poly.flatten()), "crop": crop_rel,
                "exposure": exposure(crop),
                **fields, **plate,
            })
        label = OUT / f"detection/labels/{split}/{variant}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(yolo) + "\n")


def build_data1(rows: list) -> None:
    """data1: zdjęcia z Polski, adnotacje CVAT (prostokąt + numer), bez podziału - losujemy 70/15/15."""
    src = DATA_DIR / "data1"
    rng = random.Random(SEED)
    for image in ET.parse(src / "annotations.xml").getroot().iter("image"):
        img = cv2.imread(str(src / "photos" / image.get("name")))
        plates = []
        for box in image.iter("box"):
            x1, y1, x2, y2 = (float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            raw = box.findtext("attribute[@name='plate number']", "")
            plates.append({
                "poly": np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
                "text": normalize(raw), "text_raw": raw, "blurred": 0,
                "country": "pl", "lighting": "unknown", "lines": 1,
            })
        add_photo(rows, "data1", Path(image.get("name")).stem, split_for(rng, 0.15, 0.15), img, plates)


def build_data2(rows: list) -> None:
    """data2: UC3M-LP (Hiszpania), oficjalny podział train/test, z train wydzielamy 15% na val."""
    src = DATA_DIR / "data2"
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
                "text": normalize(raw), "text_raw": lp["lp_id"], "blurred": int("*" in raw),
                "country": "es", "lighting": {"D": "day", "N": "night"}[code[1]],
                "lines": 2 if code[0] == "B" else 1,
            })
        add_photo(rows, "data2", stem, split, img, plates)


def build_data3(rows: list) -> None:
    """data3: AUTO.RIA OCR EU, gotowe wycięte tablice z oficjalnym podziałem train/val/test."""
    src = DATA_DIR / "data3"
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
            "text": normalize(plate["text"]), "text_raw": plate["text"], "blurred": 0,
            "country": plate["country"], "lighting": "unknown", "lines": plate["lines"] or 1,
        }
        for plate_id, image, fields in (
            (stem, img, {"artificial_corruption": "none", "artificial_severity": 0}),
            (f"{stem}_{corruption}{severity}", corrupted,
             {"artificial_corruption": corruption, "artificial_severity": severity}),
        ):
            crop_rel = f"ocr/{split}/{plate_id}.png"
            (OUT / crop_rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(OUT / crop_rel), image)
            rows.append({"plate_id": plate_id, "crop": crop_rel, "exposure": exposure(image), **common, **fields})


def main() -> int:
    missing = [d for d in ("data1", "data2", "data3") if not (DATA_DIR / d / "photos").is_dir()]
    if missing:
        print(f"Brak zdjęć dla: {', '.join(missing)}. Uruchom najpierw scripts/download_data.py")
        return 1
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(ROOT / "scripts" / "combined_template", OUT)

    rows = []
    for name, build in (("data1", build_data1), ("data2", build_data2), ("data3", build_data3)):
        print(f"[{name}]")
        before = len(rows)
        build(rows)
        print(f"  {len(rows) - before} wierszy (oryginał + zniekształcona kopia)")

    with open(OUT / "plates.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        key = (r["sample_type"], r["split"], r["artificial_corruption"] != "none")
        counts[key] = counts.get(key, 0) + 1
    print(f"Razem {len(rows)} tablic w {OUT.relative_to(ROOT)} (oryginał / zniekształcona kopia):")
    for kind, split in sorted({k[:2] for k in counts}):
        print(f"  {kind} {split}: {counts.get((kind, split, False), 0)} / {counts.get((kind, split, True), 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
