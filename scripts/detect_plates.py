#!/usr/bin/env python3
"""Wykrywa tablice na zdjęciach, zapisuje wyprostowane wycinki i CSV z detekcjami (wejście dla OCR).

Użycie:
    python3 scripts/detect_plates.py --weights runs/detect/plates/weights/best.pt --images zdjecia/
    python3 scripts/detect_plates.py --weights best.pt --split test          # całe zdjęcia z plates.csv
    python3 scripts/detect_plates.py --weights best.pt --split test --no-align   # same ramki, bez prostowania

Wynik w --out (domyślnie results/detections/<nazwa>):
    detections.csv     image, det_id, conf, x1, y1, x2, y2, polygon, align, crop
                       (image względem --images albo zbioru, crop względem --out;
                       align: quad / rect / box, patrz alpr.detection.align_plate)
    crops/*.png        wycinki tablic, <nazwa zdjęcia>_<det_id>.png; gdy zdjęcia są w różnych
                       podfolderach, nazwa zawiera też podfolder ("a/b.jpg" -> "a__b_0.png")

Zdjęcia, na których nic nie wykryto, też są w CSV (jeden wiersz z pustym det_id), żeby potok
mógł je policzyć jako brak odczytu.
"""

import argparse
import csv
import os
import sys
from pathlib import Path, PurePosixPath

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # działa też bez pip install -e .

from alpr.config import RESULTS, default_dataset
from alpr.dataset import SPLITS, format_polygon, list_images, load_plates
from alpr.detection import align_plate, detect, load_model
from alpr.geometry import box_corners, warp

FIELDS = ["image", "det_id", "conf", "x1", "y1", "x2", "y2", "polygon", "align", "crop"]


def crop_name(rel: str, k: int, common: str) -> str:
    """Nazwa wycinka ze ścieżki zdjęcia względem wspólnego katalogu wszystkich zdjęć (common),
    żeby zdjęcia o tej samej nazwie w różnych podfolderach nie nadpisywały sobie wycinków."""
    stem = PurePosixPath(rel).relative_to(common).with_suffix("").as_posix()
    return f"crops/{stem.replace('/', '__')}_{k}.png"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="wagi YOLO (best.pt z train_yolo.py)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="folder ze zdjęciami (przeszukiwany rekurencyjnie)")
    source.add_argument("--split", choices=SPLITS, help="całe zdjęcia z plates.csv")
    parser.add_argument("--dataset", type=Path, default=default_dataset(), help="z --split: katalog zbioru (z plates.csv)")
    parser.add_argument("--originals-only", action="store_true", help="z --split: pomiń kopie ze sztucznym zniekształceniem")
    parser.add_argument("--conf", type=float, default=0.25, help="próg pewności detekcji")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-align", action="store_true", help="nie prostuj, zapisz zwykły wycinek ramki")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.images:
        base = args.images.resolve()
        images = [p.relative_to(base).as_posix() for p in list_images(base)]
        name = base.name
    else:
        base = args.dataset.resolve()
        df = load_plates(base, sample_type="full_photo", split=args.split, originals_only=args.originals_only)
        images = sorted(df.image.unique())
        name = args.split
    if not images:
        print(f"Brak zdjęć w {base}")
        return 1
    common = os.path.commonpath([PurePosixPath(rel).parent.as_posix() for rel in images])
    out = (args.out or RESULTS / "detections" / name).resolve()
    (out / "crops").mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights)
    rows, methods = [], {}
    for i, rel in enumerate(images, 1):
        img = cv2.imread(str(base / rel))
        if img is None:
            print(f"  pominięto (nie da się wczytać): {rel}")
            continue
        dets = detect(model, img, conf=args.conf, imgsz=args.imgsz)
        if not dets:
            rows.append({"image": rel})
        for k, det in enumerate(dets):
            if args.no_align:
                poly = box_corners(det.box)
                crop, method = warp(img, poly), "box"
            else:
                crop, poly, method = align_plate(img, det.box)
            methods[method] = methods.get(method, 0) + 1
            crop_rel = crop_name(rel, k, common)
            cv2.imwrite(str(out / crop_rel), crop)
            x1, y1, x2, y2 = det.box
            rows.append({
                "image": rel, "det_id": k, "conf": round(det.conf, 4),
                "x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2),
                "polygon": format_polygon(poly), "align": method, "crop": crop_rel,
            })
        if i % 100 == 0:
            print(f"  {i}/{len(images)}")

    with open(out / "detections.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    found = sum(methods.values())
    empty = sum(1 for r in rows if "det_id" not in r)
    print(f"{len(images)} zdjęć, {found} tablic, {empty} zdjęć bez detekcji; wyrównanie: {methods}")
    print(f"Wynik: {out / 'detections.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
