#!/usr/bin/env python3
"""Wykrywa tablice, zapisuje wyprostowane wycinki i detections.csv (wejście dla OCR).

Użycie:
    python3 scripts/detect_plates.py --weights best.pt --images zdjecia/
    python3 scripts/detect_plates.py --weights best.pt --split test              # całe zdjęcia z plates.csv
    python3 scripts/detect_plates.py --weights best.pt --split test --no-align   # bez prostowania

Wynik w --out (domyślnie results/detections/<nazwa>):
    detections.csv   image, det_id, conf, x1, y1, x2, y2, polygon, align (quad / rect / box), crop
                     (zdjęcie bez detekcji: jeden wiersz z pustym det_id)
    crops/*.png      <ścieżka zdjęcia z "/" -> "__">_<det_id>.png
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from alpr.config import RESULTS, default_dataset
from alpr.dataset import SPLITS, list_images, load_plates, write_csv
from alpr.detection import PlateAligner, PlateDetector
from alpr.detection.pipeline import DETECTION_FIELDS, crop_plates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="wagi YOLO (best.pt z train_yolo.py)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="folder ze zdjęciami (rekurencyjnie)")
    source.add_argument("--split", choices=SPLITS, help="całe zdjęcia z plates.csv")
    parser.add_argument("--dataset", type=Path, default=default_dataset(), help="z --split: katalog zbioru")
    parser.add_argument("--originals-only", action="store_true", help="z --split: bez zniekształconych kopii")
    parser.add_argument("--conf", type=float, default=0.25, help="próg pewności")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-align", action="store_true", help="zwykły wycinek ramki")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.images:
        base = args.images.resolve()
        images = [p.relative_to(base).as_posix() for p in list_images(base)]
    else:
        base = args.dataset.resolve()
        df = load_plates(base, sample_type="full_photo", split=args.split, originals_only=args.originals_only)
        images = sorted(df.image.unique())
    if not images:
        print(f"Brak zdjęć w {base}")
        return 1
    out = (args.out or RESULTS / "detections" / (base.name if args.images else args.split)).resolve()

    detector = PlateDetector(args.weights, args.conf, args.imgsz)
    rows = crop_plates(detector, images, base, out, None if args.no_align else PlateAligner())
    write_csv(out / "detections.csv", rows, DETECTION_FIELDS)

    methods = Counter(r["align"] for r in rows if "align" in r)
    empty = sum("det_id" not in r for r in rows)
    print(f"{len(images)} zdjęć, {methods.total()} tablic, {empty} zdjęć bez detekcji; wyrównanie: {dict(methods)}")
    print(f"Wynik: {out / 'detections.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
