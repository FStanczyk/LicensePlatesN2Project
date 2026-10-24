#!/usr/bin/env python3
"""Odczytuje wycinki z detections.csv (wynik detect_plates.py) silnikiem OCR i zapisuje odczyty do CSV.

Użycie:
    python3 scripts/ocr_plates.py --detections results/detections/test/detections.csv
    python3 scripts/ocr_plates.py --detections results/detections/test/detections.csv --engine easyocr --force

Wynik obok detections.csv: ocr_<silnik>.csv z kolumnami detections.csv oraz
    text      odczyt znormalizowany jak kolumna text w plates.csv (A-Z, 0-9)
    conf_ocr  pewność silnika 0-1
(zdjęcie bez detekcji: puste text i conf_ocr). Istniejący wynik nie jest liczony ponownie bez --force.
"""

import argparse
import sys
from pathlib import Path

from alpr.detection.pipeline import read_detections
from alpr.ocr import ENGINES, read_crops


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detections", type=Path, required=True, help="detections.csv z detect_plates.py")
    parser.add_argument("--engine", default="easyocr", choices=sorted(ENGINES))
    parser.add_argument("--out", type=Path, help="domyślnie ocr_<silnik>.csv obok detections.csv")
    parser.add_argument("--force", action="store_true", help="policz ponownie, nawet jeśli wynik już jest")
    args = parser.parse_args()

    out = args.out or args.detections.with_name(f"ocr_{args.engine}.csv")
    if out.exists() and not args.force:
        print(f"Wynik już jest: {out} (--force, aby policzyć ponownie)")
        return 0
    rows = read_detections(args.detections)
    print(f"OCR ({args.engine}) dla {rows.crop.notna().sum()} wycinków z {args.detections}")
    result = read_crops(rows, args.detections.parent, args.engine)
    result.to_csv(out, index=False)

    read = result[result.crop.notna()]
    print(f"{len(read)} wycinków, {(read.text.str.len() > 0).sum()} z niepustym odczytem")
    print(f"Wynik: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
