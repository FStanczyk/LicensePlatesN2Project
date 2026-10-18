#!/usr/bin/env python3
"""Trenuje detektor tablic YOLO na data/combined/detection (fine-tuning wag z COCO, wybór epoki na val).

Użycie:
    python3 scripts/train_yolo.py                                          # yolo11n, 100 epok, 640 px
    python3 scripts/train_yolo.py --model yolov8n.pt --epochs 50 --imgsz 960
    python3 scripts/train_yolo.py --epochs 1 --fraction 0.1 --device cpu   # szybki test
    python3 scripts/train_yolo.py --project /kaggle/working/runs           # Kaggle/Colab

Wynik: <project>/<name>/weights/best.pt oraz krzywe i metryki ultralytics.
"""

import argparse
import sys
from pathlib import Path

from alpr.config import RUNS, default_dataset
from alpr.detection.training import train_detector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=default_dataset() / "detection" / "data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="wagi startowe (yolo11n.pt, yolo11s.pt, yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="tablice są małe, warto sprawdzić też 960")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="wczesne zatrzymanie po tylu epokach bez poprawy")
    parser.add_argument("--fraction", type=float, default=1.0, help="ułamek zbioru train")
    parser.add_argument("--device", default=None, help="np. 0, cpu; domyślnie GPU, jeśli jest")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=RUNS / "detect")
    parser.add_argument("--name", default="plates")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Brak {args.data}. Pobierz zbiór: python3 scripts/get_data.py")
        return 1
    params = vars(args)
    best = train_detector(params.pop("data"), params.pop("model"), params.pop("project"), params.pop("name"), **params)
    print(f"Najlepsze wagi: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
