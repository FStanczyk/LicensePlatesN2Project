#!/usr/bin/env python3
"""Trenuje detektor tablic YOLO na data/combined/detection (fine-tuning wag z COCO, wybór epoki na val).

Z --pose trenuje YOLO-pose: ramka i 4 narożniki tablicy (zbiór eksportowany do data/pose, narożniki z data2),
więc detect_plates.py i run_experiment.py prostują wycinki narożnikami modelu zamiast PlateAligner.

Użycie:
    python3 scripts/train_yolo.py                                          # yolo11n, 100 epok, 640 px
    python3 scripts/train_yolo.py --model yolov8n.pt --epochs 50 --imgsz 960
    python3 scripts/train_yolo.py --epochs 1 --fraction 0.1 --device cpu --cache ram   # szybki test
    python3 scripts/train_yolo.py --project /kaggle/working/runs           # Kaggle/Colab
    python3 scripts/train_yolo.py --pose --imgsz 960                       # yolo11n-pose, runs/detect/plates_pose

Wynik: <project>/<name>/weights/best.pt oraz krzywe i metryki ultralytics.
"""

import argparse
import sys
from pathlib import Path

from alpr.config import POSE, RUNS, default_dataset
from alpr.detection.training import train_detector, train_pose_detector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=default_dataset() / "detection" / "data.yaml")
    parser.add_argument("--pose", action="store_true", help="YOLO-pose z 4 narożnikami tablicy")
    parser.add_argument("--dataset", type=Path, default=default_dataset(), help="z --pose: katalog zbioru")
    parser.add_argument("--pose-dir", type=Path, default=POSE, help="z --pose: gdzie zapisać zbiór pose")
    parser.add_argument("--model", help="wagi startowe; domyślnie yolo11n.pt, z --pose yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="tablice są małe, warto sprawdzić też 960")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="wczesne zatrzymanie po tylu epokach bez poprawy")
    parser.add_argument("--fraction", type=float, default=1.0, help="ułamek zbioru train")
    parser.add_argument("--device", default=None, help="np. 0, cpu; domyślnie GPU, jeśli jest")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cache", choices=["ram", "disk"], help="cache zdjęć (ram: szybsze epoki, gdy starczy pamięci)"
    )
    parser.add_argument("--project", type=Path, default=RUNS / "detect")
    parser.add_argument("--name", help="domyślnie plates, z --pose plates_pose")
    args = parser.parse_args()

    required = args.dataset / "plates.csv" if args.pose else args.data
    if not required.exists():
        print(f"Brak {required}. Pobierz zbiór: python3 scripts/get_data.py")
        return 1
    params = vars(args)
    pose, dataset, pose_dir, data = (params.pop(k) for k in ("pose", "dataset", "pose_dir", "data"))
    model = params.pop("model") or ("yolo11n-pose.pt" if pose else "yolo11n.pt")
    name = params.pop("name") or ("plates_pose" if pose else "plates")
    project = params.pop("project")
    if params["cache"] is None:
        del params["cache"]
    if pose:
        best = train_pose_detector(dataset.resolve(), model, project, name, pose_dir, **params)
    else:
        best = train_detector(data, model, project, name, **params)
    print(f"Najlepsze wagi: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
