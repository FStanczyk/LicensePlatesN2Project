#!/usr/bin/env python3
"""Trenuje detektor tablic YOLO na data/combined/detection (klasa 0 = tablica).

Użycie:
    python3 scripts/train_yolo.py                                  # yolo11n, 100 epok, 640 px
    python3 scripts/train_yolo.py --model yolov8n.pt --epochs 50 --imgsz 960
    python3 scripts/train_yolo.py --epochs 1 --fraction 0.1 --device cpu   # szybki test lokalnie
    # na Kaggle/Colab (GPU), dataset podpięty jako input:
    python3 scripts/train_yolo.py --data /kaggle/input/eu-license-plates-alpr/detection/data.yaml \\
        --project /kaggle/working/runs

Startujemy z wag pre-trenowanych na COCO (fine-tuning). Trenujemy na train (oryginały i zniekształcone
kopie), wybór najlepszej epoki na val; test zostaje do eval_detection.py.
Wynik: <project>/<name>/weights/best.pt oraz krzywe i metryki ultralytics w tym samym katalogu.

data.yaml w zbiorze ma "path: .", a ultralytics rozwiązuje względną ścieżkę względem swojego
katalogu datasets, nie względem pliku yaml. Dlatego zapisujemy kopię yaml z bezwzględną ścieżką.
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "combined" / "detection" / "data.yaml"
SEED = 0


def resolved_yaml(data: Path, out_dir: Path) -> Path:
    """Kopia data.yaml z bezwzględnym "path" (katalog pliku yaml), zapisana w out_dir."""
    cfg = yaml.safe_load(data.read_text())
    cfg["path"] = str(data.parent.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "data_resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA, help="plik data.yaml zbioru detekcji")
    parser.add_argument("--model", default="yolo11n.pt", help="wagi startowe (yolo11n.pt, yolo11s.pt, yolov8n.pt, ...)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="rozmiar wejścia; tablice są małe, warto sprawdzić też 960")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="wczesne zatrzymanie po tylu epokach bez poprawy")
    parser.add_argument("--fraction", type=float, default=1.0, help="ułamek zbioru train (do szybkich testów)")
    parser.add_argument("--device", default=None, help="np. 0, cpu; domyślnie GPU, jeśli jest")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "detect")
    parser.add_argument("--name", default="plates")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Brak {args.data}. Pobierz zbiór: python3 scripts/get_data.py")
        return 1
    from ultralytics import YOLO

    data = resolved_yaml(args.data, args.project)
    model = YOLO(args.model)
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
        fraction=args.fraction, device=args.device, workers=args.workers,
        project=str(args.project.resolve()), name=args.name, exist_ok=True,
        seed=SEED, deterministic=True,
    )
    best = args.project.resolve() / args.name / "weights" / "best.pt"
    print(f"Najlepsze wagi: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
