#!/usr/bin/env python3
"""Ewaluacja detektora: P, R, F1, AP50, AP50-95 w podziale na źródło, oświetlenie, naświetlenie i zniekształcenia.

Użycie:
    python3 scripts/eval_detection.py --weights best.pt
    python3 scripts/eval_detection.py --weights best.pt --split val --name val_baseline
    python3 scripts/eval_detection.py --weights best.pt --images-root data/filtered/clahe --name clahe
    python3 scripts/eval_detection.py --preds results/detection_eval/baseline/predictions.csv   # bez modelu
    python3 scripts/eval_detection.py --classical --name classical   # klasyczny detektor bez uczenia

Predykcje liczone z progiem 0.001 (do AP), P/R/F1 przy --conf. Zdjęcie z tablicami o różnych
atrybutach trafia do grupy "mixed". Skrypt podaje też czas na zdjęcie i próg pewności maksymalizujący F1
na danym podziale (dobieraj go na val i podawaj jako --conf przy test).

Wynik w results/detection_eval/<name>/:
    predictions.csv   image, conf, x1, y1, x2, y2
    metrics.csv       group, value, images, plates, P, R, F1, AP50, AP50_95
    plates.csv        plate_id, image, detected (IoU >= 0.5 przy --conf), iou, conf
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from alpr.config import RESULTS, default_dataset
from alpr.dataset import SPLITS, load_plates
from alpr.detection import ClassicalDetector, PlateDetector
from alpr.detection.metrics import best_conf, evaluate_by_group
from alpr.detection.pipeline import predict_boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", help="wagi YOLO")
    source.add_argument("--preds", type=Path, help="gotowy predictions.csv")
    source.add_argument("--classical", action="store_true", help="klasyczny detektor (alpr.detection.classical)")
    parser.add_argument("--dataset", type=Path, default=default_dataset(), help="katalog zbioru")
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--images-root", type=Path, help="obrazy o tych samych ścieżkach co zbiór (np. po filtrze)")
    parser.add_argument("--conf", type=float, default=0.25, help="próg pewności dla P/R/F1")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    gt = load_plates(args.dataset, sample_type="full_photo", split=args.split)
    out = RESULTS / "detection_eval" / args.name
    out.mkdir(parents=True, exist_ok=True)
    if args.preds:
        preds = pd.read_csv(args.preds)
    else:
        images_root = (args.images_root or args.dataset).resolve()
        images = sorted(gt.image.unique())
        print(f"Predykcje dla {len(images)} zdjęć ({args.split}) z {images_root}")
        detector = ClassicalDetector() if args.classical else PlateDetector(args.weights, 0.001, args.imgsz)
        start = time.perf_counter()
        preds = predict_boxes(detector, images, images_root)
        print(f"Czas: {(time.perf_counter() - start) / len(images) * 1000:.0f} ms/zdjęcie (z wczytaniem)")
        preds.to_csv(out / "predictions.csv", index=False)

    table, plates = evaluate_by_group(gt, preds, args.conf)
    table.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    plates.to_csv(out / "plates.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    conf, f1 = best_conf(gt, preds)
    print(f"Próg pewności maksymalizujący F1 na {args.split}: {conf:.3f} (F1 {f1:.3f})")
    print(f"Wynik: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
