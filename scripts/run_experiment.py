#!/usr/bin/env python3
"""Detekcja i OCR na wszystkich wersjach zbioru eksperymentu oraz zestawienie z ground truth.

Użycie:
    python3 scripts/run_experiment.py --weights runs/detect/plates/weights/best.pt
    python3 scripts/run_experiment.py --weights best.pt --name yolo11s --force

Wejście: data/experiment (scripts/build_experiment.py). Kroki z gotowym wynikiem są pomijane bez --force.

Wynik w results/experiment/<name>/:
    detections.csv     jak w detect_plates.py, crops/ to wyprostowane wycinki
    ocr_<silnik>.csv   detections.csv z odczytem (text, conf_ocr)
    plates.csv         jedna linia na (wersja, tablica): text_true, detected (IoU >= 0.5), iou, det_conf, align,
                       crop, text_pred, conf_ocr, text_ok (odczyt bezbłędny)
    summary.csv        odsetek wykrytych i bezbłędnie odczytanych tablic dla każdej wersji
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from alpr.config import EXPERIMENT, RESULTS
from alpr.dataset import write_csv
from alpr.detection import PlateAligner, PlateDetector
from alpr.detection.pipeline import DETECTION_FIELDS, crop_plates, read_detections
from alpr.experiment import match_results, summary
from alpr.ocr import ENGINES, read_crops


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="wagi YOLO (best.pt z train_yolo.py)")
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT, help="katalog z build_experiment.py")
    parser.add_argument("--engine", default="easyocr", choices=sorted(ENGINES))
    parser.add_argument("--conf", type=float, default=0.25, help="próg pewności detekcji")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--force", action="store_true", help="policz wszystko od nowa")
    args = parser.parse_args()

    gt_path = args.experiment / "ground_truth.csv"
    if not gt_path.exists():
        print(f"Brak {gt_path}. Zbuduj zbiór: python3 scripts/build_experiment.py")
        return 1
    gt = pd.read_csv(gt_path)
    out = RESULTS / "experiment" / args.name
    dets_path, ocr_path = out / "detections.csv", out / f"ocr_{args.engine}.csv"

    if args.force or not dets_path.exists():
        images = sorted(gt.image.unique())
        print(f"Detekcja na {len(images)} zdjęciach ({gt.variant.nunique()} wersji)")
        detector = PlateDetector(args.weights, args.conf, args.imgsz)
        rows = crop_plates(detector, images, args.experiment.resolve(), out, PlateAligner())
        write_csv(dets_path, rows, DETECTION_FIELDS)
    if args.force or not ocr_path.exists():
        print(f"OCR ({args.engine})")
        read_crops(read_detections(dets_path), out, args.engine).to_csv(ocr_path, index=False)

    results = match_results(gt, read_detections(ocr_path))
    results.to_csv(out / "plates.csv", index=False)
    table = summary(results)
    table.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Wynik: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
