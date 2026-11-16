#!/usr/bin/env python3
"""Detekcja i OCR na wszystkich wersjach zbioru eksperymentu oraz zestawienie z ground truth.

Użycie:
    python3 scripts/run_experiment.py --weights runs/detect/plates/weights/best.pt
    python3 scripts/run_experiment.py --weights best.pt --name yolo11s --force
    python3 scripts/run_experiment.py --weights best.pt --batch 8 --no-align --deblur --engine parseq \
        --ocr-weights runs/ocr/parseq_s0/best.pt runs/ocr/parseq_dec_s0/best.pt --name parseq_deblur

--deblur: kąt ruchu z widma całego zdjęcia (alpr.deblur), 16 widoków każdego wycinka (bez zmian, 5 dysków, 10 ruchów),
odczyt widoku po dekonwolucji tylko, gdy podnosi pewność o --margin. --ocr-weights: kilka wag silnika sieciowego
(crnn, lprnet, parseq), wygrywa pewniejszy odczyt. Najlepszy układ z eksperymentu: jak w przykładzie wyżej.

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
from alpr.deblur import estimate_blur_files
from alpr.detection import PlateAligner, PlateDetector
from alpr.detection.pipeline import DETECTION_FIELDS, crop_plates, read_detections
from alpr.experiment import images_by_source, match_results, summary
from alpr.ocr import engine_names, parse_params, read_crops, read_crops_deblurred


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="wagi YOLO (best.pt z train_yolo.py)")
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT, help="katalog z build_experiment.py")
    parser.add_argument("--engine", default="easyocr", choices=engine_names())
    parser.add_argument("--param", nargs="*", default=[], help="parametry silnika OCR, np. decoder=beamsearch")
    parser.add_argument("--conf", type=float, default=0.25, help="próg pewności detekcji")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch", type=int, default=1, help="zdjęcia na przebieg modelu; 21 = wszystkie wersje zdjęcia"
    )
    parser.add_argument("--no-align", action="store_true", help="wycinek z samej ramki (przy rozmyciu lepszy dla OCR)")
    parser.add_argument("--deblur", action="store_true", help="dekonwolucja z jądrem oszacowanym z widma zdjęcia")
    parser.add_argument("--ocr-weights", nargs="+", help="wagi silnika sieciowego; kilka: wygrywa pewniejszy odczyt")
    parser.add_argument("--margin", type=float, default=0.1, help="z --deblur: o ile pewniejszy musi być inny widok")
    parser.add_argument("--workers", type=int, default=8, help="z --deblur: procesy estymacji jądra")
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

    crops = {}
    if args.force or not dets_path.exists():
        images = images_by_source(gt)
        print(f"Detekcja na {len(images)} zdjęciach ({gt.variant.nunique()} wersji)")
        detector = PlateDetector(args.weights, args.conf, args.imgsz)
        aligner = None if args.no_align else PlateAligner()
        rows = crop_plates(detector, images, args.experiment.resolve(), out, aligner, args.batch, crops)
        write_csv(dets_path, sorted(rows, key=lambda r: r["image"]), DETECTION_FIELDS)
    if args.force or not ocr_path.exists():
        print(f"OCR ({args.engine})")
        params, dets = parse_params(args.param), read_detections(dets_path)
        if args.deblur or args.ocr_weights:
            images = sorted(dets.image.unique())
            estimates = estimate_blur_files([args.experiment / i for i in images], args.workers) if args.deblur else []
            angles = {i: e.motion_angle for i, e in zip(images, estimates, strict=False)}
            weights = [{"weights": w} for w in args.ocr_weights] if args.ocr_weights else [{}]
            configs = [{"engine": args.engine, **params, **w} for w in weights]
            ocr = read_crops_deblurred(dets, out, angles, configs, crops, args.margin, deblur=args.deblur)
        else:
            ocr = read_crops(dets, out, args.engine, crops, **params)
        ocr.to_csv(ocr_path, index=False)

    results = match_results(gt, read_detections(ocr_path))
    results.to_csv(out / "plates.csv", index=False)
    table = summary(results)
    table.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Wynik: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
