#!/usr/bin/env python3
"""Porównuje czas i jakość ustawień potoku (detekcja + wyrównanie + OCR) na zbiorze eksperymentu.

Użycie:
    python3 scripts/benchmark.py --grid detection --weights n640=runs/detect/plates/weights/best.pt
    python3 scripts/benchmark.py --grid detection --weights n640=best.pt s960=s960.pt --variants clean motion_blur_1
    python3 scripts/benchmark.py --grid ocr_sweep --weights n640=best.pt --experiment data/experiment_tune
    python3 scripts/benchmark.py --grid ocr_sweep --weights n640=best.pt --base mag_ratio=1.5 --name sweep2
    python3 scripts/benchmark.py --grid engines --weights n640=best.pt     # EasyOCR, Tesseract, CRNN, LPRNet
    python3 scripts/benchmark.py --grid ocr_sweep --engine crnn --weights n640=best.pt \
        --base weights=runs/ocr/crnn_s0/best.pt --space "weights=[runs/ocr/crnn_s1/best.pt]"

Siatka detection: dla każdych wag kilka wsadów (OCR domyślny); pierwsza konfiguracja jest odniesieniem dla kolumn
same_boxes, max_box_diff, max_conf_diff. Siatki ocr_sweep i engines: detekcja raz (pierwsze wagi, wsad 1), potem
silniki i parametry na tych samych wycinkach; ocr_sweep to odniesienie (--base) i po jednej zmianie parametru
(alpr.benchmark.OCR_SWEEP albo --space). Stroić na zbiorze z train i val (data/experiment_tune), nie na test.

Wynik w results/benchmark/<name>/timings.csv: konfiguracja, czasy w s, zdjęcia/s, pamięć GPU w MB, detected /
text_ok / car średnio po wersjach i na clean, 95% przedział bootstrap dla text_ok po tablicach.
--plates: także plates_<konfiguracja>.csv dla każdej konfiguracji (duże pliki).
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from alpr.benchmark import (
    OCR_SWEEP,
    as_detections,
    compare_detections,
    detect_stage,
    detection_grid,
    engines_grid,
    file_name,
    run_pipeline,
    sweep,
)
from alpr.config import EXPERIMENT, RESULTS
from alpr.experiment import images_by_source
from alpr.ocr import engine_names, parse_params

COLUMNS = ["name", "detect_s", "ocr_s", "img_per_s", "detected_all", "text_ok_all", "car_all", "text_ok_clean"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", choices=["detection", "ocr_sweep", "engines"], required=True)
    parser.add_argument("--base", nargs="*", default=[], help="z ocr_sweep: parametry odniesienia, np. mag_ratio=1.5")
    parser.add_argument("--space", nargs="*", help="z ocr_sweep: zamiast OCR_SWEEP, np. low_text=[0.45,0.55]")
    parser.add_argument("--engine", default="easyocr", choices=engine_names(), help="z ocr_sweep: silnik OCR")
    parser.add_argument("--weights", nargs="+", required=True, help="nazwa=ścieżka; imgsz 960, gdy nazwa zawiera 960")
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT, help="katalog z build_experiment.py")
    parser.add_argument("--variants", nargs="+", help="tylko te wersje (domyślnie wszystkie)")
    parser.add_argument("--only", nargs="+", help="tylko konfiguracje o tych nazwach")
    parser.add_argument("--plates", action="store_true", help="zapisz też plates.csv każdej konfiguracji")
    parser.add_argument("--name", help="domyślnie nazwa siatki")
    args = parser.parse_args()

    gt_path = args.experiment / "ground_truth.csv"
    if not gt_path.exists():
        print(f"Brak {gt_path}. Zbuduj zbiór: python3 scripts/build_experiment.py")
        return 1
    gt = pd.read_csv(gt_path)
    if args.variants:
        gt = gt[gt.variant.isin(args.variants)]
    weights = dict(w.split("=", 1) for w in args.weights)
    first = next(iter(weights.values()))
    if args.grid == "detection":
        configs = detection_grid(weights)
    elif args.grid == "engines":
        configs = engines_grid(first)
    else:
        space = parse_params(args.space) if args.space else OCR_SWEEP
        configs = sweep(first, space, parse_params(args.base), args.engine)
    if args.only:
        configs = [c for c in configs if c.name in args.only]
    out = RESULTS / "benchmark" / (args.name or args.grid)
    base = args.experiment.resolve()
    print(f"{len(configs)} konfiguracji, {gt.image.nunique()} zdjęć, {gt.variant.nunique()} wersji")

    dets = crops = None
    if args.grid != "detection":
        rows, crops, seconds = detect_stage(configs[0], images_by_source(gt), base, out / "work")
        dets = as_detections(rows)
        print(f"detekcja: {seconds:.1f} s, {len(crops)} wycinków")

    table, ref = [], None
    for cfg in configs:
        work = out / "work"
        row, results, ocr = run_pipeline(cfg, gt, base, work, dets, crops)
        if args.grid == "detection":
            ref = ocr if ref is None else ref
            row |= compare_detections(ref, ocr)
            shutil.rmtree(work, ignore_errors=True)
        if args.plates:
            results.to_csv(out / f"plates_{file_name(cfg.name)}.csv", index=False)
        table.append(row)
        print(pd.DataFrame([row])[COLUMNS].to_string(index=False, header=len(table) == 1, float_format="%.3f"))
        pd.DataFrame(table).to_csv(out / "timings.csv", index=False, float_format="%.4f")
    shutil.rmtree(out / "work", ignore_errors=True)
    print(f"Wynik: {out / 'timings.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
