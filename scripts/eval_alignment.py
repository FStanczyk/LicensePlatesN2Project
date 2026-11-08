#!/usr/bin/env python3
"""Ocena wyrównania: zwykły wycinek ramki (box) a PlateAligner (aligned) względem narożników z plates.csv.

Użycie:
    python3 scripts/eval_alignment.py                                         # ramki z etykiet
    python3 scripts/eval_alignment.py --preds results/detection_eval/baseline/predictions.csv   # z detektora
    python3 scripts/eval_alignment.py --split test --examples 30
    python3 scripts/eval_alignment.py --preds results/detection_eval/pose/predictions.csv --name pose   # YOLO-pose

Predykcje YOLO-pose (kolumna polygon w predictions.csv) dają narożniki modelu zamiast PlateAligner (method = pose).

Metryki: IoU z prawdziwym czworokątem i błąd narożnika w wysokościach tablicy. Narożniki z perspektywą
ma tylko data2; w data1 są prostokąty ramek, więc te tablice są pomijane (alpr.dataset.with_corners).

Wynik w results/alignment_eval/<name>/: plates.csv, summary.csv, examples/ (ramka, wyrównanie, prawda).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from alpr.config import RESULTS, default_dataset
from alpr.dataset import SPLITS, load_plates
from alpr.detection import PlateAligner
from alpr.detection.metrics import boxes_by_image, summarize_alignment
from alpr.detection.pipeline import evaluate_alignment
from alpr.detection.pose import corners_by_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=default_dataset(), help="katalog zbioru")
    parser.add_argument("--split", default="all", choices=["all", *SPLITS])
    parser.add_argument("--preds", type=Path, help="predictions.csv z eval_detection.py; bez tego ramki z etykiet")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--examples", type=int, default=0, help="ile przykładowych obrazków zapisać")
    parser.add_argument("--name")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    df = load_plates(dataset, sample_type="full_photo", split=None if args.split == "all" else args.split)
    preds = pd.read_csv(args.preds) if args.preds else None
    pred_boxes = boxes_by_image(preds, args.conf) if preds is not None else None
    pred_corners = corners_by_image(preds, args.conf) if preds is not None else None
    out = RESULTS / "alignment_eval" / (args.name or ("detector" if args.preds else "gt_boxes"))
    examples = out / "examples" if args.examples else None
    (examples or out).mkdir(parents=True, exist_ok=True)

    res = evaluate_alignment(df, dataset, PlateAligner(), pred_boxes, examples, args.examples, pred_corners)
    if res.empty:
        print("Brak tablic do oceny")
        return 1
    res.to_csv(out / "plates.csv", index=False, float_format="%.4f")
    summary = summarize_alignment(res)
    summary.reset_index().to_csv(out / "summary.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200):
        print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print("better = odsetek tablic, na których wyrównanie poprawia IoU o ponad 0.01")
    print(f"Wynik: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
