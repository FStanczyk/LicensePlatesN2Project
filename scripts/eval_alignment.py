#!/usr/bin/env python3
"""Ocena wyrównania perspektywy: jak dobrze align_plate odtwarza 4 narożniki tablicy z samej ramki.

Użycie:
    python3 scripts/eval_alignment.py                          # ramki z etykiet (sama ocena wyrównania)
    python3 scripts/eval_alignment.py --preds results/detection_eval/baseline/predictions.csv
                                                               # ramki z detektora (realny przypadek)
    python3 scripts/eval_alignment.py --split test --examples 30

Porównujemy dwie metody na każdej tablicy z narożnikami w plates.csv:
    box       zwykły wycinek prostokątnej ramki (bez prostowania),
    aligned   alpr.detection.align_plate (szukanie czworokąta, fallback na ramkę).
Metryki: IoU wielokąta z prawdziwym czworokątem i średni błąd narożnika w wysokościach tablicy.
Uwaga: w data1 etykiety są prostokątami, więc tylko data2 (UC3M-LP) ma prawdziwe narożniki
z perspektywą; wyniki podajemy osobno dla każdego źródła.

Wynik w results/alignment_eval/<name>/:
    plates.csv     plate_id, source, ..., iou_box, iou_aligned, err_box, err_aligned, method
    summary.csv    średnie w grupach
    examples/      (z --examples) od góry: ramka, align_plate, prawdziwe narożniki
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # działa też bez pip install -e .

from alpr.config import RESULTS, default_dataset
from alpr.dataset import SPLITS, load_plates, parse_polygon
from alpr.detection import align_plate
from alpr.detection.metrics import BOX, best_matching_box, corner_error
from alpr.geometry import box_corners, polygon_iou, warp


def example(img: np.ndarray, box: np.ndarray, aligned: np.ndarray, gt: np.ndarray, height: int = 64) -> np.ndarray:
    tiles = [warp(img, box_corners(box)), aligned, warp(img, gt)]
    tiles = [cv2.resize(t, (int(height * 4.5), height)) for t in tiles]
    return np.vstack(tiles)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=default_dataset(), help="katalog zbioru (z plates.csv)")
    parser.add_argument("--split", default="all", choices=["all", *SPLITS])
    parser.add_argument("--preds", type=Path, help="predictions.csv z eval_detection.py; bez tego ramki z etykiet")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--examples", type=int, default=0, help="ile przykładowych obrazków zapisać")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    df = load_plates(dataset, sample_type="full_photo", split=None if args.split == "all" else args.split)
    df = df[df.polygon.notna()]
    pred_boxes = None
    if args.preds:
        preds = pd.read_csv(args.preds)
        preds = preds[preds.conf >= args.conf]
        pred_boxes = {image: g[BOX].to_numpy(float) for image, g in preds.groupby("image")}
    name = args.name or ("detector" if pred_boxes is not None else "gt_boxes")
    out = RESULTS / "alignment_eval" / name
    out.mkdir(parents=True, exist_ok=True)
    if args.examples:
        (out / "examples").mkdir(exist_ok=True)

    rows, saved = [], 0
    for image, g in df.groupby("image"):
        img = cv2.imread(str(dataset / image))
        if img is None:
            print(f"  BŁĄD: nie da się wczytać {image}")
            continue
        for _, r in g.iterrows():
            gt = parse_polygon(r.polygon)
            box = np.array([r.x1, r.y1, r.x2, r.y2], dtype=np.float32)
            if pred_boxes is not None:
                box = best_matching_box(pred_boxes.get(image, np.zeros((0, 4))), box)
                if box is None:
                    continue  # tablica niewykryta: to błąd detekcji, liczony w eval_detection.py
            aligned, poly, method = align_plate(img, box)
            rows.append({
                "plate_id": r.plate_id, "source": r.source, "lighting": r.lighting,
                "exposure": r.exposure, "corrupted": r.corrupted,
                "iou_box": polygon_iou(box_corners(box), gt), "iou_aligned": polygon_iou(poly, gt),
                "err_box": corner_error(box_corners(box), gt), "err_aligned": corner_error(poly, gt),
                "method": method,
            })
            if saved < args.examples:
                cv2.imwrite(str(out / "examples" / f"{r.plate_id}.png"), example(img, box, aligned, gt))
                saved += 1
    if not rows:
        print("Brak tablic do oceny")
        return 1

    res = pd.DataFrame(rows)
    res.to_csv(out / "plates.csv", index=False, float_format="%.4f")
    metrics = ["iou_box", "iou_aligned", "err_box", "err_aligned"]
    parts = [res.assign(group="all", value="all")]
    for col in ["source", "lighting", "exposure", "corrupted", "method"]:
        parts.append(res.assign(group=col, value=res[col]))
    long = pd.concat(parts)
    summary = long.groupby(["group", "value"], sort=False)[metrics].mean()
    summary.insert(0, "plates", long.groupby(["group", "value"], sort=False).size())
    summary["better"] = long.assign(b=long.iou_aligned > long.iou_box + 0.01).groupby(["group", "value"], sort=False).b.mean()
    summary.reset_index().to_csv(out / "summary.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200):
        print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print("better = odsetek tablic, na których align_plate poprawia IoU o więcej niż 0.01")
    print(f"Wynik: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
