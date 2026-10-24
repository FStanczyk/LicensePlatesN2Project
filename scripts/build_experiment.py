#!/usr/bin/env python3
"""Buduje zbiór eksperymentu: czyste zdjęcia z etykietami (ground truth) i ich zniekształcone wersje.

Użycie:
    python3 scripts/build_experiment.py                         # val + test, 4 zniekształcenia x siła 1-5
    python3 scripts/build_experiment.py --splits test --corruptions darken brightness
    python3 scripts/build_experiment.py --force                 # zbuduj od nowa

Wymaga pip install -e ".[build]" (imagecorruptions). Bierze tylko oryginalne całe zdjęcia z czytelnym
tekstem (bez blurred, czyli bez UC3M-LP) z podziałów nieużytych do uczenia wag YOLO.

Wynik w --out (domyślnie data/experiment):
    images/clean/*.jpg                    kopie oryginałów
    images/<zniekształcenie>_<siła>/*.jpg  te same zdjęcia po zniekształceniu
    ground_truth.csv                      variant, artificial_corruption, artificial_severity, image, source_image,
                                          plate_id, split, text, country, exposure, x1, y1, x2, y2, polygon
"""

import argparse
import shutil
import sys
from pathlib import Path

from alpr.config import EXPERIMENT, default_dataset
from alpr.dataset import SPLITS
from alpr.experiment import CORRUPTIONS, build_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=default_dataset(), help="katalog zbioru")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=SPLITS)
    parser.add_argument("--corruptions", nargs="+", default=list(CORRUPTIONS))
    parser.add_argument("--out", type=Path, default=EXPERIMENT)
    parser.add_argument("--force", action="store_true", help="usuń istniejący zbiór i zbuduj od nowa")
    args = parser.parse_args()

    gt_path = args.out / "ground_truth.csv"
    if gt_path.exists() and not args.force:
        print(f"Zbiór już jest: {gt_path} (--force, aby zbudować od nowa)")
        return 0
    shutil.rmtree(args.out, ignore_errors=True)
    gt = build_experiment(args.dataset.resolve(), args.out, tuple(args.splits), tuple(args.corruptions))
    gt.to_csv(gt_path, index=False)

    clean = gt[gt.variant == "clean"]
    print(f"{clean.image.nunique()} zdjęć, {len(clean)} tablic, {gt.variant.nunique()} wersji, razem {len(gt)} wierszy")
    print(f"Wynik: {gt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
