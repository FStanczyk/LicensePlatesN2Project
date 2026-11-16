#!/usr/bin/env python3
"""Porównuje dwa przebiegi eksperymentu (plates.csv z run_experiment.py) na wspólnych wersjach i tablicach.

Użycie:
    python3 scripts/compare_results.py --a results/experiment/ref_poprawka/plates.csv \
        --b results/experiment/parseq_deblur/plates.csv
    python3 scripts/compare_results.py --a a.csv --b b.csv --ground-truth data/experiment/ground_truth.csv

--ground-truth: text_ok liczony od nowa według aktualnych etykiet (np. po poprawce SBE5754C), dla obu plików.
Wynik: odsetek bezbłędnych odczytów w a i b, różnica b - a i jej 95% przedział bootstrap (tablice losowane razem
ze wszystkimi swoimi wersjami), osobno dla czystych, zniekształconych i wszystkich wersji.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from alpr.benchmark import compare_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, required=True, help="plates.csv pierwszego przebiegu")
    parser.add_argument("--b", type=Path, required=True, help="plates.csv drugiego przebiegu")
    parser.add_argument("--ground-truth", type=Path, help="ground_truth.csv z aktualnymi etykietami")
    args = parser.parse_args()

    gt = pd.read_csv(args.ground_truth) if args.ground_truth else None
    table = compare_runs(pd.read_csv(args.a), pd.read_csv(args.b), gt)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
