#!/usr/bin/env python3
"""Pobiera wytrenowane wagi (YOLO, CRNN, PARSeq) z wydania GitHub do runs/modele/ i sprawdza sumy SHA256.

Użycie:
    python3 scripts/get_models.py                              # wydanie z alpr.config.MODELS_TAG
    python3 scripts/get_models.py --tag modele-2026-11-16      # starsze wydanie (Włodarczyk, bez CRNN)

Pobiera tylko brakujące albo zmienione pliki. Użycie wag: README, sekcja "Gotowe modele".
"""

import argparse
import sys
from pathlib import Path

from alpr.config import MODELS, MODELS_TAG, MODELS_URL
from alpr.download import fetch_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=MODELS_TAG, help="tag wydania na GitHub")
    parser.add_argument("--dest", type=Path, default=MODELS, help="katalog docelowy")
    args = parser.parse_args()

    dest = args.dest / args.tag
    bad = fetch_release(f"{MODELS_URL}/{args.tag}", dest)
    if bad:
        print(f"BŁĄD: zła suma SHA256: {bad}")
        return 1
    print(f"OK: wagi w {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
