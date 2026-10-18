#!/usr/bin/env python3
"""Składa data1..3 w jeden zbiór w data/combined/ (do wrzucenia na Kaggle).

Użycie:
    pip install -e ".[build]"
    python3 scripts/download_data.py
    python3 scripts/build_dataset.py

Wynik: plates.csv, detection/ (zdjęcia, etykiety YOLO, data.yaml), ocr/<split>/ (wycinki).
"""

import argparse
import sys
from pathlib import Path

from alpr.build.sources import SOURCES
from alpr.build.writer import CombinedWriter
from alpr.config import COMBINED


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=COMBINED, help="katalog wynikowy")
    args = parser.parse_args()

    missing = [name for name, source in SOURCES.items() if not source.photos.is_dir()]
    if missing:
        print(f"Brak zdjęć dla: {', '.join(missing)}. Uruchom najpierw scripts/download_data.py")
        return 1
    writer = CombinedWriter.create(args.out.resolve())
    for name, source in SOURCES.items():
        before = len(writer.rows)
        source.build(writer)
        print(f"[{name}] {len(writer.rows) - before} wierszy (oryginał + kopia)")
    path = writer.save()

    counts = writer.summary()
    print(f"Razem {len(writer.rows)} tablic w {path} (oryginał / kopia):")
    for kind, split in sorted({k[:2] for k in counts}):
        print(f"  {kind} {split}: {counts[kind, split, False]} / {counts[kind, split, True]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
