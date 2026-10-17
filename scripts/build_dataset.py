#!/usr/bin/env python3
"""Składa data1, data2 i data3 w jeden znormalizowany zbiór w data/combined/ (do wrzucenia na Kaggle).

Użycie:
    pip install -e ".[build]"                   # imagecorruptions==1.1.2, scikit-image
    python3 scripts/download_data.py            # najpierw pobierz zbiory źródłowe
    python3 scripts/build_dataset.py            # zbuduj data/combined/

Logika (zniekształcenia, wycinki, etykiety YOLO, exposure): alpr/build/combine.py.
Wynik: plates.csv, detection/ (zdjęcia, etykiety YOLO, data.yaml), ocr/<split>/ (wycinki).
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # działa też bez pip install -e .

from alpr.build.combine import BUILDERS, SOURCES
from alpr.config import COMBINED, DATA_DIR
from alpr.dataset import FIELDS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=COMBINED, help="katalog wynikowy (domyślnie data/combined)")
    args = parser.parse_args()
    out = args.out.resolve()

    missing = [d for d in SOURCES if not (DATA_DIR / d / "photos").is_dir()]
    if missing:
        print(f"Brak zdjęć dla: {', '.join(missing)}. Uruchom najpierw scripts/download_data.py")
        return 1
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "scripts" / "combined_template", out)

    rows = []
    for name in SOURCES:
        print(f"[{name}]")
        before = len(rows)
        BUILDERS[name](DATA_DIR / name, out, rows)
        print(f"  {len(rows) - before} wierszy (oryginał + zniekształcona kopia)")

    with open(out / "plates.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        key = (r["sample_type"], r["split"], r["artificial_corruption"] != "none")
        counts[key] = counts.get(key, 0) + 1
    print(f"Razem {len(rows)} tablic w {out} (oryginał / zniekształcona kopia):")
    for kind, split in sorted({k[:2] for k in counts}):
        print(f"  {kind} {split}: {counts.get((kind, split, False), 0)} / {counts.get((kind, split, True), 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
