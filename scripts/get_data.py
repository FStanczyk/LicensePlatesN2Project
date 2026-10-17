#!/usr/bin/env python3
"""Pobiera gotowy zbiór z Kaggle do data/combined/ (wystarczy raz).

Użycie:
    python3 scripts/get_data.py            # pobierz, jeśli jeszcze nie ma (przerwane pobieranie się wznawia)
    python3 scripts/get_data.py --force    # pobierz ponownie (np. nowa wersja na Kaggle)

Wymaga tokenu API Kaggle w ~/.kaggle/ (kaggle.com -> Settings -> API). Dostęp do zbioru nadaje Stańczyk.
Stary zbiór jest podmieniany dopiero po udanym pobraniu i rozpakowaniu.
"""

import argparse
import sys
from pathlib import Path

from alpr.config import COMBINED
from alpr.download import fetch_kaggle, missing_files


def check(dest: Path) -> int:
    df, missing = missing_files(dest)
    if missing:
        print(f"BŁĄD: brakuje {len(missing)} plików, np. {missing[:3]}")
        return 1
    print(f"OK: {len(df)} tablic ({(df.sample_type == 'full_photo').sum()} na całych zdjęciach), wszystkie pliki są")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=COMBINED, help="katalog docelowy")
    parser.add_argument("--force", action="store_true", help="pobierz ponownie")
    args = parser.parse_args()
    dest = args.dest.expanduser().resolve()

    if (dest / "plates.csv").exists() and not args.force:
        print(f"Zbiór już jest w {dest} (--force, aby pobrać ponownie)")
        return check(dest)
    if dest.exists() and any(dest.iterdir()) and not (dest / "plates.csv").exists():
        print(f"BŁĄD: {dest} nie jest pusty i nie zawiera zbioru, wybierz inny --dest")
        return 1
    fetch_kaggle(dest)
    return check(dest)


if __name__ == "__main__":
    sys.exit(main())
