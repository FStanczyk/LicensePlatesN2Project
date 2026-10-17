#!/usr/bin/env python3
"""Pobiera gotowy, znormalizowany zbiór z Kaggle do data/combined/ (wystarczy raz).

Użycie:
    pip install -e .
    python3 scripts/get_data.py            # pobierz, jeśli jeszcze nie ma
    python3 scripts/get_data.py --force    # pobierz ponownie (np. po nowej wersji na Kaggle)

Wymaga tokenu API Kaggle: kaggle.com -> Settings -> API -> Create New Token, plik zapisz
w ~/.kaggle/ (kaggle.json albo access_token). Zbiór jest prywatny, dostęp nadaje Stańczyk.
Opis zawartości: README.md w data/combined/ i sekcja "Dane" w README repozytorium.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # działa też bez pip install -e .

from alpr.config import COMBINED, KAGGLE_DATASET


def check(dest: Path) -> bool:
    """Sprawdza, czy każdy plik wskazany w plates.csv istnieje."""
    with open(dest / "plates.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    paths = {r["crop"] for r in rows} | {r["image"] for r in rows if r["image"]}
    missing = [p for p in paths if not (dest / p).exists()]
    if missing:
        print(f"BŁĄD: brakuje {len(missing)} z {len(paths)} plików, np. {missing[:3]}")
        return False
    photos = sum(r["sample_type"] == "full_photo" for r in rows)
    print(f"OK: {len(rows)} tablic ({photos} na całych zdjęciach), wszystkie {len(paths)} plików są na miejscu")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=COMBINED, help=f"katalog docelowy (domyślnie {COMBINED.relative_to(ROOT)})")
    parser.add_argument("--force", action="store_true", help="usuń i pobierz ponownie")
    args = parser.parse_args()
    dest = args.dest.expanduser().resolve()

    if (dest / "plates.csv").exists() and not args.force:
        print(f"Zbiór już jest w {dest} (użyj --force, aby pobrać ponownie)")
        return 0 if check(dest) else 1

    from kaggle.api.kaggle_api_extended import KaggleApi  # import uwierzytelnia, więc dopiero tutaj

    if (dest / "plates.csv").exists():  # usuwamy tylko katalog, w którym naprawdę jest zbiór
        shutil.rmtree(dest)
    elif dest.exists() and any(dest.iterdir()):
        print(f"BŁĄD: {dest} nie jest pusty i nie zawiera zbioru, wybierz inny --dest")
        return 1
    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Pobieranie {KAGGLE_DATASET} do {dest} (ok. 4.5 GB)")
    api.dataset_download_files(KAGGLE_DATASET, path=str(dest), unzip=True, quiet=False)
    # na Kaggle wycinki leżą w ocr/ocr/<split>/ (tak rozpakował się ocr.zip), a plates.csv
    # wskazuje ocr/<split>/, więc podnosimy je o poziom wyżej
    nested = dest / "ocr" / "ocr"
    if nested.is_dir():
        for split in nested.iterdir():
            split.rename(dest / "ocr" / split.name)
        nested.rmdir()
    return 0 if check(dest) else 1


if __name__ == "__main__":
    sys.exit(main())
