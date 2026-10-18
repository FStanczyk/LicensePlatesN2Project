#!/usr/bin/env python3
"""Pobiera zbiory źródłowe (Kaggle, Zenodo, Nomeroff Net) do data/<nazwa>/; tylko do przebudowy zbioru.

Użycie:
    python3 scripts/download_data.py                                  # wszystkie
    python3 scripts/download_data.py data1 --zip ~/Downloads/archive.zip   # już pobrane archiwum
    python3 scripts/download_data.py --force                          # dociągnij, nawet gdy zdjęcia już są

Z dużych archiwów pobierany jest tylko podzbiór z data/<nazwa>/labels.csv; przerwane pobieranie
wystarczy uruchomić ponownie. Na końcu zdjęcia są sprawdzane z data/<nazwa>/SHA256SUMS.
"""

import argparse
import sys
from pathlib import Path

from alpr.build.sources import SOURCES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", default=list(SOURCES), help=f"zbiory: {', '.join(SOURCES)}")
    parser.add_argument("--zip", type=Path, help="lokalne archiwum (tylko dla jednego zbioru)")
    parser.add_argument("--force", action="store_true", help="pobierz, nawet jeśli zdjęcia już są")
    args = parser.parse_args()
    if unknown := set(args.datasets) - set(SOURCES):
        parser.error(f"nieznane zbiory: {', '.join(sorted(unknown))}")
    if args.zip and len(args.datasets) != 1:
        parser.error("--zip wymaga podania dokładnie jednego zbioru")

    ok = True
    for name in args.datasets:
        source = SOURCES[name]
        print(f"[{name}] -> {source.dir}")
        if source.has_photos() and not args.force:
            print("  zdjęcia już są (--force, aby dociągnąć)")
        else:
            try:
                source.fetch(args.zip.expanduser() if args.zip else None)
            except (OSError, ValueError) as e:
                print(f"  BŁĄD: {e}\n  pobierz ręcznie z {source.page} i użyj --zip")
                ok = False
                continue
        bad = source.verify()
        if bad is None:
            print("  brak SHA256SUMS, pomijam weryfikację")
        elif bad:
            print(f"  BŁĄD: {len(bad)} plików brakuje lub ma złą sumę, np. {bad[:5]}")
            ok = False
        else:
            print("  OK: zdjęcia zgodne z SHA256SUMS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
