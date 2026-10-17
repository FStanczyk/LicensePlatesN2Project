#!/usr/bin/env python3
"""Pobiera zbiory zdjęć (Kaggle, Zenodo, Nomeroff Net) i rozpakowuje je do data/<nazwa>/.

Zbiory z kluczem "sample" nie są pobierane w całości: skrypt czyta spis archiwum zdalnie
(HTTP Range), losuje podzbiór i ściąga tylko wybrane pliki. Wybór jest zapisany
w data/<nazwa>/labels.csv (w gicie), więc każdy dostaje te same zdjęcia. Przerwane
pobieranie wznawia się przez --force (pobrane już pliki są pomijane).

Użycie:
    python3 scripts/download_data.py                       # pobierz wszystko
    python3 scripts/download_data.py --zip ~/Downloads/archive.zip   # użyj już pobranego archiwum
    python3 scripts/download_data.py --force               # nadpisz istniejące zdjęcia

Po rozpakowaniu sumy SHA-256 zdjęć są sprawdzane względem data/<nazwa>/SHA256SUMS.
Konfiguracja zbiorów i logika pobierania: alpr/build/download.py.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # działa też bez pip install -e .

from alpr.build.download import DATASETS, DOWNLOADS_DIR, download, extract, md5sum, sample_zip, verify
from alpr.config import DATA_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", default=list(DATASETS), help="nazwy zbiorów (domyślnie wszystkie)")
    parser.add_argument("--zip", type=Path, help="lokalne archiwum zamiast pobierania (tylko dla jednego zbioru)")
    parser.add_argument("--force", action="store_true", help="rozpakuj ponownie, nawet jeśli zdjęcia już są")
    args = parser.parse_args()

    if args.zip and len(args.datasets) != 1:
        parser.error("--zip wymaga podania dokładnie jednego zbioru")

    ok = True
    for name in args.datasets:
        if name not in DATASETS:
            parser.error(f"nieznany zbiór: {name} (dostępne: {', '.join(DATASETS)})")
        cfg = DATASETS[name]
        target = DATA_DIR / name
        print(f"[{name}] -> {target.relative_to(ROOT)}")

        if (target / "photos").exists() and any((target / "photos").iterdir()) and not args.force:
            print("  zdjęcia już są (użyj --force, aby nadpisać)")
        elif "sample" in cfg and not cfg.get("full_download"):
            sample_zip(cfg, target, args.zip.expanduser() if args.zip else None)
        else:
            zip_path = args.zip.expanduser() if args.zip else DOWNLOADS_DIR / f"{name}.zip"
            if not zip_path.exists():
                try:
                    download(cfg["url"], zip_path)
                except OSError as e:
                    print(
                        f"  nie udało się pobrać ({e}). Pobierz ręcznie z {cfg['page']}\n"
                        f"  i uruchom: python3 scripts/download_data.py {name} --zip <ścieżka>"
                    )
                    ok = False
                    continue
            if "md5" in cfg and md5sum(zip_path) != cfg["md5"]:
                print(f"  BŁĄD: archiwum {zip_path} ma złą sumę MD5, usuń je i pobierz ponownie")
                ok = False
                continue
            if "sample" in cfg:
                sample_zip(cfg, target, zip_path)
            else:
                print(f"  rozpakowywanie {zip_path}")
                extract(zip_path, cfg["members"], target)
        ok &= verify(target)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
