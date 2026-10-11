#!/usr/bin/env python3
"""Pobiera zbiory zdjęć z Kaggle i rozpakowuje je do data/<nazwa>/.

Użycie:
    python scripts/download_data.py                       # pobierz wszystko
    python scripts/download_data.py --zip ~/Downloads/archive.zip   # użyj już pobranego archiwum
    python scripts/download_data.py --force               # nadpisz istniejące zdjęcia

Po rozpakowaniu sumy SHA-256 zdjęć są sprawdzane względem data/<nazwa>/SHA256SUMS.
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "_downloads"

DATASETS = {
    "data1": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/"
        "piotrstefaskiue/poland-vehicle-license-plate-dataset",
        "page": "https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset",
        # ścieżka w archiwum -> ścieżka w data/<nazwa>/
        "members": {"photos/photos/": "photos/", "annotations.xml": "annotations.xml"},
    },
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Pobieranie {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def extract(zip_path: Path, members: dict[str, str], target: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            for src, dst in members.items():
                if info.filename == src or (src.endswith("/") and info.filename.startswith(src)):
                    rel = dst + info.filename[len(src):] if src.endswith("/") else dst
                    out = target / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as f, open(out, "wb") as o:
                        shutil.copyfileobj(f, o)
                    break


def verify(target: Path) -> bool:
    sums_file = target / "SHA256SUMS"
    if not sums_file.exists():
        print(f"  brak {sums_file.relative_to(ROOT)}, pomijam weryfikację")
        return True
    bad = []
    for line in sums_file.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = target / "photos" / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            bad.append(name)
    if bad:
        print(f"  BŁĄD: {len(bad)} plików brakuje lub ma złą sumę, np. {bad[:5]}")
        return False
    print(f"  OK: {len(sums_file.read_text().splitlines())} plików zgodnych z SHA256SUMS")
    return True


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
        else:
            zip_path = args.zip.expanduser() if args.zip else DOWNLOADS_DIR / f"{name}.zip"
            if not zip_path.exists():
                try:
                    download(cfg["url"], zip_path)
                except OSError as e:
                    print(f"  nie udało się pobrać ({e}). Pobierz ręcznie z {cfg['page']}\n"
                          f"  i uruchom: python scripts/download_data.py {name} --zip <ścieżka>")
                    ok = False
                    continue
            print(f"  rozpakowywanie {zip_path}")
            extract(zip_path, cfg["members"], target)
        ok &= verify(target)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
