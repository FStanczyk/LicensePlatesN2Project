#!/usr/bin/env python3
"""Pobiera zbiory zdjęć (Kaggle, Zenodo, Nomeroff Net) i rozpakowuje je do data/<nazwa>/.

Zbiory z kluczem "sample" nie są pobierane w całości: skrypt czyta spis archiwum zdalnie
(HTTP Range), losuje podzbiór i ściąga tylko wybrane pliki. Wybór jest zapisany
w data/<nazwa>/labels.csv (w gicie), więc każdy dostaje te same zdjęcia. Przerwane
pobieranie wznawia się przez --force (pobrane już pliki są pomijane).

Użycie:
    python scripts/download_data.py                       # pobierz wszystko
    python scripts/download_data.py --zip ~/Downloads/archive.zip   # użyj już pobranego archiwum
    python scripts/download_data.py --force               # nadpisz istniejące zdjęcia

Po rozpakowaniu sumy SHA-256 zdjęć są sprawdzane względem data/<nazwa>/SHA256SUMS.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "_downloads"


def download(url: str, dest: Path, parts: int = 8, chunk: int = 32 << 20) -> None:
    """Pobiera plik; jeśli serwer obsługuje Range, to kawałkami w kilku połączeniach naraz
    (Zenodo ogranicza prędkość pojedynczego połączenia do ok. 1 MB/s). Gotowe kawałki zostają
    w <plik>.chunks/, więc po przerwaniu wystarczy uruchomić skrypt ponownie."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Pobieranie {url}")
    try:  # zapytanie o 1 bajt: 206 = serwer obsługuje Range (nagłówek Accept-Ranges bywa pomijany)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"})) as resp:
            ranges = resp.status == 206
            total = int(resp.headers.get("Content-Range", "/0").split("/")[-1])
    except (OSError, ValueError):
        total, ranges = 0, False

    if ranges and total > chunk:
        chunks = dest.with_suffix(".chunks")
        chunks.mkdir(exist_ok=True)
        starts = range(0, total, chunk)
        todo = [s for s in starts if not (chunks / f"{s:013d}").exists()]
        done = [total - sum(min(chunk, total - s) for s in todo)]

        def get(start: int) -> None:
            data = range_get(url, start, min(start + chunk, total) - 1)
            part = chunks / f"{start:013d}"
            part.with_suffix(".tmp").write_bytes(data)
            part.with_suffix(".tmp").rename(part)
            done[0] += len(data)
            print(f"\r  {done[0] / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)

        with ThreadPoolExecutor(parts) as pool:
            list(pool.map(get, todo))
        with open(dest.with_suffix(".part"), "wb") as out:
            for s in starts:
                out.write((chunks / f"{s:013d}").read_bytes())
        shutil.rmtree(chunks)
    else:
        with urllib.request.urlopen(url) as resp, open(dest.with_suffix(".part"), "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while data := resp.read(1 << 20):
                out.write(data)
                done += len(data)
                if total:
                    print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()
    dest.with_suffix(".part").rename(dest)


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


def range_get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except OSError as e:
            if attempt == 4:
                raise
            retry_after = getattr(e, "headers", None) and e.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
            time.sleep(min(wait, 120))


class RemoteFile(io.RawIOBase):
    """Plik zdalny do odczytu przez zipfile; każdy odczyt to jedno zapytanie HTTP Range."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
            self.size = int(resp.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = [offset, self.pos + offset, self.size + offset][whence]
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size:
            return 0
        data = range_get(self.url, self.pos, min(self.pos + len(buf), self.size) - 1)
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_member(url: str, info: zipfile.ZipInfo) -> bytes:
    """Pobiera jeden plik ze zdalnego zipa jednym zapytaniem (nagłówek lokalny + dane)."""
    start = info.header_offset
    data = range_get(url, start, start + 30 + 4 * len(info.filename) + 1024 + info.compress_size)
    name_len, extra_len = struct.unpack("<HH", data[26:30])
    begin = 30 + name_len + extra_len
    if begin + info.compress_size > len(data):
        data += range_get(url, start + len(data), start + begin + info.compress_size - 1)
    raw = data[begin : begin + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(raw, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"nieobsługiwana kompresja w {info.filename}")
    if zlib.crc32(raw) != info.CRC:
        raise ValueError(f"zła suma CRC dla {info.filename}")
    return raw


def select_uc3m(src: "Source", opts: dict) -> tuple[list[dict], dict[str, bytes]]:
    """UC3M-LP: wszystkie zdjęcia z nocnymi tablicami + losowe dzienne, aż będzie "plates" tablic."""
    split_of = {}
    for split in ("train", "test"):
        for stem in src.zf.read(f"UC3M-LP/{split}.txt").decode().split():
            split_of[stem] = split
    infos = [i for i in src.zf.infolist() if i.filename.endswith(".json")]
    images = []
    for info, raw in zip(infos, src.read_many(infos)):
        split, name = info.filename.split("/")[1:]
        stem = name[: -len(".json")]
        lps = json.loads(raw)["lps"]
        images.append({
            "file": f"{stem}.jpg",
            "member": f"UC3M-LP/{split}/{stem}.jpg",
            "split": split_of[stem],
            "plates": len(lps),
            "night": sum(lp["lp_id"][1] == "N" for lp in lps),
            "_ann": raw,
        })
    night = [r for r in images if r["night"]]
    day = [r for r in images if not r["night"]]
    random.Random(opts["seed"]).shuffle(day)
    chosen, total = list(night), sum(r["plates"] for r in night)
    for r in day:
        if total >= opts["plates"]:
            break
        chosen.append(r)
        total += r["plates"]
    print(f"  {len(images)} zdjęć w archiwum, wybrano {len(chosen)} ({total} tablic, "
          f"{sum(r['night'] for r in chosen)} nocnych)")
    extra = {f"annotations/{r['file'][:-4]}.json": r.pop("_ann") for r in chosen}
    return sorted(chosen, key=lambda r: r["file"]), extra


def select_autoria(src: "Source", opts: dict) -> tuple[list[dict], dict[str, bytes]]:
    """AUTO.RIA OCR: wszystkie tablice z "prefer", resztę po równo z krajów z "countries"."""
    prefix = src.zf.namelist()[0].split("/")[0]
    by_country = {}
    for info in src.zf.infolist():
        parts = info.filename.split("/")
        if len(parts) != 4 or parts[2] != "ann" or not parts[3].endswith(".json"):
            continue
        ann = json.loads(src.zf.read(info).decode("utf-8-sig"))
        stem = parts[3][: -len(".json")]
        country = ann.get("country", "")
        if country in opts["countries"]:
            by_country.setdefault(country, []).append({
                "file": f"{parts[1]}/{stem}.png",
                "member": f"{prefix}/{parts[1]}/img/{stem}.png",
                "text": ann["description"],
                "country": country,
                "lines": ann.get("count_lines", ""),
            })
    chosen = [r for c in opts["prefer"] for r in by_country.pop(c, [])]
    rng = random.Random(opts["seed"])
    queues = [by_country[c] for c in sorted(by_country)]
    for q in queues:
        rng.shuffle(q)
    # po jednej tablicy z każdego kraju na rundę, aż do "size" (małe kraje kończą się wcześniej)
    while len(chosen) < opts["size"] and any(queues):
        for q in queues:
            if q and len(chosen) < opts["size"]:
                chosen.append(q.pop())
    counts = {}
    for r in chosen:
        counts[r["country"]] = counts.get(r["country"], 0) + 1
    print(f"  wybrano {len(chosen)} tablic: " + ", ".join(f"{c} {n}" for c, n in sorted(counts.items())))
    return sorted(chosen, key=lambda r: r["file"]), {"LICENSE.txt": src.zf.read(f"{prefix}/license.txt")}


class Source:
    """Archiwum zip: lokalne albo zdalne (wtedy pliki pobierane są pojedynczo, równolegle)."""

    def __init__(self, url: str, zip_path: Path | None):
        self.url, self.local = url, zip_path is not None
        if self.local:
            self.zf = zipfile.ZipFile(zip_path)
        else:
            print(f"  czytanie spisu archiwum {url}")
            self.zf = zipfile.ZipFile(io.BufferedReader(RemoteFile(url), 1 << 20))

    def read_many(self, infos: list[zipfile.ZipInfo], on_done=None):
        if self.local:
            for info in infos:
                yield self.zf.read(info)
                if on_done:
                    on_done()
            return
        with ThreadPoolExecutor(8) as pool:
            for data in pool.map(lambda info: fetch_member(self.url, info), infos):
                yield data
                if on_done:
                    on_done()


def sample_zip(cfg: dict, target: Path, zip_path: Path | None) -> None:
    labels = target / "labels.csv"
    src = Source(cfg["url"], zip_path)
    if labels.exists():
        with open(labels, newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"  wybór z {labels.relative_to(ROOT)}: {len(rows)} plików")
    else:
        rows, extra = cfg["select"](src, cfg["sample"])
        for rel, data in extra.items():
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            (target / rel).write_bytes(data)
        with open(labels, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # pliki już pobrane pomijamy, więc przerwane pobieranie można wznowić
    todo = [r for r in rows if not (target / "photos" / r["file"]).exists()]
    print(f"  do pobrania {len(todo)} z {len(rows)} zdjęć")
    done = [0]

    def progress():
        done[0] += 1
        if done[0] % 100 == 0 or done[0] == len(todo):
            print(f"\r  pobrano {done[0]}/{len(todo)}", end="", flush=True)

    infos = [src.zf.getinfo(r["member"]) for r in todo]
    for row, data in zip(todo, src.read_many(infos, progress)):
        out = target / "photos" / row["file"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    if todo:
        print()


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


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


DATASETS = {
    "data1": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/"
        "piotrstefaskiue/poland-vehicle-license-plate-dataset",
        "page": "https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset",
        # ścieżka w archiwum -> ścieżka w data/<nazwa>/
        "members": {"photos/photos/": "photos/", "annotations.xml": "annotations.xml"},
    },
    "data2": {
        "url": "https://zenodo.org/api/records/17152029/files/UC3M-LP.zip/content",
        "page": "https://zenodo.org/records/17152029",
        # bierzemy wszystkie nocne zdjęcia + losowe dzienne do ~1000 tablic; Zenodo ogranicza
        # liczbę zapytań, więc archiwum (4.5 GB) pobieramy w całości i wybieramy z niego lokalnie
        "full_download": True,
        "md5": "496e3b5e66b78d231f71d85c0e5e11a3",
        "select": select_uc3m,
        "sample": {"plates": 1000, "seed": 0},
    },
    "data3": {
        "url": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrEu-2023-06-26.zip",
        "page": "https://github.com/ria-com/nomeroff-net#autoria-numberplate-ocr-datasets",
        # archiwum ma 12 GB i 74 tys. wyciętych tablic; bierzemy wszystkie polskie, a resztę
        # do "size" po równo z pozostałych krajów UE (tablice bez kraju i spoza UE pomijamy)
        "select": select_autoria,
        "sample": {
            "size": 7000,
            "seed": 0,
            "prefer": ["pl"],
            "countries": ["at", "be", "bg", "cy", "cz", "de", "dk", "ee", "es", "fi", "fr", "gr", "hr",
                          "hu", "ie", "it", "lt", "lu", "lv", "mt", "nl", "pl", "pt", "ro", "se", "si", "sk"],
        },
    },
}


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
                    print(f"  nie udało się pobrać ({e}). Pobierz ręcznie z {cfg['page']}\n"
                          f"  i uruchom: python scripts/download_data.py {name} --zip <ścieżka>")
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
