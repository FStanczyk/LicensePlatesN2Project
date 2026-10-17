"""Pobieranie zbiorów źródłowych (Kaggle, Zenodo, Nomeroff Net) do data/<nazwa>/.

Zbiory z kluczem "sample" nie są pobierane w całości: czytamy spis archiwum zdalnie (HTTP Range),
losujemy podzbiór i ściągamy tylko wybrane pliki. Wybór jest zapisany w data/<nazwa>/labels.csv
(w gicie), więc każdy dostaje te same zdjęcia. Sumy SHA-256 zdjęć: data/<nazwa>/SHA256SUMS.
"""

import csv
import hashlib
import io
import shutil
import struct
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from alpr.build.sources import select_autoria, select_uc3m
from alpr.config import DATA_DIR, ROOT

DOWNLOADS_DIR = DATA_DIR / "_downloads"

DATASETS = {
    "data1": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/piotrstefaskiue/poland-vehicle-license-plate-dataset",
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
            "countries": [
                "at",
                "be",
                "bg",
                "cy",
                "cz",
                "de",
                "dk",
                "ee",
                "es",
                "fi",
                "fr",
                "gr",
                "hr",
                "hu",
                "ie",
                "it",
                "lt",
                "lu",
                "lv",
                "mt",
                "nl",
                "pl",
                "pt",
                "ro",
                "se",
                "si",
                "sk",
            ],
        },
    },
}


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
    """Rozpakowuje wybrane pliki i katalogi (klucz kończący się "/") z archiwum do target."""
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            for src, dst in members.items():
                if info.filename == src or (src.endswith("/") and info.filename.startswith(src)):
                    rel = dst + info.filename[len(src) :] if src.endswith("/") else dst
                    out = target / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as f, open(out, "wb") as o:
                        shutil.copyfileobj(f, o)
                    break


def range_get(url: str, start: int, end: int) -> bytes:
    """Pobiera bajty start..end (włącznie); ponawia do 5 razy, z uwzględnieniem Retry-After."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except OSError as e:
            if attempt == 4:
                raise
            retry_after = getattr(e, "headers", None) and e.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
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
    """Pobiera podzbiór archiwum: wybór z labels.csv (albo losuje go i zapisuje), potem brakujące zdjęcia."""
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
    for row, data in zip(todo, src.read_many(infos, progress), strict=True):
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
    """Sprawdza zdjęcia w target/photos względem target/SHA256SUMS."""
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
