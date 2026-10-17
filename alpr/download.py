"""Pobieranie: HTTP z wznawianiem, zdalne archiwa zip (HTTP Range), zbiór z Kaggle."""

import hashlib
import io
import os
import shutil
import struct
import time
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from alpr.config import DOWNLOADS, KAGGLE_DATASET

WORKERS = 8
CHUNK = 32 << 20


def range_get(url: str, start: int, end: int, retries: int = 5) -> bytes:
    """Bajty start..end włącznie, z ponawianiem i obsługą Retry-After."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except OSError as e:
            if attempt == retries - 1:
                raise
            retry_after = getattr(e, "headers", None) and e.headers.get("Retry-After")
            time.sleep(min(int(retry_after) if retry_after and retry_after.isdigit() else 2**attempt, 120))


def _progress(done: int, total: int) -> None:
    print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)


def download(url: str, dest: Path) -> None:
    """Pobiera plik równoległymi kawałkami (gdy serwer obsługuje Range); kawałki przeżywają przerwanie."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Pobieranie {url}")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"})) as resp:
            total = int(resp.headers.get("Content-Range", "/0").split("/")[-1]) if resp.status == 206 else 0
    except (OSError, ValueError):
        total = 0
    part = dest.with_suffix(".part")
    if total > CHUNK:
        _download_chunks(url, part, total)
    else:
        with urllib.request.urlopen(url) as resp, open(part, "wb") as out:
            size, done = int(resp.headers.get("Content-Length", 0)), 0
            while data := resp.read(1 << 20):
                out.write(data)
                done += len(data)
                if size:
                    _progress(done, size)
    print()
    part.rename(dest)


def _download_chunks(url: str, out: Path, total: int) -> None:
    chunks = out.with_suffix(".chunks")
    chunks.mkdir(exist_ok=True)
    starts = range(0, total, CHUNK)
    todo = [s for s in starts if not (chunks / f"{s:013d}").exists()]
    done = total - sum(min(CHUNK, total - s) for s in todo)

    def fetch(start: int) -> int:
        data = range_get(url, start, min(start + CHUNK, total) - 1)
        tmp = chunks / f"{start:013d}.tmp"
        tmp.write_bytes(data)
        tmp.rename(chunks / f"{start:013d}")
        return len(data)

    with ThreadPoolExecutor(WORKERS) as pool:
        for size in pool.map(fetch, todo):
            done += size
            _progress(done, total)
    with open(out, "wb") as f:
        for s in starts:
            with open(chunks / f"{s:013d}", "rb") as chunk:
                shutil.copyfileobj(chunk, f)
    shutil.rmtree(chunks)


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def cached_md5(path: Path) -> str:
    """MD5 liczone raz i zapisywane obok pliku (<plik>.md5)."""
    cache = path.with_name(path.name + ".md5")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return cache.read_text().strip()
    digest = file_hash(path, "md5")
    cache.write_text(digest)
    return digest


def verify_sums(sums_file: Path, folder: Path) -> list[str]:
    """Pliki z sums_file (format sha256sum), których brakuje albo mają złą sumę; liczone równolegle."""
    entries = [line.split(maxsplit=1) for line in sums_file.read_text().splitlines()]

    def bad(entry: list[str]) -> bool:
        path = folder / entry[1]
        return not path.exists() or file_hash(path) != entry[0]

    with ThreadPoolExecutor(WORKERS) as pool:
        return [name for (_, name), wrong in zip(entries, pool.map(bad, entries), strict=True) if wrong]


def _target(name: str, members: dict[str, str]) -> str | None:
    for src, dst in members.items():
        if name == src:
            return dst
        if (not src or src.endswith("/")) and name.startswith(src):
            return dst + name[len(src) :]
    return None


def extract(zip_path: Path, target: Path, members: dict[str, str]) -> None:
    """Rozpakowuje równolegle pliki pasujące do members: {plik albo prefiks/ (""=wszystko): ścieżka docelowa}."""
    with zipfile.ZipFile(zip_path) as zf:
        jobs = [(info, _target(info.filename, members)) for info in zf.infolist() if not info.is_dir()]
    jobs = [(info, target / rel) for info, rel in jobs if rel is not None]
    for parent in {out.parent for _, out in jobs}:
        parent.mkdir(parents=True, exist_ok=True)

    def work(part: list[tuple[zipfile.ZipInfo, Path]]) -> None:
        with zipfile.ZipFile(zip_path) as zf:
            for info, out in part:
                with zf.open(info) as f, open(out, "wb") as o:
                    shutil.copyfileobj(f, o, 1 << 20)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(work, [jobs[i::WORKERS] for i in range(WORKERS)]))


class RemoteFile(io.RawIOBase):
    """Plik zdalny tylko do odczytu, każdy odczyt to jedno zapytanie Range."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
            self.size = int(resp.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = [offset, self.pos + offset, self.size + offset][whence]
        return self.pos

    def readinto(self, buf) -> int:
        if self.pos >= self.size:
            return 0
        data = range_get(self.url, self.pos, min(self.pos + len(buf), self.size) - 1)
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_member(url: str, info: zipfile.ZipInfo) -> bytes:
    """Jeden plik ze zdalnego zipa: nagłówek lokalny i dane w jednym zapytaniu."""
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


class Archive:
    """Archiwum zip lokalne albo zdalne (pliki zdalne pobierane pojedynczo, równolegle)."""

    def __init__(self, url: str, path: Path | None = None):
        self.url, self.local = url, path is not None
        if path is None:
            print(f"  czytanie spisu archiwum {url}")
        self.zf = zipfile.ZipFile(path if path else io.BufferedReader(RemoteFile(url), 1 << 20))

    def read_many(self, infos: list[zipfile.ZipInfo], on_done: Callable[[], None] | None = None) -> Iterator[bytes]:
        with ThreadPoolExecutor(WORKERS) as pool:
            results = map(self.zf.read, infos) if self.local else pool.map(lambda i: fetch_member(self.url, i), infos)
            for data in results:
                yield data
                if on_done:
                    on_done()


def fetch_kaggle(dest: Path) -> None:
    """Pobiera zbiór z Kaggle (z wznawianiem) i podmienia dest dopiero po udanym rozpakowaniu."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    zip_path = DOWNLOADS / f"{KAGGLE_DATASET.split('/')[1]}.zip"
    print(f"Pobieranie {KAGGLE_DATASET} (ok. 4.5 GB) do {zip_path}")
    api.dataset_download_files(KAGGLE_DATASET, path=str(DOWNLOADS), quiet=False)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Rozpakowywanie do {dest}")
    extract(zip_path, tmp, {"ocr/ocr/": "ocr/", "": ""})
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    zip_path.unlink()


def missing_files(dest: Path) -> tuple[pd.DataFrame, list[str]]:
    """plates.csv i ścieżki z niego, których nie ma w dest."""
    df = pd.read_csv(dest / "plates.csv", usecols=["sample_type", "image", "crop"])
    expected = set(df.crop) | set(df.image.dropna())
    present = {
        os.path.relpath(os.path.join(root, name), dest).replace(os.sep, "/")
        for root, _, names in os.walk(dest)
        for name in names
    }
    return df, sorted(expected - present)
