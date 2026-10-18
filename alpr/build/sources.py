"""Zbiory źródłowe data1..3: pobieranie do data/<nazwa>/ i składanie w data/combined.

Duże archiwa nie są pobierane w całości: losowy podzbiór jest zapisany w data/<nazwa>/labels.csv
(w gicie), więc każdy dostaje te same zdjęcia. Sumy SHA-256 zdjęć: data/<nazwa>/SHA256SUMS.
"""

from __future__ import annotations

import csv
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from alpr.config import DATA_DIR, DOWNLOADS, SEED
from alpr.dataset import normalize_text, write_csv
from alpr.download import Archive, cached_md5, download, extract, verify_sums
from alpr.geometry import four_corners

if TYPE_CHECKING:
    from alpr.build.writer import CombinedWriter

EU = "at be bg cy cz de dk ee es fi fr gr hr hu ie it lt lu lv mt nl pl pt ro se si sk".split()


def split_for(rng: random.Random, val: float, test: float) -> str:
    r = rng.random()
    return "test" if r < test else "val" if r < test + val else "train"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class SourceDataset:
    name: str
    url: str
    page: str
    md5: str | None = None

    def __init__(self, root: Path = DATA_DIR):
        self.dir = root / self.name
        self.photos = self.dir / "photos"

    def has_photos(self) -> bool:
        return self.photos.is_dir() and any(self.photos.iterdir())

    def fetch(self, zip_path: Path | None = None) -> None:
        """Pobiera zdjęcia; zip_path: już pobrane archiwum."""
        self._unpack(self._local_archive(zip_path))

    def _local_archive(self, zip_path: Path | None) -> Path:
        path = zip_path or DOWNLOADS / f"{self.name}.zip"
        if not path.exists():
            download(self.url, path)
        if self.md5 and cached_md5(path) != self.md5:
            raise ValueError(f"archiwum {path} ma złą sumę MD5, usuń je i pobierz ponownie")
        return path

    def _unpack(self, zip_path: Path) -> None:
        raise NotImplementedError

    def verify(self) -> list[str] | None:
        """Zdjęcia niezgodne z SHA256SUMS albo None, gdy nie ma pliku z sumami."""
        sums = self.dir / "SHA256SUMS"
        return verify_sums(sums, self.photos) if sums.exists() else None

    def build(self, writer: CombinedWriter) -> None:
        raise NotImplementedError


class PolishPlates(SourceDataset):
    """data1: zdjęcia z Polski, adnotacje CVAT (prostokąt i numer), losowy podział 70/15/15."""

    name = "data1"
    url = "https://www.kaggle.com/api/v1/datasets/download/piotrstefaskiue/poland-vehicle-license-plate-dataset"
    page = "https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset"

    def _unpack(self, zip_path: Path) -> None:
        print(f"  rozpakowywanie {zip_path}")
        extract(zip_path, self.dir, {"photos/photos/": "photos/", "annotations.xml": "annotations.xml"})

    def build(self, writer: CombinedWriter) -> None:
        rng = random.Random(SEED)
        for image in ET.parse(self.dir / "annotations.xml").getroot().iter("image"):
            img = cv2.imread(str(self.photos / image.get("name")))
            plates = []
            for box in image.iter("box"):
                x1, y1, x2, y2 = (float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
                raw = box.findtext("attribute[@name='plate number']", "")
                plates.append(
                    {
                        "poly": np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
                        "text": normalize_text(raw),
                        "text_raw": raw,
                        "blurred": 0,
                        "country": "pl",
                        "lighting": "unknown",
                        "lines": 1,
                    }
                )
            key = f"{self.name}_{Path(image.get('name')).stem}"
            writer.add_photo(key, split_for(rng, 0.15, 0.15), img, plates)


class SampledSource(SourceDataset):
    """Źródło, z którego bierzemy podzbiór zapisany w labels.csv."""

    full_download = False

    def fetch(self, zip_path: Path | None = None) -> None:
        self._unpack(self._local_archive(zip_path) if zip_path or self.full_download else None)

    def select(self, archive: Archive) -> tuple[list[dict], dict[str, bytes]]:
        """(wiersze labels.csv, dodatkowe pliki {ścieżka w data/<nazwa>/: zawartość})."""
        raise NotImplementedError

    def _unpack(self, zip_path: Path | None) -> None:
        labels = self.dir / "labels.csv"
        archive = Archive(self.url, zip_path)
        if labels.exists():
            rows = read_csv(labels)
            print(f"  wybór z {labels}: {len(rows)} plików")
        else:
            rows, extra = self.select(archive)
            for rel, data in extra.items():
                (self.dir / rel).parent.mkdir(parents=True, exist_ok=True)
                (self.dir / rel).write_bytes(data)
            write_csv(labels, rows, list(rows[0]))

        todo = [r for r in rows if not (self.photos / r["file"]).exists()]
        print(f"  do pobrania {len(todo)} z {len(rows)} zdjęć")
        infos = [archive.zf.getinfo(r["member"]) for r in todo]
        for i, (row, data) in enumerate(zip(todo, archive.read_many(infos), strict=True), 1):
            out = self.photos / row["file"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            if i % 100 == 0 or i == len(todo):
                print(f"\r  pobrano {i}/{len(todo)}", end="", flush=True)
        if todo:
            print()


class Uc3mLp(SampledSource):
    """data2: UC3M-LP (Hiszpania), wszystkie nocne zdjęcia i losowe dzienne do ok. 1000 tablic.

    Zenodo ogranicza liczbę zapytań, więc archiwum (4.5 GB) pobieramy w całości.
    """

    name = "data2"
    url = "https://zenodo.org/api/records/17152029/files/UC3M-LP.zip/content"
    page = "https://zenodo.org/records/17152029"
    md5 = "496e3b5e66b78d231f71d85c0e5e11a3"
    full_download = True
    plates = 1000

    def select(self, archive: Archive) -> tuple[list[dict], dict[str, bytes]]:
        split_of = {
            stem: split
            for split in ("train", "test")
            for stem in archive.zf.read(f"UC3M-LP/{split}.txt").decode().split()
        }
        infos = [i for i in archive.zf.infolist() if i.filename.endswith(".json")]
        images = []
        for info, raw in zip(infos, archive.read_many(infos), strict=True):
            split, name = info.filename.split("/")[1:]
            stem = name.removesuffix(".json")
            lps = json.loads(raw)["lps"]
            images.append(
                {
                    "file": f"{stem}.jpg",
                    "member": f"UC3M-LP/{split}/{stem}.jpg",
                    "split": split_of[stem],
                    "plates": len(lps),
                    "night": sum(lp["lp_id"][1] == "N" for lp in lps),
                    "_ann": raw,
                }
            )
        chosen = [r for r in images if r["night"]]
        day = [r for r in images if not r["night"]]
        random.Random(SEED).shuffle(day)
        total = sum(r["plates"] for r in chosen)
        for r in day:
            if total >= self.plates:
                break
            chosen.append(r)
            total += r["plates"]
        night = sum(r["night"] for r in chosen)
        print(f"  {len(images)} zdjęć, wybrano {len(chosen)} ({total} tablic, {night} nocnych)")
        extra = {f"annotations/{r['file'][:-4]}.json": r.pop("_ann") for r in chosen}
        return sorted(chosen, key=lambda r: r["file"]), extra

    def build(self, writer: CombinedWriter) -> None:
        rng = random.Random(SEED)
        for photo in read_csv(self.dir / "labels.csv"):
            stem = photo["file"].removesuffix(".jpg")
            split = photo["split"] if photo["split"] == "test" else split_for(rng, 0.15, 0)
            img = cv2.imread(str(self.photos / photo["file"]))
            plates = []
            for lp in json.loads((self.dir / "annotations" / f"{stem}.json").read_text())["lps"]:
                code, raw = lp["lp_id"].split("-", 1)
                plates.append(
                    {
                        "poly": four_corners(np.array(lp["poly_coord"], dtype=float)),
                        "text": normalize_text(raw),
                        "text_raw": lp["lp_id"],
                        "blurred": int("*" in raw),
                        "country": "es",
                        "lighting": {"D": "day", "N": "night"}[code[1]],
                        "lines": 2 if code[0] == "B" else 1,
                    }
                )
            writer.add_photo(f"{self.name}_{stem}", split, img, plates)


class AutoRia(SampledSource):
    """data3: AUTO.RIA OCR EU (12 GB wyciętych tablic): wszystkie polskie, reszta po równo z krajów UE."""

    name = "data3"
    url = "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrEu-2023-06-26.zip"
    page = "https://github.com/ria-com/nomeroff-net#autoria-numberplate-ocr-datasets"
    size = 7000
    prefer = ("pl",)

    def select(self, archive: Archive) -> tuple[list[dict], dict[str, bytes]]:
        prefix = archive.zf.namelist()[0].split("/")[0]
        by_country: dict[str, list[dict]] = {}
        for info in archive.zf.infolist():
            parts = info.filename.split("/")
            if len(parts) != 4 or parts[2] != "ann" or not parts[3].endswith(".json"):
                continue
            ann = json.loads(archive.zf.read(info).decode("utf-8-sig"))
            stem = parts[3].removesuffix(".json")
            country = ann.get("country", "")
            if country in EU:
                by_country.setdefault(country, []).append(
                    {
                        "file": f"{parts[1]}/{stem}.png",
                        "member": f"{prefix}/{parts[1]}/img/{stem}.png",
                        "text": ann["description"],
                        "country": country,
                        "lines": ann.get("count_lines", ""),
                    }
                )
        chosen = [r for c in self.prefer for r in by_country.pop(c, [])]
        rng = random.Random(SEED)
        queues = [by_country[c] for c in sorted(by_country)]
        for q in queues:
            rng.shuffle(q)
        while len(chosen) < self.size and any(queues):
            for q in queues:
                if q and len(chosen) < self.size:
                    chosen.append(q.pop())
        counts = Counter(r["country"] for r in chosen)
        print(f"  wybrano {len(chosen)} tablic: " + ", ".join(f"{c} {n}" for c, n in sorted(counts.items())))
        return sorted(chosen, key=lambda r: r["file"]), {"LICENSE.txt": archive.zf.read(f"{prefix}/license.txt")}

    def build(self, writer: CombinedWriter) -> None:
        for i, plate in enumerate(read_csv(self.dir / "labels.csv")):
            img = cv2.imread(str(self.photos / plate["file"]))
            fields = {
                "text": normalize_text(plate["text"]),
                "text_raw": plate["text"],
                "blurred": 0,
                "country": plate["country"],
                "lighting": "unknown",
                "lines": plate["lines"] or 1,
            }
            writer.add_crop(f"{self.name}_{i:05d}", plate["file"].split("/")[0], img, fields)


SOURCES: dict[str, SourceDataset] = {s.name: s for s in (PolishPlates(), Uc3mLp(), AutoRia())}
