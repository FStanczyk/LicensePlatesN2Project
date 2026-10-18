"""Powtarzalność składania zbioru: hash wyniku na syntetycznych źródłach, zapisany z kodu sprzed refaktoryzacji."""

import csv
import hashlib
import json

import cv2
import numpy as np
import pytest

pytest.importorskip("imagecorruptions", reason='wymaga extras [build]: pip install -e ".[build]"')

from alpr.build.sources import AutoRia, PolishPlates, Uc3mLp  # noqa: E402
from alpr.build.writer import CombinedWriter  # noqa: E402

GOLDEN_ROWS = 24
GOLDEN = "1f4b17edebdcaa74"


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_sources(root):
    rng = np.random.default_rng(1)
    d1 = root / "data1"
    (d1 / "photos").mkdir(parents=True)
    xml = ["<annotations>"]
    for i, (name, (h, w)) in enumerate({"a.jpg": (300, 400), "b.jpg": (1200, 2400), "c.jpg": (200, 300)}.items()):
        cv2.imwrite(str(d1 / "photos" / name), rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        boxes = [(0.2, 0.3, 0.5, 0.45, f"wa 12{i}45")] + ([(0.6, 0.6, 0.9, 0.7, f"KR-99{i}")] if i == 1 else [])
        xml.append(f'<image id="{i}" name="{name}">')
        for x1, y1, x2, y2, text in boxes:
            xml.append(
                f'<box label="plate" xtl="{w * x1}" ytl="{h * y1}" xbr="{w * x2}" ybr="{h * y2}">'
                f'<attribute name="plate number">{text}</attribute></box>'
            )
        xml.append("</image>")
    (d1 / "annotations.xml").write_text("\n".join([*xml, "</annotations>"]))

    d2 = root / "data2"
    (d2 / "photos").mkdir(parents=True)
    (d2 / "annotations").mkdir()
    specs = [
        ("p1", "train", [("AN-61*7*DX", [[50, 60], [150, 65], [148, 95], [52, 90]])]),
        ("p2", "test", [("BD-1234ABC", [[20, 20], [80, 22], [100, 30], [98, 60], [40, 62], [18, 50]])]),
        (
            "p3",
            "train",
            [
                ("AD-5555XYZ", [[100, 100], [200, 100], [200, 140], [100, 140]]),
                ("AD-1111AAA", [[10, 150], [90, 150], [90, 180], [10, 180]]),
            ],
        ),
        ("p4", "train", [("AN-2222BBB", [[30, 30], [120, 35], [120, 70], [30, 65]])]),
    ]
    rows = []
    for stem, split, lps in specs:
        cv2.imwrite(str(d2 / "photos" / f"{stem}.jpg"), rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
        ann = {"lps": [{"lp_id": lp_id, "poly_coord": poly} for lp_id, poly in lps]}
        (d2 / "annotations" / f"{stem}.json").write_text(json.dumps(ann))
        rows.append({"file": f"{stem}.jpg", "member": "", "split": split, "plates": len(lps), "night": 0})
    write_rows(d2 / "labels.csv", rows)

    d3 = root / "data3"
    rows = []
    for i, (split, text, country, lines) in enumerate(
        [("train", "WA 123", "pl", "1"), ("val", "AB-12 CD", "de", ""), ("test", "XY777", "fr", "2")]
    ):
        (d3 / "photos" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d3 / "photos" / split / f"f{i}.png"), rng.integers(0, 256, (40, 160, 4), dtype=np.uint8))
        rows.append({"file": f"{split}/f{i}.png", "member": "", "text": text, "country": country, "lines": lines})
    write_rows(d3 / "labels.csv", rows)


def test_zbior_identyczny(tmp_path):
    make_sources(tmp_path / "src")
    writer = CombinedWriter(tmp_path / "out")
    for source in (PolishPlates, Uc3mLp, AutoRia):
        source(tmp_path / "src").build(writer)
    writer.save()
    digest = hashlib.sha256()
    for path in sorted((tmp_path / "out").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(tmp_path / "out").as_posix().encode() + path.read_bytes())
    assert (len(writer.rows), digest.hexdigest()[:16]) == (GOLDEN_ROWS, GOLDEN)


def test_writer_create_i_summary(tmp_path):
    writer = CombinedWriter.create(tmp_path / "out")
    assert (tmp_path / "out" / "detection" / "data.yaml").exists()
    writer.add_crop("k1", "test", np.full((40, 80, 3), 128, np.uint8), {"text": "A"})
    assert writer.summary() == {("plate_crop", "test", False): 1, ("plate_crop", "test", True): 1}
