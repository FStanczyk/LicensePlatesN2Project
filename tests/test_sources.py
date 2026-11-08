"""Źródła data1..3: wybór podzbioru z archiwum i przekazanie tablic do writera (writer podmieniony)."""

import csv
import json
import random
import zipfile
from collections import Counter

import cv2
import numpy as np
import pytest

from alpr.build.sources import (
    EU,
    SOURCES,
    AutoRia,
    PolishPlates,
    SourceDataset,
    Uc3mLp,
    read_csv,
    split_for,
)
from alpr.download import Archive


class FakeWriter:
    def __init__(self):
        self.photos, self.crops = [], []

    def add_photo(self, key, split, img, plates):
        self.photos.append((key, split, img.shape, plates))

    def add_crop(self, key, split, img, plate):
        self.crops.append((key, split, img.shape, plate))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image(h=60, w=100) -> np.ndarray:
    return np.full((h, w, 3), 128, np.uint8)


def test_split_for_deterministyczny_i_proporcje():
    draws = [split_for(random.Random(0), 0.15, 0.15) for _ in range(3)]
    assert len(set(draws)) == 1
    rng = random.Random(0)
    counts = Counter(split_for(rng, 0.15, 0.15) for _ in range(20000))
    assert counts["test"] / 20000 == pytest.approx(0.15, abs=0.01)
    assert counts["val"] / 20000 == pytest.approx(0.15, abs=0.01)


def test_split_for_bez_testu():
    rng = random.Random(0)
    assert {split_for(rng, 0.15, 0) for _ in range(1000)} == {"train", "val"}


def test_read_csv(tmp_path):
    write_rows(tmp_path / "a.csv", [{"a": "1", "b": "ż"}])
    assert read_csv(tmp_path / "a.csv") == [{"a": "1", "b": "ż"}]


def test_sources_rejestr_i_katalogi(tmp_path):
    assert list(SOURCES) == ["data1", "data2", "data3"]
    source = Uc3mLp(tmp_path)
    assert source.dir == tmp_path / "data2" and source.photos == tmp_path / "data2" / "photos"
    assert not source.has_photos()
    source.photos.mkdir(parents=True)
    assert not source.has_photos()


def test_klasa_bazowa_wymaga_implementacji(tmp_path):
    class Pusty(SourceDataset):
        name = "x"

    with pytest.raises(NotImplementedError):
        Pusty(tmp_path).build(FakeWriter())
    with pytest.raises(NotImplementedError):
        Pusty(tmp_path)._unpack(tmp_path / "a.zip")


def test_polish_plates_build(tmp_path):
    d = tmp_path / "data1"
    (d / "photos").mkdir(parents=True)
    cv2.imwrite(str(d / "photos" / "a.jpg"), image())
    cv2.imwrite(str(d / "photos" / "b.jpg"), image(80, 120))
    (d / "annotations.xml").write_text(
        "<annotations>"
        '<image id="0" name="a.jpg"><box xtl="10" ytl="20" xbr="50.5" ybr="35">'
        '<attribute name="plate number">wa 123-45</attribute></box></image>'
        '<image id="1" name="b.jpg"><box xtl="1" ytl="2" xbr="3" ybr="4"></box>'
        '<box xtl="5" ytl="6" xbr="7" ybr="8"><attribute name="plate number">KR 1</attribute></box></image>'
        "</annotations>"
    )
    writer = FakeWriter()
    PolishPlates(tmp_path).build(writer)
    (key_a, split_a, shape_a, plates_a), (key_b, _, shape_b, plates_b) = writer.photos
    assert (key_a, key_b) == ("data1_a", "data1_b")
    assert split_a in ("train", "val", "test") and (shape_a, shape_b) == ((60, 100, 3), (80, 120, 3))
    plate = plates_a[0]
    np.testing.assert_array_equal(plate["poly"], [[10, 20], [50.5, 20], [50.5, 35], [10, 35]])
    assert {k: plate[k] for k in ("text", "text_raw", "blurred", "country", "lighting", "lines")} == {
        "text": "WA12345",
        "text_raw": "wa 123-45",
        "blurred": 0,
        "country": "pl",
        "lighting": "unknown",
        "lines": 1,
    }
    assert [p["text"] for p in plates_b] == ["", "KR1"]


def test_polish_plates_podzial_powtarzalny(tmp_path):
    d = tmp_path / "data1"
    (d / "photos").mkdir(parents=True)
    xml = ["<annotations>"]
    for i in range(40):
        cv2.imwrite(str(d / "photos" / f"{i}.jpg"), image(8, 8))
        xml.append(f'<image id="{i}" name="{i}.jpg"></image>')
    (d / "annotations.xml").write_text("".join([*xml, "</annotations>"]))
    first, second = FakeWriter(), FakeWriter()
    PolishPlates(tmp_path).build(first)
    PolishPlates(tmp_path).build(second)
    assert [p[1] for p in first.photos] == [p[1] for p in second.photos]
    assert len({p[1] for p in first.photos}) == 3


def uc3m_annotations(tmp_path, specs):
    d = tmp_path / "data2"
    (d / "photos").mkdir(parents=True)
    (d / "annotations").mkdir()
    rows = []
    for stem, split, lps in specs:
        cv2.imwrite(str(d / "photos" / f"{stem}.jpg"), image())
        ann = {"lps": [{"lp_id": lp_id, "poly_coord": poly} for lp_id, poly in lps]}
        (d / "annotations" / f"{stem}.json").write_text(json.dumps(ann))
        rows.append({"file": f"{stem}.jpg", "member": "", "split": split, "plates": len(lps), "night": 0})
    write_rows(d / "labels.csv", rows)


def test_uc3m_build_pola_tablicy(tmp_path):
    hexagon = [[20, 20], [80, 22], [100, 30], [98, 60], [40, 62], [18, 50]]
    uc3m_annotations(
        tmp_path,
        [
            ("p1", "test", [("AN-61*7*DX", [[1, 1], [9, 1], [9, 5], [1, 5]])]),
            ("p2", "train", [("BD-1234ABC", hexagon)]),
        ],
    )
    writer = FakeWriter()
    Uc3mLp(tmp_path).build(writer)
    (k1, s1, _, [lp1]), (k2, s2, _, [lp2]) = writer.photos
    assert (k1, s1) == ("data2_p1", "test") and k2 == "data2_p2" and s2 in ("train", "val")
    assert (lp1["text"], lp1["blurred"], lp1["lighting"], lp1["lines"]) == ("61*7*DX", 1, "night", 1)
    assert (lp2["text"], lp2["blurred"], lp2["lighting"], lp2["lines"]) == ("1234ABC", 0, "day", 2)
    assert lp1["text_raw"] == "AN-61*7*DX" and lp1["country"] == "es"
    assert lp2["poly"].shape == (4, 2)


def test_uc3m_test_zostaje_testem_a_train_nigdy_nie_trafia_do_testu(tmp_path):
    specs = [(f"t{i}", "test", []) for i in range(10)] + [(f"r{i}", "train", []) for i in range(40)]
    uc3m_annotations(tmp_path, specs)
    writer = FakeWriter()
    Uc3mLp(tmp_path).build(writer)
    split = {key: s for key, s, _, _ in writer.photos}
    assert all(split[f"data2_t{i}"] == "test" for i in range(10))
    assert {split[f"data2_r{i}"] for i in range(40)} == {"train", "val"}


def test_autoria_build(tmp_path):
    rows = []
    for i, (split, text, lines) in enumerate(
        [("train", "WA 123", "1"), ("val", "AB-12 CD", ""), ("test", "X", "2"), ("test", "Y", "0")]
    ):
        (tmp_path / "data3" / "photos" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path / "data3" / "photos" / split / f"f{i}.png"), image(20, 80))
        rows.append({"file": f"{split}/f{i}.png", "member": "", "text": text, "country": "de", "lines": lines})
    write_rows(tmp_path / "data3" / "labels.csv", rows)
    writer = FakeWriter()
    AutoRia(tmp_path).build(writer)
    assert [(k, s) for k, s, _, _ in writer.crops] == [
        ("data3_00000", "train"),
        ("data3_00001", "val"),
        ("data3_00002", "test"),
        ("data3_00003", "test"),
    ]
    assert [p["lines"] for *_, p in writer.crops] == [1, 1, 2, 1]
    assert writer.crops[1][3]["text"] == "AB12CD" and writer.crops[1][3]["text_raw"] == "AB-12 CD"


def make_uc3m_zip(path, images):
    with zipfile.ZipFile(path, "w") as zf:
        for split in ("train", "test"):
            zf.writestr(f"UC3M-LP/{split}.txt", "\n".join(s for s, sp, _ in images if sp == split))
        for stem, split, lps in images:
            zf.writestr(f"UC3M-LP/{split}/{stem}.json", json.dumps({"lps": [{"lp_id": c} for c in lps]}))
            zf.writestr(f"UC3M-LP/{split}/{stem}.jpg", f"jpg {stem}")
    return path


def test_uc3m_select_wszystkie_nocne_i_dzienne_do_limitu(tmp_path):
    images = [
        ("n1", "train", ["AN-1"]),
        ("n2", "test", ["AD-1", "AN-2"]),
        *[(f"d{i}", "train", ["AD-1", "AD-2"]) for i in range(10)],
    ]
    source = Uc3mLp(tmp_path)
    source.plates = 7
    rows, extra = source.select(Archive("", make_uc3m_zip(tmp_path / "u.zip", images)))
    files = [r["file"] for r in rows]
    assert {"n1.jpg", "n2.jpg"} <= set(files) and files == sorted(files)
    assert sum(r["plates"] for r in rows) == 7
    assert {r["file"]: r["split"] for r in rows}["n2.jpg"] == "test"
    assert set(extra) == {f"annotations/{f[:-4]}.json" for f in files}
    assert json.loads(extra["annotations/n2.json"])["lps"][1]["lp_id"] == "AN-2"
    assert all("_ann" not in r for r in rows)
    again, _ = source.select(Archive("", tmp_path / "u.zip"))
    assert again == rows


def make_autoria_zip(path, plates, prefix="autoria"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{prefix}/license.txt", "licencja")
        for i, (split, country) in enumerate(plates):
            ann = {"description": f"T{i}", "country": country, "count_lines": 1}
            zf.writestr(f"{prefix}/{split}/ann/{i}.json", "﻿" + json.dumps(ann))
            zf.writestr(f"{prefix}/{split}/img/{i}.png", f"png {i}")
        zf.writestr(f"{prefix}/{split}/ann/meta.txt", "pomijany")
    return path


def test_autoria_select_polskie_najpierw_reszta_po_rowno(tmp_path):
    plates = [("train", "pl")] * 3 + [("train", "de")] * 10 + [("val", "fr")] * 10 + [("test", "ua")] * 5
    source = AutoRia(tmp_path)
    source.size = 9
    rows, extra = source.select(Archive("", make_autoria_zip(tmp_path / "a.zip", plates)))
    counts = Counter(r["country"] for r in rows)
    assert counts == {"pl": 3, "de": 3, "fr": 3}
    assert all(r["country"] in EU for r in rows)
    assert extra == {"LICENSE.txt": b"licencja"}
    row = next(r for r in rows if r["country"] == "fr")
    assert row["file"].startswith("val/") and row["member"] == f"autoria/val/img/{row['file'].split('/')[1]}"


def test_autoria_select_mniej_tablic_niz_limit(tmp_path):
    plates = [("train", "de")] * 2 + [("val", "it")]
    rows, _ = AutoRia(tmp_path).select(Archive("", make_autoria_zip(tmp_path / "a.zip", plates)))
    assert len(rows) == 3


def test_sampled_fetch_z_lokalnego_archiwum(tmp_path, capsys):
    plates = [("train", "pl"), ("val", "de"), ("test", "cz")]
    zip_path = make_autoria_zip(tmp_path / "a.zip", plates)
    source = AutoRia(tmp_path)
    source.fetch(zip_path)
    labels = read_csv(tmp_path / "data3" / "labels.csv")
    assert len(labels) == 3
    assert (tmp_path / "data3" / "LICENSE.txt").read_text() == "licencja"
    assert (tmp_path / "data3" / "photos" / "val" / "1.png").read_text() == "png 1"
    assert source.has_photos()
    (tmp_path / "data3" / "photos" / "val" / "1.png").unlink()
    capsys.readouterr()
    source.fetch(zip_path)
    assert "do pobrania 1 z 3" in capsys.readouterr().out
    assert (tmp_path / "data3" / "photos" / "val" / "1.png").exists()


def test_sampled_fetch_uzywa_istniejacego_labels(tmp_path):
    zip_path = make_autoria_zip(tmp_path / "a.zip", [("train", "pl"), ("val", "de")])
    write_rows(
        tmp_path / "data3" / "labels.csv",
        [{"file": "val/1.png", "member": "autoria/val/img/1.png", "text": "T1", "country": "de", "lines": "1"}],
    )
    AutoRia(tmp_path).fetch(zip_path)
    files = [
        p.relative_to(tmp_path / "data3" / "photos").as_posix() for p in (tmp_path / "data3" / "photos").rglob("*.png")
    ]
    assert files == ["val/1.png"]


def test_fetch_zla_suma_md5(tmp_path):
    zip_path = make_uc3m_zip(tmp_path / "u.zip", [("n1", "train", ["AN-1"])])
    with pytest.raises(ValueError, match="MD5"):
        Uc3mLp(tmp_path).fetch(zip_path)


def test_polish_plates_fetch_rozpakowuje(tmp_path):
    with zipfile.ZipFile(tmp_path / "p.zip", "w") as zf:
        zf.writestr("photos/photos/a.jpg", "a")
        zf.writestr("annotations.xml", "<annotations/>")
        zf.writestr("README", "x")
    PolishPlates(tmp_path).fetch(tmp_path / "p.zip")
    assert (tmp_path / "data1" / "photos" / "a.jpg").read_text() == "a"
    assert (tmp_path / "data1" / "annotations.xml").exists()
    assert not (tmp_path / "data1" / "README").exists()


def test_polish_plates_to_samo_auto_w_jednym_podziale(tmp_path):
    d = tmp_path / "data1"
    (d / "photos").mkdir(parents=True)
    xml = ["<annotations>"]
    for i in range(60):
        cv2.imwrite(str(d / "photos" / f"{i}.jpg"), image(8, 8))
        box = f'<box xtl="1" ytl="1" xbr="5" ybr="4"><attribute name="plate number">WA {i // 3}</attribute></box>'
        xml.append(f'<image id="{i}" name="{i}.jpg">{box}</image>')
    (d / "annotations.xml").write_text("".join([*xml, "</annotations>"]))
    writer = FakeWriter()
    PolishPlates(tmp_path).build(writer)
    split_of_car = {}
    for _, split, _, plates in writer.photos:
        assert split_of_car.setdefault(plates[0]["text"], split) == split
    assert len(set(split_of_car.values())) > 1


def test_polish_plates_unikalne_auta_bez_zmian_w_losowaniu(tmp_path):
    d = tmp_path / "data1"
    (d / "photos").mkdir(parents=True)
    xml = ["<annotations>"]
    for i in range(30):
        cv2.imwrite(str(d / "photos" / f"{i}.jpg"), image(8, 8))
        box = f'<box xtl="1" ytl="1" xbr="5" ybr="4"><attribute name="plate number">WA {i}</attribute></box>'
        xml.append(f'<image id="{i}" name="{i}.jpg">{box}</image>')
    (d / "annotations.xml").write_text("".join([*xml, "</annotations>"]))
    writer = FakeWriter()
    PolishPlates(tmp_path).build(writer)
    rng = random.Random(0)
    assert [split for _, split, _, _ in writer.photos] == [split_for(rng, 0.15, 0.15) for _ in range(30)]
