import numpy as np
import pandas as pd
import pytest

from alpr.dataset import (
    FIELDS,
    crop_path,
    exposure,
    format_polygon,
    list_images,
    load_plates,
    normalize_text,
    parse_polygon,
    write_csv,
)


@pytest.mark.parametrize(("raw", "text"), [("wa 123-45", "WA12345"), ("61*7*dx", "61*7*DX"), ("ÄB 1", "B1"), ("", "")])
def test_normalize_text(raw, text):
    assert normalize_text(raw) == text


def test_polygon_w_obie_strony():
    poly = np.array([[1.4, 2.6], [10, 2], [10, 8], [1, 8]])
    text = format_polygon(poly)
    assert text == "1 3 10 2 10 8 1 8"
    assert parse_polygon(text).shape == (4, 2)


def test_exposure():
    img = np.random.default_rng(0).integers(0, 256, (20, 60, 3), dtype=np.uint8)
    assert (exposure(img), exposure(img // 4), exposure(255 - img // 4)) == ("normal", "under", "over")


@pytest.fixture
def dataset(tmp_path):
    rows = []
    for i, (kind, split, corr, blurred) in enumerate(
        [
            ("full_photo", "train", "none", 0),
            ("full_photo", "test", "none", 1),
            ("full_photo", "test", "fog", 1),
            ("plate_crop", "test", "none", 0),
        ]
    ):
        rows.append(
            {f: "" for f in FIELDS}
            | {
                "plate_id": f"data{1 + i % 3}_{i}",
                "sample_type": kind,
                "split": split,
                "artificial_corruption": corr,
                "blurred": blurred,
            }
        )
    pd.DataFrame(rows, columns=FIELDS).to_csv(tmp_path / "plates.csv", index=False)
    return tmp_path


def test_load_plates_filtry(dataset):
    assert len(load_plates(dataset)) == 4
    assert len(load_plates(dataset, sample_type="full_photo", split="test")) == 2
    assert len(load_plates(dataset, sample_type="full_photo", split="test", originals_only=True)) == 1
    assert load_plates(dataset, split="test", ocr_only=True).sample_type.tolist() == ["plate_crop"]


def test_load_plates_kolumny_pochodne(dataset):
    df = load_plates(dataset)
    assert df.source.tolist() == ["data1", "data2", "data3", "data1"]
    assert df.corrupted.tolist() == ["original", "original", "corrupted", "original"]


def test_crop_path_lokalnie_i_na_kaggle(tmp_path):
    (tmp_path / "ocr" / "test").mkdir(parents=True)
    (tmp_path / "ocr" / "test" / "a.png").touch()
    assert crop_path(tmp_path, "ocr/test/a.png") == tmp_path / "ocr" / "test" / "a.png"
    kaggle = tmp_path / "kaggle"
    (kaggle / "ocr" / "ocr" / "test").mkdir(parents=True)
    (kaggle / "ocr" / "ocr" / "test" / "a.png").touch()
    assert crop_path(kaggle, "ocr/test/a.png") == kaggle / "ocr" / "ocr" / "test" / "a.png"


def test_write_csv_puste_pola(tmp_path):
    write_csv(tmp_path / "a.csv", [{"a": 1, "b": 2}, {"a": 3}], ["a", "b"])
    assert (tmp_path / "a.csv").read_text().splitlines() == ["a,b", "1,2", "3,"]


def test_list_images(tmp_path):
    for name in ["b/x.JPG", "a.png", "c.txt"]:
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).touch()
    assert [p.relative_to(tmp_path).as_posix() for p in list_images(tmp_path)] == ["a.png", "b/x.JPG"]
