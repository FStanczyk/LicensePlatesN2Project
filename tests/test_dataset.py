import numpy as np
import pandas as pd
import pytest

from alpr.dataset import FIELDS, exposure, format_polygon, load_plates, normalize_text, parse_polygon


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
    for i, (kind, split, corr, blurred) in enumerate([
        ("full_photo", "train", "none", 0), ("full_photo", "test", "none", 1),
        ("full_photo", "test", "fog", 1), ("plate_crop", "test", "none", 0),
    ]):
        rows.append({f: "" for f in FIELDS} | {
            "plate_id": f"data{1 + i % 3}_{i}", "sample_type": kind, "split": split,
            "artificial_corruption": corr, "blurred": blurred,
        })
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
