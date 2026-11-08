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
    with_corners,
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


@pytest.mark.parametrize("raw", ["wa 123-45", "Łódź 1", "  ab*c ", "X-Y_Z.9", "ŻÓŁW"])
def test_normalize_text_idempotentne_i_tylko_dozwolone_znaki(raw):
    text = normalize_text(raw)
    assert normalize_text(text) == text
    assert set(text) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*")


@pytest.mark.parametrize(
    ("value", "label"), [(0, "under"), (99, "under"), (100, "normal"), (101, "over"), (255, "over")]
)
def test_exposure_progi_na_jednolitym_obrazie(value, label):
    assert exposure(np.full((10, 30, 3), value, np.uint8)) == label


def test_exposure_wysoki_kontrast_to_normal():
    img = np.zeros((10, 30, 3), np.uint8)
    img[:, 15:] = 255
    assert exposure(img) == "normal"


def test_polygon_roundtrip_dla_calkowitych():
    poly = np.array([[0, 5], [100, 7], [98, 40], [2, 38]], np.float32)
    np.testing.assert_array_equal(parse_polygon(format_polygon(poly)), poly)
    assert parse_polygon(format_polygon(poly)).dtype == np.float32


def test_parse_polygon_zla_liczba_wspolrzednych():
    with pytest.raises(ValueError):
        parse_polygon("1 2 3 4 5 6")


def test_crop_path_gdy_pliku_nie_ma_nigdzie(tmp_path):
    assert crop_path(tmp_path, "ocr/test/x.png") == tmp_path / "ocr/test/x.png"


def test_crop_path_preferuje_uklad_lokalny(tmp_path):
    for rel in ("ocr/test/a.png", "ocr/ocr/test/a.png"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    assert crop_path(str(tmp_path), "ocr/test/a.png") == tmp_path / "ocr/test/a.png"


def test_load_plates_pusty_wynik_ma_kolumny_pochodne(dataset):
    df = load_plates(dataset, split="val")
    assert df.empty and {"source", "corrupted"} <= set(df.columns)


def test_load_plates_nie_zmienia_kolejnosci_ani_kolumn(dataset):
    df = load_plates(dataset, split="test")
    assert list(df.columns) == [*FIELDS, "source", "corrupted"]
    assert df.plate_id.tolist() == ["data2_1", "data3_2", "data1_3"]


def test_load_plates_przyjmuje_str(dataset):
    assert len(load_plates(str(dataset))) == 4


def test_fields_bez_duplikatow():
    assert len(FIELDS) == len(set(FIELDS))
    assert FIELDS[0] == "plate_id" and {"x1", "y1", "x2", "y2", "crop", "image"} <= set(FIELDS)


def test_write_csv_kolejnosc_kolumn_z_fields(tmp_path):
    write_csv(tmp_path / "a.csv", [{"b": 2, "a": 1}], ["a", "b"])
    assert (tmp_path / "a.csv").read_text().splitlines() == ["a,b", "1,2"]


def test_write_csv_nieznane_pole_to_blad(tmp_path):
    with pytest.raises(ValueError):
        write_csv(tmp_path / "a.csv", [{"a": 1, "c": 3}], ["a"])


def test_write_csv_cudzyslowy_i_polskie_znaki(tmp_path):
    write_csv(tmp_path / "a.csv", [{"a": "x,y", "b": "żółw"}], ["a", "b"])
    assert pd.read_csv(tmp_path / "a.csv").iloc[0].tolist() == ["x,y", "żółw"]


def test_list_images_pusty_folder(tmp_path):
    assert list_images(tmp_path) == []


def test_with_corners_pomija_prostokaty_data1_i_brak_wielokata():
    df = pd.DataFrame(
        {"plate_id": ["data1_a_0", "data2_b_0", "data2_c_0", "data3_d"], "polygon": ["1 2 3 4 5 6 7 8"] * 3 + [None]}
    )
    df = df.assign(source=df.plate_id.str.split("_").str[0])
    assert with_corners(df).plate_id.tolist() == ["data2_b_0", "data2_c_0"]
