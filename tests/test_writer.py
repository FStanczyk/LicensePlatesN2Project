import cv2
import numpy as np
import pytest

pytest.importorskip("imagecorruptions", reason='wymaga extras [build]: pip install -e ".[build]"')

from alpr.build.corruptions import CORRUPTIONS, apply_corruption  # noqa: E402
from alpr.build.writer import MAX_SIDE, CombinedWriter  # noqa: E402
from alpr.dataset import FIELDS, load_plates, parse_polygon  # noqa: E402

FIELDS_PLATE = {"text": "WA1", "text_raw": "wa 1", "blurred": 0, "country": "pl", "lighting": "unknown", "lines": 1}


def photo(h=200, w=300) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def plate(poly) -> dict:
    return {"poly": np.array(poly, float), **FIELDS_PLATE}


def test_add_photo_dwie_wersje_i_etykiety_yolo(tmp_path):
    writer = CombinedWriter(tmp_path)
    writer.add_photo("data1_a", "train", photo(), [plate([[30, 40], [130, 40], [130, 70], [30, 70]])])
    original, copy = writer.rows
    assert original["plate_id"] == "data1_a_0" and original["artificial_corruption"] == "none"
    assert copy["plate_id"].startswith("data1_a_") and copy["artificial_corruption"] in CORRUPTIONS
    assert copy["plate_id"] == f"data1_a_{copy['artificial_corruption']}{copy['artificial_severity']}_0"
    assert (original["x1"], original["y1"], original["x2"], original["y2"]) == (30, 40, 130, 70)
    for row in writer.rows:
        assert (tmp_path / row["image"]).exists() and (tmp_path / row["crop"]).exists()
        label = tmp_path / row["image"].replace("images", "labels").replace(".jpg", ".txt")
        cls, cx, cy, bw, bh = label.read_text().split()
        assert cls == "0"
        assert [float(v) for v in (cx, cy, bw, bh)] == pytest.approx(
            [80 / 300, 55 / 200, 100 / 300, 30 / 200], abs=1e-6
        )


def test_add_photo_zmniejsza_duze_zdjecia(tmp_path):
    writer = CombinedWriter(tmp_path)
    poly = [[1000, 400], [2000, 400], [2000, 600], [1000, 600]]
    writer.add_photo("k", "val", photo(1000, 3840), [plate(poly)])
    row = writer.rows[0]
    assert cv2.imread(str(tmp_path / row["image"])).shape[:2] == (500, MAX_SIDE)
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (500, 200, 1000, 300)
    np.testing.assert_array_equal(parse_polygon(row["polygon"]), np.array(poly) / 2)
    assert cv2.imread(str(tmp_path / row["crop"])).shape[:2] == (100, 500)


def test_add_photo_ramka_przycieta_do_obrazu(tmp_path):
    writer = CombinedWriter(tmp_path)
    writer.add_photo("k", "test", photo(), [plate([[-10, -5], [100, -5], [100, 30], [-10, 30]])])
    row = writer.rows[0]
    assert (row["x1"], row["y1"]) == (0, 0)
    label = (tmp_path / "detection/labels/test/k.txt").read_text().split()
    assert all(0.0 <= float(v) <= 1.0 for v in label[1:])


def test_add_photo_wiele_tablic_i_puste_zdjecie(tmp_path):
    writer = CombinedWriter(tmp_path)
    plates = [plate([[10, 10], [60, 10], [60, 30], [10, 30]]), plate([[100, 100], [200, 100], [200, 140], [100, 140]])]
    writer.add_photo("k", "train", photo(), plates)
    writer.add_photo("pusty", "train", photo(), [])
    assert len(writer.rows) == 4
    assert len((tmp_path / "detection/labels/train/k.txt").read_text().splitlines()) == 2
    assert (tmp_path / "detection/labels/train/pusty.txt").read_text() == "\n"


def test_kopia_ma_te_same_etykiety_co_oryginal(tmp_path):
    writer = CombinedWriter(tmp_path)
    writer.add_photo("k", "train", photo(), [plate([[30, 40], [130, 40], [130, 70], [30, 70]])])
    original, copy = writer.rows
    for col in ("x1", "y1", "x2", "y2", "polygon", "split", "text"):
        assert original[col] == copy[col]


def test_add_crop(tmp_path):
    writer = CombinedWriter(tmp_path)
    writer.add_crop("data3_00000", "val", np.full((40, 100, 3), 200, np.uint8), dict(FIELDS_PLATE))
    original, copy = writer.rows
    assert original["sample_type"] == copy["sample_type"] == "plate_crop"
    assert original["crop"] == "ocr/val/data3_00000.png" and original["exposure"] == "over"
    assert "image" not in original and (tmp_path / copy["crop"]).exists()


def test_save_i_load_plates(tmp_path):
    writer = CombinedWriter.create(tmp_path / "out")
    writer.add_photo("data1_a", "train", photo(), [plate([[30, 40], [130, 40], [130, 70], [30, 70]])])
    writer.add_crop("data3_00000", "test", np.full((40, 100, 3), 60, np.uint8), dict(FIELDS_PLATE))
    path = writer.save()
    assert path.read_text().splitlines()[0] == ",".join(FIELDS)
    df = load_plates(tmp_path / "out")
    assert len(df) == 4 and set(df.source) == {"data1", "data3"}
    assert len(load_plates(tmp_path / "out", originals_only=True)) == 2
    assert load_plates(tmp_path / "out", sample_type="full_photo").image.notna().all()


def test_create_czysci_stary_katalog(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "stary.txt").touch()
    CombinedWriter.create(tmp_path / "out")
    assert not (tmp_path / "out" / "stary.txt").exists()
    assert (tmp_path / "out" / "dataset-metadata.json").exists()


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_apply_corruption_kazde_znieksztalcenie(name):
    img = photo(60, 80)
    out = apply_corruption(img, name, 3, "k")
    assert out.shape == img.shape and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(apply_corruption(img, name, 3, "k"), out)


@pytest.mark.parametrize("name", ["brightness", "darken", "contrast", "gaussian_noise", "motion_blur"])
def test_apply_corruption_sila_rosnie(name):
    img = np.full((60, 80, 3), 128, np.uint8)
    img[20:40, 20:60] = 30
    diffs = [np.abs(apply_corruption(img, name, s, "k").astype(int) - img).mean() for s in (1, 5)]
    assert diffs[1] > diffs[0]


def test_apply_corruption_bgr_kanaly_zachowane():
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, :, 0] = 200
    out = apply_corruption(img, "brightness", 1, "k")
    assert out[:, :, 0].mean() > out[:, :, 2].mean()


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_apply_corruption_maly_obraz(name):
    img = photo(20, 90)
    out = apply_corruption(img, name, 3, "k")
    assert out.shape == img.shape and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]


def test_add_crop_niski_wycinek(tmp_path):
    writer = CombinedWriter(tmp_path)
    for i in range(20):
        writer.add_crop(f"k{i}", "test", photo(18, 70), dict(FIELDS_PLATE))
    assert len(writer.rows) == 40
