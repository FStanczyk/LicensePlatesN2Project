import cv2
import numpy as np
import pandas as pd
import pytest

from alpr import ocr


def quad(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class FakeReader:
    def __init__(self, found):
        self.found = found

    def readtext(self, crop, allowlist, detail, paragraph):
        return self.found


@pytest.fixture
def fake_reader(monkeypatch):
    def use(found):
        monkeypatch.setattr(ocr, "easyocr_reader", lambda: FakeReader(found))

    return use


def test_reading_order_dwa_wiersze():
    rects = [(60, 40, 90, 60), (0, 0, 50, 20), (0, 42, 50, 60), (55, 2, 90, 20)]
    boxes = [np.array(quad(*r), np.float32) for r in rects]
    assert ocr.reading_order(boxes) == [1, 3, 2, 0]


def test_easyocr_skleja_fragmenty(fake_reader):
    fake_reader([(quad(60, 0, 100, 20), "12345", 0.5), (quad(0, 0, 50, 20), "WA", 1.0)])
    text, conf = ocr.easyocr_engine(np.zeros((20, 100, 3), np.uint8))
    assert text == "WA12345"
    assert conf == pytest.approx((2 * 1.0 + 5 * 0.5) / 7)


def test_easyocr_pusty_odczyt(fake_reader):
    fake_reader([])
    assert ocr.easyocr_engine(np.zeros((20, 100, 3), np.uint8)) == ("", 0.0)


def test_read_crops(tmp_path, monkeypatch):
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((10, 40, 3), np.uint8))
    monkeypatch.setitem(ocr.ENGINES, "fake", lambda crop: ("ab-12", 0.91234))
    rows = pd.DataFrame({"image": ["x.jpg", "y.jpg"], "crop": ["a.png", None]})
    out = ocr.read_crops(rows, tmp_path, "fake")
    assert out.text.iloc[0] == "AB12" and out.conf_ocr.iloc[0] == 0.9123
    assert out.iloc[1][["text", "conf_ocr"]].isna().all()


def test_reading_order_przypadki_brzegowe():
    assert ocr.reading_order([]) == []
    assert ocr.reading_order([np.array(quad(0, 0, 10, 10), np.float32)]) == [0]


def test_reading_order_jeden_wiersz_z_drganiem_w_pionie():
    rects = [(70, 3, 100, 23), (0, 0, 30, 20), (35, 2, 65, 21)]
    assert ocr.reading_order([np.array(quad(*r), np.float32) for r in rects]) == [1, 2, 0]


def test_reading_order_trzy_wiersze():
    rects = [(0, 100, 40, 130), (50, 0, 90, 30), (0, 50, 40, 80), (0, 0, 40, 30), (50, 100, 90, 130)]
    assert ocr.reading_order([np.array(quad(*r), np.float32) for r in rects]) == [3, 1, 2, 0, 4]


def test_reading_order_skosne_ramki():
    boxes = [np.array([[60, 4], [100, 6], [100, 26], [60, 24]], np.float32), np.array(quad(0, 0, 50, 20), np.float32)]
    assert ocr.reading_order(boxes) == [1, 0]


def test_easyocr_pusty_fragment_ma_wage_1(fake_reader):
    fake_reader([(quad(0, 0, 10, 20), "", 0.0), (quad(20, 0, 40, 20), "AB", 1.0)])
    text, conf = ocr.easyocr_engine(np.zeros((20, 40, 3), np.uint8))
    assert text == "AB" and conf == pytest.approx(2 / 3)


def test_easyocr_dwuwierszowa_tablica(fake_reader):
    fake_reader([(quad(0, 30, 60, 55), "1234", 0.8), (quad(0, 0, 30, 25), "AB", 0.9)])
    assert ocr.easyocr_engine(np.zeros((60, 60, 3), np.uint8))[0] == "AB1234"


def test_read_plate_przez_easyocr_normalizuje(fake_reader):
    fake_reader([(quad(0, 0, 50, 20), "wa-12", 0.7)])
    assert ocr.read_plate(np.zeros((20, 50, 3), np.uint8)) == ("WA12", 0.7)


def test_read_plate_nieznany_silnik():
    with pytest.raises(NotImplementedError, match="tesseract"):
        ocr.read_plate(np.zeros((10, 40, 3), np.uint8), "tesseract")


def test_read_crops_zachowuje_kolumny_i_indeks(tmp_path, monkeypatch):
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((10, 40, 3), np.uint8))
    (tmp_path / "zly.png").write_bytes(b"x")
    monkeypatch.setitem(ocr.ENGINES, "fake", lambda crop: ("AB", 0.5))
    rows = pd.DataFrame(
        {"image": ["x", "y", "z"], "crop": ["a.png", "zly.png", "brak.png"], "conf": 1}, index=[5, 7, 9]
    )
    out = ocr.read_crops(rows, tmp_path, "fake")
    assert list(out.index) == [5, 7, 9] and list(out.columns) == ["image", "crop", "conf", "text", "conf_ocr"]
    assert out.text.tolist()[0] == "AB" and out.text.isna().tolist() == [False, True, True]
    assert "text" not in rows


def test_read_crops_przekazuje_wycinek_bgr(tmp_path, monkeypatch):
    img = np.zeros((10, 40, 3), np.uint8)
    img[:, :, 2] = 255
    cv2.imwrite(str(tmp_path / "a.png"), img)
    seen = []
    monkeypatch.setitem(ocr.ENGINES, "fake", lambda crop: (seen.append(crop) or "A", 1.0))
    ocr.read_crops(pd.DataFrame({"crop": ["a.png"]}), tmp_path, "fake")
    assert seen[0].shape == (10, 40, 3) and seen[0][0, 0].tolist() == [0, 0, 255]


METRIC_TODO = pytest.mark.xfail(raises=NotImplementedError, reason="metryki OCR do uzupełnienia (Golba)")


@METRIC_TODO
@pytest.mark.parametrize(
    ("pred", "true", "acc"),
    [("WA12345", "WA12345", 1.0), ("WA1234", "WA12345", 1 - 1 / 7), ("WB12345", "WA12345", 1 - 1 / 7), ("", "AB", 0.0)],
)
def test_character_accuracy(pred, true, acc):
    assert ocr.character_accuracy(pred, true) == pytest.approx(acc)


@METRIC_TODO
def test_plate_accuracy():
    assert ocr.plate_accuracy(["WA1", "KR2", ""], ["WA1", "KR3", "X"]) == pytest.approx(1 / 3)
    assert ocr.plate_accuracy(["A"], ["A"]) == 1.0


def test_plate_fragments_odrzuca_niskie_napisy():
    boxes = [
        np.array(quad(*r), np.float32) for r in [(0, 0, 10, 8), (12, 0, 90, 30), (0, 32, 90, 40), (95, 2, 120, 28)]
    ]
    assert ocr.plate_fragments(boxes) == [1, 3]
    assert ocr.plate_fragments(boxes, min_height=0) == [0, 1, 2, 3]


def test_plate_fragments_dwa_rowne_wiersze_zostaja():
    boxes = [np.array(quad(0, 0, 60, 25), np.float32), np.array(quad(0, 30, 60, 54), np.float32)]
    assert ocr.plate_fragments(boxes) == [0, 1]


def test_easyocr_pomija_pl_z_paska_i_napis_dealera(fake_reader):
    fake_reader(
        [
            (quad(0, 10, 8, 18), "PL", 0.9),
            (quad(10, 0, 100, 30), "SBL58012", 0.8),
            (quad(20, 32, 90, 38), "BRUTSCH", 0.7),
        ]
    )
    text, conf = ocr.easyocr_engine(np.zeros((40, 100, 3), np.uint8))
    assert (text, conf) == ("SBL58012", pytest.approx(0.8))
