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
