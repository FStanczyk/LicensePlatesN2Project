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
    with pytest.raises(NotImplementedError, match="paddleocr"):
        ocr.read_plate(np.zeros((10, 40, 3), np.uint8), "paddleocr")


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


@pytest.mark.parametrize(
    ("pred", "true", "acc"),
    [("WA12345", "WA12345", 1.0), ("WA1234", "WA12345", 1 - 1 / 7), ("WB12345", "WA12345", 1 - 1 / 7), ("", "AB", 0.0)],
)
def test_character_accuracy(pred, true, acc):
    assert ocr.character_accuracy(pred, true) == pytest.approx(acc)


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


def test_easyocr_przekazuje_parametry_do_readtext(monkeypatch):
    seen = {}

    class Reader:
        def readtext(self, crop, **params):
            seen.update(params)
            return []

    monkeypatch.setattr(ocr, "easyocr_reader", lambda: Reader())
    ocr.easyocr_engine(np.zeros((5, 5, 3), np.uint8), mag_ratio=2.0)
    assert seen == {"allowlist": ocr.ALLOWLIST, "detail": 1, "paragraph": False, "mag_ratio": 2.0}


def test_read_crops_z_pamieci_i_porcjami(tmp_path, monkeypatch):
    cv2.imwrite(str(tmp_path / "dysk.png"), np.full((10, 40, 3), 7, np.uint8))
    calls = []
    monkeypatch.setitem(ocr.BATCH_ENGINES, "fake", lambda crops: calls.append(len(crops)) or [("x", 0.5)] * len(crops))
    rows = pd.DataFrame({"crop": ["pam.png", None, "dysk.png", "pam.png", "brak.png"]})
    out = ocr.read_crops(rows, tmp_path, "fake", crops={"pam.png": np.zeros((4, 4, 3), np.uint8)}, chunk=2)
    assert calls == [2, 1]
    assert out.text.isna().tolist() == [False, True, False, False, True]


def test_read_crops_z_pamieci_jak_z_dysku(tmp_path, monkeypatch):
    img = np.random.default_rng(0).integers(0, 255, (10, 40, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), img)
    monkeypatch.setitem(ocr.ENGINES, "fake", lambda crop: (str(int(crop.sum())), 1.0))
    rows = pd.DataFrame({"crop": ["a.png"]})
    pd.testing.assert_frame_equal(
        ocr.read_crops(rows, tmp_path, "fake"), ocr.read_crops(rows, tmp_path / "x", "fake", crops={"a.png": img})
    )


@pytest.mark.parametrize(("a", "b", "d"), [("", "", 0), ("ABC", "", 3), ("KITTEN", "SITTING", 3), ("AB", "BA", 2)])
def test_edit_distance(a, b, d):
    assert ocr.edit_distance(a, b) == d == ocr.edit_distance(b, a)


def test_character_accuracy_nie_ujemna():
    assert ocr.character_accuracy("ABCDEFGH", "A") == 0.0


def test_easyocr_min_height_nie_trafia_do_readtext(monkeypatch):
    seen = {}

    class Reader:
        def readtext(self, crop, **params):
            seen.update(params)
            return [(quad(0, 10, 8, 18), "PL", 0.9), (quad(10, 0, 100, 30), "AB1", 0.8)]

    monkeypatch.setattr(ocr, "easyocr_reader", lambda: Reader())
    assert ocr.easyocr_engine(np.zeros((40, 100, 3), np.uint8))[0] == "AB1"
    assert ocr.easyocr_engine(np.zeros((40, 100, 3), np.uint8), min_height=0.2)[0] == "PLAB1"
    assert "min_height" not in seen


def test_gate_wybiera_widok_tylko_przy_przewadze():
    assert ocr.gate([("A", 0.5), ("B", 0.55), ("C", 0.65)], 0.1) == ("C", 0.65)
    assert ocr.gate([("A", 0.5), ("B", 0.55)], 0.1) == ("A", 0.5)
    assert ocr.gate([("A", 0.9)]) == ("A", 0.9)


def test_read_views_jednym_wsadem(monkeypatch):
    calls = []

    def engine(crops, **params):
        calls.append((len(crops), params))
        return [(str(int(c.mean())), float(c.mean()) / 10) for c in crops]

    monkeypatch.setitem(ocr.BATCH_ENGINES, "fake", engine)
    views = [[np.full((2, 2, 3), v, np.uint8) for v in vs] for vs in ([1, 5], [3], [2, 9, 4])]
    out = ocr.read_views(views, "fake", margin=0.2, k=1)
    assert out == [("5", 0.5), ("3", 0.3), ("9", 0.9)]
    assert calls == [(6, {"k": 1})]


def test_read_deblurred_pewniejszy_silnik(monkeypatch):
    import alpr.deblur

    monkeypatch.setattr(alpr.deblur, "deconv_views", lambda c, a, iters: [c, c + 1])
    monkeypatch.setitem(ocr.BATCH_ENGINES, "lo", lambda crops: [("lo", 0.2 + c.mean() / 100) for c in crops])
    monkeypatch.setitem(ocr.BATCH_ENGINES, "hi", lambda crops: [("hi", 0.5) for c in crops])
    crops = [np.zeros((2, 2, 3), np.uint8), np.full((2, 2, 3), 50, np.uint8)]
    out = ocr.read_deblurred(crops, [0.0, 0.0], [{"engine": "lo"}, {"engine": "hi"}])
    assert [t for t, _ in out] == ["HI", "LO"]
    seen = []
    monkeypatch.setitem(ocr.BATCH_ENGINES, "spy", lambda crops: seen.append(len(crops)) or [("x", 1.0)] * len(crops))
    ocr.read_deblurred(crops, [0.0, 0.0], [{"engine": "spy"}], deblur=False)
    assert seen == [2]


def test_read_crops_deblurred(tmp_path, monkeypatch):
    seen = {}

    def fake(crops, angles, configs, margin, iters, deblur):
        seen.update(angles=angles, margin=margin, deblur=deblur)
        return [("AB1", 0.71234)] * len(crops)

    monkeypatch.setattr(ocr, "read_deblurred", fake)
    rows = pd.DataFrame({"image": ["a.jpg", "b.jpg", "c.jpg"], "crop": ["x.png", None, "y.png"]})
    crops = {"x.png": np.zeros((4, 4, 3), np.uint8), "y.png": np.zeros((4, 4, 3), np.uint8)}
    out = ocr.read_crops_deblurred(rows, tmp_path, {"a.jpg": 12.0}, [{"engine": "e"}], crops, margin=0.3)
    assert out.text.tolist()[0] == "AB1" and pd.isna(out.text.iloc[1]) and out.conf_ocr.iloc[2] == 0.7123
    assert seen == {"angles": [12.0, 0.0], "margin": 0.3, "deblur": True}


def test_parseq_zarejestrowany():
    assert "parseq" in ocr.engine_names()


def test_read_batch_normalizuje_i_przekazuje_parametry(monkeypatch):
    monkeypatch.setitem(ocr.ENGINES, "fake", lambda crop, **params: (f"a-{params['k']}", 1.0))
    assert ocr.read_batch([np.zeros((5, 5, 3))] * 2, "fake", k=7) == [("A7", 1.0), ("A7", 1.0)]
    monkeypatch.setitem(ocr.BATCH_ENGINES, "wsad", lambda crops, n: [("t-0", 0.5)] * len(crops) * n)
    assert ocr.read_batch([np.zeros((5, 5, 3))], "wsad", n=1) == [("T0", 0.5)]
    assert {"easyocr", "tesseract", "crnn", "lprnet", "parseq"} <= set(ocr.engine_names())
