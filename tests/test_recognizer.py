import sys
import types

import cv2
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from alpr import ocr, recognizer  # noqa: E402
from alpr.dataset import FIELDS, write_csv  # noqa: E402
from alpr.recognizer import (  # noqa: E402
    ALPHABET,
    CRNN,
    LPRNet,
    Recognizer,
    TrainConfig,
    augment,
    ctc_decode,
    encode,
    load_split,
    normalize,
    to_tensor,
    train_recognizer,
)


def test_encode_pomija_znaki_spoza_alfabetu():
    assert encode("AB-9") == [1, 2, ALPHABET.index("9") + 1]


def one_hot(seq):
    logits = torch.full((1, len(seq), len(ALPHABET) + 1), -10.0)
    for t, k in enumerate(seq):
        logits[0, t, k] = 10.0
    return logits


def test_ctc_decode_skleja_powtorzenia_i_usuwa_puste():
    a, b = encode("A")[0], encode("B")[0]
    text, conf = ctc_decode(one_hot([a, a, 0, a, b, b, 0]))[0]
    assert text == "AAB" and conf == pytest.approx(1.0)
    assert ctc_decode(one_hot([0, 0, 0])) == [("", 0.0)]


@pytest.mark.parametrize(("model", "size", "steps"), [(CRNN, (32, 128), 32), (LPRNet, (24, 94), 18)])
def test_ksztalt_wyjscia(model, size, steps):
    out = model().eval()(torch.zeros(3, 3, *size))
    assert out.shape == (3, steps, len(ALPHABET) + 1)


def test_lprnet_odczyt_nie_zalezy_od_wsadu():
    torch.manual_seed(0)
    model = LPRNet().eval()
    x = torch.randn(4, 3, 24, 94)
    with torch.no_grad():
        assert torch.allclose(model(x)[:1], model(x[:1]), atol=1e-5)


def test_to_tensor_i_normalize():
    crop = np.zeros((10, 40, 3), np.uint8)
    crop[:, :, 2] = 255
    x = to_tensor([crop, crop], (24, 94))
    assert x.shape == (2, 3, 24, 94) and x.dtype == torch.uint8 and x[0, 0, 0, 0] == 255
    assert normalize(x).min() == -1 and normalize(x).max() == 1


def test_augment_powtarzalne_i_w_zakresie():
    x = normalize(torch.randint(0, 255, (4, 3, 24, 94), dtype=torch.uint8))
    a = augment(x, torch.Generator().manual_seed(1))
    b = augment(x, torch.Generator().manual_seed(1))
    assert torch.equal(a, b) and a.shape == x.shape and a.abs().max() <= 1
    assert torch.allclose(augment(x, torch.Generator().manual_seed(1), strength=0), x, atol=1e-5)


class Arr:
    def __init__(self, a):
        self.a = np.asarray(a, np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def synthetic(n, size):
    rng = np.random.default_rng(0)
    texts = ["".join(rng.choice(list("AB12"), 3)) for _ in range(n)]
    return torch.randint(0, 255, (n, 3, *size), dtype=torch.uint8), texts


def test_train_recognizer_i_odczyt(tmp_path):
    xt, tt = synthetic(16, (32, 128))
    cfg = TrainConfig(epochs=2, batch=8, amp=False)
    best, history = train_recognizer("crnn", tmp_path, tmp_path / "out", cfg, "cpu", (xt, tt, xt[:4], tt[:4]))
    assert best.exists() and list(history.epoch) == [1, 2]
    assert set(history.columns) == {"epoch", "loss", "plate_acc", "car"}
    reader = Recognizer(best, "cpu")
    found = reader.read([np.zeros((20, 80, 3), np.uint8)] * 3)
    assert len(found) == 3 and all(isinstance(t, str) and 0 <= c <= 1 for t, c in found)
    assert reader.read([]) == []


def test_load_split_wycinki_i_zdjecia(tmp_path, monkeypatch):
    (tmp_path / "ocr" / "train").mkdir(parents=True)
    (tmp_path / "detection").mkdir()
    cv2.imwrite(str(tmp_path / "ocr/train/a.png"), np.full((20, 80, 3), 50, np.uint8))
    cv2.imwrite(str(tmp_path / "detection/b.jpg"), np.full((100, 200, 3), 90, np.uint8))
    base = {"split": "train", "blurred": 0, "artificial_corruption": "none", "artificial_severity": 0}
    rows = [
        dict.fromkeys(FIELDS, "")
        | base
        | {"plate_id": "c", "sample_type": "plate_crop", "crop": "ocr/train/a.png", "text": "AB1"},
        dict.fromkeys(FIELDS, "")
        | base
        | {
            "plate_id": "d",
            "sample_type": "full_photo",
            "image": "detection/b.jpg",
            "text": "XY",
            "polygon": "10 10 90 10 90 30 10 30",
            **dict(zip(["x1", "y1", "x2", "y2"], [10, 10, 90, 30], strict=True)),
        },
        dict.fromkeys(FIELDS, "")
        | base
        | {"plate_id": "e", "sample_type": "plate_crop", "crop": "brak.png", "text": "Q"},
    ]
    write_csv(tmp_path / "plates.csv", rows, FIELDS)
    x, texts = load_split(tmp_path, "train", (24, 94))
    assert x.shape == (2, 3, 24, 94) and texts == ["AB1", "XY"]
    assert x[1].float().mean() == pytest.approx(90, abs=1)
    x, texts = load_split(tmp_path, "train", (24, 94), jitter=2)
    assert texts == ["AB1", "XY", "XY", "XY"]

    class Yolo:
        def __init__(self, weights):
            pass

        def predict(self, imgs, **params):
            boxes = types.SimpleNamespace(xyxy=Arr([[11, 9, 89, 31]]), conf=Arr([0.9]))
            return [types.SimpleNamespace(boxes=boxes) for _ in imgs]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=Yolo))
    x, texts = load_split(tmp_path, "train", (24, 94), detector="x.pt")
    assert texts == ["AB1", "XY", "XY"]
    exp = tmp_path / "exp"
    (exp / "images" / "v").mkdir(parents=True)
    cv2.imwrite(str(exp / "images/v/b.jpg"), np.full((100, 200, 3), 30, np.uint8))
    gt = {"image": "images/v/b.jpg", "split": "train", "text": "XY", "x1": 10, "y1": 10, "x2": 90, "y2": 30}
    pd.DataFrame([gt, gt | {"split": "test"}]).to_csv(exp / "ground_truth.csv", index=False)
    x, texts = load_split(tmp_path, "train", (24, 94), detector="x.pt", extra=exp)
    assert texts == ["AB1", "XY", "XY", "XY"]
    x, texts = load_split(tmp_path, "train", (24, 94), jitter=1, detector="x.pt", extra=exp)
    assert texts == ["AB1", "XY", "XY", "XY", "XY"]
    x, texts = load_split(tmp_path, "train", (24, 94), detector="x.pt", align=False)
    assert texts == ["AB1", "XY", "XY"] and x[2].float().mean() == pytest.approx(90, abs=1)


def test_silnik_sieci_domyslne_wagi(monkeypatch, tmp_path):
    seen = []

    class Fake:
        def __init__(self, weights):
            seen.append(weights)

        def read(self, crops, batch):
            return [("ab", 0.5)] * len(crops)

    monkeypatch.setattr(recognizer, "Recognizer", Fake)
    ocr.recognizer.cache_clear()
    assert ocr.read_batch([np.zeros((5, 5, 3), np.uint8)], "lprnet") == [("AB", 0.5)]
    ocr.read_batch([np.zeros((5, 5, 3), np.uint8)], "crnn", weights=str(tmp_path / "w.pt"))
    assert seen[0].endswith("runs/ocr/lprnet/best.pt") and seen[1] == str(tmp_path / "w.pt")
    ocr.recognizer.cache_clear()


def test_tesseract_input():
    out = ocr.tesseract_input(np.full((20, 80, 3), 100, np.uint8), height=40, pad=5)
    assert out.shape == (50, 170) and out.dtype == np.uint8
    otsu = ocr.tesseract_input(np.full((20, 80, 3), 100, np.uint8), binarize=True)
    assert set(np.unique(otsu)) <= {0, 255}


def test_tesseract_engine(monkeypatch):
    seen = {}

    def image_to_data(img, config, output_type):
        seen["config"] = config
        return {"text": ["", "WA", "123"], "conf": [-1, 80, 60]}

    fake = types.SimpleNamespace(image_to_data=image_to_data, Output=types.SimpleNamespace(DICT="dict"))
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    assert ocr.tesseract_engine(np.zeros((20, 80, 3), np.uint8)) == ("WA123", pytest.approx(0.7))
    assert "--psm 7" in seen["config"] and "whitelist" in seen["config"]
    ocr.tesseract_engine(np.zeros((20, 80, 3), np.uint8), psm=8, whitelist=False)
    assert "--psm 8" in seen["config"] and "whitelist" not in seen["config"]
    assert ocr.read_batch([np.zeros((20, 80, 3), np.uint8)] * 3, "tesseract", workers=2) == [("WA123", 0.7)] * 3


def test_tesseract_pusty_odczyt(monkeypatch):
    fake = types.SimpleNamespace(
        image_to_data=lambda *a, **k: {"text": ["", " "], "conf": [-1, -1]},
        Output=types.SimpleNamespace(DICT="dict"),
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    assert ocr.tesseract_engine(np.zeros((20, 80, 3), np.uint8)) == ("", 0.0)


def test_silniki_zarejestrowane():
    assert {"easyocr", "tesseract", "crnn", "lprnet", "parseq"} <= set(ocr.engine_names())


def test_jitter_poly_w_granicach():
    from alpr.recognizer import jitter_poly

    poly = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], np.float32)
    out = jitter_poly(poly, np.random.default_rng(0))
    assert out.shape == (4, 2) and np.abs(out - poly)[:, 0].max() <= 6 and np.abs(out - poly)[:, 1].max() <= 3


def test_blur_kernels_znormalizowane_i_niepuste():
    from alpr.recognizer import blur_kernels

    k = blur_kernels(16, torch.Generator().manual_seed(0), "cpu")
    assert k.shape == (16, 1, 9, 9)
    assert torch.allclose(k.sum((1, 2, 3)), torch.ones(16)) and (k >= 0).all()


def test_blur_prob_zero_nie_zmienia_i_jeden_rozmywa():
    from alpr.recognizer import blur

    x = normalize(torch.randint(0, 255, (4, 3, 32, 128), dtype=torch.uint8))
    assert torch.allclose(blur(x, torch.Generator().manual_seed(0), 0.0), x, atol=1e-6)
    out = blur(x, torch.Generator().manual_seed(0), 1.0)
    assert out.shape == x.shape and out.std() < x.std()
