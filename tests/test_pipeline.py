import sys
import types

import cv2
import numpy as np
import pytest

from alpr.detection import PlateDetector
from alpr.detection.pipeline import crop_name, crop_plates, predict_boxes
from alpr.detection.training import train_detector


class Array:
    def __init__(self, a):
        self.a = np.asarray(a, np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeYOLO:
    trained = {}

    def __init__(self, weights):
        self.weights = weights

    def predict(self, img, conf, imgsz, verbose):
        boxes = [[10, 10, 50, 30], [60, 40, 90, 55]] if img.mean() > 0 else np.zeros((0, 4))
        confs = [0.3, 0.9][: len(boxes)]
        return [types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=Array(boxes), conf=Array(confs)))]

    def train(self, **params):
        FakeYOLO.trained = params


@pytest.fixture
def detector(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    return PlateDetector("x.pt")


@pytest.fixture
def images(tmp_path):
    (tmp_path / "a").mkdir()
    cv2.imwrite(str(tmp_path / "a" / "jasne.png"), np.full((80, 120, 3), 200, np.uint8))
    cv2.imwrite(str(tmp_path / "a" / "ciemne.png"), np.zeros((80, 120, 3), np.uint8))
    return tmp_path, ["a/ciemne.png", "a/jasne.png", "a/brak.png"]


def test_detect_malejaco_po_pewnosci(detector):
    assert [d.conf for d in detector.detect(np.ones((80, 120, 3), np.uint8))] == pytest.approx([0.9, 0.3])


def test_predict_boxes(detector, images):
    base, names = images
    preds = predict_boxes(detector, names, base)
    assert list(preds.columns) == ["image", "conf", "x1", "y1", "x2", "y2"]
    assert preds.image.tolist() == ["a/jasne.png", "a/jasne.png"]


def test_crop_plates(detector, images, tmp_path):
    base, names = images
    out = tmp_path / "out"
    rows = crop_plates(detector, names, base, out)
    assert rows[0] == {"image": "a/ciemne.png"}
    assert [r["crop"] for r in rows[1:]] == ["crops/jasne_0.png", "crops/jasne_1.png"]
    assert rows[1]["x1"] == 60 and rows[1]["align"] == "box"
    assert cv2.imread(str(out / rows[1]["crop"])).shape[:2] == (15, 30)


def test_crop_name():
    assert crop_name("a/b/c.jpg", 2, "a") == "crops/b__c_2.png"


def test_train_detector(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    (tmp_path / "data.yaml").write_text("path: .\nnames: {0: plate}\n")
    best = train_detector(tmp_path / "data.yaml", "yolo11n.pt", tmp_path / "runs", "x", epochs=1)
    assert best == tmp_path / "runs" / "x" / "weights" / "best.pt"
    assert FakeYOLO.trained["epochs"] == 1 and FakeYOLO.trained["seed"] == 0
    assert f"path: {tmp_path}" in (tmp_path / "runs" / "data_resolved.yaml").read_text()
