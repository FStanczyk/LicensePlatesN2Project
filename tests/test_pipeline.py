import os
import sys
import types
from pathlib import PurePosixPath

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from alpr.dataset import add_derived, write_csv
from alpr.detection import Detection, PlateAligner, PlateDetector
from alpr.detection.pipeline import (
    DETECTION_FIELDS,
    alignment_example,
    crop_name,
    crop_plates,
    evaluate_alignment,
    predict_boxes,
    read_detections,
    read_images,
)
from alpr.detection.training import resolved_yaml, train_detector

PLATE = np.array([[120, 110], [290, 125], [285, 175], [115, 162]], np.float32)


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
    assert list(preds.columns) == ["image", "conf", "x1", "y1", "x2", "y2", "polygon"]
    assert preds.polygon.isna().all()
    assert preds.image.tolist() == ["a/jasne.png", "a/jasne.png"]


def test_crop_plates(detector, images, tmp_path):
    base, names = images
    out = tmp_path / "out"
    rows = crop_plates(detector, names, base, out)
    assert rows[0] == {"image": "a/ciemne.png"}
    assert [r["crop"] for r in rows[1:]] == ["crops/jasne.png_0.png", "crops/jasne.png_1.png"]
    assert rows[1]["x1"] == 60 and rows[1]["align"] == "box"
    assert cv2.imread(str(out / rows[1]["crop"])).shape[:2] == (15, 30)


def test_crop_name():
    assert crop_name("a/b/c.jpg", 2, "a") == "crops/b__c.jpg_2.png"


def test_train_detector(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    (tmp_path / "data.yaml").write_text("path: .\nnames: {0: plate}\n")
    best = train_detector(tmp_path / "data.yaml", "yolo11n.pt", tmp_path / "runs", "x", epochs=1)
    assert best == tmp_path / "runs" / "x" / "weights" / "best.pt"
    assert FakeYOLO.trained["epochs"] == 1 and FakeYOLO.trained["seed"] == 0
    assert f"path: {tmp_path}" in (tmp_path / "runs" / "data_resolved.yaml").read_text()


def test_detect_puste():
    detector = PlateDetector.__new__(PlateDetector)
    detector.model, detector.conf, detector.imgsz = FakeYOLO("x"), 0.25, 640
    assert detector.detect(np.zeros((10, 10, 3), np.uint8)) == []
    assert all(isinstance(d, Detection) for d in detector.detect(np.ones((80, 120, 3), np.uint8)))


def test_detector_przekazuje_parametry(monkeypatch):
    seen = {}

    class Spy(FakeYOLO):
        def predict(self, img, conf, imgsz, verbose):
            seen.update(conf=conf, imgsz=imgsz, verbose=verbose)
            return super().predict(img, conf, imgsz, verbose)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=Spy))
    PlateDetector("x.pt", conf=0.4, imgsz=320).detect(np.ones((8, 8, 3), np.uint8))
    assert seen == {"conf": 0.4, "imgsz": 320, "verbose": False}


def test_read_images_pomija_niewczytywalne(images, capsys):
    base, names = images
    (base / "a" / "zly.png").write_bytes(b"to nie jest obraz")
    got = [rel for rel, _ in read_images([*names, "a/zly.png"], base)]
    assert got == ["a/ciemne.png", "a/jasne.png"]
    assert capsys.readouterr().out.count("pominięto") == 2


def test_predict_boxes_puste_ma_kolumny(detector, tmp_path):
    preds = predict_boxes(detector, ["brak.png"], tmp_path)
    assert preds.empty and list(preds.columns) == ["image", "conf", "x1", "y1", "x2", "y2", "polygon"]


@pytest.mark.parametrize(
    ("images", "expected"),
    [
        (["a.jpg", "b.jpg"], ["crops/a.jpg_0.png", "crops/b.jpg_0.png"]),
        (["b/x.jpg", "b/x.png"], ["crops/x.jpg_0.png", "crops/x.png_0.png"]),
        (["x/a.jpg"], ["crops/a.jpg_0.png"]),
        (["images/clean/a.jpg", "images/darken_5/a.jpg"], ["crops/clean__a.jpg_0.png", "crops/darken_5__a.jpg_0.png"]),
        (["clean/a.jpg", "dark/a.jpg"], ["crops/clean__a.jpg_0.png", "crops/dark__a.jpg_0.png"]),
    ],
)
def test_crop_name_unikalne_dla_wersji_o_tej_samej_nazwie(images, expected):
    common = os.path.commonpath([PurePosixPath(r).parent.as_posix() for r in images])
    assert [crop_name(r, 0, common) for r in images] == expected


def test_crop_plates_z_alignerem(detector, tmp_path):
    img = np.full((80, 120, 3), 40, np.uint8)
    img[42:53, 62:88] = 230
    cv2.imwrite(str(tmp_path / "p.png"), img)
    rows = crop_plates(detector, ["p.png"], tmp_path, tmp_path / "out", PlateAligner())
    assert [r["det_id"] for r in rows] == [0, 1]
    assert all(r["align"] in ("quad", "rect", "box") for r in rows)
    assert all(len(r["polygon"].split()) == 8 for r in rows)
    assert all((tmp_path / "out" / r["crop"]).exists() for r in rows)


def test_crop_plates_wiersze_zgodne_z_detection_fields(detector, images, tmp_path):
    base, names = images
    rows = crop_plates(detector, names, base, tmp_path / "out")
    assert all(set(r) <= set(DETECTION_FIELDS) for r in rows)
    write_csv(tmp_path / "det.csv", rows, DETECTION_FIELDS)
    df = read_detections(tmp_path / "det.csv")
    assert str(df.det_id.dtype) == "Int64" and str(df.x1.dtype) == "Int64"
    assert df.det_id.isna().tolist() == [True, False, False]
    assert df.x1.tolist()[1:] == [60, 10]


def test_alignment_example_trzy_kafelki():
    img = np.zeros((100, 200, 3), np.uint8)
    gt = np.array([[10, 10], [90, 12], [88, 40], [12, 38]], np.float32)
    out = alignment_example(img, np.array([5, 5, 95, 45], np.float32), np.zeros((20, 60, 3), np.uint8), gt)
    assert out.shape == (192, 288, 3)


def alignment_dataset(tmp_path):
    img = np.full((300, 400, 3), 40, np.uint8)
    cv2.fillPoly(img, [PLATE.astype(np.int32)], (230, 230, 230))
    cv2.putText(img, "AB123", (140, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
    cv2.imwrite(str(tmp_path / "s.png"), img)
    x1, y1 = PLATE.min(axis=0) - 4
    x2, y2 = PLATE.max(axis=0) + 4
    plates = pd.DataFrame(
        [
            {
                "plate_id": "data2_s_0",
                "image": "s.png",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "polygon": " ".join(str(int(v)) for v in PLATE.flatten()),
                "lighting": "unknown",
                "exposure": "normal",
                "artificial_corruption": "none",
            }
        ]
    )
    return add_derived(plates)


def test_evaluate_alignment_poprawia_ramke(tmp_path):
    (tmp_path / "ex").mkdir()
    res = evaluate_alignment(
        alignment_dataset(tmp_path), tmp_path, PlateAligner(), examples=tmp_path / "ex", n_examples=1
    )
    row = res.iloc[0]
    assert row.method == "quad" and row.source == "data2"
    assert row.iou_aligned > row.iou_box and row.err_aligned < row.err_box
    assert (tmp_path / "ex" / "data2_s_0.png").exists()


def test_evaluate_alignment_z_ramkami_detektora(tmp_path):
    plates = alignment_dataset(tmp_path)
    box = plates[["x1", "y1", "x2", "y2"]].to_numpy(float)
    assert len(evaluate_alignment(plates, tmp_path, PlateAligner(), {"s.png": box + 2})) == 1
    assert evaluate_alignment(plates, tmp_path, PlateAligner(), {"s.png": box + 200}).empty
    assert evaluate_alignment(plates, tmp_path, PlateAligner(), {}).empty


def test_resolved_yaml_zachowuje_reszte_konfiguracji(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "data.yaml").write_text("path: .\ntrain: images/train\nnames: {0: tablica}\n")
    out = resolved_yaml(tmp_path / "d" / "data.yaml", tmp_path / "o" / "p")
    cfg = yaml.safe_load(out.read_text())
    assert cfg == {"path": str((tmp_path / "d").resolve()), "train": "images/train", "names": {0: "tablica"}}


def test_evaluate_alignment_pomija_prostokaty_z_data1(tmp_path):
    plates = alignment_dataset(tmp_path)
    plates = add_derived(plates.assign(plate_id="data1_s_0"))
    assert evaluate_alignment(plates, tmp_path, PlateAligner()).empty


def test_read_images_w_tle_zachowuje_kolejnosc_i_pamiec(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("alpr.detection.pipeline.READ_WORKERS", 2)
    names = [f"{i:03d}.png" for i in range(250)]
    for i, name in enumerate(names):
        if i != 7:
            cv2.imwrite(str(tmp_path / name), np.full((4, 4, 3), i % 256, np.uint8))
    calls = []
    imread = cv2.imread
    monkeypatch.setattr(cv2, "imread", lambda path: calls.append(path) or imread(path))
    reader = read_images(names, tmp_path)
    first = next(reader)
    assert first[0] == "000.png"
    assert len(calls) <= 2 * 2 + 1
    got = [first, *reader]
    assert [rel for rel, _ in got] == [n for i, n in enumerate(names) if i != 7]
    assert all(img[0, 0, 0] == int(rel[:3]) % 256 for rel, img in got)
    out = capsys.readouterr().out
    assert "pominięto (nie da się wczytać): 007.png" in out
    assert "  100/250" in out and "  200/250" in out


def test_read_images_puste():
    assert list(read_images([], ".")) == []
