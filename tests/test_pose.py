import sys
import types

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from alpr.dataset import FIELDS, add_derived, write_csv
from alpr.detection import AlignedPlate, PlateAligner, PlateDetector
from alpr.detection.align import equalize
from alpr.detection.pipeline import crop_plates, evaluate_alignment, predict_boxes
from alpr.detection.pose import FLIP_IDX, KPT_SHAPE, corners_by_image, export_pose_dataset, pose_label
from alpr.detection.training import train_pose_detector
from alpr.geometry import polygon_iou

PLATE = np.array([[120, 110], [290, 125], [285, 175], [115, 162]], np.float32)


def scene() -> np.ndarray:
    img = np.full((300, 400, 3), 40, np.uint8)
    cv2.fillPoly(img, [PLATE.astype(np.int32)], (230, 230, 230))
    return img


class Array:
    def __init__(self, a):
        self.a = np.asarray(a, np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakePose:
    trained: dict = {}

    def __init__(self, weights):
        self.weights = weights

    def predict(self, img, conf, imgsz, verbose):
        boxes = types.SimpleNamespace(xyxy=Array([[112, 108, 293, 178]]), conf=Array([0.8]))
        keypoints = types.SimpleNamespace(xy=Array([PLATE[[1, 2, 3, 0]]]))
        return [types.SimpleNamespace(boxes=boxes, keypoints=keypoints)]

    def train(self, **params):
        FakePose.trained = params


@pytest.fixture
def pose_detector(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakePose))
    return PlateDetector("pose.pt")


def test_pose_label_widoczne_i_ukryte_narozniki():
    row = types.SimpleNamespace(x1=100, y1=50, x2=300, y2=100, polygon="300 50 300 100 100 100 100 50")
    visible = pose_label(row, 400, 200, True).split()
    assert visible[:5] == ["0", "0.500000", "0.375000", "0.500000", "0.250000"]
    assert visible[5:] == ["0.250000", "0.250000", "2", "0.750000", "0.250000", "2"] + [
        "0.750000",
        "0.500000",
        "2",
        "0.250000",
        "0.500000",
        "2",
    ]
    hidden = pose_label(row, 400, 200, False).split()
    assert hidden[5:] == ["0"] * 12 and len(hidden) == 5 + KPT_SHAPE[0] * KPT_SHAPE[1]


@pytest.fixture
def dataset(tmp_path):
    rows = []
    for i, (plate_id, split, polygon) in enumerate(
        [
            ("data2_a_0", "train", "120 110 115 162 285 175 290 125"),
            ("data1_b_0", "val", "10 20 50 20 50 35 10 35"),
            ("data2_c_0", "test", "120 110 290 125 285 175 115 162"),
        ]
    ):
        image = f"detection/images/{split}/{plate_id[:-2]}.jpg"
        (tmp_path / image).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path / image), scene())
        rows.append(
            dict.fromkeys(FIELDS, "")
            | {
                "plate_id": plate_id,
                "sample_type": "full_photo",
                "split": split,
                "artificial_corruption": "none",
                "blurred": 0,
                "image": image,
                "x1": 112 if i != 1 else 10,
                "y1": 108 if i != 1 else 20,
                "x2": 293 if i != 1 else 50,
                "y2": 178 if i != 1 else 35,
                "polygon": polygon,
            }
        )
    write_csv(tmp_path / "plates.csv", rows, FIELDS)
    return tmp_path


def test_export_pose_dataset(dataset, tmp_path):
    out = tmp_path / "pose"
    cfg = yaml.safe_load(export_pose_dataset(dataset, out).read_text())
    assert cfg["kpt_shape"] == KPT_SHAPE and cfg["flip_idx"] == FLIP_IDX and cfg["names"] == {0: "plate"}
    assert cfg["path"] == str(out.resolve()) and cfg["train"] == "images/train"
    link = out / "images" / "train" / "data2_a.jpg"
    assert link.is_symlink() and link.resolve() == (dataset / "detection/images/train/data2_a.jpg").resolve()
    ccw, cw = ((out / "labels" / s / f"data2_{k}.txt").read_text().split() for s, k in [("train", "a"), ("test", "c")])
    assert ccw == cw
    assert (out / "labels" / "val" / "data1_b.txt").read_text().split()[5:] == ["0"] * 12
    export_pose_dataset(dataset, out)
    assert len(list((out / "labels").rglob("*.txt"))) == 3


def test_corners_by_image():
    preds = pd.DataFrame(
        {
            "image": ["a", "a", "b"],
            "conf": [0.9, 0.1, 0.8],
            "x1": 0,
            "y1": 0,
            "x2": 1,
            "y2": 1,
            "polygon": ["1 2 3 4 5 6 7 8", "0 0 0 0 0 0 0 0", None],
        }
    )
    corners = corners_by_image(preds, 0.25)
    assert list(corners) == ["a", "b"] and corners["b"] == [None]
    np.testing.assert_array_equal(corners["a"][0], [[1, 2], [3, 4], [5, 6], [7, 8]])
    assert corners_by_image(preds.drop(columns="polygon"), 0.25) is None
    assert corners_by_image(preds.assign(polygon=None), 0.25) is None


def test_from_corners_porzadkuje_narozniki():
    plate = AlignedPlate.from_corners(scene(), PLATE[[2, 1, 0, 3]])
    assert plate.method == "pose"
    np.testing.assert_allclose(plate.poly, PLATE)
    assert plate.crop.shape[1] > 3 * plate.crop.shape[0]


def test_detektor_pose_zwraca_narozniki(pose_detector):
    det = pose_detector.detect(scene())[0]
    np.testing.assert_allclose(det.corners, PLATE[[1, 2, 3, 0]])


def test_predict_boxes_zapisuje_narozniki(pose_detector, tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), scene())
    preds = predict_boxes(pose_detector, ["a.png"], tmp_path)
    assert preds.polygon.iloc[0] == "290 125 285 175 115 162 120 110"
    corners = corners_by_image(preds, 0.25)["a.png"][0]
    assert polygon_iou(AlignedPlate.from_corners(scene(), corners).poly, PLATE) > 0.99


def test_crop_plates_narozniki_pose_przed_alignerem(pose_detector, tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), scene())
    rows = crop_plates(pose_detector, ["a.png"], tmp_path, tmp_path / "out", PlateAligner())
    assert rows[0]["align"] == "pose" and rows[0]["polygon"] == "120 110 290 125 285 175 115 162"


def test_evaluate_alignment_z_naroznikami_pose(tmp_path):
    cv2.imwrite(str(tmp_path / "s.png"), scene())
    plates = add_derived(
        pd.DataFrame(
            [
                {
                    "plate_id": "data2_s_0",
                    "image": "s.png",
                    "x1": 112,
                    "y1": 108,
                    "x2": 293,
                    "y2": 178,
                    "polygon": "120 110 290 125 285 175 115 162",
                    "lighting": "day",
                    "exposure": "normal",
                    "artificial_corruption": "none",
                }
            ]
        )
    )
    boxes = {"s.png": np.array([[0, 0, 10, 10], [112, 108, 293, 178]], float)}
    corners = {"s.png": [None, PLATE[[3, 0, 1, 2]]]}
    row = evaluate_alignment(plates, tmp_path, PlateAligner(), boxes, pred_corners=corners).iloc[0]
    assert row.method == "pose" and row.iou_aligned > 0.99 and row.err_aligned < 0.01
    row = evaluate_alignment(plates, tmp_path, PlateAligner(), boxes).iloc[0]
    assert row.method != "pose"


def test_train_pose_detector(dataset, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakePose))
    best = train_pose_detector(dataset, "yolo11n-pose.pt", tmp_path / "runs", "p", tmp_path / "pose", epochs=1)
    assert best == tmp_path / "runs" / "p" / "weights" / "best.pt"
    cfg = yaml.safe_load(open(FakePose.trained["data"]))
    assert cfg["kpt_shape"] == KPT_SHAPE and cfg["path"] == str((tmp_path / "pose").resolve())
    assert FakePose.trained["epochs"] == 1 and FakePose.trained["seed"] == 0


def test_equalize_podnosi_kontrast():
    img = np.full((40, 120, 3), 60, np.uint8)
    img[10:30, 20:100] = 75
    out = equalize(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.ptp(out[:, :, 0].astype(int)) > np.ptp(img[:, :, 0].astype(int))


def test_aligner_bez_clahe_nadal_dziala():
    plate = PlateAligner(clahe=False).align(scene(), np.array([112, 108, 293, 178], np.float32))
    assert plate.method == "quad"
