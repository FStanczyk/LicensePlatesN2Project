import cv2
import numpy as np
import pandas as pd
import pytest

from alpr.dataset import FIELDS, write_csv
from alpr.detection.pipeline import read_detections
from alpr.experiment import build_experiment, ground_truth, match_results, summary, variants


@pytest.fixture
def dataset(tmp_path):
    rows = []
    for i, (kind, split, corr, blurred) in enumerate(
        [
            ("full_photo", "test", "none", 0),
            ("full_photo", "val", "none", 0),
            ("full_photo", "train", "none", 0),
            ("full_photo", "test", "fog", 0),
            ("full_photo", "test", "none", 1),
            ("plate_crop", "test", "none", 0),
        ]
    ):
        image = f"detection/images/{split}/d{i}.jpg" if kind == "full_photo" else ""
        if image:
            (tmp_path / image).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tmp_path / image), np.full((60, 100, 3), 120, np.uint8))
        rows.append(
            dict.fromkeys(FIELDS, "")
            | {
                "plate_id": f"data1_{i}_0",
                "sample_type": kind,
                "split": split,
                "text": "WA12345",
                "blurred": blurred,
                "artificial_corruption": corr,
                "artificial_severity": 0 if corr == "none" else 3,
                "image": image,
                "x1": 10,
                "y1": 20,
                "x2": 50,
                "y2": 35,
            }
        )
    write_csv(tmp_path / "plates.csv", rows, FIELDS)
    return tmp_path


def test_variants():
    names = [v for v, _, _ in variants(("darken",))]
    assert names == ["clean", "darken_1", "darken_2", "darken_3", "darken_4", "darken_5"]


def test_ground_truth_tylko_czyste_oryginaly(dataset):
    assert sorted(ground_truth(dataset).plate_id) == ["data1_0_0", "data1_1_0"]


def test_build_experiment(dataset, tmp_path):
    pytest.importorskip("imagecorruptions", reason='wymaga extras [build]: pip install -e ".[build]"')
    out = tmp_path / "exp"
    gt = build_experiment(dataset, out, ("test",), ("darken",))
    assert len(gt) == 6 and set(gt.variant) == {v for v, _, _ in variants(("darken",))}
    clean, dark = (cv2.imread(str(out / f"images/{v}/d0.jpg")) for v in ("clean", "darken_5"))
    assert dark.shape == clean.shape and dark.mean() < clean.mean()
    assert (gt[["x1", "y1", "x2", "y2"]] == [10, 20, 50, 35]).all().all()


def test_match_results_i_summary(tmp_path):
    gt = pd.DataFrame(
        {
            "variant": ["clean", "darken_5"],
            "artificial_corruption": ["none", "darken"],
            "artificial_severity": [0, 5],
            "image": ["images/clean/a.jpg", "images/darken_5/a.jpg"],
            "plate_id": ["p", "p"],
            "split": "test",
            "exposure": "normal",
            "text": "WA12345",
            "x1": 10,
            "y1": 20,
            "x2": 50,
            "y2": 35,
        }
    )
    (tmp_path / "ocr.csv").write_text(
        "image,det_id,conf,x1,y1,x2,y2,polygon,align,crop,text,conf_ocr\n"
        "images/clean/a.jpg,0,0.9,11,20,50,35,,quad,crops/a_0.png,WA12345,0.8\n"
        "images/clean/a.jpg,1,0.3,200,200,240,215,,box,crops/a_1.png,XX,0.1\n"
        "images/darken_5/a.jpg,,,,,,,,,,,\n"
    )
    res = match_results(gt, read_detections(tmp_path / "ocr.csv"))
    assert res.detected.tolist() == [1, 0] and res.text_ok.tolist() == [1, 0]
    assert res.crop.iloc[0] == "crops/a_0.png" and res["align"].tolist() == ["quad", ""] and res.text_pred.iloc[1] == ""
    table = summary(res)
    assert table.detected.tolist() == [1.0, 0.0]
