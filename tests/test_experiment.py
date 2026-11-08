import cv2
import numpy as np
import pandas as pd
import pytest

from alpr.dataset import FIELDS, write_csv
from alpr.detection.pipeline import read_detections
from alpr.experiment import (
    CORRUPTIONS,
    GT_FIELDS,
    RESULT_FIELDS,
    SEVERITIES,
    SPLITS,
    build_experiment,
    ground_truth,
    match_results,
    summary,
    variants,
)


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
    assert ground_truth(dataset).plate_id.tolist() == ["data1_0_0"]
    assert sorted(ground_truth(dataset, ("val", "test")).plate_id) == ["data1_0_0", "data1_1_0"]


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


def test_variants_domyslne():
    vs = variants()
    assert vs[0] == ("clean", "none", 0)
    assert len(vs) == 1 + len(CORRUPTIONS) * len(SEVERITIES)
    assert len({v for v, _, _ in vs}) == len(vs)
    assert all(v == f"{c}_{s}" for v, c, s in vs[1:])


def test_ground_truth_bez_treningu_i_kopii(dataset):
    gt = ground_truth(dataset)
    assert SPLITS == ("test",) and set(gt.split) == {"test"}
    assert (gt.artificial_corruption == "none").all() and (gt.blurred == 0).all()
    assert (gt.sample_type == "full_photo").all()


def test_ground_truth_inne_podzialy(dataset):
    assert ground_truth(dataset, ("train",)).plate_id.tolist() == ["data1_2_0"]
    assert ground_truth(dataset, ()).empty


def test_build_experiment_kolumny_i_zrodla(dataset, tmp_path):
    pytest.importorskip("imagecorruptions", reason='wymaga extras [build]: pip install -e ".[build]"')
    gt = build_experiment(dataset, tmp_path / "exp", ("val", "test"), ("darken",))
    assert list(gt.columns) == GT_FIELDS
    assert set(gt.source_image) == {"detection/images/test/d0.jpg", "detection/images/val/d1.jpg"}
    clean = cv2.imread(str(tmp_path / "exp" / "images" / "clean" / "d1.jpg"))
    original = cv2.imread(str(dataset / "detection/images/val/d1.jpg"))
    assert clean.shape == original.shape and np.abs(clean.astype(int) - original).mean() < 2
    assert gt.groupby("variant").size().eq(2).all()


def gt_frame(rows):
    base = {
        "variant": "clean",
        "artificial_corruption": "none",
        "artificial_severity": 0,
        "split": "test",
        "exposure": "normal",
    }
    return pd.DataFrame([base | r for r in rows])


def test_match_results_dwie_tablice_na_zdjeciu(tmp_path):
    gt = gt_frame(
        [
            {"image": "a.jpg", "plate_id": "p0", "text": "AAA", "x1": 0, "y1": 0, "x2": 40, "y2": 10},
            {"image": "a.jpg", "plate_id": "p1", "text": "BBB", "x1": 100, "y1": 0, "x2": 140, "y2": 10},
        ]
    )
    (tmp_path / "ocr.csv").write_text(
        "image,det_id,conf,x1,y1,x2,y2,polygon,align,crop,text,conf_ocr\n"
        "a.jpg,0,0.9,100,0,140,10,,quad,c0.png,BBB,0.9\n"
        "a.jpg,1,0.8,0,0,40,10,,box,c1.png,AAX,0.4\n"
    )
    res = match_results(gt, read_detections(tmp_path / "ocr.csv"))
    assert list(res.columns) == RESULT_FIELDS
    assert res.text_pred.tolist() == ["AAX", "BBB"] and res.text_ok.tolist() == [0, 1]
    assert res.crop.tolist() == ["c1.png", "c0.png"] and res.iou.tolist() == [1.0, 1.0]


def test_match_results_zdjecie_bez_wiersza_i_pusty_odczyt(tmp_path):
    gt = gt_frame(
        [
            {"image": "a.jpg", "plate_id": "p0", "text": "AAA", "x1": 0, "y1": 0, "x2": 40, "y2": 10},
            {"image": "b.jpg", "plate_id": "p1", "text": "BBB", "x1": 0, "y1": 0, "x2": 40, "y2": 10},
        ]
    )
    (tmp_path / "ocr.csv").write_text(
        "image,det_id,conf,x1,y1,x2,y2,polygon,align,crop,text,conf_ocr\na.jpg,0,0.9,0,0,40,10,,quad,c0.png,,\n"
    )
    res = match_results(gt, read_detections(tmp_path / "ocr.csv"))
    assert res.detected.tolist() == [1, 0]
    assert res.text_pred.tolist() == ["", ""] and res.text_ok.tolist() == [0, 0]
    assert res.iou.tolist() == [1.0, 0.0] and np.isnan(res.det_conf.iloc[1])


def test_match_results_slaba_ramka_nie_jest_wykryciem(tmp_path):
    gt = gt_frame([{"image": "a.jpg", "plate_id": "p0", "text": "AAA", "x1": 0, "y1": 0, "x2": 40, "y2": 10}])
    (tmp_path / "ocr.csv").write_text(
        "image,det_id,conf,x1,y1,x2,y2,polygon,align,crop,text,conf_ocr\na.jpg,0,0.9,25,0,65,10,,box,c.png,AAA,0.9\n"
    )
    res = match_results(gt, read_detections(tmp_path / "ocr.csv"))
    assert res.detected.tolist() == [0] and res.text_ok.tolist() == [0]
    assert 0 < res.iou.iloc[0] < 0.5


def test_summary_kolejnosc_wersji_i_srednie():
    res = pd.DataFrame(
        {
            "artificial_corruption": ["none", "none", "darken", "darken"],
            "artificial_severity": [0, 0, 5, 5],
            "plate_id": ["a", "b", "a", "b"],
            "detected": [1, 1, 1, 0],
            "text_ok": [1, 0, 0, 0],
        }
    )
    table = summary(res)
    assert table.artificial_corruption.tolist() == ["none", "darken"]
    assert table.plates.tolist() == [2, 2]
    assert table.detected.tolist() == [1.0, 0.5] and table.text_ok.tolist() == [0.5, 0.0]
