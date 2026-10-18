import numpy as np
import pandas as pd
import pytest

from alpr.detection.metrics import (
    average_precision,
    best_matching_box,
    boxes_by_image,
    corner_error,
    evaluate,
    evaluate_by_group,
    image_groups,
    match,
    summarize_alignment,
)


def test_average_precision_przypadki_brzegowe():
    assert average_precision(np.array([True, True]), np.array([0.9, 0.8]), 2) == pytest.approx(0.995)
    assert average_precision(np.zeros(0, bool), np.zeros(0), 3) == 0.0
    assert np.isnan(average_precision(np.zeros(0, bool), np.zeros(0), 0))


def test_average_precision_kolejnosc_po_pewnosci():
    good = average_precision(np.array([True, False]), np.array([0.9, 0.5]), 2)
    bad = average_precision(np.array([True, False]), np.array([0.5, 0.9]), 2)
    assert good == pytest.approx(0.6225)
    assert bad < good


def test_match_zachlanny_kazda_tablica_raz():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    pred = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [20, 0, 30, 10]], float)
    tp, owner = match(pred, gt, 0.5)
    assert tp.tolist() == [True, False, True]
    assert owner.tolist() == [0, 2]


def make_gt():
    return pd.DataFrame(
        [
            {
                "plate_id": "data1_a_0",
                "image": "a.jpg",
                "x1": 0,
                "y1": 0,
                "x2": 100,
                "y2": 30,
                "lighting": "day",
                "exposure": "normal",
                "artificial_corruption": "none",
                "artificial_severity": 0,
            },
            {
                "plate_id": "data2_b_0",
                "image": "b.jpg",
                "x1": 50,
                "y1": 50,
                "x2": 150,
                "y2": 80,
                "lighting": "night",
                "exposure": "under",
                "artificial_corruption": "fog",
                "artificial_severity": 3,
            },
            {
                "plate_id": "data2_b_1",
                "image": "b.jpg",
                "x1": 200,
                "y1": 50,
                "x2": 300,
                "y2": 80,
                "lighting": "night",
                "exposure": "normal",
                "artificial_corruption": "fog",
                "artificial_severity": 3,
            },
        ]
    )


def test_evaluate_idealne_predykcje():
    gt = make_gt()
    preds = gt[["image", "x1", "y1", "x2", "y2"]].assign(conf=0.9)
    metrics, plates = evaluate(gt, preds, 0.25)
    assert metrics["P"] == metrics["R"] == metrics["F1"] == 1.0
    assert metrics["AP50"] == pytest.approx(0.995)
    assert metrics["AP50_95"] == pytest.approx(0.995)
    assert plates.detected.tolist() == [1, 1, 1]


def test_evaluate_brak_predykcji_i_prog_pewnosci():
    gt = make_gt()
    preds = pd.DataFrame([{"image": "a.jpg", "conf": 0.1, "x1": 0, "y1": 0, "x2": 100, "y2": 30}])
    metrics, plates = evaluate(gt, preds, 0.25)
    assert metrics["R"] == 0.0
    assert metrics["AP50"] == pytest.approx(average_precision(np.array([True]), np.array([0.1]), 3))
    assert plates.detected.tolist() == [0, 0, 0]


def test_image_groups_mixed():
    groups = image_groups(make_gt())
    assert groups.loc["a.jpg", "source"] == "data1"
    assert groups.loc["b.jpg", "corrupted"] == "corrupted"
    assert groups.loc["b.jpg", "exposure"] == "mixed"


def test_corner_error_w_wysokosciach_tablicy():
    gt = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], float)
    assert corner_error(gt, gt) == 0.0
    assert corner_error(gt + [0, 10], gt) == pytest.approx(0.5)


def test_best_matching_box():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    np.testing.assert_array_equal(best_matching_box(boxes, np.array([1, 1, 11, 11.0])), boxes[1])
    assert best_matching_box(boxes, np.array([100, 100, 110, 110.0])) is None


def test_evaluate_by_group():
    gt = make_gt()
    preds = gt[["image", "x1", "y1", "x2", "y2"]].iloc[:1].assign(conf=0.9)
    table, plates = evaluate_by_group(gt, preds, 0.25)
    rows = table.set_index(["group", "value"])
    assert rows.loc[("all", "all"), "plates"] == 3
    assert rows.loc[("source", "data1"), "R"] == 1.0
    assert rows.loc[("source", "data2"), "R"] == 0.0
    assert rows.loc[("exposure", "mixed"), "images"] == 1
    assert plates.detected.tolist() == [1, 0, 0]


def test_summarize_alignment():
    res = pd.DataFrame(
        {
            "source": ["data1", "data2"],
            "lighting": ["day", "night"],
            "exposure": ["normal", "normal"],
            "corrupted": ["original", "original"],
            "method": ["quad", "box"],
            "iou_box": [0.5, 0.8],
            "iou_aligned": [0.9, 0.8],
            "err_box": [0.4, 0.1],
            "err_aligned": [0.1, 0.1],
        }
    )
    summary = summarize_alignment(res)
    assert summary.loc[("all", "all"), "plates"] == 2
    assert summary.loc[("all", "all"), "better"] == 0.5
    assert summary.loc[("method", "quad"), "iou_aligned"] == pytest.approx(0.9)
    assert list(summary.columns) == ["plates", "iou_box", "iou_aligned", "err_box", "err_aligned", "better"]


def test_boxes_by_image():
    preds = pd.DataFrame(
        {"image": ["a", "a", "b"], "conf": [0.9, 0.1, 0.1], "x1": [1, 2, 3], "y1": 0, "x2": 5, "y2": 5}
    )
    boxes = boxes_by_image(preds, 0.25)
    assert list(boxes) == ["a"] and boxes["a"].tolist() == [[1, 0, 5, 5]]
