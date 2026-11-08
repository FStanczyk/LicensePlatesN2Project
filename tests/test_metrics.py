import numpy as np
import pandas as pd
import pytest

from alpr.detection.metrics import (
    average_precision,
    best_conf,
    best_match,
    boxes_by_image,
    corner_error,
    desc_order,
    evaluate,
    evaluate_by_group,
    greedy_match,
    image_groups,
    image_matches,
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


def test_best_match():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert best_match(boxes, np.array([1, 1, 11, 11.0])) == 1
    assert best_match(boxes, np.array([100, 100, 110, 110.0])) is None


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


def test_match_prog_iou_wlacznie():
    gt = np.array([[0, 0, 10, 10]], float)
    pred = np.array([[0, 0, 10, 5]], float)
    assert match(pred, gt, 0.5)[0].tolist() == [True]
    assert match(pred, gt, 0.51)[0].tolist() == [False]


def test_match_puste_wejscia():
    tp, owner = match(np.zeros((0, 4)), np.array([[0, 0, 10, 10]], float), 0.5)
    assert tp.tolist() == [] and owner.tolist() == [-1]
    tp, owner = match(np.array([[0, 0, 10, 10]], float), np.zeros((0, 4)), 0.5)
    assert tp.tolist() == [False] and owner.tolist() == []


def test_match_predykcja_wybiera_najlepiej_pokryta_tablice():
    gt = np.array([[0, 0, 10, 10], [4, 0, 14, 10]], float)
    tp, owner = match(np.array([[5, 0, 15, 10]], float), gt, 0.5)
    assert tp.tolist() == [True] and owner.tolist() == [-1, 0]


def test_average_precision_polowa_tablic():
    assert average_precision(np.array([True]), np.array([0.9]), 2) == pytest.approx(0.75)


def test_average_precision_same_bledy():
    assert average_precision(np.array([False, False]), np.array([0.9, 0.8]), 2) == pytest.approx(0.0)


def test_average_precision_niezalezne_od_kolejnosci_wejscia():
    rng = np.random.default_rng(0)
    tp = rng.random(30) > 0.4
    conf = rng.random(30)
    perm = rng.permutation(30)
    assert average_precision(tp[perm], conf[perm], 25) == pytest.approx(average_precision(tp, conf, 25))


def test_average_precision_w_przedziale_0_1():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(1, 40))
        tp = rng.random(n) > 0.5
        ap = average_precision(tp, rng.random(n), int(tp.sum()) + int(rng.integers(0, 5)))
        assert 0.0 <= ap <= 1.0


def single_gt():
    return make_gt().iloc[:1]


def pred(image, conf, x1, y1, x2, y2):
    return {"image": image, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_evaluate_falszywy_alarm_obniza_precyzje():
    preds = pd.DataFrame([pred("a.jpg", 0.9, 0, 0, 100, 30), pred("a.jpg", 0.8, 300, 300, 400, 330)])
    metrics, plates = evaluate(single_gt(), preds, 0.25)
    assert (metrics["P"], metrics["R"]) == (0.5, 1.0)
    assert metrics["F1"] == pytest.approx(2 / 3)
    assert metrics["AP50"] == pytest.approx(0.995)
    assert plates.conf.tolist() == [0.9]


def test_evaluate_podwojna_detekcja_to_falszywy_alarm():
    preds = pd.DataFrame([pred("a.jpg", 0.9, 0, 0, 100, 30), pred("a.jpg", 0.8, 1, 0, 101, 30)])
    metrics, _ = evaluate(single_gt(), preds, 0.25)
    assert metrics["P"] == 0.5 and metrics["R"] == 1.0


def test_evaluate_przesunieta_ramka_obniza_ap50_95_a_nie_ap50():
    preds = pd.DataFrame([pred("a.jpg", 0.9, 10, 0, 110, 30)])
    metrics, plates = evaluate(single_gt(), preds, 0.25)
    assert metrics["AP50"] == pytest.approx(0.995)
    assert metrics["AP50_95"] < metrics["AP50"]
    assert plates.iou.tolist() == [round(90 / 110, 3)]


def test_evaluate_niedetekcja_ma_iou_najlepszej_ramki_i_pusty_conf():
    preds = pd.DataFrame([pred("a.jpg", 0.9, 60, 0, 160, 30)])
    _, plates = evaluate(single_gt(), preds, 0.25)
    row = plates.iloc[0]
    assert row.detected == 0 and row.conf == "" and 0 < row.iou < 0.5


def test_evaluate_bez_zadnych_predykcji():
    metrics, plates = evaluate(make_gt(), pd.DataFrame(columns=["image", "conf", "x1", "y1", "x2", "y2"]), 0.25)
    assert (metrics["P"], metrics["R"], metrics["F1"], metrics["AP50"]) == (0.0, 0.0, 0.0, 0.0)
    assert (metrics["images"], metrics["plates"]) == (2, 3)
    assert plates.detected.tolist() == [0, 0, 0]


def test_evaluate_prog_conf_nie_zmienia_ap():
    gt = make_gt()
    preds = gt[["image", "x1", "y1", "x2", "y2"]].assign(conf=[0.9, 0.5, 0.1])
    low, _ = evaluate(gt, preds, 0.05)
    high, _ = evaluate(gt, preds, 0.6)
    assert low["AP50"] == high["AP50"]
    assert (low["R"], high["R"]) == (1.0, pytest.approx(1 / 3))


def test_evaluate_by_group_liczba_wierszy():
    gt = make_gt()
    table, _ = evaluate_by_group(gt, gt[["image", "x1", "y1", "x2", "y2"]].assign(conf=0.9), 0.25)
    expected = 1 + image_groups(gt).nunique().sum()
    assert len(table) == expected
    assert (table.R == 1.0).all()


def test_image_groups_dodaje_kolumny_pochodne_tylko_gdy_brak():
    gt = make_gt().assign(source="inne", corrupted="x")
    assert image_groups(gt).loc["a.jpg", "source"] == "inne"


def test_corner_error_niezalezny_od_skali():
    gt = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], float)
    est = gt + [3, -2]
    assert corner_error(est * 4, gt * 4) == pytest.approx(corner_error(est, gt))


def test_corner_error_zdegenerowana_tablica_nie_dzieli_przez_zero():
    gt = np.zeros((4, 2))
    assert corner_error(gt + 1, gt) == pytest.approx(np.sqrt(2))


def test_best_match_wybiera_najwyzsze_iou_nie_pierwsze():
    target = np.array([0, 0, 10, 10.0])
    boxes = np.array([[2, 0, 12, 10], [0, 0, 10, 10], [1, 0, 11, 10]], float)
    assert best_match(boxes, target) == 1
    assert best_match(np.zeros((0, 4)), target) is None


def test_summarize_alignment_better_wymaga_poprawy_ponad_0_01():
    res = pd.DataFrame(
        {
            "source": ["d", "d"],
            "lighting": "day",
            "exposure": "normal",
            "corrupted": "original",
            "method": "quad",
            "iou_box": [0.5, 0.5],
            "iou_aligned": [0.505, 0.52],
            "err_box": 0.1,
            "err_aligned": 0.1,
        }
    )
    assert summarize_alignment(res).loc[("all", "all"), "better"] == 0.5


def test_boxes_by_image_prog_wlacznie():
    preds = pd.DataFrame([pred("a", 0.25, 0, 0, 1, 1)])
    assert list(boxes_by_image(preds, 0.25)) == ["a"]


def test_best_conf_odcina_falszywe_alarmy():
    gt = make_gt()
    good = gt[["image", "x1", "y1", "x2", "y2"]].assign(conf=[0.9, 0.8, 0.7])
    noise = pd.DataFrame([pred("a.jpg", c, 300, 300, 400, 330) for c in (0.5, 0.4, 0.3)])
    conf, f1 = best_conf(gt, pd.concat([good, noise]))
    assert (conf, f1) == (0.7, 1.0)


def test_best_conf_zgodne_z_evaluate():
    gt = make_gt()
    preds = pd.DataFrame(
        [pred("a.jpg", 0.9, 0, 0, 100, 30), pred("b.jpg", 0.6, 300, 0, 400, 30), pred("b.jpg", 0.5, 50, 50, 150, 80)]
    )
    conf, f1 = best_conf(gt, preds)
    assert f1 == pytest.approx(evaluate(gt, preds, conf)[0]["F1"])
    assert all(evaluate(gt, preds, c)[0]["F1"] <= f1 + 1e-9 for c in (0.1, 0.55, 0.7, 0.95))


def test_best_conf_rowne_pewnosci_liczone_razem():
    gt = single_gt()
    preds = pd.DataFrame([pred("a.jpg", 0.5, 0, 0, 100, 30), pred("a.jpg", 0.5, 300, 300, 400, 330)])
    assert best_conf(gt, preds) == (0.5, pytest.approx(2 / 3))


def test_best_conf_bez_predykcji():
    assert best_conf(make_gt(), pd.DataFrame(columns=["image", "conf", "x1", "y1", "x2", "y2"])) == (0.0, 0.0)


def test_desc_order_jak_sort_values_z_remisami_i_nan():
    rng = np.random.default_rng(0)
    for n in (0, 1, 5, 17, 200):
        conf = rng.integers(0, 5, n) / 4
        conf[rng.random(n) < 0.1] = np.nan
        expected = pd.Series(conf).sort_values(ascending=False).index.to_numpy()
        assert np.array_equal(desc_order(conf), expected)


def test_greedy_match_zachlanny_i_prog_wlacznie():
    ious = np.array([[0.6, 0.5], [0.9, 0.5], [0.4, 0.49]])
    tp, owner = greedy_match(ious, 0.5)
    assert tp.tolist() == [True, True, False]
    assert owner.tolist() == [0, 1]
    tp, owner = greedy_match(np.zeros((3, 0)), 0.5)
    assert tp.tolist() == [False] * 3 and owner.tolist() == []


def test_image_matches_kolejnosc_i_pozycje_tablic():
    gt = pd.DataFrame(
        {"image": ["b", "a", "b"], "x1": [0, 0, 50], "y1": [0, 0, 0], "x2": [10, 10, 60], "y2": [10, 10, 10]}
    )
    preds = pd.DataFrame(
        {
            "image": ["b", "b", "c"],
            "conf": [0.3, 0.8, 0.9],
            "x1": [50, 0, 0],
            "y1": [0] * 3,
            "x2": [60, 10, 10],
            "y2": [10] * 3,
        }
    )
    a, b = image_matches(gt, preds, [0.5, 0.95])
    assert (a.image, b.image) == ("a", "b")
    assert a.plates.tolist() == [1] and len(a.conf) == 0 and a.ious.shape == (0, 1)
    assert b.plates.tolist() == [0, 2] and b.conf.tolist() == [0.8, 0.3]
    assert b.ious.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert [t.tolist() for t in b.tp] == [[True, True], [True, True]]
