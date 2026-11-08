"""Testy własności (hypothesis): niezmienniki geometrii, metryk i OCR na losowych wejściach."""

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis", reason='wymaga extras [dev]: pip install -e ".[dev]"')

from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from alpr.dataset import normalize_text  # noqa: E402
from alpr.detection.metrics import average_precision, corner_error, evaluate, match  # noqa: E402
from alpr.geometry import box_corners, box_iou, four_corners, order_corners, polygon_iou, warp  # noqa: E402
from alpr.ocr import plate_fragments, reading_order  # noqa: E402

settings.register_profile("alpr", max_examples=300, deadline=None, derandomize=True)
settings.load_profile("alpr")

coord = st.floats(0, 1000, allow_nan=False)
size = st.floats(1, 300, allow_nan=False)


@st.composite
def boxes(draw):
    x, y, w, h = draw(coord), draw(coord), draw(size), draw(size)
    return np.array([x, y, x + w, y + h])


@st.composite
def plates(draw, max_angle=80.0):
    """Obrócony prostokąt o proporcjach tablicy: (narożniki TL, TR, BR, BL)."""
    w = draw(st.floats(40, 400))
    h = w / draw(st.floats(1.3, 6.0))
    t = np.deg2rad(draw(st.floats(-max_angle, max_angle)))
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
    return corners @ rot.T + [draw(st.floats(500, 1500)), draw(st.floats(500, 1500))]


@given(boxes(), boxes())
def test_box_iou_symetryczne_i_ograniczone(a, b):
    iou = box_iou(a, b)
    assert 0.0 <= iou <= 1.0 + 1e-9
    assert iou == pytest.approx(box_iou(b, a))


@given(boxes(), st.floats(-500, 500), st.floats(-500, 500), st.floats(0.1, 10))
def test_box_iou_niezmiennicze_wzgledem_przesuniecia_i_skali(a, dx, dy, s):
    b = a + [3, 1, 7, 4]
    shift = np.array([dx, dy, dx, dy])
    assert box_iou(a, a) == pytest.approx(1.0)
    assert box_iou(a + shift, b + shift) == pytest.approx(box_iou(a, b), abs=1e-6)
    assert box_iou(a * s, b * s) == pytest.approx(box_iou(a, b), abs=1e-6)


@given(boxes(), boxes())
def test_polygon_iou_zgodne_z_box_iou(a, b):
    assert polygon_iou(box_corners(a), box_corners(b)) == pytest.approx(box_iou(a, b), abs=1e-3)


@given(plates(), st.integers(0, 3), st.booleans())
def test_four_corners_niezalezne_od_startu_i_kierunku(plate, start, reverse):
    shuffled = np.roll(plate[::-1] if reverse else plate, start, axis=0)
    np.testing.assert_allclose(four_corners(shuffled), plate, atol=1e-6)


@given(plates(max_angle=40.0), st.permutations(range(4)))
def test_order_corners_dla_umiarkowanego_obrotu(plate, perm):
    np.testing.assert_allclose(order_corners(plate[list(perm)]), plate.astype(np.float32), atol=1e-3)


@given(plates(max_angle=30.0))
def test_warp_rozmiar_z_bokow(plate):
    img = np.zeros((2100, 2100, 3), np.uint8)
    out = warp(img, plate)
    tl, tr, br, bl = plate
    assert out.shape[1] == max(1, int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    assert out.shape[0] == max(1, int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))


@given(plates(), st.floats(0.2, 10), st.floats(-5, 5), st.floats(-5, 5))
def test_corner_error_nieujemny_i_niezmienniczy_wzgledem_skali(plate, s, dx, dy):
    est = plate + [dx, dy]
    err = corner_error(est, plate)
    assert err >= 0 and corner_error(plate, plate) == 0
    assert corner_error(est * s, plate * s) == pytest.approx(err, rel=1e-6, abs=1e-9)


@given(st.text())
def test_normalize_text_idempotentne(text):
    out = normalize_text(text)
    assert normalize_text(out) == out
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*" for c in out)


@given(st.lists(boxes(), max_size=8), st.lists(boxes(), max_size=8), st.floats(0.05, 0.95))
def test_match_kazda_tablica_co_najwyzej_raz(pred, gt, threshold):
    pred, gt = np.array(pred).reshape(-1, 4), np.array(gt).reshape(-1, 4)
    tp, owner = match(pred, gt, threshold)
    owned = owner[owner >= 0]
    assert len(set(owned)) == len(owned) == tp.sum()
    assert set(np.flatnonzero(tp)) == set(owned)
    for j, o in enumerate(owner):
        if o >= 0:
            assert box_iou(pred[o], gt[j]) >= threshold


@given(st.lists(st.tuples(st.booleans(), st.floats(0, 1)), min_size=1, max_size=50), st.integers(0, 10))
def test_average_precision_w_przedziale(dets, extra):
    tp = np.array([d[0] for d in dets])
    conf = np.array([d[1] for d in dets])
    ap = average_precision(tp, conf, int(tp.sum()) + extra) if tp.sum() + extra else 0.0
    assert 0.0 <= ap <= 1.0


@given(st.integers(1, 30), st.lists(st.floats(0.01, 1), min_size=30, max_size=30))
def test_average_precision_idealne_detekcje(n, conf):
    assert average_precision(np.ones(n, bool), np.array(conf[:n]), n) == pytest.approx(0.995)


@given(st.lists(boxes(), min_size=1, max_size=6), st.data())
def test_evaluate_metryki_ograniczone(gt_boxes, data):
    gt = pd.DataFrame(
        [
            {"plate_id": f"p{i}", "image": f"i{i % 2}", **dict(zip(["x1", "y1", "x2", "y2"], b, strict=True))}
            for i, b in enumerate(gt_boxes)
        ]
    )
    preds = pd.DataFrame(
        [
            {
                "image": f"i{i % 2}",
                "conf": data.draw(st.floats(0, 1)),
                **dict(zip(["x1", "y1", "x2", "y2"], b, strict=True)),
            }
            for i, b in enumerate(data.draw(st.lists(boxes(), max_size=6)))
        ],
        columns=["image", "conf", "x1", "y1", "x2", "y2"],
    )
    metrics, table = evaluate(gt, preds, 0.25)
    for key in ("P", "R", "F1", "AP50", "AP50_95"):
        assert 0.0 <= metrics[key] <= 1.0 + 1e-9, key
    assert metrics["AP50_95"] <= metrics["AP50"] + 1e-9
    assert len(table) == len(gt) and table.detected.sum() == round(metrics["R"] * len(gt))


@st.composite
def fragments(draw):
    return [
        np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
        for x, y, w, h in draw(
            st.lists(st.tuples(st.floats(0, 200), st.floats(0, 100), st.floats(1, 80), st.floats(1, 40)), min_size=1)
        )
    ]


@given(fragments())
def test_reading_order_to_permutacja(boxes_):
    assert sorted(reading_order(boxes_)) == list(range(len(boxes_)))


@given(fragments(), st.floats(0, 1))
def test_plate_fragments_zawiera_najwyzszy(boxes_, min_height):
    kept = plate_fragments(boxes_, min_height)
    heights = [np.ptp(b[:, 1]) for b in boxes_]
    assert kept == sorted(set(kept)) and int(np.argmax(heights)) in kept
    assume(min_height == 0)
    assert kept == list(range(len(boxes_)))
