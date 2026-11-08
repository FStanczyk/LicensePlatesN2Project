"""Zoptymalizowane metryki i match_results dają to samo co poprzednie wersje (pętle po parach) na losowych danych.

Wzorce poniżej to kod sprzed optymalizacji (08.11.2026), porównanie bez tolerancji.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from alpr.detection.metrics import (
    BOX,
    GROUPS,
    IOU_THRESHOLDS,
    average_precision,
    best_conf,
    evaluate,
    evaluate_by_group,
    image_groups,
    match,
)
from alpr.experiment import RESULT_FIELDS, match_results
from alpr.geometry import box_iou


def old_match(pred, gt, threshold):
    tp = np.zeros(len(pred), dtype=bool)
    owner = np.full(len(gt), -1)
    for i, p in enumerate(pred):
        ious = [box_iou(p, g) if owner[j] < 0 else 0.0 for j, g in enumerate(gt)]
        if ious and max(ious) >= threshold:
            j = int(np.argmax(ious))
            tp[i], owner[j] = True, i
    return tp, owner


def old_evaluate(gt, preds, conf):
    tps = {t: [] for t in IOU_THRESHOLDS}
    confs, plates = [], []
    pred_groups = dict(tuple(preds.groupby("image")))
    for image, g in gt.groupby("image"):
        boxes = g[BOX].to_numpy(float)
        p = pred_groups.get(image)
        p = p.sort_values("conf", ascending=False) if p is not None else preds.iloc[:0]
        pb, pc = p[BOX].to_numpy(float), p.conf.to_numpy(float)
        confs.append(pc)
        for t in IOU_THRESHOLDS:
            tps[t].append(old_match(pb, boxes, t)[0])
        keep = pc >= conf
        _, owner = old_match(pb[keep], boxes, 0.5)
        for plate_id, o, box in zip(g.plate_id, owner, boxes, strict=True):
            best = max((box_iou(b, box) for b in pb[keep]), default=0.0)
            plates.append(
                {
                    "plate_id": plate_id,
                    "image": image,
                    "detected": int(o >= 0),
                    "iou": round(best, 3),
                    "conf": round(pc[keep][o], 3) if o >= 0 else "",
                }
            )
    conf_all = np.concatenate(confs) if confs else np.zeros(0)
    n_gt = len(gt)
    aps = [average_precision(np.concatenate(tps[t]), conf_all, n_gt) for t in IOU_THRESHOLDS]
    tp50 = np.concatenate(tps[IOU_THRESHOLDS[0]])[conf_all >= conf] if len(conf_all) else np.zeros(0)
    p = tp50.sum() / len(tp50) if len(tp50) else 0.0
    r = tp50.sum() / n_gt if n_gt else float("nan")
    metrics = {
        "images": gt.image.nunique(),
        "plates": n_gt,
        "P": p,
        "R": r,
        "F1": 2 * p * r / (p + r) if p + r > 0 else 0.0,
        "AP50": aps[0],
        "AP50_95": float(np.mean(aps)),
    }
    return metrics, pd.DataFrame(plates)


def old_image_groups(gt):
    return gt.groupby("image")[GROUPS].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed").astype(str)


def old_evaluate_by_group(gt, preds, conf):
    overall, plates = old_evaluate(gt, preds, conf)
    results = [{"group": "all", "value": "all", **overall}]
    for col, values in old_image_groups(gt).items():
        for value, imgs in values.groupby(values).groups.items():
            metrics, _ = old_evaluate(gt[gt.image.isin(imgs)], preds[preds.image.isin(imgs)], conf)
            results.append({"group": col, "value": value, **metrics})
    return pd.DataFrame(results), plates


def old_best_conf(gt, preds, iou=0.5):
    pred_groups = dict(tuple(preds.groupby("image")))
    tps, confs = [], []
    for image, g in gt.groupby("image"):
        p = pred_groups.get(image)
        if p is not None:
            p = p.sort_values("conf", ascending=False)
            tps.append(old_match(p[BOX].to_numpy(float), g[BOX].to_numpy(float), iou)[0])
            confs.append(p.conf.to_numpy(float))
    if not tps:
        return 0.0, 0.0
    order = np.argsort(-np.concatenate(confs), kind="stable")
    tp, conf = np.concatenate(tps)[order], np.concatenate(confs)[order]
    f1 = 2 * np.cumsum(tp) / (np.arange(1, len(tp) + 1) + len(gt))
    f1[np.r_[conf[1:] == conf[:-1], False]] = -1
    i = int(np.argmax(f1))
    return float(conf[i]), float(f1[i])


def old_match_results(gt, ocr):
    dets = dict(tuple(ocr[ocr.det_id.notna()].groupby("image")))
    rows = []
    for image, g in gt.groupby("image", sort=False):
        d = dets.get(image, ocr.iloc[:0]).sort_values("conf", ascending=False)
        boxes, pred = g[BOX].to_numpy(float), d[BOX].to_numpy(float)
        _, owner = old_match(pred, boxes, 0.5)
        for (_, plate), o, box in zip(g.iterrows(), owner, boxes, strict=True):
            det = d.iloc[o] if o >= 0 else None
            text_pred = det["text"] if det is not None and isinstance(det["text"], str) else ""
            rows.append(
                {
                    **plate[["variant", "artificial_corruption", "artificial_severity", "image", "plate_id"]],
                    **plate[["split", "exposure"]],
                    "text_true": plate.text,
                    "detected": int(o >= 0),
                    "iou": round(max((box_iou(p, box) for p in pred), default=0.0), 3),
                    "det_conf": det["conf"] if det is not None else np.nan,
                    "align": det["align"] if det is not None else "",
                    "crop": det["crop"] if det is not None else "",
                    "text_pred": text_pred,
                    "conf_ocr": det["conf_ocr"] if det is not None else np.nan,
                    "text_ok": int(text_pred == plate.text),
                }
            )
    return pd.DataFrame(rows, columns=RESULT_FIELDS)


def random_data(seed: int, n_images: int = 40, per_image: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """gt (1-3 tablice na zdjęcie, grupy z NaN) i predykcje z remisami pewności, część bez gt i bez predykcji."""
    rng = np.random.default_rng(seed)
    gt_rows, pred_rows = [], []
    for k in range(n_images):
        image = f"img_{k:03d}.jpg"
        boxes = []
        for n in range(rng.integers(1, 4)):
            x, y = rng.uniform(0, 800, 2)
            box = [x, y, x + rng.uniform(40, 200), y + rng.uniform(10, 60)]
            boxes.append(box)
            gt_rows.append(
                {
                    "image": image,
                    "plate_id": f"p{k}_{n}",
                    **dict(zip(BOX, box, strict=True)),
                    "source": rng.choice(["data1", "data2"]),
                    "lighting": rng.choice(["day", "night", None]),
                    "exposure": rng.choice(["under", "normal", "over"]),
                    "artificial_corruption": rng.choice(["none", "darken"]),
                    "artificial_severity": int(rng.integers(0, 3)),
                    "corrupted": int(rng.integers(0, 2)),
                }
            )
        if k % 7 == 3:
            continue
        for _ in range(rng.integers(0, per_image)):
            if rng.random() < 0.4:
                b = np.array(boxes[rng.integers(len(boxes))])
                b = b + rng.normal(0, 0.12, 4) * (b[2] - b[0])
            else:
                x, y = rng.uniform(0, 900, 2)
                b = [x, y, x + rng.uniform(5, 300), y + rng.uniform(5, 80)]
            pred_rows.append({"image": image, "conf": round(rng.random(), 1), **dict(zip(BOX, b, strict=True))})
    pred_rows += [{"image": "spoza.jpg", "conf": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    return pd.DataFrame(gt_rows), pd.DataFrame(pred_rows).sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.mark.parametrize("seed", range(8))
def test_match_jak_poprzednio(seed):
    gt, preds = random_data(seed)
    for t in (0.3, 0.5, 0.75):
        pb, gb = preds[BOX].to_numpy(float), gt[BOX].to_numpy(float)[:6]
        for new, old in zip(match(pb[:30], gb, t), old_match(pb[:30], gb, t), strict=True):
            assert np.array_equal(new, old)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("conf", [0.0, 0.25, 0.5])
def test_evaluate_i_grupy_jak_poprzednio(seed, conf):
    gt, preds = random_data(seed)
    metrics, plates = evaluate(gt, preds, conf)
    old_metrics, old_plates = old_evaluate(gt, preds, conf)
    assert metrics == old_metrics
    assert_frame_equal(plates, old_plates, check_exact=True)
    table, plates = evaluate_by_group(gt, preds, conf)
    old_table, old_plates = old_evaluate_by_group(gt, preds, conf)
    assert_frame_equal(table, old_table, check_exact=True)
    assert_frame_equal(plates, old_plates, check_exact=True)


@pytest.mark.parametrize("seed", range(8))
def test_best_conf_i_image_groups_jak_poprzednio(seed):
    gt, preds = random_data(seed)
    assert best_conf(gt, preds) == old_best_conf(gt, preds)
    assert best_conf(gt, preds, 0.7) == old_best_conf(gt, preds, 0.7)
    assert_frame_equal(image_groups(gt), old_image_groups(gt), check_exact=True)


def test_evaluate_bez_predykcji_jak_poprzednio():
    gt, preds = random_data(0)
    empty = preds.iloc[:0]
    assert evaluate(gt, empty, 0.25)[0] == old_evaluate(gt, empty, 0.25)[0]
    assert best_conf(gt, empty) == old_best_conf(gt, empty) == (0.0, 0.0)


@pytest.mark.parametrize("seed", range(8))
def test_match_results_jak_poprzednio(seed):
    gt, preds = random_data(seed)
    rng = np.random.default_rng(seed)
    gt = gt.assign(
        variant="clean",
        artificial_corruption="none",
        artificial_severity=0,
        split="test",
        text=[f"AB{i}" for i in range(len(gt))],
    )
    ocr = preds[preds.image != "spoza.jpg"].assign(
        det_id=lambda d: d.groupby("image").cumcount(),
        align="quad",
        crop=lambda d: [f"crops/{i}.png" for i in range(len(d))],
        text=lambda d: rng.choice(["AB1", "AB2", None], len(d)),
        conf_ocr=lambda d: rng.random(len(d)),
    )
    ocr = pd.concat([ocr, pd.DataFrame([{"image": gt.image.iloc[-1]}])], ignore_index=True)
    ocr = ocr.astype({"det_id": "Int64"})
    shuffled = gt.sample(frac=1, random_state=seed)
    assert_frame_equal(match_results(shuffled, ocr), old_match_results(shuffled, ocr), check_exact=True)
