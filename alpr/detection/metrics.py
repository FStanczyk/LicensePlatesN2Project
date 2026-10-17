"""Metryki detekcji (P, R, F1, AP50, AP50-95) i wyrównania (IoU, błąd narożnika).

gt: wiersze plates.csv z całymi zdjęciami, preds: image, conf, x1, y1, x2, y2.
"""

import numpy as np
import pandas as pd

from alpr.dataset import add_derived
from alpr.geometry import box_iou

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
GROUPS = ["source", "lighting", "exposure", "artificial_corruption", "artificial_severity", "corrupted"]
BOX = ["x1", "y1", "x2", "y2"]
ALIGN_METRICS = ["iou_box", "iou_aligned", "err_box", "err_aligned"]
ALIGN_GROUPS = ["source", "lighting", "exposure", "corrupted", "method"]


def match(pred: np.ndarray, gt: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Zachłanne dopasowanie predykcji (malejąco po pewności): (tp predykcji, indeks predykcji tablicy albo -1)."""
    tp = np.zeros(len(pred), dtype=bool)
    owner = np.full(len(gt), -1)
    for i, p in enumerate(pred):
        ious = [box_iou(p, g) if owner[j] < 0 else 0.0 for j, g in enumerate(gt)]
        if ious and max(ious) >= threshold:
            j = int(np.argmax(ious))
            tp[i], owner[j] = True, i
    return tp, owner


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """AP z interpolacją w 101 punktach, jak w ultralytics."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp[np.argsort(-conf, kind="stable")])
    mrec = np.concatenate([[0.0], ctp / n_gt, [1.0]])
    mpre = np.concatenate([[1.0], ctp / np.arange(1, len(tp) + 1), [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def evaluate(gt: pd.DataFrame, preds: pd.DataFrame, conf: float) -> tuple[dict, pd.DataFrame]:
    """(metryki, stan tablic przy progu conf: detected, iou, conf)."""
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
            tps[t].append(match(pb, boxes, t)[0])
        keep = pc >= conf
        _, owner = match(pb[keep], boxes, 0.5)
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


def image_groups(gt: pd.DataFrame) -> pd.DataFrame:
    """Wartość każdej grupy z GROUPS dla zdjęcia; różne wartości tablic na zdjęciu dają "mixed"."""
    if "source" not in gt or "corrupted" not in gt:
        gt = add_derived(gt)
    return gt.groupby("image")[GROUPS].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed").astype(str)


def evaluate_by_group(gt: pd.DataFrame, preds: pd.DataFrame, conf: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(metryki dla całości i każdej grupy, stan tablic)."""
    overall, plates = evaluate(gt, preds, conf)
    results = [{"group": "all", "value": "all", **overall}]
    for col, values in image_groups(gt).items():
        for value, imgs in values.groupby(values).groups.items():
            metrics, _ = evaluate(gt[gt.image.isin(imgs)], preds[preds.image.isin(imgs)], conf)
            results.append({"group": col, "value": value, **metrics})
    return pd.DataFrame(results), plates


def corner_error(est: np.ndarray, gt: np.ndarray) -> float:
    """Średnia odległość narożników w wysokościach tablicy."""
    height = (np.linalg.norm(gt[3] - gt[0]) + np.linalg.norm(gt[2] - gt[1])) / 2
    return float(np.linalg.norm(est - gt, axis=1).mean() / max(height, 1.0))


def best_matching_box(boxes: np.ndarray, gt_box: np.ndarray, min_iou: float = 0.5) -> np.ndarray | None:
    best, best_iou = None, min_iou
    for box in boxes:
        iou = box_iou(box, gt_box)
        if iou >= best_iou:
            best, best_iou = box, iou
    return best


def summarize_alignment(res: pd.DataFrame) -> pd.DataFrame:
    """Średnie w grupach; better: odsetek tablic, na których wyrównanie poprawia IoU o ponad 0.01."""
    long = pd.concat([res.assign(group="all", value="all"), *(res.assign(group=c, value=res[c]) for c in ALIGN_GROUPS)])
    long["better"] = long.iou_aligned > long.iou_box + 0.01
    by = long.groupby(["group", "value"], sort=False)
    summary = by[[*ALIGN_METRICS, "better"]].mean()
    summary.insert(0, "plates", by.size())
    return summary
