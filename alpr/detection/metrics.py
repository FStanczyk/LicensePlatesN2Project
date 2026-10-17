"""Metryki detekcji (P, R, F1, AP50, AP50-95) i wyrównania (błąd narożnika).

Wejście: gt = wiersze plates.csv z całymi zdjęciami (image, x1, y1, x2, y2, ...),
preds = predykcje (image, conf, x1, y1, x2, y2), jak w predictions.csv z eval_detection.py.
"""

import numpy as np
import pandas as pd

from alpr.dataset import add_derived
from alpr.geometry import box_iou

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
GROUPS = ["source", "lighting", "exposure", "artificial_corruption", "artificial_severity", "corrupted"]
BOX = ["x1", "y1", "x2", "y2"]


def match(pred: np.ndarray, gt: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Zachłanne dopasowanie predykcji (posortowanych malejąco po pewności) do tablic.
    Zwraca (tp dla każdej predykcji, indeks dopasowanej predykcji dla każdej tablicy albo -1)."""
    tp = np.zeros(len(pred), dtype=bool)
    owner = np.full(len(gt), -1)
    for i, p in enumerate(pred):
        ious = [box_iou(p, g) if owner[j] < 0 else 0.0 for j, g in enumerate(gt)]
        if ious and max(ious) >= threshold:
            j = int(np.argmax(ious))
            tp[i], owner[j] = True, i
    return tp, owner


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """AP z interpolacją w 101 punktach (jak ultralytics / COCO)."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    tp = tp[np.argsort(-conf, kind="stable")]
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(tp) + 1)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def evaluate(gt: pd.DataFrame, preds: pd.DataFrame, conf: float) -> tuple[dict, pd.DataFrame]:
    """Metryki dla zbioru zdjęć: gt = wiersze plates.csv, preds = predykcje na tych zdjęciach.
    Zwraca (metryki, stan każdej tablicy przy progu conf: detected, iou, conf)."""
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
            tp, _ = match(pb, boxes, t)
            tps[t].append(tp)
        # stan tablic przy roboczym progu pewności (do P/R i analizy błędów)
        keep = pc >= conf
        _, owner = match(pb[keep], boxes, 0.5)
        for (_, row), o, box in zip(g.iterrows(), owner, boxes, strict=True):
            best = max((box_iou(b, box) for b in pb[keep]), default=0.0)
            plates.append({"plate_id": row.plate_id, "image": image, "detected": int(o >= 0),
                           "iou": round(best, 3), "conf": round(pc[keep][o], 3) if o >= 0 else ""})
    conf_all = np.concatenate(confs) if confs else np.zeros(0)
    n_gt = len(gt)
    aps = [average_precision(np.concatenate(tps[t]), conf_all, n_gt) for t in IOU_THRESHOLDS]
    tp50 = np.concatenate(tps[IOU_THRESHOLDS[0]])[conf_all >= conf] if len(conf_all) else np.zeros(0)
    p = tp50.sum() / len(tp50) if len(tp50) else 0.0
    r = tp50.sum() / n_gt if n_gt else float("nan")
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    metrics = {"images": gt.image.nunique(), "plates": n_gt, "P": p, "R": r, "F1": f1,
               "AP50": aps[0], "AP50_95": float(np.mean(aps))}
    return metrics, pd.DataFrame(plates)


def image_groups(gt: pd.DataFrame) -> pd.DataFrame:
    """Dla każdego zdjęcia wartość każdej grupy z GROUPS. Atrybuty tablicy przenosimy na zdjęcie;
    jeśli tablice na jednym zdjęciu mają różne wartości, zdjęcie trafia do grupy "mixed"."""
    if "source" not in gt or "corrupted" not in gt:
        gt = add_derived(gt)
    agg = gt.groupby("image")[GROUPS].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")
    return agg.astype(str)


def corner_error(est: np.ndarray, gt: np.ndarray) -> float:
    """Średnia odległość odpowiadających sobie narożników, w wysokościach tablicy."""
    height = (np.linalg.norm(gt[3] - gt[0]) + np.linalg.norm(gt[2] - gt[1])) / 2
    return float(np.linalg.norm(est - gt, axis=1).mean() / max(height, 1.0))


def best_matching_box(boxes: np.ndarray, gt_box: np.ndarray, min_iou: float = 0.5) -> np.ndarray | None:
    """Ramka z boxes (Nx4) najlepiej pokrywająca się z gt_box (IoU >= min_iou) albo None."""
    best, best_iou = None, min_iou
    for box in boxes:
        iou = box_iou(box, gt_box)
        if iou >= best_iou:
            best, best_iou = box, iou
    return best
