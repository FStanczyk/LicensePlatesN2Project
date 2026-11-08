"""Metryki detekcji (P, R, F1, AP50, AP50-95) i wyrównania (IoU, błąd narożnika).

gt: wiersze plates.csv z całymi zdjęciami, preds: image, conf, x1, y1, x2, y2.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpr.dataset import add_derived
from alpr.geometry import box_iou, box_iou_matrix

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
GROUPS = ["source", "lighting", "exposure", "artificial_corruption", "artificial_severity", "corrupted"]
BOX = ["x1", "y1", "x2", "y2"]
ALIGN_METRICS = ["iou_box", "iou_aligned", "err_box", "err_aligned"]
ALIGN_GROUPS = ["source", "lighting", "exposure", "corrupted", "method"]


def desc_order(conf: np.ndarray) -> np.ndarray:
    """Kolejność malejąco po pewności dokładnie jak DataFrame.sort_values(ascending=False), z remisami i NaN."""
    nan = np.isnan(conf)
    idx = np.flatnonzero(~nan)[::-1]
    return np.concatenate([idx[conf[idx].argsort(kind="quicksort")][::-1], np.flatnonzero(nan)])


def greedy_match(ious: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """match na macierzy IoU (predykcje malejąco po pewności x tablice)."""
    tp = np.zeros(ious.shape[0], dtype=bool)
    owner = np.full(ious.shape[1], -1)
    if ious.shape[1] == 0:
        return tp, owner
    for i in np.flatnonzero(ious.max(axis=1) >= threshold):
        row = np.where(owner < 0, ious[i], 0.0)
        j = int(np.argmax(row))
        if row[j] >= threshold:
            tp[i], owner[j] = True, i
    return tp, owner


def match(pred: np.ndarray, gt: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Zachłanne dopasowanie predykcji (malejąco po pewności): (tp predykcji, indeks predykcji tablicy albo -1)."""
    return greedy_match(box_iou_matrix(np.asarray(pred).reshape(-1, 4), np.asarray(gt).reshape(-1, 4)), threshold)


@dataclass(frozen=True)
class ImageMatch:
    """Predykcje zdjęcia malejąco po pewności, ich IoU z tablicami (pozycje w gt) i tp dla każdego progu."""

    image: str
    plates: np.ndarray
    conf: np.ndarray
    ious: np.ndarray
    tp: list[np.ndarray]


def image_matches(gt: pd.DataFrame, preds: pd.DataFrame, thresholds) -> list[ImageMatch]:
    """Dopasowania dla każdego zdjęcia z gt (kolejność gt.groupby); liczone raz dla wszystkich progów i grup."""
    gt_box = gt[BOX].to_numpy(float)
    pred_box, pred_conf = preds[BOX].to_numpy(float), preds.conf.to_numpy(float)
    by_image = preds.groupby("image").indices
    result = []
    for image, plates in gt.groupby("image").indices.items():
        p = by_image.get(image, np.zeros(0, dtype=int))
        p = p[desc_order(pred_conf[p])]
        ious = box_iou_matrix(pred_box[p], gt_box[plates])
        result.append(ImageMatch(image, plates, pred_conf[p], ious, [greedy_match(ious, t)[0] for t in thresholds]))
    return result


def _scores(matches: list[ImageMatch], n_gt: int, conf: float) -> dict:
    conf_all = np.concatenate([m.conf for m in matches]) if matches else np.zeros(0)
    tps = [
        np.concatenate([m.tp[k] for m in matches]) if matches else np.zeros(0, bool) for k in range(len(IOU_THRESHOLDS))
    ]
    aps = [average_precision(tp, conf_all, n_gt) for tp in tps]
    tp50 = tps[0][conf_all >= conf] if len(conf_all) else np.zeros(0)
    p = tp50.sum() / len(tp50) if len(tp50) else 0.0
    r = tp50.sum() / n_gt if n_gt else float("nan")
    return {
        "images": len(matches),
        "plates": n_gt,
        "P": p,
        "R": r,
        "F1": 2 * p * r / (p + r) if p + r > 0 else 0.0,
        "AP50": aps[0],
        "AP50_95": float(np.mean(aps)),
    }


def _plate_states(gt: pd.DataFrame, matches: list[ImageMatch], conf: float) -> pd.DataFrame:
    plate_id = gt.plate_id.to_numpy()
    rows = []
    for m in matches:
        keep = m.conf >= conf
        ious, kept_conf = m.ious[keep], m.conf[keep]
        _, owner = greedy_match(ious, 0.5)
        best = ious.max(axis=0) if len(ious) else np.zeros(len(m.plates))
        for g, o, iou in zip(m.plates, owner, best, strict=True):
            rows.append(
                {
                    "plate_id": plate_id[g],
                    "image": m.image,
                    "detected": int(o >= 0),
                    "iou": round(iou, 3),
                    "conf": round(kept_conf[o], 3) if o >= 0 else "",
                }
            )
    return pd.DataFrame(rows)


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
    matches = image_matches(gt, preds, IOU_THRESHOLDS)
    return _scores(matches, len(gt), conf), _plate_states(gt, matches, conf)


def best_conf(gt: pd.DataFrame, preds: pd.DataFrame, iou: float = 0.5) -> tuple[float, float]:
    """(próg pewności, F1) maksymalizujący F1 przy IoU >= iou; dobierać na val, stosować na test."""
    matches = [m for m in image_matches(gt, preds, [iou]) if len(m.conf)]
    if not matches:
        return 0.0, 0.0
    confs = np.concatenate([m.conf for m in matches])
    order = np.argsort(-confs, kind="stable")
    tp, conf = np.concatenate([m.tp[0] for m in matches])[order], confs[order]
    f1 = 2 * np.cumsum(tp) / (np.arange(1, len(tp) + 1) + len(gt))
    f1[np.r_[conf[1:] == conf[:-1], False]] = -1
    i = int(np.argmax(f1))
    return float(conf[i]), float(f1[i])


def image_groups(gt: pd.DataFrame) -> pd.DataFrame:
    """Wartość każdej grupy z GROUPS dla zdjęcia; różne wartości tablic na zdjęciu dają "mixed"."""
    if "source" not in gt or "corrupted" not in gt:
        gt = add_derived(gt)
    first = gt.drop_duplicates("image").set_index("image")[GROUPS]
    unique = gt.groupby("image")[GROUPS].nunique()
    return first.reindex(unique.index).where(unique == 1, "mixed").astype(str)


def evaluate_by_group(gt: pd.DataFrame, preds: pd.DataFrame, conf: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(metryki dla całości i każdej grupy, stan tablic); dopasowania liczone raz i dzielone między grupy."""
    matches = image_matches(gt, preds, IOU_THRESHOLDS)
    position = {m.image: k for k, m in enumerate(matches)}
    results = [{"group": "all", "value": "all", **_scores(matches, len(gt), conf)}]
    for col, values in image_groups(gt).items():
        for value, imgs in values.groupby(values).groups.items():
            subset = [matches[k] for k in sorted(position[i] for i in imgs)]
            results.append({"group": col, "value": value, **_scores(subset, sum(len(m.plates) for m in subset), conf)})
    return pd.DataFrame(results), _plate_states(gt, matches, conf)


def corner_error(est: np.ndarray, gt: np.ndarray) -> float:
    """Średnia odległość narożników w wysokościach tablicy."""
    height = (np.linalg.norm(gt[3] - gt[0]) + np.linalg.norm(gt[2] - gt[1])) / 2
    return float(np.linalg.norm(est - gt, axis=1).mean() / max(height, 1.0))


def best_match(boxes: np.ndarray, gt_box: np.ndarray, min_iou: float = 0.5) -> int | None:
    """Indeks ramki o największym IoU z gt_box (co najmniej min_iou) albo None."""
    best, best_iou = None, min_iou
    for i, box in enumerate(boxes):
        iou = box_iou(box, gt_box)
        if iou >= best_iou:
            best, best_iou = i, iou
    return best


def boxes_by_image(preds: pd.DataFrame, conf: float) -> dict[str, np.ndarray]:
    """Ramki predykcji z pewnością >= conf dla każdego zdjęcia."""
    kept = preds[preds.conf >= conf]
    return {image: g[BOX].to_numpy(float) for image, g in kept.groupby("image")}


def summarize_alignment(res: pd.DataFrame) -> pd.DataFrame:
    """Średnie w grupach; better: odsetek tablic, na których wyrównanie poprawia IoU o ponad 0.01."""
    long = pd.concat([res.assign(group="all", value="all"), *(res.assign(group=c, value=res[c]) for c in ALIGN_GROUPS)])
    long["better"] = long.iou_aligned > long.iou_box + 0.01
    by = long.groupby(["group", "value"], sort=False)
    summary = by[[*ALIGN_METRICS, "better"]].mean()
    summary.insert(0, "plates", by.size())
    return summary
