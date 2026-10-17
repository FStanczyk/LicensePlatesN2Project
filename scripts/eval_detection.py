#!/usr/bin/env python3
"""Ewaluacja detektora tablic: Precision, Recall, F1, AP50, AP50-95 w podziale na warunki zdjęcia.

Użycie:
    python3 scripts/eval_detection.py --weights runs/detect/plates/weights/best.pt
    python3 scripts/eval_detection.py --weights best.pt --split val --name val_baseline
    # test na zdjęciach po filtrze (Stańczyk zapisuje je z tą samą strukturą katalogów co data/combined):
    python3 scripts/eval_detection.py --weights best.pt --images-root data/filtered/clahe --name clahe
    # tylko przeliczenie metryk z zapisanych predykcji (bez modelu):
    python3 scripts/eval_detection.py --preds results/detection_eval/baseline/predictions.csv

Grupy: źródło (data1 = PL, data2 = UC3M-LP), lighting, exposure, artificial_corruption, siła
zniekształcenia oraz oryginał / kopia. Atrybuty tablicy przenosimy na zdjęcie; jeśli tablice na
jednym zdjęciu mają różne wartości, zdjęcie trafia do grupy "mixed". Predykcje liczone są z progiem
0.001 (potrzebne do AP), a P/R/F1 podajemy dla progu --conf.

Wynik w results/detection_eval/<name>/:
    predictions.csv    image, conf, x1, y1, x2, y2 (wszystkie predykcje)
    metrics.csv        group, value, images, plates, P, R, F1, AP50, AP50_95
    plates.csv         plate_id, image, detected (IoU >= 0.5 przy progu --conf), iou, conf
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from plate_detection import box_iou, detect, load_model

ROOT = Path(__file__).resolve().parent.parent
COMBINED = ROOT / "data" / "combined"
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
GROUPS = ["source", "lighting", "exposure", "artificial_corruption", "artificial_severity", "corrupted"]


def predict_all(weights: str, images: list[str], base: Path, imgsz: int) -> pd.DataFrame:
    model = load_model(weights)
    rows = []
    for i, rel in enumerate(images, 1):
        img = cv2.imread(str(base / rel))
        if img is None:
            print(f"  BŁĄD: nie da się wczytać {base / rel}")
            continue
        for det in detect(model, img, conf=0.001, imgsz=imgsz):
            rows.append({"image": rel, "conf": det.conf, **dict(zip(["x1", "y1", "x2", "y2"], det.box))})
        if i % 100 == 0:
            print(f"  {i}/{len(images)}")
    return pd.DataFrame(rows, columns=["image", "conf", "x1", "y1", "x2", "y2"])


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
    """Metryki dla zbioru zdjęć: gt = wiersze plates.csv, preds = predykcje na tych zdjęciach."""
    tps = {t: [] for t in IOU_THRESHOLDS}
    confs, plates = [], []
    pred_groups = dict(tuple(preds.groupby("image")))
    for image, g in gt.groupby("image"):
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        p = pred_groups.get(image)
        p = p.sort_values("conf", ascending=False) if p is not None else preds.iloc[:0]
        pb, pc = p[["x1", "y1", "x2", "y2"]].to_numpy(float), p.conf.to_numpy(float)
        confs.append(pc)
        for t in IOU_THRESHOLDS:
            tp, _ = match(pb, boxes, t)
            tps[t].append(tp)
        # stan tablic przy roboczym progu pewności (do P/R i analizy błędów)
        keep = pc >= conf
        _, owner = match(pb[keep], boxes, 0.5)
        for (_, row), o, box in zip(g.iterrows(), owner, boxes):
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
    """Dla każdego zdjęcia wartość każdej grupy (albo "mixed", gdy tablice się różnią)."""
    gt = gt.assign(source=gt.plate_id.str.split("_").str[0],
                   corrupted=np.where(gt.artificial_corruption == "none", "original", "corrupted"))
    agg = gt.groupby("image")[GROUPS].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")
    return agg.astype(str)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", help="wagi YOLO; niepotrzebne z --preds")
    parser.add_argument("--preds", type=Path, help="gotowy predictions.csv (pomija uruchamianie modelu)")
    parser.add_argument("--dataset", type=Path, default=COMBINED, help="katalog zbioru (z plates.csv)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--images-root", type=Path, default=None,
                        help="katalog z obrazami o tych samych ścieżkach względnych co zbiór (np. po filtrze)")
    parser.add_argument("--conf", type=float, default=0.25, help="próg pewności dla P/R/F1")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()
    if not args.weights and not args.preds:
        parser.error("podaj --weights albo --preds")

    images_root = (args.images_root or args.dataset).resolve()
    df = pd.read_csv(args.dataset / "plates.csv")
    gt = df[(df.sample_type == "full_photo") & (df.split == args.split)].copy()
    images = sorted(gt.image.unique())
    out = ROOT / "results" / "detection_eval" / args.name
    out.mkdir(parents=True, exist_ok=True)

    if args.preds:
        preds = pd.read_csv(args.preds)
    else:
        print(f"Predykcje dla {len(images)} zdjęć ({args.split}) z {images_root}")
        preds = predict_all(args.weights, images, images_root, args.imgsz)
        preds.to_csv(out / "predictions.csv", index=False)

    groups = image_groups(gt)
    overall, all_plates = evaluate(gt, preds, args.conf)
    results = [{"group": "all", "value": "all", **overall}]
    for col in GROUPS:
        for value, imgs in groups.groupby(col).groups.items():
            subset = gt[gt.image.isin(imgs)]
            metrics, _ = evaluate(subset, preds[preds.image.isin(imgs)], args.conf)
            results.append({"group": col, "value": value, **metrics})

    table = pd.DataFrame(results)
    table.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    all_plates.to_csv(out / "plates.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Wynik: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
