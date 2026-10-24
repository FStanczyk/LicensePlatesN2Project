"""Zbiór eksperymentu (Golba): czyste zdjęcia z etykietami, ich zniekształcone wersje, detekcja i OCR na wszystkich.

Ground truth: całe zdjęcia z ramką i czytelnym tekstem (bez blurred), same oryginały, tylko podziały
nieużyte do uczenia wag YOLO. Zniekształcenia nie zmieniają geometrii, więc etykiety są wspólne dla wersji.

    data/experiment/images/<wersja>/<zdjęcie>.jpg   wersja: clean albo <zniekształcenie>_<siła>
    data/experiment/ground_truth.csv                jedna linia na (wersja, tablica)
"""

import shutil
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd

from alpr.dataset import load_plates
from alpr.detection.metrics import BOX, match
from alpr.geometry import box_iou

SPLITS = ("val", "test")
CORRUPTIONS = ("brightness", "darken", "motion_blur", "defocus_blur")
SEVERITIES = (1, 2, 3, 4, 5)
GT_FIELDS = [
    "variant",
    "artificial_corruption",
    "artificial_severity",
    "image",
    "source_image",
    "plate_id",
    "split",
    "text",
    "country",
    "exposure",
    *BOX,
    "polygon",
]
RESULT_FIELDS = [
    "variant",
    "artificial_corruption",
    "artificial_severity",
    "image",
    "plate_id",
    "split",
    "exposure",
    "text_true",
    "detected",
    "iou",
    "det_conf",
    "align",
    "crop",
    "text_pred",
    "conf_ocr",
    "text_ok",
]


def variants(corruptions: tuple[str, ...] = CORRUPTIONS) -> list[tuple[str, str, int]]:
    """(nazwa katalogu, zniekształcenie, siła); pierwsza zawsze clean."""
    return [("clean", "none", 0)] + [(f"{c}_{s}", c, s) for c in corruptions for s in SEVERITIES]


def ground_truth(dataset: Path, splits: tuple[str, ...] = SPLITS) -> pd.DataFrame:
    """Oryginalne całe zdjęcia z czytelnym tekstem z podanych podziałów."""
    df = load_plates(dataset, sample_type="full_photo", originals_only=True, ocr_only=True)
    return df[df.split.isin(splits)]


def build_experiment(
    dataset: Path, out: Path, splits: tuple[str, ...] = SPLITS, corruptions: tuple[str, ...] = CORRUPTIONS
) -> pd.DataFrame:
    """Zapisuje wersje zdjęć w out/images i zwraca ground truth dla wszystkich wersji."""
    from alpr.build.corruptions import apply_corruption

    gt = ground_truth(dataset, splits)
    rows = []
    for i, (image, plates) in enumerate(gt.groupby("image"), 1):
        name = PurePosixPath(image).name
        img = cv2.imread(str(dataset / image))
        for variant, corruption, severity in variants(corruptions):
            rel = f"images/{variant}/{name}"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            if severity == 0:
                shutil.copyfile(dataset / image, out / rel)
            else:
                version = apply_corruption(img, corruption, severity, f"{name}:{variant}")
                cv2.imwrite(str(out / rel), version, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows += [
                {
                    **plate,
                    "variant": variant,
                    "artificial_corruption": corruption,
                    "artificial_severity": severity,
                    "image": rel,
                    "source_image": image,
                }
                for plate in plates.to_dict("records")
            ]
        if i % 20 == 0:
            print(f"  {i}/{gt.image.nunique()}")
    return pd.DataFrame(rows, columns=GT_FIELDS).astype(dict.fromkeys(BOX, int))


def match_results(gt: pd.DataFrame, ocr: pd.DataFrame) -> pd.DataFrame:
    """Tablica z ground truth obok dopasowanej detekcji (IoU >= 0.5, jak w metrykach detekcji) i jej odczytu."""
    dets = dict(tuple(ocr[ocr.det_id.notna()].groupby("image")))
    rows = []
    for image, g in gt.groupby("image", sort=False):
        d = dets.get(image, ocr.iloc[:0]).sort_values("conf", ascending=False)
        boxes, pred = g[BOX].to_numpy(float), d[BOX].to_numpy(float)
        _, owner = match(pred, boxes, 0.5)
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


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """Odsetek wykrytych i bezbłędnie odczytanych tablic dla każdej wersji."""
    return (
        results.groupby(["artificial_corruption", "artificial_severity"], sort=False)
        .agg(plates=("plate_id", "size"), detected=("detected", "mean"), text_ok=("text_ok", "mean"))
        .reset_index()
    )
