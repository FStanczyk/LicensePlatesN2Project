"""Zbiór eksperymentu (Golba): czyste zdjęcia z etykietami, ich zniekształcone wersje, detekcja i OCR na wszystkich.

Ground truth: całe zdjęcia z ramką i czytelnym tekstem (bez blurred), same oryginały, tylko test (val wybiera
best.pt i wczesne zatrzymanie YOLO). Zniekształcenia nie zmieniają geometrii, więc etykiety są wspólne dla wersji.

    data/experiment/images/<wersja>/<zdjęcie>.jpg   wersja: clean albo <zniekształcenie>_<siła>
    data/experiment/ground_truth.csv                jedna linia na (wersja, tablica)
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd

from alpr.config import MP_START
from alpr.dataset import load_plates
from alpr.detection.metrics import BOX, desc_order, greedy_match
from alpr.geometry import box_iou_matrix

SPLITS = ("test",)
CORRUPTIONS = ("brightness", "darken", "motion_blur", "defocus_blur")
SEVERITIES = (1, 2, 3, 4, 5)
BLURS = ("motion_blur", "defocus_blur")
BLUR_SEVERITIES = tuple(range(1, 11))
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


def variants(corruptions: tuple[str, ...] = CORRUPTIONS, levels: int = 5) -> list[tuple[str, str, int]]:
    """(nazwa katalogu, zniekształcenie, siła); pierwsza zawsze clean.

    levels 10: tylko rozmycia (BLURS) w poziomach 1-10 (alpr.build.corruptions.BLUR_LEVELS), katalogi <rozmycie>_01..10.
    """
    if levels == 5:
        return [("clean", "none", 0)] + [(f"{c}_{s}", c, s) for c in corruptions for s in SEVERITIES]
    if levels != 10 or not set(corruptions) <= set(BLURS):
        raise ValueError(f"10 poziomów jest tylko dla {BLURS}")
    return [("clean", "none", 0)] + [(f"{c}_{s:02d}", c, s) for c in corruptions for s in BLUR_SEVERITIES]


def ground_truth(dataset: Path, splits: tuple[str, ...] = SPLITS) -> pd.DataFrame:
    """Oryginalne całe zdjęcia z czytelnym tekstem z podanych podziałów."""
    df = load_plates(dataset, sample_type="full_photo", originals_only=True, ocr_only=True)
    return df[df.split.isin(splits)]


def images_by_source(gt: pd.DataFrame) -> list[str]:
    """Zdjęcia z wersjami jednego zdjęcia źródłowego obok siebie (ten sam kształt, więc wspólny wsad detekcji)."""
    return gt.drop_duplicates("image").sort_values(["source_image", "image"]).image.tolist()


def build_image(
    item: tuple[str, list[dict]],
    dataset: Path,
    out: Path,
    corruptions: tuple[str, ...] = CORRUPTIONS,
    levels: int = 5,
) -> list[dict]:
    """Wszystkie wersje jednego zdjęcia zapisane w out/images; wiersze ground truth dla nich."""
    from alpr.build.corruptions import apply_blur, apply_corruption

    corrupt = apply_corruption if levels == 5 else apply_blur

    image, plates = item
    name = PurePosixPath(image).name
    img = cv2.imread(str(dataset / image))
    rows = []
    for variant, corruption, severity in variants(corruptions, levels):
        rel = f"images/{variant}/{name}"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        version = img if severity == 0 else corrupt(img, corruption, severity, f"{name}:{variant}")
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
            for plate in plates
        ]
    return rows


def build_experiment(
    dataset: Path,
    out: Path,
    splits: tuple[str, ...] = SPLITS,
    corruptions: tuple[str, ...] = CORRUPTIONS,
    workers: int = 1,
    levels: int = 5,
) -> pd.DataFrame:
    """Zapisuje wersje zdjęć w out/images i zwraca ground truth dla wszystkich wersji.

    workers > 1: zdjęcia w osobnych procesach; losowość zniekształceń zależy tylko od nazwy, więc wynik ten sam.
    """
    gt = ground_truth(dataset, splits)
    items = [(image, plates.to_dict("records")) for image, plates in gt.groupby("image")]
    variants(corruptions, levels)
    build = partial(build_image, dataset=dataset, out=out, corruptions=corruptions, levels=levels)
    rows = []
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context(MP_START)) as pool:
        for i, part in enumerate(pool.map(build, items) if workers > 1 else map(build, items), 1):
            rows += part
            if i % 20 == 0:
                print(f"  {i}/{len(items)}")
    return pd.DataFrame(rows, columns=GT_FIELDS).astype(dict.fromkeys(BOX, int))


def match_results(gt: pd.DataFrame, ocr: pd.DataFrame) -> pd.DataFrame:
    """Tablica z ground truth obok dopasowanej detekcji (IoU >= 0.5, jak w metrykach detekcji) i jej odczytu."""
    dets = ocr[ocr.det_id.notna()]
    by_image = dets.groupby("image").indices
    det_box, det_conf, det_rows = dets[BOX].to_numpy(float), dets.conf.to_numpy(float), dets.to_dict("records")
    gt_box, gt_rows = gt[BOX].to_numpy(float), gt.to_dict("records")
    rows = []
    for image, plates in gt.groupby("image", sort=False).indices.items():
        d = by_image.get(image, np.zeros(0, dtype=int))
        d = d[desc_order(det_conf[d])]
        ious = box_iou_matrix(det_box[d], gt_box[plates])
        _, owner = greedy_match(ious, 0.5)
        best = ious.max(axis=0) if len(d) else np.zeros(len(plates))
        for g, o, iou in zip(plates, owner, best, strict=True):
            plate, det = gt_rows[g], det_rows[d[o]] if o >= 0 else None
            text_pred = det["text"] if det is not None and isinstance(det["text"], str) else ""
            rows.append(
                {
                    **{c: plate[c] for c in [*RESULT_FIELDS[:5], "split", "exposure"]},
                    "text_true": plate["text"],
                    "detected": int(o >= 0),
                    "iou": round(iou, 3),
                    "det_conf": det["conf"] if det is not None else np.nan,
                    "align": det["align"] if det is not None else "",
                    "crop": det["crop"] if det is not None else "",
                    "text_pred": text_pred,
                    "conf_ocr": det["conf_ocr"] if det is not None else np.nan,
                    "text_ok": int(text_pred == plate["text"]),
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
