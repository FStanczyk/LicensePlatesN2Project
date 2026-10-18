"""Przebiegi po zdjęciach: predykcje, wycinki dla OCR, ocena wyrównania."""

import os
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd

from alpr.dataset import format_polygon, parse_polygon
from alpr.detection.align import AlignedPlate, PlateAligner
from alpr.detection.detector import PlateDetector
from alpr.detection.metrics import BOX, best_matching_box, corner_error
from alpr.geometry import box_corners, polygon_iou, warp

DETECTION_FIELDS = ["image", "det_id", "conf", "x1", "y1", "x2", "y2", "polygon", "align", "crop"]


def read_images(images: list[str], base: Path) -> Iterator[tuple[str, np.ndarray]]:
    for i, rel in enumerate(images, 1):
        img = cv2.imread(str(base / rel))
        if img is None:
            print(f"  pominięto (nie da się wczytać): {rel}")
        else:
            yield rel, img
        if i % 100 == 0:
            print(f"  {i}/{len(images)}")


def predict_boxes(detector: PlateDetector, images: list[str], base: Path) -> pd.DataFrame:
    rows = [
        {"image": rel, "conf": det.conf, **dict(zip(BOX, det.box, strict=True))}
        for rel, img in read_images(images, base)
        for det in detector.detect(img)
    ]
    return pd.DataFrame(rows, columns=["image", "conf", *BOX])


def crop_name(rel: str, k: int, common: str) -> str:
    """crops/<ścieżka względem common z "/" zamienionym na "__">_<k>.png."""
    stem = PurePosixPath(rel).relative_to(common).with_suffix("").as_posix()
    return f"crops/{stem.replace('/', '__')}_{k}.png"


def crop_plates(
    detector: PlateDetector, images: list[str], base: Path, out: Path, aligner: PlateAligner | None = None
) -> list[dict]:
    """Zapisuje wycinki w out/crops; zdjęcie bez detekcji daje wiersz z samym image."""
    common = os.path.commonpath([PurePosixPath(rel).parent.as_posix() for rel in images])
    (out / "crops").mkdir(parents=True, exist_ok=True)
    rows = []
    for rel, img in read_images(images, base):
        dets = detector.detect(img)
        if not dets:
            rows.append({"image": rel})
        for k, det in enumerate(dets):
            plate = aligner.align(img, det.box) if aligner else AlignedPlate.from_box(img, det.box)
            crop_rel = crop_name(rel, k, common)
            cv2.imwrite(str(out / crop_rel), plate.crop)
            rows.append(
                {
                    "image": rel,
                    "det_id": k,
                    "conf": round(det.conf, 4),
                    **{c: round(v) for c, v in zip(BOX, det.box, strict=True)},
                    "polygon": format_polygon(plate.poly),
                    "align": plate.method,
                    "crop": crop_rel,
                }
            )
    return rows


def alignment_example(img: np.ndarray, box: np.ndarray, aligned: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Od góry: wycinek ramki, wyrównany, z prawdziwych narożników."""
    tiles = [warp(img, box_corners(box)), aligned, warp(img, gt)]
    return np.vstack([cv2.resize(t, (288, 64)) for t in tiles])


def evaluate_alignment(
    plates: pd.DataFrame,
    dataset: Path,
    aligner: PlateAligner,
    pred_boxes: dict[str, np.ndarray] | None = None,
    examples: Path | None = None,
    n_examples: int = 0,
) -> pd.DataFrame:
    """Ramka a wyrównanie względem narożników z plates.csv; pred_boxes: ramki detektora zamiast etykiet."""
    rows = []
    groups = dict(tuple(plates.groupby("image")))
    for image, img in read_images(sorted(groups), dataset):
        for r in groups[image].itertuples():
            gt = parse_polygon(r.polygon)
            box = np.array([r.x1, r.y1, r.x2, r.y2], dtype=np.float32)
            if pred_boxes is not None:
                box = best_matching_box(pred_boxes.get(image, np.zeros((0, 4))), box)
                if box is None:
                    continue
            plate = aligner.align(img, box)
            rows.append(
                {
                    "plate_id": r.plate_id,
                    "source": r.source,
                    "lighting": r.lighting,
                    "exposure": r.exposure,
                    "corrupted": r.corrupted,
                    "iou_box": polygon_iou(box_corners(box), gt),
                    "iou_aligned": polygon_iou(plate.poly, gt),
                    "err_box": corner_error(box_corners(box), gt),
                    "err_aligned": corner_error(plate.poly, gt),
                    "method": plate.method,
                }
            )
            if examples and len(rows) <= n_examples:
                cv2.imwrite(str(examples / f"{r.plate_id}.png"), alignment_example(img, box, plate.crop, gt))
    return pd.DataFrame(rows)
