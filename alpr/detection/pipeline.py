"""Przebiegi po zdjęciach: predykcje, wycinki dla OCR, ocena wyrównania."""

import os
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd

from alpr.dataset import format_polygon, parse_polygon, with_corners
from alpr.detection.align import AlignedPlate, PlateAligner
from alpr.detection.detector import Detection, Detector, detect_batch
from alpr.detection.metrics import BOX, best_match, corner_error
from alpr.geometry import box_corners, polygon_iou, warp

READ_WORKERS = 8
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]
DETECTION_FIELDS = ["image", "det_id", "conf", "x1", "y1", "x2", "y2", "polygon", "align", "crop"]


def read_images(images: list[str], base: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Zdjęcia w kolejności images; wczytywane w tle (cv2 zwalnia GIL), najwyżej 2 * READ_WORKERS naraz w pamięci."""
    with ThreadPoolExecutor(READ_WORKERS) as pool:
        pending = deque()
        for i, rel in enumerate(images, 1):
            pending.append((i, rel, pool.submit(cv2.imread, str(base / rel))))
            while pending and (len(pending) > 2 * READ_WORKERS or i == len(images)):
                k, name, future = pending.popleft()
                img = future.result()
                if img is None:
                    print(f"  pominięto (nie da się wczytać): {name}")
                else:
                    yield name, img
                if k % 100 == 0:
                    print(f"  {k}/{len(images)}")


def shape_groups(imgs: list[np.ndarray]) -> list[list[int]]:
    """Indeksy obrazów pogrupowane po kształcie, w kolejności pierwszego wystąpienia."""
    groups: dict[tuple, list[int]] = {}
    for i, img in enumerate(imgs):
        groups.setdefault(img.shape, []).append(i)
    return list(groups.values())


def detect_images(
    detector: Detector, images: list[str], base: Path, batch: int = 1
) -> Iterator[tuple[str, np.ndarray, list[Detection]]]:
    """(image, obraz, detekcje) w kolejności images; po batch zdjęć naraz, wsad tylko z obrazów tego samego kształtu.

    Przy jednakowym kształcie YOLO dopełnia obraz tak samo jak pojedynczy, więc wynik nie zależy od batch
    (z dokładnością do numeryki wsadowych jąder GPU).
    """
    chunk: list[tuple[str, np.ndarray]] = []
    for item in read_images(images, base):
        chunk.append(item)
        if len(chunk) < batch:
            continue
        yield from _detect_chunk(detector, chunk)
        chunk = []
    yield from _detect_chunk(detector, chunk)


def _detect_chunk(detector: Detector, chunk: list[tuple[str, np.ndarray]]):
    dets: list[list[Detection]] = [[] for _ in chunk]
    for group in shape_groups([img for _, img in chunk]):
        for i, found in zip(group, detect_batch(detector, [chunk[i][1] for i in group]), strict=True):
            dets[i] = found
    for (rel, img), found in zip(chunk, dets, strict=True):
        yield rel, img, found


def predict_boxes(detector: Detector, images: list[str], base: Path, batch: int = 1) -> pd.DataFrame:
    """image, conf, x1, y1, x2, y2, polygon (narożniki z YOLO-pose, inaczej puste)."""
    rows = [
        {
            "image": rel,
            "conf": det.conf,
            **dict(zip(BOX, det.box, strict=True)),
            "polygon": format_polygon(det.corners) if det.corners is not None else None,
        }
        for rel, _, dets in detect_images(detector, images, base, batch)
        for det in dets
    ]
    return pd.DataFrame(rows, columns=["image", "conf", *BOX, "polygon"])


def crop_name(rel: str, k: int, common: str) -> str:
    """crops/<ścieżka względem common, "/" -> "__", z rozszerzeniem>_<k>.png; x.jpg i x.png się nie nadpisują."""
    name = PurePosixPath(rel).relative_to(common).as_posix()
    return f"crops/{name.replace('/', '__')}_{k}.png"


def crop_plates(
    detector: Detector,
    images: list[str],
    base: Path,
    out: Path,
    aligner: PlateAligner | None = None,
    batch: int = 1,
    crops: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """Zapisuje wycinki w out/crops; narożniki z YOLO-pose mają pierwszeństwo przed alignerem.

    Zdjęcie bez detekcji daje wiersz z samym image. crops: słownik do uzupełnienia wycinkami (klucz jak kolumna crop),
    żeby OCR nie czytał ich ponownie z dysku.
    """
    common = os.path.commonpath([PurePosixPath(rel).parent.as_posix() for rel in images]).replace(os.sep, "/")
    (out / "crops").mkdir(parents=True, exist_ok=True)
    rows = []
    for rel, img, dets in detect_images(detector, images, base, batch):
        if not dets:
            rows.append({"image": rel})
        for k, det in enumerate(dets):
            if det.corners is not None:
                plate = AlignedPlate.from_corners(img, det.corners)
            else:
                plate = aligner.align(img, det.box) if aligner else AlignedPlate.from_box(img, det.box)
            crop_rel = crop_name(rel, k, common)
            cv2.imwrite(str(out / crop_rel), plate.crop, PNG_PARAMS)
            if crops is not None:
                crops[crop_rel] = plate.crop
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


def read_detections(path: Path) -> pd.DataFrame:
    """detections.csv z całkowitymi det_id i ramkami mimo pustych wierszy zdjęć bez detekcji."""
    return pd.read_csv(path, dtype=dict.fromkeys(["det_id", *BOX], "Int64"))


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
    pred_corners: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Ramka a wyrównanie względem narożników (with_corners).

    pred_boxes: ramki detektora zamiast etykiet; pred_corners (corners_by_image): narożniki YOLO-pose zamiast alignera.
    """
    rows = []
    groups = dict(tuple(with_corners(plates).groupby("image")))
    for image, img in read_images(sorted(groups), dataset):
        for r in groups[image].itertuples():
            gt = parse_polygon(r.polygon)
            box = np.array([r.x1, r.y1, r.x2, r.y2], dtype=np.float32)
            corners = None
            if pred_boxes is not None:
                boxes = pred_boxes.get(image, np.zeros((0, 4)))
                i = best_match(boxes, box)
                if i is None:
                    continue
                box = boxes[i]
                corners = pred_corners[image][i] if pred_corners and image in pred_corners else None
            plate = AlignedPlate.from_corners(img, corners) if corners is not None else aligner.align(img, box)
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
