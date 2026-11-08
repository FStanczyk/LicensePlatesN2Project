"""Zbiór YOLO-pose: ramka i 4 narożniki tablicy (TL, TR, BR, BL), wyrównanie w jednym przejściu z detekcją.

Narożniki tylko ze źródeł z prawdziwymi narożnikami (with_corners); pozostałe tablice mają widoczność 0,
więc strata punktów ich nie liczy, a ramka uczy się jak w zwykłej detekcji. Zdjęcia to dowiązania do zbioru.

    <out>/images/<split>/<zdjęcie>   dowiązanie do data/combined/detection/images/...
    <out>/labels/<split>/<zdjęcie>.txt   0 cx cy w h  x1 y1 v1 ... x4 y4 v4 (znormalizowane)
    <out>/data.yaml                  kpt_shape [4, 3], flip_idx [1, 0, 3, 2]
"""

import os
from pathlib import Path, PurePosixPath

import cv2
import pandas as pd
import yaml

from alpr.dataset import SPLITS, load_plates, parse_polygon, with_corners
from alpr.geometry import four_corners

KPT_SHAPE = [4, 3]
FLIP_IDX = [1, 0, 3, 2]


def pose_label(row, width: int, height: int, visible: bool) -> str:
    """Linia etykiety YOLO-pose dla jednej tablicy."""
    cx, cy = (row.x1 + row.x2) / 2 / width, (row.y1 + row.y2) / 2 / height
    bw, bh = (row.x2 - row.x1) / width, (row.y2 - row.y1) / height
    if visible:
        points = [f"{x / width:.6f} {y / height:.6f} 2" for x, y in four_corners(parse_polygon(row.polygon))]
    else:
        points = ["0 0 0"] * KPT_SHAPE[0]
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(points)


def export_pose_dataset(dataset: Path, out: Path) -> Path:
    """Etykiety i dowiązania dla całych zdjęć ze wszystkich podziałów; zwraca ścieżkę data.yaml."""
    plates = load_plates(dataset, sample_type="full_photo")
    visible = set(with_corners(plates).plate_id)
    for image, rows in plates.groupby("image"):
        split = rows.split.iloc[0]
        name = PurePosixPath(image).name
        link = out / "images" / split / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            os.symlink((dataset / image).resolve(), link)
        height, width = cv2.imread(str(dataset / image)).shape[:2]
        label = out / "labels" / split / f"{PurePosixPath(name).stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(
            "\n".join(pose_label(r, width, height, r.plate_id in visible) for r in rows.itertuples()) + "\n"
        )
    cfg = {
        "path": str(out.resolve()),
        **{split: f"images/{split}" for split in SPLITS},
        "names": {0: "plate"},
        "kpt_shape": KPT_SHAPE,
        "flip_idx": FLIP_IDX,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return out / "data.yaml"


def corners_by_image(preds: pd.DataFrame, conf: float) -> dict[str, list] | None:
    """Narożniki (4x2) predykcji z pewnością >= conf, w kolejności boxes_by_image; None bez kolumny polygon."""
    if "polygon" not in preds or preds.polygon.isna().all():
        return None
    kept = preds[preds.conf >= conf]
    return {
        image: [parse_polygon(p) if isinstance(p, str) else None for p in g.polygon]
        for image, g in kept.groupby("image")
    }
