"""Model YOLO (ultralytics): ładowanie, detekcja na obrazie i konfiguracja zbioru do treningu.

ultralytics importujemy dopiero w funkcjach, żeby reszta pakietu działała bez torcha.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Detection:
    box: np.ndarray  # x1, y1, x2, y2 w pikselach
    conf: float


def load_model(weights: str | Path):
    """Ładuje model YOLO z pliku wag (best.pt z train_yolo.py albo np. yolo11n.pt)."""
    from ultralytics import YOLO

    return YOLO(str(weights))


def detect(model, img: np.ndarray, conf: float = 0.25, imgsz: int = 640) -> list[Detection]:
    """Wykrywa tablice na obrazie BGR, zwraca ramki posortowane malejąco po pewności."""
    result = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    dets = [Detection(b, float(c)) for b, c in zip(boxes, confs, strict=True)]
    return sorted(dets, key=lambda d: -d.conf)


def resolved_yaml(data: Path, out_dir: Path) -> Path:
    """Kopia data.yaml z bezwzględnym "path" (katalog pliku yaml), zapisana w out_dir.

    data.yaml w zbiorze ma "path: .", a ultralytics rozwiązuje względną ścieżkę względem swojego
    katalogu datasets, nie względem pliku yaml.
    """
    cfg = yaml.safe_load(data.read_text())
    cfg["path"] = str(data.parent.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "data_resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return out
