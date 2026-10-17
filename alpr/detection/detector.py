"""Detektor tablic YOLO (ultralytics importowany leniwie, reszta pakietu działa bez torcha)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    box: np.ndarray
    conf: float


class PlateDetector:
    def __init__(self, weights: str | Path, conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, img: np.ndarray) -> list[Detection]:
        """Ramki x1, y1, x2, y2 na obrazie BGR, malejąco po pewności."""
        boxes = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0].boxes
        dets = [Detection(b, float(c)) for b, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), strict=True)]
        return sorted(dets, key=lambda d: -d.conf)
