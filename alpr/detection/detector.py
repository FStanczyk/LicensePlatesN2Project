"""Detektor tablic YOLO (ultralytics importowany leniwie, reszta pakietu działa bez torcha).

Wagi detekcji dają ramki; wagi YOLO-pose (alpr.detection.pose) dają też 4 narożniki tablicy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    box: np.ndarray
    conf: float
    corners: np.ndarray | None = None


class Detector(Protocol):
    def detect(self, img: np.ndarray) -> list[Detection]: ...


def detect_batch(detector: Detector, imgs: list[np.ndarray]) -> list[list[Detection]]:
    """Wsadowo, jeśli detektor to umie (PlateDetector), inaczej po jednym zdjęciu."""
    if hasattr(detector, "detect_batch"):
        return detector.detect_batch(imgs)
    return [detector.detect(img) for img in imgs]


def to_detections(result) -> list[Detection]:
    """Wynik ultralytics -> detekcje malejąco po pewności."""
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    keypoints = getattr(result, "keypoints", None)
    corners = keypoints.xy.cpu().numpy() if keypoints is not None else [None] * len(boxes)
    dets = [Detection(b, float(c), k) for b, c, k in zip(boxes, confs, corners, strict=True)]
    return sorted(dets, key=lambda d: -d.conf)


class PlateDetector:
    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect_batch(self, imgs: list[np.ndarray]) -> list[list[Detection]]:
        """Jeden przebieg modelu na listę obrazów BGR; ten sam wynik co detect tylko przy jednakowych kształtach."""
        if not imgs:
            return []
        results = self.model.predict(
            imgs,
            conf=self.conf,
            imgsz=self.imgsz,
            batch=len(imgs),
            device=self.device,
            verbose=False,
        )
        return [to_detections(r) for r in results]

    def detect(self, img: np.ndarray) -> list[Detection]:
        """Ramki x1, y1, x2, y2 na obrazie BGR (z modelem pose także narożniki), malejąco po pewności."""
        return self.detect_batch([img])[0]
