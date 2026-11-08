"""Klasyczny detektor tablic (bez uczenia), punkt odniesienia dla YOLO.

Kandydaci: black-hat (ciemne znaki na jasnym tle), pionowe krawędzie Sobela, domknięcie morfologiczne w kilku
skalach i filtr proporcji. Pewność: weryfikacja znaków (plamy po Otsu o wysokości znaku, w jednym rzędzie)
razem z dopasowaniem proporcji. Ten sam interfejs co PlateDetector: detect(img) -> list[Detection].
"""

from dataclasses import dataclass

import cv2
import numpy as np

from alpr.detection.detector import Detection
from alpr.geometry import box_iou

CHAR_HEIGHT = 40


def char_score(gray: np.ndarray, box: np.ndarray) -> float:
    """0-1: ile plam o wysokości znaku leży w jednym rzędzie (3 to minimum, 8 i więcej daje 1)."""
    x1, y1, x2, y2 = box.astype(int)
    crop = gray[y1:y2, x1:x2]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return 0.0
    crop = cv2.resize(crop, (max(1, round(CHAR_HEIGHT * crop.shape[1] / crop.shape[0])), CHAR_HEIGHT))
    best = 0
    for mode in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, binary = cv2.threshold(crop, 0, 255, mode + cv2.THRESH_OTSU)
        stats = cv2.connectedComponentsWithStats(binary)[2][1:]
        chars = [
            s for s in stats if 0.35 * CHAR_HEIGHT <= s[3] <= 0.95 * CHAR_HEIGHT and 2 <= s[2] <= 0.3 * crop.shape[1]
        ]
        if len(chars) < 2:
            continue
        centers = np.array([s[1] + s[3] / 2 for s in chars])
        heights = np.array([s[3] for s in chars])
        in_row = int((abs(centers - np.median(centers)) < 0.25 * CHAR_HEIGHT).sum())
        same = int((abs(heights - np.median(heights)) < 0.2 * np.median(heights)).sum())
        best = max(best, min(in_row, same))
    return min(best, 8) / 8 if best >= 3 else 0.05 * best


@dataclass(frozen=True)
class ClassicalDetector:
    work_width: int = 1024
    kernels: tuple[tuple[int, int], ...] = ((9, 3), (17, 5), (25, 7))
    aspect: tuple[float, float] = (1.5, 7.0)
    min_height: int = 6
    max_width: float = 0.5
    nms_iou: float = 0.3
    max_dets: int = 20

    def candidates(self, gray: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Ramki w skali gray z pewnością przed NMS."""
        h, w = gray.shape
        found = []
        for kernel in self.kernels:
            rect = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect)
            grad = np.abs(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3))
            grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, edges = cv2.threshold(cv2.GaussianBlur(grad, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            closed = cv2.dilate(cv2.erode(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rect), None), None)
            for contour in cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                x, y, bw, bh = cv2.boundingRect(contour)
                if bh < self.min_height or bw > self.max_width * w or not self.aspect[0] <= bw / bh <= self.aspect[1]:
                    continue
                box = np.array([x, y, x + bw, y + bh], np.float32)
                shape = np.exp(-abs(np.log(bw / bh / 4.0))) ** 0.5
                found.append((box, char_score(gray, box) * float(shape)))
        return found

    def detect(self, img: np.ndarray) -> list[Detection]:
        """Ramki x1, y1, x2, y2 na obrazie BGR, malejąco po pewności (0-1), po NMS."""
        scale = min(1.0, self.work_width / img.shape[1])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if scale < 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        kept: list[Detection] = []
        for box, score in sorted(self.candidates(gray), key=lambda c: -c[1]):
            if len(kept) < self.max_dets and all(box_iou(box / scale, d.box) < self.nms_iou for d in kept):
                kept.append(Detection(box / scale, score))
        return kept
