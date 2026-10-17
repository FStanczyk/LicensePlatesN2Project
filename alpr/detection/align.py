"""Wyrównanie perspektywy: 4 narożniki tablicy szukane wewnątrz ramki z YOLO, fallback na samą ramkę."""

from dataclasses import dataclass

import cv2
import numpy as np

from alpr.geometry import box_corners, box_iou, order_corners, warp


@dataclass(frozen=True)
class AlignedPlate:
    crop: np.ndarray
    poly: np.ndarray
    method: str

    @classmethod
    def from_box(cls, img: np.ndarray, box: np.ndarray) -> "AlignedPlate":
        poly = box_corners(np.asarray(box, dtype=np.float32))
        return cls(warp(img, poly), poly, "box")


def plate_masks(bgr: np.ndarray) -> list[np.ndarray]:
    """Krawędzie i jasne tło (Otsu z doklejonym niebieskim paskiem UE, sklejone ze znakami)."""
    gray = cv2.bilateralFilter(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 7, 50, 50)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blue = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), (95, 80, 30), (135, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    return [edges, cv2.morphologyEx(bright | blue, cv2.MORPH_CLOSE, kernel)]


def refine_quad(quad: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Narożniki jako przecięcia prostych dopasowanych do środkowych części boków konturu."""
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        side = b - a
        length = np.linalg.norm(side)
        if length < 4:
            return quad
        rel = points - a
        t = rel @ side / length**2
        dist = np.abs(side[0] * rel[:, 1] - side[1] * rel[:, 0]) / length
        near = points[(t > 0.15) & (t < 0.85) & (dist < max(2.0, 0.05 * length))]
        if len(near) < 5:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    refined = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        det = d1[0] * -d2[1] + d2[0] * d1[1]
        if abs(det) < 1e-6:
            return quad
        s = ((p2 - p1)[0] * -d2[1] + d2[0] * (p2 - p1)[1]) / det
        refined.append(p1 + s * d1)
    refined = np.array(refined, dtype=np.float32)
    limit = 0.25 * min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0]))
    return refined if np.linalg.norm(refined - quad, axis=1).max() < limit else quad


@dataclass(frozen=True)
class PlateAligner:
    margin: float = 0.15
    work_width: int = 320
    min_iou: float = 0.6
    aspect: tuple[float, float] = (1.0, 7.0)

    def align(self, img: np.ndarray, box: np.ndarray) -> AlignedPlate:
        """Metoda: quad (kontur z 4 wierzchołkami), rect (obrócony prostokąt) albo box (nie znaleziono)."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        mx, my = self.margin * (x2 - x1), self.margin * (y2 - y1)
        cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
        cx2, cy2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
        crop = img[cy1:cy2, cx1:cx2]
        if crop.shape[0] < 8 or crop.shape[1] < 16:
            return AlignedPlate.from_box(img, box)
        found = self.find_corners(crop, np.array([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], dtype=np.float32))
        if found is None:
            return AlignedPlate.from_box(img, box)
        poly = found[0] + np.array([cx1, cy1], dtype=np.float32)
        return AlignedPlate(warp(img, poly), poly, found[1])

    def find_corners(self, crop: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, str] | None:
        """Czworokąt o proporcjach tablicy najlepiej pokrywający ramkę target (IoU >= min_iou)."""
        scale = self.work_width / crop.shape[1]
        small = cv2.resize(
            crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        )
        target = target * scale
        min_area = 0.3 * (target[2] - target[0]) * (target[3] - target[1])

        best, best_iou = None, self.min_iou
        for mask in plate_masks(small):
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                hull = cv2.convexHull(contour)
                if cv2.contourArea(hull) < min_area:
                    continue
                approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
                if len(approx) == 4:
                    quad, method = approx.reshape(4, 2).astype(np.float32), "quad"
                else:
                    quad, method = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32), "rect"
                quad = refine_quad(order_corners(quad), contour.reshape(-1, 2).astype(np.float32))
                tl, tr, br, bl = quad
                width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
                height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
                if height < 1 or not self.aspect[0] <= width / height <= self.aspect[1]:
                    continue
                iou = box_iou(np.r_[quad.min(axis=0), quad.max(axis=0)], target)
                if iou > best_iou:
                    best, best_iou = (quad / scale, method), iou
        return best
