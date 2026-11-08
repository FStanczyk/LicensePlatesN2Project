"""Narożniki (lewy górny, prawy górny, prawy dolny, lewy dolny), prostowanie perspektywy, IoU."""

import cv2
import numpy as np


def _corner_index(points: np.ndarray) -> list[int]:
    s, d = points.sum(axis=1), points[:, 0] - points[:, 1]
    return [s.argmin(), d.argmax(), s.argmax(), d.argmin()]


def order_corners(points: np.ndarray) -> np.ndarray:
    return points[_corner_index(points)].astype(np.float32)


def four_corners(points: np.ndarray) -> np.ndarray:
    """Narożniki od lewego górnego zgodnie z ruchem wskazówek zegara; górny bok to wyżej leżący z dłuższej pary.

    Wielokąt z 5-6 punktami (data2) jest najpierw sprowadzany do 4 narożników. Działa dla obrotów poniżej 90 stopni,
    także tam, gdzie order_corners zawodzi (ok. 45 stopni) i dla adnotacji zapisanych przeciwnie do zegara.
    """
    quad = points if len(points) == 4 else points[_corner_index(points)]
    x, y = quad[:, 0], quad[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) < 0:
        quad = quad[[0, 3, 2, 1]]
    sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    pair = [0, 2] if sides[0] + sides[2] >= sides[1] + sides[3] else [1, 3]
    top = min(pair, key=lambda i: quad[i, 1] + quad[(i + 1) % 4, 1])
    return np.roll(quad, -top, axis=0)


def box_corners(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def warp(img: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Prostuje czworokąt; tą funkcją powstały wycinki w zbiorze."""
    tl, tr, br, bl = poly
    w = max(1, int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = max(1, int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(poly.astype(np.float32), dst), (w, h))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float32), b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0
