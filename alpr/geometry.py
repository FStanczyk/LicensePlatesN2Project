"""Geometria tablic: narożniki, prostowanie perspektywy i IoU.

Narożniki zawsze w kolejności: lewy górny, prawy górny, prawy dolny, lewy dolny (tablica 4x2).
Ramki: x1, y1, x2, y2 w pikselach.
"""

import cv2
import numpy as np


def _corner_index(points: np.ndarray) -> list[int]:
    """Indeksy punktów będących narożnikami: najmniejsza i największa suma oraz różnica x - y."""
    s, d = points.sum(axis=1), points[:, 0] - points[:, 1]
    return [s.argmin(), d.argmax(), s.argmax(), d.argmin()]


def order_corners(points: np.ndarray) -> np.ndarray:
    """Porządkuje punkty w 4 narożniki: lewy górny, prawy górny, prawy dolny, lewy dolny."""
    return points[_corner_index(points)].astype(np.float32)


def four_corners(points: np.ndarray) -> np.ndarray:
    """Sprowadza wielokąt z adnotacji do 4 narożników. 4 punkty zostają w kolejności z adnotacji
    (i w oryginalnym typie), więcej punktów porządkujemy jak order_corners. W data2 ok. 5% tablic
    ma 5-6 punktów (powtórzone punkty albo ucięty róg)."""
    return points if len(points) == 4 else points[_corner_index(points)]


def box_corners(box: np.ndarray) -> np.ndarray:
    """4 narożniki prostokątnej ramki."""
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def warp(img: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Prostuje tablicę z 4 narożników. Tą samą funkcją powstały wycinki w zbiorze (build_dataset.py),
    więc wycinki z detekcji wyglądają jak wycinki z etykiet."""
    tl, tr, br, bl = poly
    w = max(1, int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = max(1, int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(poly.astype(np.float32), dst), (w, h))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU dwóch ramek x1, y1, x2, y2."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU dwóch wypukłych czworokątów (4x2)."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0
