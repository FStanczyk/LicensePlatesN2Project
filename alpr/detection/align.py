"""Wyrównanie perspektywy wykrytej tablicy.

YOLO zwraca tylko prostokątną ramkę, a tablica na zdjęciu jest zwykle czworokątem (perspektywa,
obrót). Szukamy więc 4 narożników tablicy wewnątrz ramki powiększonej o margines i prostujemy
wycinek przekształceniem perspektywicznym (geometry.warp, jak przy budowie zbioru).
Gdy narożników nie da się wiarygodnie znaleźć, zwracamy zwykły wycinek ramki.
"""

import cv2
import numpy as np

from alpr.geometry import box_corners, box_iou, order_corners, warp

MARGIN = 0.15  # margines wokół ramki przy szukaniu narożników (ułamek szerokości/wysokości)
WORK_WIDTH = 320  # szerokość, do której skalujemy wycinek przed szukaniem narożników
MIN_IOU = 0.6  # minimalne IoU ramki znalezionego czworokąta z ramką z YOLO
ASPECT = (1.0, 7.0)  # dopuszczalny stosunek szerokości do wysokości tablicy (2 wiersze ok. 1.5, 1 wiersz ok. 4.7)


def _masks(bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Maski binarne, na których szukamy konturu tablicy: krawędzie (ramka tablicy) i progowanie
    Otsu (jasne tło tablicy). Do jasnego tła dokładamy niebieski pasek UE: w skali szarości jest
    ciemny i bez tego lewa krawędź tablicy wypadała na granicy paska. Zamknięcie szerokim jądrem
    skleja znaki z tłem w jedną plamę."""
    gray = cv2.bilateralFilter(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 7, 50, 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blue = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), (95, 80, 30), (135, 255, 255))
    bright = cv2.morphologyEx(bright | blue, cv2.MORPH_CLOSE, kernel)
    return [("edges", edges), ("otsu", bright)]


def refine_quad(quad: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Doprecyzowuje narożniki: do punktów konturu leżących wzdłuż każdego boku (bez okolic rogów,
    które bywają zaokrąglone) dopasowuje prostą, a narożniki to przecięcia sąsiednich prostych.
    Wierzchołki z approxPolyDP potrafią być przesunięte o kilka pikseli, co pochyla wycinek."""
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        side = b - a
        length = np.linalg.norm(side)
        if length < 4:
            return quad
        t = (points - a) @ side / length**2  # położenie wzdłuż boku (0..1)
        rel = points - a
        dist = np.abs(side[0] * rel[:, 1] - side[1] * rel[:, 0]) / length  # odległość od boku
        near = points[(t > 0.15) & (t < 0.85) & (dist < max(2.0, 0.05 * length))]
        if len(near) < 5:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    refined = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]  # narożnik i leży między bokami i-1 oraz i
        det = d1[0] * -d2[1] + d2[0] * d1[1]
        if abs(det) < 1e-6:
            return quad
        s = ((p2 - p1)[0] * -d2[1] + d2[0] * (p2 - p1)[1]) / det
        refined.append(p1 + s * d1)
    refined = np.array(refined, dtype=np.float32)
    # zabezpieczenie: doprecyzowanie nie może przesunąć narożnika daleko od pierwotnego
    limit = 0.25 * min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0]))
    return refined if np.linalg.norm(refined - quad, axis=1).max() < limit else quad


def find_corners(crop: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, str] | None:
    """Szuka 4 narożników tablicy na wycinku BGR.

    target: ramka z detektora we współrzędnych wycinka; znaleziony czworokąt musi się z nią pokrywać
    (IoU >= MIN_IOU), co odrzuca np. zderzak albo całą maskę auta. Zwraca (narożniki we współrzędnych
    wycinka, metoda) albo None. Metoda "quad": kontur przybliżony dokładnie 4 punktami,
    "rect": najmniejszy obrócony prostokąt (gdy róg jest ucięty lub zaokrąglony).
    """
    scale = WORK_WIDTH / crop.shape[1]
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    target = target * scale
    min_area = 0.3 * (target[2] - target[0]) * (target[3] - target[1])

    best, best_iou = None, MIN_IOU
    for _, mask in _masks(small):
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
            if height < 1 or not ASPECT[0] <= width / height <= ASPECT[1]:
                continue
            iou = box_iou(np.r_[quad.min(axis=0), quad.max(axis=0)], target)
            if iou > best_iou:
                best, best_iou = (quad / scale, method), iou
    return best


def align_plate(img: np.ndarray, box: np.ndarray, margin: float = MARGIN) -> tuple[np.ndarray, np.ndarray, str]:
    """Wycina i prostuje tablicę z ramki box (x1, y1, x2, y2) na obrazie BGR.

    Zwraca (wyprostowany wycinek, 4 narożniki na całym obrazie, metoda), gdzie metoda to "quad",
    "rect" albo "box" (narożników nie znaleziono, zwykły wycinek ramki).
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    cx2, cy2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    crop = img[cy1:cy2, cx1:cx2]
    found = None
    if crop.shape[0] >= 8 and crop.shape[1] >= 16:
        found = find_corners(crop, np.array([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], dtype=np.float32))
    if found is None:
        poly, method = box_corners(np.array(box, dtype=np.float32)), "box"
    else:
        poly, method = found[0] + np.array([cx1, cy1], dtype=np.float32), found[1]
    return warp(img, poly), poly, method
