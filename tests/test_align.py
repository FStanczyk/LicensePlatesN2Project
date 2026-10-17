import cv2
import numpy as np

from alpr.detection.align import align_plate
from alpr.detection.metrics import corner_error

PLATE = np.array([[120, 110], [290, 125], [285, 175], [115, 162]], np.float32)
BOX = np.array([112, 108, 293, 178], np.float32)


def scene() -> np.ndarray:
    """Ciemne tło z jasną, lekko obróconą tablicą i ciemnymi znakami."""
    img = np.full((300, 400, 3), 40, np.uint8)
    cv2.fillPoly(img, [PLATE.astype(np.int32)], (230, 230, 230))
    cv2.putText(img, "AB123", (140, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
    return img


def test_align_plate_znajduje_czworokat():
    aligned, poly, method = align_plate(scene(), BOX)
    assert method == "quad"
    assert corner_error(poly, PLATE) < 0.1
    assert aligned.shape[1] > 3 * aligned.shape[0]  # tablica jednorzędowa: szeroka


def test_align_plate_bez_tablicy_zostaje_ramka():
    img = np.full((300, 400, 3), 40, np.uint8)
    _, poly, method = align_plate(img, BOX)
    assert method == "box"
    np.testing.assert_array_equal(poly.min(axis=0), BOX[:2])


def test_align_plate_ramka_przy_krawedzi_obrazu():
    aligned, _, method = align_plate(scene(), np.array([0, 0, 5, 3], np.float32))
    assert method == "box"
    assert aligned.size > 0
