import cv2
import numpy as np

from alpr.detection import AlignedPlate, PlateAligner
from alpr.detection.metrics import corner_error

PLATE = np.array([[120, 110], [290, 125], [285, 175], [115, 162]], np.float32)
BOX = np.array([112, 108, 293, 178], np.float32)


def scene() -> np.ndarray:
    img = np.full((300, 400, 3), 40, np.uint8)
    cv2.fillPoly(img, [PLATE.astype(np.int32)], (230, 230, 230))
    cv2.putText(img, "AB123", (140, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
    return img


def test_align_znajduje_czworokat():
    plate = PlateAligner().align(scene(), BOX)
    assert plate.method == "quad"
    assert corner_error(plate.poly, PLATE) < 0.1
    assert plate.crop.shape[1] > 3 * plate.crop.shape[0]


def test_align_bez_tablicy_zostaje_ramka():
    plate = PlateAligner().align(np.full((300, 400, 3), 40, np.uint8), BOX)
    assert plate.method == "box"
    np.testing.assert_array_equal(plate.poly.min(axis=0), BOX[:2])


def test_align_ramka_przy_krawedzi_obrazu():
    plate = PlateAligner().align(scene(), np.array([0, 0, 5, 3], np.float32))
    assert plate.method == "box"
    assert plate.crop.size > 0


def test_from_box():
    plate = AlignedPlate.from_box(scene(), BOX)
    assert (plate.method, plate.crop.shape[:2]) == ("box", (70, 181))
