import dataclasses

import cv2
import numpy as np
import pytest

from alpr.detection import AlignedPlate, PlateAligner
from alpr.detection.align import plate_masks, refine_quad
from alpr.detection.metrics import corner_error
from alpr.geometry import box_corners, polygon_iou, warp

PLATE = np.array([[120, 110], [290, 125], [285, 175], [115, 162]], np.float32)
BOX = np.array([112, 108, 293, 178], np.float32)
RECT = np.array([[100, 100], [300, 100], [300, 150], [100, 150]], np.float32)


def scene() -> np.ndarray:
    img = np.full((300, 400, 3), 40, np.uint8)
    cv2.fillPoly(img, [PLATE.astype(np.int32)], (230, 230, 230))
    cv2.putText(img, "AB123", (140, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
    return img


def draw(plate: np.ndarray, bg: int = 40, fg: int = 230, scale: float = 1.0, noise: float = 0) -> np.ndarray:
    img = np.full((int(300 * scale), int(400 * scale), 3), bg, np.uint8)
    cv2.fillPoly(img, [plate.astype(np.int32)], (fg, fg, fg))
    x, y = (plate[0] + np.array([20, 45]) * scale).astype(int)
    cv2.putText(img, "AB123", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), max(1, int(3 * scale)))
    if noise:
        img = np.clip(img + np.random.default_rng(0).normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def box_of(plate: np.ndarray, pad: float = 4) -> np.ndarray:
    return np.r_[plate.min(axis=0) - pad, plate.max(axis=0) + pad].astype(np.float32)


def rotated(poly: np.ndarray, degrees: float) -> np.ndarray:
    t = np.deg2rad(degrees)
    center = poly.mean(axis=0)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return ((poly - center) @ rot.T + center).astype(np.float32)


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


@pytest.mark.parametrize(
    ("name", "plate", "kwargs"),
    [
        ("skala x3", PLATE * 3, {"scale": 3}),
        ("skala x0.5", PLATE * 0.5, {"scale": 0.5}),
        ("prostokat", RECT, {}),
        ("obrot 10", rotated(RECT, 10), {}),
        ("obrot -15", rotated(RECT, -15), {}),
        ("noc", PLATE, {"bg": 5, "fg": 90}),
        ("szum", PLATE, {"noise": 10}),
        ("srednie tlo", PLATE, {"bg": 150, "fg": 250}),
    ],
)
def test_align_rozne_sceny(name, plate, kwargs):
    img = draw(plate, **kwargs)
    box = box_of(plate)
    aligned = PlateAligner().align(img, box)
    assert aligned.method in ("quad", "rect"), name
    assert corner_error(aligned.poly, plate) < 0.1, name
    assert polygon_iou(aligned.poly, plate) > polygon_iou(box_corners(box), plate), name


@pytest.mark.parametrize(("bg", "fg"), [(200, 250), (180, 230)])
def test_align_jasna_tablica_na_jasnym_tle_nie_gorsza_od_ramki(bg, fg):
    img = draw(PLATE, bg=bg, fg=fg)
    box = box_of(PLATE)
    aligned = PlateAligner().align(img, box)
    assert polygon_iou(aligned.poly, PLATE) >= polygon_iou(box_corners(box), PLATE)


def test_align_crop_to_warp_poly():
    img = scene()
    plate = PlateAligner().align(img, BOX)
    np.testing.assert_array_equal(plate.crop, warp(img, plate.poly))


def test_align_poly_w_ukladzie_calego_zdjecia():
    plate = PlateAligner().align(scene(), BOX)
    assert plate.poly.dtype == np.float32 and plate.poly.shape == (4, 2)
    assert (plate.poly.min(axis=0) >= BOX[:2] - 0.15 * (BOX[2:] - BOX[:2])).all()
    assert (plate.poly.max(axis=0) <= BOX[2:] + 0.15 * (BOX[2:] - BOX[:2])).all()


def test_align_proporcje_poza_zakresem_daja_ramke():
    plate = PlateAligner(aspect=(10.0, 20.0)).align(scene(), BOX)
    assert plate.method == "box"


def test_align_wysoki_prog_iou_daje_ramke():
    plate = PlateAligner(min_iou=0.99).align(scene(), BOX)
    assert plate.method == "box"


def test_align_ramka_float64():
    plate = PlateAligner().align(scene(), BOX.astype(np.float64))
    assert plate.method == "quad"


def test_align_deterministyczne():
    a, b = PlateAligner().align(scene(), BOX), PlateAligner().align(scene(), BOX)
    np.testing.assert_array_equal(a.poly, b.poly)


def test_aligned_plate_i_aligner_niemutowalne():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PlateAligner().margin = 0.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        AlignedPlate.from_box(scene(), BOX).method = "quad"


def test_plate_masks_rozmiar_i_typ():
    masks = plate_masks(scene())
    assert len(masks) == 2
    for mask in masks:
        assert mask.shape == (300, 400) and mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 255}


def test_plate_masks_niebieski_pasek_ue_w_masce_jasnego_tla():
    img = np.full((40, 160, 3), 30, np.uint8)
    img[5:35, 10:20] = (200, 50, 0)
    img[5:35, 20:150] = 240
    bright = plate_masks(img)[1]
    assert bright[20, 15] == 255 and bright[20, 100] == 255 and bright[20, 155] == 0


def perimeter(quad: np.ndarray, n: int = 200) -> np.ndarray:
    return np.concatenate([np.linspace(quad[i], quad[(i + 1) % 4], n) for i in range(4)]).astype(np.float32)


def test_refine_quad_poprawia_narozniki_do_konturu():
    rect = np.array([[0, 0], [100, 0], [100, 30], [0, 30]], np.float32)
    rough = rect + np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]], np.float32)
    np.testing.assert_allclose(refine_quad(rough, perimeter(rect)), rect, atol=0.05)


def test_refine_quad_za_krotki_bok_zwraca_wejscie():
    quad = np.array([[0, 0], [3, 0], [3, 2], [0, 2]], np.float32)
    assert refine_quad(quad, perimeter(quad)) is quad


def test_refine_quad_za_malo_punktow_zwraca_wejscie():
    quad = np.array([[0, 0], [100, 0], [100, 30], [0, 30]], np.float32)
    assert refine_quad(quad, quad.copy()) is quad


def test_refine_quad_kontur_daleko_od_bokow_zwraca_wejscie():
    rect = np.array([[0, 0], [100, 0], [100, 30], [0, 30]], np.float32)
    shifted = rect + [0, 12]
    assert refine_quad(rect, perimeter(shifted)) is rect


def test_find_corners_odrzuca_obrys_calego_okna():
    crop = np.full((60, 200, 3), 230, np.uint8)
    target = np.array([20, 8, 180, 52], np.float32)
    assert PlateAligner(min_iou=0.3).find_corners(crop, target) is None
