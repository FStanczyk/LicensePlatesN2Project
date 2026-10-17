import numpy as np
import pytest

from alpr.geometry import box_corners, box_iou, four_corners, order_corners, polygon_iou, warp

QUAD = np.array([[10, 20], [150, 30], [140, 90], [15, 80]], dtype=np.float32)


def test_order_corners_niezalezne_od_kolejnosci():
    shuffled = QUAD[[2, 0, 3, 1]]
    np.testing.assert_array_equal(order_corners(shuffled), QUAD)


def test_four_corners_zostawia_4_punkty_w_kolejnosci_z_adnotacji():
    reversed_quad = QUAD[::-1].astype(float)
    np.testing.assert_array_equal(four_corners(reversed_quad), reversed_quad)
    assert four_corners(reversed_quad).dtype == np.float64


def test_four_corners_redukuje_wielokat_do_narozników():
    pentagon = np.array([[5, 5], [0, 0], [9, 0], [9, 9], [0, 9]], dtype=float)
    assert four_corners(pentagon).tolist() == [[0, 0], [9, 0], [9, 9], [0, 9]]


def test_box_corners():
    assert box_corners(np.array([1, 2, 3, 4])).tolist() == [[1, 2], [3, 2], [3, 4], [1, 4]]


def test_warp_rozmiar_i_narozniki():
    img = np.random.default_rng(0).integers(0, 256, (50, 80, 3), dtype=np.uint8)
    out = warp(img, box_corners(np.array([10, 5, 60, 25])))
    assert out.shape == (20, 50, 3)
    # narożniki ramki trafiają w narożniki wycinku (środek jest lekko przeskalowany: 50 px -> 0..49)
    np.testing.assert_array_equal(out[0, 0], img[5, 10])


def test_warp_zdegenerowany_czworokat_daje_co_najmniej_1_piksel():
    img = np.zeros((10, 10, 3), np.uint8)
    assert warp(img, np.zeros((4, 2), np.float32)).shape[:2] == (1, 1)


@pytest.mark.parametrize(
    ("a", "b", "iou"),
    [
        ([0, 0, 10, 10], [0, 0, 10, 10], 1.0),
        ([0, 0, 10, 10], [5, 0, 15, 10], 50 / 150),
        ([0, 0, 10, 10], [20, 20, 30, 30], 0.0),
        ([0, 0, 0, 0], [0, 0, 0, 0], 0.0),
    ],
)
def test_box_iou(a, b, iou):
    assert box_iou(np.array(a, float), np.array(b, float)) == pytest.approx(iou)


def test_polygon_iou_zgodne_z_box_iou_dla_prostokatow():
    a, b = np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])
    assert polygon_iou(box_corners(a), box_corners(b)) == pytest.approx(box_iou(a, b))
    assert polygon_iou(QUAD, QUAD) == pytest.approx(1.0)
