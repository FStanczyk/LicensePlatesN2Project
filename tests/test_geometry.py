import numpy as np
import pytest

from alpr.geometry import box_corners, box_iou, box_iou_matrix, four_corners, order_corners, polygon_iou, warp

QUAD = np.array([[10, 20], [150, 30], [140, 90], [15, 80]], dtype=np.float32)


def test_order_corners_niezalezne_od_kolejnosci():
    shuffled = QUAD[[2, 0, 3, 1]]
    np.testing.assert_array_equal(order_corners(shuffled), QUAD)


def test_four_corners_porzadkuje_adnotacje_przeciwnie_do_zegara():
    ccw = QUAD[[0, 3, 2, 1]].astype(float)
    np.testing.assert_array_equal(four_corners(ccw), QUAD)
    np.testing.assert_array_equal(four_corners(QUAD[[2, 3, 0, 1]].astype(float)), QUAD)
    assert four_corners(ccw).dtype == np.float64


def test_four_corners_poprawna_kolejnosc_bez_zmian():
    np.testing.assert_array_equal(four_corners(QUAD.astype(float)), QUAD)


@pytest.mark.parametrize("angle", [-60, -45, -30, 0, 30, 45, 60])
def test_four_corners_obrocona_tablica(angle):
    import cv2

    t = np.deg2rad(angle)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    plate = np.array([[-50, -10], [50, -10], [50, 10], [-50, 10]], float) @ rot.T + 200
    for start in range(4):
        for order in (plate, plate[::-1]):
            q = four_corners(np.roll(order, start, axis=0))
            np.testing.assert_allclose(q, plate)
    assert cv2.contourArea(four_corners(plate).astype(np.float32)) > 0


def test_four_corners_redukuje_wielokat_do_narozników():
    pentagon = np.array([[5, 5], [0, 0], [9, 0], [9, 9], [0, 9]], dtype=float)
    assert four_corners(pentagon).tolist() == [[0, 0], [9, 0], [9, 9], [0, 9]]


def test_box_corners():
    assert box_corners(np.array([1, 2, 3, 4])).tolist() == [[1, 2], [3, 2], [3, 4], [1, 4]]


def test_warp_rozmiar_i_narozniki():
    img = np.random.default_rng(0).integers(0, 256, (50, 80, 3), dtype=np.uint8)
    out = warp(img, box_corners(np.array([10, 5, 60, 25])))
    assert out.shape == (20, 50, 3)
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


def random_boxes(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (n, 2))
    return np.hstack([xy, xy + rng.uniform(1, 50, (n, 2))])


def test_box_iou_symetryczne_i_w_przedziale_0_1():
    boxes = random_boxes(40)
    for a in boxes:
        for b in boxes[:10]:
            iou = box_iou(a, b)
            assert 0.0 <= iou <= 1.0
            assert iou == pytest.approx(box_iou(b, a))


def test_box_iou_ramka_wewnatrz_i_stykajace_sie():
    assert box_iou(np.array([0, 0, 10, 10.0]), np.array([0, 0, 5, 5.0])) == pytest.approx(0.25)
    assert box_iou(np.array([0, 0, 10, 10.0]), np.array([10, 0, 20, 10.0])) == 0.0


def test_box_iou_zgodne_z_polygon_iou_na_losowych_ramkach():
    boxes = random_boxes(20, seed=1)
    for a, b in zip(boxes[:-1], boxes[1:], strict=True):
        assert polygon_iou(box_corners(a), box_corners(b)) == pytest.approx(box_iou(a, b), abs=1e-4)


def test_polygon_iou_rozlaczne_zawarte_symetryczne():
    big = box_corners(np.array([0, 0, 20, 20]))
    small = box_corners(np.array([5, 5, 15, 15]))
    far = box_corners(np.array([100, 100, 110, 110]))
    assert polygon_iou(big, far) == 0.0
    assert polygon_iou(big, small) == pytest.approx(0.25)
    assert polygon_iou(QUAD, big) == pytest.approx(polygon_iou(big, QUAD))


@pytest.mark.parametrize("angle", [-20, -5, 0, 5, 20])
def test_order_corners_obrocony_prostokat(angle):
    import cv2

    pts = cv2.boxPoints(((100, 60), (80, 20), angle)).astype(np.float32)
    ordered = order_corners(pts[::-1])
    tl, tr, br, bl = ordered
    assert tl[0] < tr[0] and bl[0] < br[0]
    assert tl[1] < bl[1] and tr[1] < br[1]
    assert np.linalg.norm(tr - tl) == pytest.approx(80, abs=0.5)
    assert ordered.dtype == np.float32


def test_four_corners_szesciokat_z_data2():
    hexagon = np.array([[20, 20], [80, 22], [100, 30], [98, 60], [40, 62], [18, 50]], float)
    corners = four_corners(hexagon)
    assert corners.shape == (4, 2)
    assert corners.tolist() == [[20, 20], [100, 30], [98, 60], [18, 50]]


def test_warp_wymiary_z_dluzszych_bokow():
    img = np.zeros((100, 200, 3), np.uint8)
    poly = np.array([[10, 10], [110, 10], [100, 50], [20, 40]], np.float32)
    out = warp(img, poly)
    assert out.shape == (41, 100, 3) and out.dtype == np.uint8


def test_warp_zachowuje_kolor_i_kanaly():
    img = np.full((60, 80, 3), (10, 20, 30), np.uint8)
    out = warp(img, np.array([[5, 5], [70, 8], [68, 40], [6, 38]], np.float32))
    assert (out == (10, 20, 30)).all()
    gray = warp(img[:, :, 0], box_corners(np.array([0, 0, 40, 20])))
    assert gray.shape == (20, 40)


def test_warp_kolejnosc_naroznikow_ma_znaczenie():
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, :20] = 255
    poly = box_corners(np.array([0, 0, 39, 39]))
    assert warp(img, poly)[:, 0].mean() == 255
    mirrored = poly[[1, 0, 3, 2]]
    assert warp(img, mirrored)[:, 0].mean() == 0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_box_iou_matrix_identyczne_z_box_iou(dtype):
    a, b = random_boxes(30, 1).astype(dtype), random_boxes(20, 2).astype(dtype)
    b[:5] = a[:5]
    b[5] = [3, 3, 3, 9]
    matrix = box_iou_matrix(a, b)
    assert matrix.shape == (30, 20) and matrix.dtype == dtype
    assert all(matrix[i, j] == box_iou(a[i], b[j]) for i in range(30) for j in range(20))


def test_box_iou_matrix_puste():
    assert box_iou_matrix(np.zeros((0, 4)), random_boxes(3)).shape == (0, 3)
    assert box_iou_matrix(random_boxes(2), np.zeros((0, 4))).shape == (2, 0)
