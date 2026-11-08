import cv2
import numpy as np
import pytest

from alpr.detection import ClassicalDetector, Detection
from alpr.detection.classical import char_score
from alpr.geometry import box_iou

PLATE = np.array([300, 200, 520, 250], np.float32)


def car_scene(plate=PLATE, text="WA 12345", bg=90, size=(480, 800), noise=0.0) -> np.ndarray:
    img = np.full((*size, 3), bg, np.uint8)
    cv2.rectangle(img, (150, 120), (670, 330), (bg - 40,) * 3, -1)
    x1, y1, x2, y2 = plate.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (235, 235, 235), -1)
    cv2.rectangle(img, (x1, y1), (x1 + 14, y2), (160, 60, 0), -1)
    scale = (y2 - y1) / 40
    cv2.putText(img, text, (x1 + int(20 * scale), y2 - int(8 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (20,) * 3, 2)
    if noise:
        img = np.clip(img + np.random.default_rng(0).normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("dzien", {}),
        ("ciemno", {"bg": 40}),
        ("jasno", {"bg": 170}),
        ("szum", {"noise": 8}),
        ("mala tablica", {"plate": np.array([360, 220, 470, 245], np.float32)}),
        ("duze zdjecie", {"size": (1080, 1920), "plate": PLATE * 2}),
    ],
)
def test_wykrywa_tablice_jako_najpewniejsza(name, kwargs):
    img = car_scene(**{k: v for k, v in kwargs.items() if k != "plate"}, plate=kwargs.get("plate", PLATE))
    dets = ClassicalDetector().detect(img)
    assert dets, name
    assert on_plate(dets[0].box, kwargs.get("plate", PLATE)), name


def on_plate(box: np.ndarray, plate: np.ndarray, tol: float = 4) -> bool:
    """Ramka w obrębie tablicy i pokrywa co najmniej 70% jej szerokości (detektor obejmuje obszar znaków)."""
    inside = (box[:2] >= plate[:2] - tol).all() and (box[2:] <= plate[2:] + tol).all()
    return bool(inside and (box[2] - box[0]) >= 0.7 * (plate[2] - plate[0]))


def test_pusty_obraz_bez_detekcji():
    assert ClassicalDetector().detect(np.full((480, 800, 3), 90, np.uint8)) == []


def test_wynik_posortowany_po_nms_i_w_zakresie():
    dets = ClassicalDetector().detect(car_scene(noise=8))
    confs = [d.conf for d in dets]
    assert confs == sorted(confs, reverse=True) and all(0 <= c <= 1 for c in confs)
    assert all(isinstance(d, Detection) and d.corners is None for d in dets)
    for i, a in enumerate(dets):
        for b in dets[i + 1 :]:
            assert box_iou(a.box, b.box) < ClassicalDetector().nms_iou


def test_ramki_w_ukladzie_oryginalnego_obrazu():
    small, big = (
        ClassicalDetector().detect(car_scene(size=s, plate=p))
        for s, p in [((480, 800), PLATE), ((960, 1600), PLATE * 2)]
    )
    np.testing.assert_allclose(big[0].box, small[0].box * 2, atol=6)


def test_limit_detekcji_i_obraz_szary():
    gray = cv2.cvtColor(car_scene(noise=20), cv2.COLOR_BGR2GRAY)
    assert len(ClassicalDetector(max_dets=2).detect(gray)) <= 2


def test_deterministyczny():
    a, b = ClassicalDetector().detect(car_scene()), ClassicalDetector().detect(car_scene())
    assert [(d.box.tolist(), d.conf) for d in a] == [(d.box.tolist(), d.conf) for d in b]


def test_char_score_znaki_a_jednolite_tlo():
    img = cv2.cvtColor(car_scene(), cv2.COLOR_BGR2GRAY)
    assert char_score(img, PLATE) >= 0.75
    assert char_score(img, np.array([20, 20, 120, 60], np.float32)) == 0.0
    assert char_score(img, np.array([10, 10, 11, 11], np.float32)) == 0.0


def test_char_score_za_malo_znakow():
    img = cv2.cvtColor(car_scene(text="AB"), cv2.COLOR_BGR2GRAY)
    assert char_score(img, PLATE) < 0.2
