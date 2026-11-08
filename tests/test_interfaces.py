"""Interfejsy szkieletów preprocessing i ocr (implementacje uzupełniają Stańczyk i Golba)."""

import numpy as np
import pytest

from alpr.ocr import ENGINES, read_plate
from alpr.preprocessing import METHODS, enhance


def test_enhance_none_bez_zmian():
    img = np.zeros((4, 4, 3), np.uint8)
    assert enhance(img) is img


def test_enhance_nieznana_metoda():
    with pytest.raises(ValueError):
        enhance(np.zeros((4, 4, 3), np.uint8), "nie_ma")


def test_metody_zachowuja_rozmiar_i_typ():
    img = np.random.default_rng(0).integers(0, 256, (40, 120, 3), dtype=np.uint8)
    for name in METHODS:
        try:
            out = enhance(img, name)
        except NotImplementedError:
            continue
        assert out.shape == img.shape and out.dtype == np.uint8, name


def test_read_plate_normalizuje(monkeypatch):
    monkeypatch.setitem(ENGINES, "fake", lambda crop: ("wa 123-45", 0.9))
    assert read_plate(np.zeros((10, 40, 3), np.uint8), "fake") == ("WA12345", 0.9)


def test_metody_maja_nazwy_zgodne_z_kluczem():
    assert {"none", "clahe_rgb", "clahe_lab", "gamma", "hist_eq", "denoise", "unsharp", "clahe_unsharp"} <= set(METHODS)
    assert all(method.__name__ == name for name, method in METHODS.items())


def test_niezaimplementowana_metoda_podaje_nazwe():
    for name in METHODS:
        try:
            enhance(np.zeros((8, 8, 3), np.uint8), name)
        except NotImplementedError as e:
            assert name in str(e)


def test_enhance_przekazuje_parametry(monkeypatch):
    monkeypatch.setitem(METHODS, "spy", lambda img, **params: img + params["k"])
    assert enhance(np.zeros((2, 2, 3), np.uint8), "spy", k=3).max() == 3


IMAGES = {
    "losowy": np.random.default_rng(0).integers(0, 256, (40, 120, 3), dtype=np.uint8),
    "czarny": np.zeros((40, 120, 3), np.uint8),
    "bialy": np.full((40, 120, 3), 255, np.uint8),
    "ciemny": np.random.default_rng(1).integers(0, 40, (40, 120, 3), dtype=np.uint8),
    "maly": np.random.default_rng(2).integers(0, 256, (3, 5, 3), dtype=np.uint8),
}


@pytest.mark.parametrize("image", IMAGES)
@pytest.mark.parametrize("name", METHODS)
def test_kontrakt_metody(name, image):
    img = IMAGES[image]
    before = img.copy()
    try:
        out = enhance(img, name)
    except NotImplementedError:
        pytest.skip(f"{name} niezaimplementowana")
    assert out.shape == img.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(img, before)
    np.testing.assert_array_equal(enhance(img, name), out)


def test_read_plate_nieznany_silnik_to_not_implemented():
    assert "easyocr" in ENGINES
    with pytest.raises(NotImplementedError):
        read_plate(np.zeros((10, 40, 3), np.uint8), "nie_ma")
