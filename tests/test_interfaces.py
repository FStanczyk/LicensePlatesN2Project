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
