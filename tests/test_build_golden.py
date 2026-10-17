"""Powtarzalność budowy zbioru: wyniki przed refaktoryzacją (scripts/build_dataset.py z commita 92285a4).

Jeśli któryś test przestanie przechodzić po zmianie kodu albo wersji bibliotek, build_dataset.py
da inny zbiór niż ten na Kaggle. Trzeba to wtedy świadomie zaakceptować i wypuścić nową wersję zbioru.
"""

import hashlib
import random

import numpy as np
import pytest

pytest.importorskip("imagecorruptions", reason='wymaga extras [build]: pip install -e ".[build]"')

from alpr.build.corruptions import CORRUPTIONS, add_corruption, darken  # noqa: E402
from alpr.config import SEED  # noqa: E402
from alpr.geometry import warp  # noqa: E402


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def img():
    return np.random.default_rng(0).integers(0, 256, (120, 200, 3), dtype=np.uint8)


# klucz -> (zniekształcenie, siła, hash obrazu); klucze dobrane tak, żeby pokryć wszystkie typy
GOLDEN = {
    "k0": ("defocus_blur", 2, "fe8c0d82045433bd"),
    "k1": ("contrast", 5, "579f730fe2b463c3"),
    "k2": ("motion_blur", 5, "e79d747dd17f73d5"),
    "k3": ("brightness", 4, "13c607bd13107613"),
    "k6": ("gaussian_noise", 1, "4146b1a78a06f25d"),
    "k7": ("shot_noise", 2, "2bd22e14a847a4ea"),
    "k12": ("darken", 2, "cf7daae76701d634"),
    "k15": ("fog", 3, "8f558f95549845ad"),
    "k17": ("jpeg_compression", 4, "77bc4a8416fbff4b"),
    "data1_a": ("contrast", 3, "8bab0722f8086428"),
}


def test_seed_i_lista_znieksztalcen():
    assert SEED == 0
    assert CORRUPTIONS == [
        "brightness",
        "darken",
        "contrast",
        "fog",
        "gaussian_noise",
        "shot_noise",
        "motion_blur",
        "defocus_blur",
        "jpeg_compression",
    ]


@pytest.mark.parametrize("key", GOLDEN)
def test_losowanie_znieksztalcenia(key):
    rng = random.Random(f"{SEED}:{key}")
    assert (rng.choice(CORRUPTIONS), rng.randint(1, 5)) == GOLDEN[key][:2]


@pytest.mark.parametrize("key", GOLDEN)
def test_obraz_po_znieksztalceniu(img, key):
    out, name, severity = add_corruption(img, key)
    assert (name, severity, digest(out)) == GOLDEN[key]


def test_darken(img):
    assert [digest(darken(img, s)) for s in range(1, 6)] == [
        "1438cbd8d7696f1d",
        "cf7daae76701d634",
        "d0308fddd821c03f",
        "4d4d968532e3e5dc",
        "5590da603ef71a69",
    ]


def test_warp(img):
    poly = np.array([[10, 20], [150, 30], [140, 90], [15, 80]], float)
    assert digest(warp(img, poly)) == "ad0658d40dba63b3"
