"""Sztuczne zniekształcenia: imagecorruptions (łatany pod numpy 2 i nowy scikit-image) i własne darken.

Wymaga extras [build]; import łata pakiet globalnie, więc importuje go tylko budowa zbioru.
"""

import random
import warnings
import zlib

import cv2
import numpy as np
import skimage.filters

from alpr.config import SEED

np.float_ = np.float64
from imagecorruptions import corrupt, corruptions  # noqa: E402

_gaussian = skimage.filters.gaussian
corruptions.gaussian = lambda img, sigma=1, multichannel=None, **kw: _gaussian(
    img, sigma=sigma, channel_axis=-1 if multichannel else None, **kw
)
warnings.filterwarnings("ignore", module="imagecorruptions")

MIN_SIDE = 32
CORRUPTIONS = [
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


def darken(img: np.ndarray, severity: int) -> np.ndarray:
    """Niedoświetlenie: gamma > 1 i obniżenie jasności."""
    gamma, gain = [(1.4, 0.85), (1.8, 0.7), (2.2, 0.55), (2.6, 0.42), (3.0, 0.3)][severity - 1]
    return (255 * gain * (img / 255.0) ** gamma).astype(np.uint8)


def apply_corruption(img: np.ndarray, name: str, severity: int, key: str) -> np.ndarray:
    """Zniekształcenie name o sile 1-5, losowość powtarzalna na podstawie key; img w BGR.

    imagecorruptions wymaga boków >= MIN_SIDE: mniejszy obraz jest dopełniany krawędzią i po zniekształceniu przycinany.
    """
    if name == "darken":
        return darken(img, severity)
    np.random.seed(zlib.crc32(key.encode()))
    h, w = img.shape[:2]
    pad = ((0, max(0, MIN_SIDE - h)), (0, max(0, MIN_SIDE - w)), (0, 0))
    rgb = np.ascontiguousarray(np.pad(img[:, :, ::-1], pad, mode="edge"))
    out = corrupt(rgb, corruption_name=name, severity=severity)
    return np.ascontiguousarray(out[:h, :w, ::-1])


BLUR_LEVELS = {
    "motion_blur": [(5, 2), (10, 3), (12, 4), (15, 5), (15, 6), (15, 8), (15, 10), (15, 12), (18, 13), (20, 15)],
    "defocus_blur": [
        (2, 0.1),
        (3, 0.1),
        (3.5, 0.3),
        (4, 0.5),
        (5, 0.5),
        (6, 0.5),
        (7, 0.5),
        (8, 0.5),
        (9, 0.5),
        (10, 0.5),
    ],
}


def apply_blur(img: np.ndarray, name: str, level: int, key: str) -> np.ndarray:
    """Rozmycie name na poziomie 1-10 (BLUR_LEVELS); poziom 2k to siła k z imagecorruptions, nieparzyste leżą między.

    Ta sama losowość (kąt ruchu) co apply_corruption dla tego samego key; img w BGR.
    """
    a, b = BLUR_LEVELS[name][level - 1]
    np.random.seed(zlib.crc32(key.encode()))
    h, w = img.shape[:2]
    pad = ((0, max(0, MIN_SIDE - h)), (0, max(0, MIN_SIDE - w)), (0, 0))
    rgb = np.pad(img[:, :, ::-1], pad, mode="edge")
    if name == "motion_blur":
        out = np.clip(corruptions._motion_blur(rgb, radius=a, sigma=b, angle=np.random.uniform(-45, 45)), 0, 255)
    else:
        kernel = corruptions.disk(radius=a, alias_blur=b)
        x = rgb / 255.0
        out = np.clip(np.stack([cv2.filter2D(x[:, :, d], -1, kernel) for d in range(3)], axis=2), 0, 1) * 255
    return np.ascontiguousarray(np.uint8(out)[:h, :w, ::-1])


def add_corruption(img: np.ndarray, key: str) -> tuple[np.ndarray, str, int]:
    """Powtarzalnie losowane (na podstawie key) zniekształcenie i siła 1-5; img w BGR."""
    rng = random.Random(f"{SEED}:{key}")
    name, severity = rng.choice(CORRUPTIONS), rng.randint(1, 5)
    return apply_corruption(img, name, severity, key), name, severity
