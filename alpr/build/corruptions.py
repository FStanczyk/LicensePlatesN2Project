"""Sztuczne zniekształcenia zdjęć: imagecorruptions (Hendrycks i Dietterich) plus własne "darken".

Wymaga extras [build]: pip install -e ".[build]" (imagecorruptions==1.1.2, scikit-image).
Import modułu łata imagecorruptions pod numpy 2 i nowy scikit-image, dlatego importuje go
tylko budowa zbioru, a nie reszta pakietu.
"""

import random
import warnings
import zlib

import numpy as np
import skimage.filters

from alpr.config import SEED

# imagecorruptions 1.1.2 nie działa z numpy 2 i scikit-image >= 0.19 (fog, gaussian_blur, glass_blur)
np.float_ = np.float64
from imagecorruptions import corrupt, corruptions  # noqa: E402

_gaussian = skimage.filters.gaussian
corruptions.gaussian = lambda img, sigma=1, multichannel=None, **kw: _gaussian(
    img, sigma=sigma, channel_axis=-1 if multichannel else None, **kw
)
warnings.filterwarnings("ignore", module="imagecorruptions")

# z imagecorruptions, plus "darken" (brak w pakiecie: brightness tylko rozjaśnia);
# pomijamy przekształcenia geometryczne (elastic_transform, zoom_blur), bo przesuwają tablicę
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
    """Niedoświetlenie: korekcja gamma > 1 i obniżenie jasności."""
    gamma, gain = [(1.4, 0.85), (1.8, 0.7), (2.2, 0.55), (2.6, 0.42), (3.0, 0.3)][severity - 1]
    return (255 * gain * (img / 255.0) ** gamma).astype(np.uint8)


def add_corruption(img: np.ndarray, key: str) -> tuple[np.ndarray, str, int]:
    """Losuje (powtarzalnie, na podstawie key) zniekształcenie i siłę 1-5; img w BGR."""
    rng = random.Random(f"{SEED}:{key}")
    name, severity = rng.choice(CORRUPTIONS), rng.randint(1, 5)
    if name == "darken":
        return darken(img, severity), name, severity
    np.random.seed(zlib.crc32(key.encode()))  # imagecorruptions losuje przez np.random
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    out = corrupt(rgb, corruption_name=name, severity=severity)
    return np.ascontiguousarray(out[:, :, ::-1]), name, severity
