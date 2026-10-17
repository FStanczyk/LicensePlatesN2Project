"""Metody poprawy jakości obrazu przed detekcją i OCR (Stańczyk, bloki 3-4).

Szkielet: ustalony interfejs, implementacje do uzupełnienia. Każda metoda to funkcja
(obraz BGR uint8, **parametry) -> obraz BGR uint8 tego samego rozmiaru, zarejestrowana w METHODS.

    from alpr.preprocessing import enhance
    out = enhance(img, "clahe_lab", clip_limit=2.0)

Potok end-to-end wywołuje enhance(img, method) przed detekcją, więc nazwa metody wystarcza, żeby
porównać warianty. Zdjęcia po filtrze zapisujemy z tą samą strukturą katalogów co data/combined,
wtedy eval_detection.py --images-root testuje je bez zmian w kodzie.
"""

from collections.abc import Callable

import numpy as np

Method = Callable[..., np.ndarray]


def none(img: np.ndarray) -> np.ndarray:
    """Bez zmian (baseline)."""
    return img


def _todo(name: str) -> Method:
    def method(img: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError(f"metoda {name!r} nie jest jeszcze zaimplementowana (alpr/preprocessing)")

    method.__name__ = name
    return method


# Warianty z README ("Wnioski dla naszego projektu"): CLAHE na RGB i na kanale jasności, gamma,
# wyrównanie histogramu, odszumianie, unsharp masking oraz ich połączenie.
METHODS: dict[str, Method] = {
    "none": none,
    "clahe_rgb": _todo("clahe_rgb"),
    "clahe_lab": _todo("clahe_lab"),
    "gamma": _todo("gamma"),
    "hist_eq": _todo("hist_eq"),
    "denoise": _todo("denoise"),
    "unsharp": _todo("unsharp"),
    "clahe_unsharp": _todo("clahe_unsharp"),
}


def enhance(img: np.ndarray, method: str = "none", **params) -> np.ndarray:
    """Stosuje metodę z METHODS do obrazu BGR."""
    if method not in METHODS:
        raise ValueError(f"nieznana metoda: {method} (dostępne: {', '.join(METHODS)})")
    return METHODS[method](img, **params)
