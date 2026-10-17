"""Poprawa obrazu przed detekcją i OCR (Stańczyk). Szkielet: interfejs ustalony, metody do uzupełnienia.

Metoda: (obraz BGR uint8, **parametry) -> obraz BGR uint8 tego samego rozmiaru, zarejestrowana w METHODS.

    out = enhance(img, "clahe_lab", clip_limit=2.0)
"""

from collections.abc import Callable

import numpy as np

Method = Callable[..., np.ndarray]


def none(img: np.ndarray) -> np.ndarray:
    return img


def _todo(name: str) -> Method:
    def method(img: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError(f"metoda {name!r} nie jest jeszcze zaimplementowana (alpr/preprocessing.py)")

    method.__name__ = name
    return method


METHODS: dict[str, Method] = {"none": none} | {
    name: _todo(name) for name in ("clahe_rgb", "clahe_lab", "gamma", "hist_eq", "denoise", "unsharp", "clahe_unsharp")
}


def enhance(img: np.ndarray, method: str = "none", **params) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"nieznana metoda: {method} (dostępne: {', '.join(METHODS)})")
    return METHODS[method](img, **params)
