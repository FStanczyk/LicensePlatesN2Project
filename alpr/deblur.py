"""Dekonwolucja rozmytych wycinków przed OCR: jądra rozmycia, estymacja jądra z widma zdjęcia, Richardson-Lucy na GPU.

Jądra to rodzina z imagecorruptions (dysk dla defocus_blur, odcinek z wagami Gaussa dla motion_blur, siły 1-5),
odtworzona bez tej biblioteki. Kąt ruchu szacujemy z uśrednionego widma mocy całego zdjęcia: przy dobrym jądrze
log|Y| - log|H| wygląda jak widmo naturalnego zdjęcia (izotropowe, liniowe w log częstotliwości).

    est = estimate_blur(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    views = deconv_views(crop, est.motion_angle)        # 16 wycinków: bez, 5 dysków, 5 sił ruchu x 2 kierunki
"""

import math
import multiprocessing
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd

from alpr.config import SEED

DISK = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]
MOTION = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]
ANGLES = np.arange(-90, 90, 2.0)
TILE = 512


def disk_kernel(radius: int, alias_blur: float) -> np.ndarray:
    """Dysk nieostrości dokładnie jak imagecorruptions.corruptions.disk (po wygładzeniu suma nieco ponad 1)."""
    size, ksize = (8, (3, 3)) if radius <= 8 else (radius, (5, 5))
    ys, xs = np.mgrid[-size : size + 1, -size : size + 1]
    disk = ((xs**2 + ys**2) <= radius**2).astype(np.float32)
    return cv2.GaussianBlur(disk / disk.sum(), ksize=ksize, sigmaX=alias_blur).astype(np.float64)


def motion_kernel(radius: int, sigma: float, angle: float) -> np.ndarray:
    """Jądro splotu, którym imagecorruptions._motion_blur rozmywa obraz (odcinek pod kątem, wagi Gaussa od początku)."""
    width = 2 * radius + 1
    weights = np.exp(-(np.arange(width) ** 2) / (2 * sigma**2))
    weights /= weights.sum()
    sin, cos = math.sin(math.radians(angle)), math.cos(math.radians(angle))
    k = np.zeros((2 * width + 1, 2 * width + 1))
    for i, w in enumerate(weights):
        k[width - math.ceil(i * sin - 0.5), width - math.ceil(i * cos - 0.5)] += w
    return trim(k)


def trim(kernel: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Najmniejszy kwadrat o nieparzystym boku wokół środka z niezerową częścią jądra; suma 1."""
    c = kernel.shape[0] // 2
    ys, xs = np.nonzero(kernel > eps)
    r = max(int(np.abs(ys - c).max()), int(np.abs(xs - c).max()), 1)
    k = kernel[c - r : c + r + 1, c - r : c + r + 1]
    return k / k.sum()


def candidate_kernels(motion_angle: float) -> list[np.ndarray]:
    """5 dysków i po 5 sił ruchu w obu kierunkach (moduł widma nie rozróżnia kąta a od a + 180)."""
    opposite = motion_angle - 180 if motion_angle >= 0 else motion_angle + 180
    return [disk_kernel(*d) for d in DISK] + [
        motion_kernel(*MOTION[s], a) for s in range(len(MOTION)) for a in (motion_angle, opposite)
    ]


@dataclass(frozen=True)
class BlurEstimate:
    kind: str
    severity: int
    angle: float
    motion_angle: float


def _polar_bins(n: int = TILE):
    fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    rb, ab = np.linspace(0.04, 0.45, 25), np.linspace(0, 180, 37)
    ri = np.digitize(np.hypot(fy, fx), rb) - 1
    ai = np.digitize(np.degrees(np.arctan2(fy, fx)) % 180, ab) - 1
    valid = (ri >= 0) & (ri < len(rb) - 1) & (ai >= 0) & (ai < len(ab) - 1)
    flat = ri[valid] * (len(ab) - 1) + ai[valid]
    counts = np.bincount(flat, minlength=(len(rb) - 1) * (len(ab) - 1))
    return valid, flat, counts, (len(rb) - 1, len(ab) - 1), np.log(0.5 * (rb[:-1] + rb[1:]))


def polar(values: np.ndarray) -> np.ndarray:
    """Średnie wartości widma [n, n] w przedziałach (promień, kąt)."""
    valid, flat, counts, shape, _ = _bins()
    return (np.bincount(flat, values[valid], minlength=counts.size) / np.maximum(counts, 1)).reshape(shape)


@cache
def _bins():
    return _polar_bins()


def log_spectrum(kernel: np.ndarray, n: int = TILE) -> np.ndarray:
    h = np.zeros((n, n))
    r = kernel.shape[0] // 2
    h[: kernel.shape[0], : kernel.shape[1]] = kernel
    return polar(np.log(np.maximum(np.abs(np.fft.fft2(np.roll(h, (-r, -r), (0, 1)))) ** 2, 1e-5)))


@cache
def _candidates() -> tuple[list[tuple[str, int, float]], np.ndarray]:
    cands = [("none", 0, 0.0)] + [("defocus_blur", s + 1, 0.0) for s in range(len(DISK))]
    cands += [("motion_blur", s + 1, float(a)) for s in range(len(MOTION)) for a in ANGLES]
    kernels = [np.ones((1, 1))] + [disk_kernel(*d) for d in DISK]
    kernels += [motion_kernel(*MOTION[s], a) for s in range(len(MOTION)) for a in ANGLES]
    return cands, np.stack([log_spectrum(k) for k in kernels])


def image_log_spectrum(gray: np.ndarray, n: int = TILE) -> np.ndarray:
    """Uśrednione widmo mocy okien n x n (Welch, okno Hanna, zakładka połowy) w przedziałach biegunowych."""
    g = gray.astype(np.float64) / 255
    win = np.outer(np.hanning(n), np.hanning(n))
    tiles = [
        g[y : y + n, x : x + n]
        for y in range(0, g.shape[0] - n + 1, n // 2)
        for x in range(0, g.shape[1] - n + 1, n // 2)
    ]
    if not tiles:
        tiles = [
            cv2.copyMakeBorder(g, 0, max(0, n - g.shape[0]), 0, max(0, n - g.shape[1]), cv2.BORDER_REFLECT)[:n, :n]
        ]
    acc = np.mean([np.abs(np.fft.fft2((t - t.mean()) * win)) ** 2 for t in tiles], axis=0)
    return polar(np.log(acc + 1e-10))


def estimate_blur(gray: np.ndarray) -> BlurEstimate:
    """Jądro z kandydatów (brak, dyski, ruch co 2 stopnie), po którym widmo najlepiej pasuje do naturalnego zdjęcia."""
    cands, spectra = _candidates()
    *_, logr = _bins()
    resid = image_log_spectrum(gray)[None] - spectra
    design = np.stack([np.ones_like(logr), logr], 1)
    coef = np.linalg.lstsq(design, resid.mean(2).T, rcond=None)[0]
    fit = (design @ coef).T[:, :, None]
    score = ((resid - fit) ** 2).mean((1, 2))
    best = int(score.argmin())
    motion = [i for i, c in enumerate(cands) if c[0] == "motion_blur"]
    return BlurEstimate(*cands[best], motion_angle=cands[min(motion, key=lambda i: score[i])][2])


def kernel_spectra(kernels: list[np.ndarray], shape: tuple[int, int], device):
    """rfft2 jąder [K, H, W//2+1] ze środkiem jądra w (0, 0), czyli splot bez przesunięcia."""
    import torch

    h = np.zeros((len(kernels), *shape), np.float32)
    for i, k in enumerate(kernels):
        r = k.shape[0] // 2
        h[i, : k.shape[0], : k.shape[1]] = k
        h[i] = np.roll(h[i], (-r, -r), (0, 1))
    return torch.fft.rfft2(torch.as_tensor(h, device=device))


def richardson_lucy(image: np.ndarray, kernels: list[np.ndarray], iters: int = 30, device: str | None = None):
    """Richardson-Lucy dla jednego obrazu uint8 [H, W, C] i wielu jąder naraz; splot przez FFT na GPU, jeśli jest.

    Obraz dopełniany powieleniem krawędzi o największy bok jądra, więc splot kołowy nie miesza przeciwnych brzegów.
    Zwraca listę obrazów uint8 w kolejności kernels.
    """
    import torch
    from torch.nn import functional as F

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pad = max(k.shape[0] for k in kernels)
    y = torch.as_tensor(image, dtype=torch.float32, device=dev).permute(2, 0, 1)[None] / 255
    y = F.pad(y, (pad,) * 4, mode="replicate")[0]
    shape = y.shape[-2:]
    spec = kernel_spectra(kernels, tuple(shape), dev)[:, None]
    conj = spec.conj()

    def conv(x, s):
        return torch.fft.irfft2(torch.fft.rfft2(x) * s, s=shape)

    y = y[None].expand(len(kernels), -1, -1, -1)
    u = torch.full_like(y, 0.5)
    for _ in range(iters):
        u = u * conv(y / conv(u, spec).clamp_min(1e-6), conj)
    u = u[..., pad:-pad, pad:-pad].clamp(0, 1)
    return list((u.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy())


def deconv_views(crop: np.ndarray, motion_angle: float, iters: int = 30, device: str | None = None) -> list[np.ndarray]:
    """[wycinek bez zmian, 5 dysków, 5 sił ruchu x 2 kierunki] - 16 widoków do wyboru po pewności OCR."""
    return [crop] + richardson_lucy(crop, candidate_kernels(motion_angle), iters, device)


def true_kernel(corruption: str, severity: int, key: str) -> np.ndarray | None:
    """Jądro, którym alpr.build.corruptions.apply_corruption rozmyło obraz o kluczu key (kąt z tego samego ziarna).

    None dla zniekształceń, które nie są rozmyciem. Tylko do budowy danych treningowych; przy odczycie jądro szacujemy.
    """
    if corruption == "defocus_blur":
        return disk_kernel(*DISK[severity - 1])
    if corruption == "motion_blur":
        angle = np.random.RandomState(zlib.crc32(key.encode())).uniform(-45, 45)
        return motion_kernel(*MOTION[severity - 1], angle)
    return None


def deconv_training_crops(
    experiment: Path, split: str = "train", iters: int = 30, seed: int = SEED, copies: int = 2
) -> tuple[list[np.ndarray], list[str]]:
    """Wycinki rozmytych wersji zdjęć z podziału split zbioru eksperymentu po dekonwolucji prawdziwym jądrem.

    Ramka z etykiety losowo przesunięta (4% szerokości, 10% wysokości), copies razy na tablicę, jak przy detektorze.
    """
    from alpr.geometry import box_corners, warp

    gt = pd.read_csv(experiment / "ground_truth.csv")
    gt = gt[(gt.split == split) & gt.artificial_corruption.isin(["defocus_blur", "motion_blur"])]
    rng = np.random.default_rng(seed)
    crops, texts = [], []
    for r in gt.itertuples():
        img = cv2.imread(str(experiment / r.image))
        key = f"{PurePosixPath(r.source_image).name}:{r.variant}"
        kernel = true_kernel(r.artificial_corruption, int(r.artificial_severity), key)
        box = np.array([r.x1, r.y1, r.x2, r.y2], np.float32)
        size = np.array([0.04, 0.1, 0.04, 0.1], np.float32) * np.tile(box[2:] - box[:2], 2)
        for _ in range(copies):
            jittered = box + rng.uniform(-1, 1, 4).astype(np.float32) * size
            crops.append(richardson_lucy(warp(img, box_corners(jittered)), [kernel], iters)[0])
            texts.append(r.text)
    return crops, texts


def _estimate_file(path: str) -> BlurEstimate:
    return estimate_blur(cv2.imread(path, cv2.IMREAD_GRAYSCALE))


def estimate_blur_files(paths: list[Path], workers: int = 8) -> list[BlurEstimate]:
    """estimate_blur dla wielu zdjęć w osobnych procesach (CPU, ok. 0,1 s na zdjęcie)."""
    if workers <= 1:
        return [_estimate_file(str(p)) for p in paths]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("forkserver")) as pool:
        return list(pool.map(_estimate_file, [str(p) for p in paths], chunksize=4))
