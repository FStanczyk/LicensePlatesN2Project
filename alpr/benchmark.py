"""Pomiar czasu i jakości ustawień potoku (detekcja, wyrównanie, OCR) na zbiorze eksperymentu.

Konfiguracja to wagi, imgsz, fp16, wsad detekcji, silnik OCR i jego parametry. Czas liczony z synchronizacją
CUDA; jakość przez match_results i summary z alpr.experiment, więc liczby są porównywalne z run_experiment.py.

    timings, results = run_pipeline(PipelineConfig("wsad8", "best.pt", batch=8), gt, EXPERIMENT, out)
"""

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from alpr.config import SEED
from alpr.detection import PlateAligner, PlateDetector
from alpr.detection.metrics import BOX
from alpr.detection.pipeline import DETECTION_FIELDS, crop_plates
from alpr.experiment import images_by_source, match_results
from alpr.ocr import character_accuracy, read_crops


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    weights: str
    imgsz: int = 640
    batch: int = 1
    conf: float = 0.25
    engine: str = "easyocr"
    ocr: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "ocr"},
            "ocr": " ".join(f"{k}={v}" for k, v in self.ocr.items()),
        }


def file_name(name: str) -> str:
    """Nazwa konfiguracji jako nazwa pliku: wszystko poza literami, cyframi, ".", "=" i "-" -> "_"."""
    return re.sub(r"[^\w.=-]+", "_", name).strip("_")


def cuda_sync() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, *args, **kwargs):
    """(wynik, sekundy) z synchronizacją CUDA przed i po."""
    cuda_sync()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    cuda_sync()
    return result, time.perf_counter() - start


def gpu_peak_mb() -> float:
    try:
        import torch
    except ImportError:
        return float("nan")
    if not torch.cuda.is_available():
        return float("nan")
    peak = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    return peak


def detect_stage(cfg: PipelineConfig, images: list[str], base: Path, out: Path, warmup: int = 8):
    """Wiersze detections.csv, wycinki w pamięci i czas (bez ładowania wag i rozgrzewki)."""
    detector = PlateDetector(cfg.weights, cfg.conf, cfg.imgsz)
    crop_plates(detector, images[:warmup], base, out / "warmup", PlateAligner(), cfg.batch)
    gpu_peak_mb()
    crops: dict[str, np.ndarray] = {}
    rows, seconds = timed(crop_plates, detector, images, base, out, PlateAligner(), cfg.batch, crops)
    return rows, crops, seconds


def ocr_stage(cfg: PipelineConfig, dets: pd.DataFrame, out: Path, crops: dict[str, np.ndarray], warmup: int = 8):
    """detections z odczytem i czas (bez ładowania modelu i rozgrzewki)."""
    sample = dets[dets.crop.notna()].head(warmup)
    read_crops(sample, out, cfg.engine, crops, **cfg.ocr)
    return timed(read_crops, dets, out, cfg.engine, crops, **cfg.ocr)


def as_detections(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=DETECTION_FIELDS).astype(dict.fromkeys(["det_id", *BOX], "Int64"))


def bootstrap_ci(
    results: pd.DataFrame, col: str = "text_ok", n: int = 2000, level: float = 0.95, seed: int = SEED
) -> tuple[float, float]:
    """Przedział ufności średniej col: tablice (plate_id) losowane ze zwracaniem, wszystkie wersje tablicy razem."""
    per_plate = results.groupby("plate_id")[col].agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(per_plate), (n, len(per_plate)))
    means = per_plate["sum"].to_numpy()[idx].sum(1) / per_plate["count"].to_numpy()[idx].sum(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def quality(results: pd.DataFrame) -> dict:
    """detected, text_ok i CAR średnio po wszystkich wersjach oraz na clean; 95% przedziały dla text_ok."""
    car = [character_accuracy(p, t) for p, t in zip(results.text_pred, results.text_true, strict=True)]
    results = results.assign(car=car)
    clean = results[results.artificial_corruption == "none"]
    (all_lo, all_hi), (clean_lo, clean_hi) = bootstrap_ci(results), bootstrap_ci(clean)
    return {
        **{f"{c}_all": results[c].mean() for c in ("detected", "text_ok", "car")},
        **{f"{c}_clean": clean[c].mean() for c in ("detected", "text_ok", "car")},
        "text_ok_all_lo": all_lo,
        "text_ok_all_hi": all_hi,
        "text_ok_clean_lo": clean_lo,
        "text_ok_clean_hi": clean_hi,
    }


def run_pipeline(
    cfg: PipelineConfig, gt: pd.DataFrame, base: Path, out: Path, dets: pd.DataFrame | None = None, crops=None
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """(wiersz z czasami i jakością, plates.csv jak w run_experiment, detections z odczytem).

    dets i crops: gotowe detekcje (siatka OCR na tych samych wycinkach), wtedy czas detekcji to NaN.
    """
    images = images_by_source(gt)
    timing = {"images": len(images), "detect_s": np.nan, "detect_gpu_mb": np.nan}
    if dets is None:
        rows, crops, timing["detect_s"] = detect_stage(cfg, images, base, out)
        timing["detect_gpu_mb"] = gpu_peak_mb()
        dets = as_detections(rows)
    ocr, timing["ocr_s"] = ocr_stage(cfg, dets, out, crops or {})
    timing["ocr_gpu_mb"] = gpu_peak_mb()
    timing["crops"] = int(dets.crop.notna().sum())
    timing["img_per_s"] = len(images) / np.nansum([timing["detect_s"], timing["ocr_s"]])
    results = match_results(gt, ocr)
    return {**cfg.row(), **timing, **quality(results)}, results, ocr


def compare_detections(ref: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Zgodność detekcji z odniesieniem: te same zdjęcia i liczby ramek, największa różnica ramki i pewności."""
    ref, other = (d[d.det_id.notna()].sort_values(["image", "det_id"]).reset_index(drop=True) for d in (ref, other))
    same = ref[["image", "det_id"]].equals(other[["image", "det_id"]])
    if not same:
        return {"same_boxes": False, "max_box_diff": np.nan, "max_conf_diff": np.nan}
    box = np.abs(ref[BOX].to_numpy(float) - other[BOX].to_numpy(float))
    conf = np.abs(ref.conf.to_numpy(float) - other.conf.to_numpy(float))
    return {
        "same_boxes": True,
        "max_box_diff": float(box.max(initial=0)),
        "max_conf_diff": float(conf.max(initial=0)),
    }


def detection_grid(weights: dict[str, str]) -> list[PipelineConfig]:
    """weights: nazwa -> ścieżka best.pt; dla każdych wag kilka wsadów (imgsz 960, gdy nazwa zawiera 960)."""
    return [
        PipelineConfig(f"{key}_b{b}", path, 960 if "960" in key else 640, b)
        for key, path in weights.items()
        for b in (1, 8, 21, 32)
    ]


OCR_SWEEP = {
    "mag_ratio": [1.25, 1.5, 1.75, 2.0, 2.5],
    "min_height": [0.5, 0.7, 0.8],
    "text_threshold": [0.5, 0.6, 0.8],
    "low_text": [0.3, 0.5],
    "adjust_contrast": [0.3, 0.7, 0.9],
    "add_margin": [0.05, 0.2],
    "width_ths": [0.7, 1.0],
    "decoder": ["beamsearch"],
}


def sweep(
    weights: str, space: dict[str, list], base: dict | None = None, engine: str = "easyocr"
) -> list[PipelineConfig]:
    """Odniesienie (base) i po jednej zmianie parametru względem niego; nazwa to zmienione parametry."""
    base = ({"batch_size": 16} if engine == "easyocr" else {}) | (base or {})
    label = " ".join(f"{k}={v}" for k, v in base.items() if k != "batch_size") or "domyslne"
    configs = [PipelineConfig(label, weights, engine=engine, ocr=base)]
    for key, values in space.items():
        for value in values:
            if base.get(key) != value:
                configs.append(PipelineConfig(f"{key}={value}", weights, engine=engine, ocr={**base, key: value}))
    return configs


EASYOCR_TUNED = {"batch_size": 16, "mag_ratio": 1.5, "low_text": 0.5, "add_margin": 0.05}


def engines_grid(weights: str) -> list[PipelineConfig]:
    """Porównanie silników: EasyOCR domyślny i strojony, Tesseract w kilku wariantach, CRNN, LPRNet."""
    base = {"weights": weights}
    tess = [
        ("tesseract", {}),
        ("tesseract_bez_listy", {"whitelist": False}),
        ("tesseract_h96", {"height": 96}),
        ("tesseract_otsu", {"binarize": True}),
        ("tesseract_psm8", {"psm": 8}),
    ]
    return [
        PipelineConfig("easyocr", **base, ocr={"batch_size": 16}),
        PipelineConfig("easyocr_strojony", **base, ocr=EASYOCR_TUNED),
        *[PipelineConfig(name, **base, engine="tesseract", ocr=params) for name, params in tess],
        PipelineConfig("crnn", **base, engine="crnn"),
        PipelineConfig("lprnet", **base, engine="lprnet"),
    ]


class DeblurScorer:
    """Zbiór eksperymentu przygotowany raz: wycinki z samej ramki detektora, kąty ruchu z widma i 16 widoków na wycinek.

    Potem ocena dowolnego czytnika (lista obrazów -> lista (tekst, pewność)) trwa sekundy, np. po każdej epoce.
    """

    def __init__(self, experiment: Path, detector, iters: int = 30, batch: int = 8, workers: int = 8):
        import tempfile

        from alpr.deblur import deconv_views, estimate_blur_files

        self.gt = pd.read_csv(experiment / "ground_truth.csv")
        images = images_by_source(self.gt)
        crops: dict[str, np.ndarray] = {}
        with tempfile.TemporaryDirectory() as tmp:
            rows = crop_plates(detector, images, experiment.resolve(), Path(tmp), None, batch, crops)
        self.dets = as_detections(rows)
        estimates = estimate_blur_files([experiment / i for i in images], workers)
        angles = {i: e.motion_angle for i, e in zip(images, estimates, strict=True)}
        present = self.dets[self.dets.crop.notna()]
        self.keys = present.crop.tolist()
        self.views = [
            deconv_views(crops[k], angles[img], iters) for k, img in zip(self.keys, present.image, strict=True)
        ]

    def results(self, read, margin: float = 0.1) -> pd.DataFrame:
        """plates.csv jak w run_experiment dla czytnika read z wyborem widoku alpr.ocr.gate."""
        from alpr.ocr import gate

        flat = read([v for vs in self.views for v in vs])
        texts, i = {}, 0
        for key, vs in zip(self.keys, self.views, strict=True):
            texts[key] = gate(flat[i : i + len(vs)], margin)[0]
            i += len(vs)
        return match_results(self.gt, self.dets.assign(text=self.dets.crop.map(texts), conf_ocr=1.0))

    def score(self, read, margin: float = 0.1) -> float:
        """Odsetek bezbłędnie odczytanych tablic na wersjach zniekształconych."""
        r = self.results(read, margin)
        return float(r[r.artificial_corruption != "none"].text_ok.mean())


def relabel(results: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """text_true i text_ok według aktualnego ground_truth (np. po poprawce etykiet) dla tych samych wersji i tablic."""
    text = gt.drop_duplicates(["variant", "plate_id"]).set_index(["variant", "plate_id"]).text
    true = [text.get((v, p), t) for v, p, t in zip(results.variant, results.plate_id, results.text_true, strict=True)]
    pred = results.text_pred.fillna("").astype(str)
    return results.assign(text_true=true, text_ok=[int(a == b) for a, b in zip(pred, true, strict=True)])


def paired_difference(
    a: pd.DataFrame, b: pd.DataFrame, n: int = 2000, level: float = 0.95, seed: int = SEED
) -> dict[str, float]:
    """text_ok w b minus w a na wspólnych (variant, plate_id) i przedział bootstrap po tablicach (losowane razem)."""
    keys = ["variant", "plate_id"]
    m = a[[*keys, "text_ok"]].merge(b[[*keys, "text_ok"]], on=keys, suffixes=("_a", "_b"))
    per_plate = m.assign(d=m.text_ok_b - m.text_ok_a).groupby("plate_id").d.agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(per_plate), (n, len(per_plate)))
    diffs = per_plate["sum"].to_numpy()[idx].sum(1) / per_plate["count"].to_numpy()[idx].sum(1)
    lo, hi = np.quantile(diffs, [(1 - level) / 2, (1 + level) / 2])
    return {
        "wspolne": len(m),
        "tablice": len(per_plate),
        "a": float(m.text_ok_a.mean()),
        "b": float(m.text_ok_b.mean()),
        "roznica": float(m.text_ok_b.mean() - m.text_ok_a.mean()),
        "lo": float(lo),
        "hi": float(hi),
    }


def compare_runs(a: pd.DataFrame, b: pd.DataFrame, gt: pd.DataFrame | None = None) -> pd.DataFrame:
    """paired_difference osobno dla czystych, zniekształconych i wszystkich wersji; gt: relabel obu przebiegów."""
    if gt is not None:
        a, b = relabel(a, gt), relabel(b, gt)
    groups = {
        "czyste": lambda d: d[d.artificial_corruption == "none"],
        "zniekształcone": lambda d: d[d.artificial_corruption != "none"],
        "wszystkie": lambda d: d,
    }
    return pd.DataFrame([{"wersje": k, **paired_difference(f(a), f(b))} for k, f in groups.items()])
