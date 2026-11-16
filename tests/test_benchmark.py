import sys
import types

import cv2
import numpy as np
import pandas as pd
import pytest

from alpr import benchmark, ocr
from alpr.benchmark import (
    PipelineConfig,
    as_detections,
    compare_detections,
    detection_grid,
    quality,
    run_pipeline,
    timed,
)
from tests.test_pipeline import FakeYOLO


def test_config_row_splaszcza_parametry_ocr():
    row = PipelineConfig("x", "w.pt", ocr={"decoder": "beamsearch", "mag_ratio": 1.5}).row()
    assert row["ocr"] == "decoder=beamsearch mag_ratio=1.5" and row["batch"] == 1 and "half" not in row


def test_timed_zwraca_wynik_i_czas():
    result, seconds = timed(lambda a, b=0: a + b, 2, b=3)
    assert result == 5 and seconds >= 0


def dets(rows):
    fields = ["image", "det_id", "conf", "x1", "y1", "x2", "y2", "crop"]
    return as_detections([dict(zip(fields, r, strict=True)) for r in rows])


def test_compare_detections():
    ref = dets([("a", 0, 0.9, 1, 2, 3, 4, "c"), ("b", 0, 0.5, 5, 5, 9, 9, "c")])
    near = dets([("b", 0, 0.52, 5, 6, 9, 9, "c"), ("a", 0, 0.9, 1, 2, 3, 4, "c")])
    assert compare_detections(ref, ref) == {"same_boxes": True, "max_box_diff": 0.0, "max_conf_diff": 0.0}
    got = compare_detections(ref, near)
    assert got["same_boxes"] and got["max_box_diff"] == 1 and got["max_conf_diff"] == pytest.approx(0.02)
    assert not compare_detections(ref, near.iloc[:1])["same_boxes"]


def test_quality():
    results = pd.DataFrame(
        {
            "artificial_corruption": ["none", "none", "blur"],
            "detected": [1, 1, 0],
            "text_ok": [1, 0, 0],
            "text_pred": ["AB12", "AB13", ""],
            "text_true": ["AB12", "AB12", "AB12"],
        }
    )
    q = quality(results.assign(plate_id=["a", "b", "a"]))
    assert (
        q["text_ok_clean"] == 0.5
        and q["text_ok_clean_lo"] <= 0.5 <= q["text_ok_clean_hi"]
        and q["detected_all"] == pytest.approx(2 / 3)
    )
    assert q["car_clean"] == pytest.approx(0.875) and q["car_all"] == pytest.approx(1.75 / 3)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setitem(ocr.ENGINES, "fake", lambda crop, **params: ("WA" + str(crop.shape[1]), 0.8))
    rows = []
    for variant in ("clean", "blur_1"):
        rel = f"images/{variant}/x.jpg"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path / rel), np.full((80, 120, 3), 200, np.uint8))
        rows.append(
            {
                "variant": variant,
                "artificial_corruption": "none" if variant == "clean" else "blur",
                "artificial_severity": 0 if variant == "clean" else 1,
                "image": rel,
                "source_image": "x.jpg",
                "plate_id": "p",
                "split": "test",
                "exposure": "normal",
                "text": "WA30",
                **dict(zip(["x1", "y1", "x2", "y2"], [60, 40, 90, 55], strict=True)),
            }
        )
    return pd.DataFrame(rows), tmp_path


@pytest.mark.parametrize("batch", [1, 2])
def test_run_pipeline(experiment, tmp_path, batch):
    gt, base = experiment
    row, results, read = run_pipeline(PipelineConfig("x", "w.pt", batch=batch, engine="fake"), gt, base, tmp_path / "o")
    assert row["images"] == 2 and row["crops"] == 4 and row["detect_s"] >= 0 and row["ocr_s"] >= 0
    assert row["text_ok_all"] == 1.0 and row["detected_clean"] == 1.0 and row["batch"] == batch
    assert results.text_pred.tolist() == ["WA30", "WA30"] and read.text.notna().all()


def test_run_pipeline_z_gotowymi_detekcjami(experiment, tmp_path, monkeypatch):
    gt, base = experiment
    _, _, read = run_pipeline(PipelineConfig("x", "w.pt", engine="fake"), gt, base, tmp_path / "o")
    monkeypatch.setattr(benchmark, "detect_stage", lambda *a: pytest.fail("detekcja nie powinna ruszyć"))
    crops = {c: np.zeros((5, 7, 3), np.uint8) for c in read.crop.dropna()}
    cfg = PipelineConfig("y", "w.pt", engine="fake")
    row, results, _ = run_pipeline(cfg, gt, base, tmp_path / "o", read.drop(columns=["text", "conf_ocr"]), crops)
    assert np.isnan(row["detect_s"]) and results.text_pred.tolist() == ["WA7", "WA7"]


def test_sweep_po_jednej_zmianie():
    configs = benchmark.sweep("w.pt", {"mag_ratio": [1.0, 1.5], "decoder": ["beamsearch"]}, base={"mag_ratio": 1.0})
    assert [c.name for c in configs] == ["mag_ratio=1.0", "mag_ratio=1.5", "decoder=beamsearch"]
    assert configs[2].ocr == {"batch_size": 16, "mag_ratio": 1.0, "decoder": "beamsearch"}
    assert benchmark.sweep("w.pt", {})[0].name == "domyslne"
    crnn = benchmark.sweep("w.pt", {"weights": ["b.pt"]}, engine="crnn")
    assert [(c.name, c.engine, c.ocr) for c in crnn] == [
        ("domyslne", "crnn", {}),
        ("weights=b.pt", "crnn", {"weights": "b.pt"}),
    ]


def test_bootstrap_ci_wersje_tablicy_razem():
    results = pd.DataFrame({"plate_id": ["a", "a", "b", "b"], "text_ok": [1, 1, 0, 0]})
    lo, hi = benchmark.bootstrap_ci(results, n=500)
    assert lo == 0.0 and hi == 1.0
    same = pd.DataFrame({"plate_id": list("abcd"), "text_ok": [1, 1, 1, 1]})
    assert benchmark.bootstrap_ci(same) == (1.0, 1.0)


def test_file_name():
    assert benchmark.file_name("weights=runs/ocr/crnn_s0/best.pt") == "weights=runs_ocr_crnn_s0_best.pt"
    assert benchmark.file_name("mag_ratio=2.0 low_text=0.5") == "mag_ratio=2.0_low_text=0.5"


def test_deblur_scorer(experiment, tmp_path, monkeypatch):
    import alpr.deblur

    gt, base = experiment
    gt.to_csv(base / "ground_truth.csv", index=False)
    monkeypatch.setattr(alpr.deblur, "deconv_views", lambda c, a, iters: [c, c])
    est = alpr.deblur.BlurEstimate("none", 0, 0.0, 5.0)
    monkeypatch.setattr(alpr.deblur, "estimate_blur_files", lambda paths, workers: [est] * len(paths))
    from alpr.detection import PlateDetector

    scorer = benchmark.DeblurScorer(base, PlateDetector("x.pt"))
    assert len(scorer.views) == 4 and all(len(v) == 2 for v in scorer.views)
    good = scorer.score(lambda imgs: [("WA30" if im.shape[1] == 30 else "X", 0.9) for im in imgs])
    assert good == 1.0
    assert scorer.score(lambda imgs: [("ZZ", 0.9)] * len(imgs)) == 0.0


def plates(ok, plate_ids=("p1", "p2"), corruptions=("none", "blur")):
    rows = [
        {
            "variant": f"v{c}",
            "plate_id": p,
            "artificial_corruption": c,
            "text_true": "AB",
            "text_pred": "AB" if o else "X",
        }
        for (p, c), o in zip([(p, c) for p in plate_ids for c in corruptions], ok, strict=True)
    ]
    return pd.DataFrame(rows).assign(text_ok=list(ok))


def test_paired_difference():
    a, b = plates([1, 0, 1, 0]), plates([1, 1, 1, 1])
    d = benchmark.paired_difference(a, b, n=500)
    assert d["wspolne"] == 4 and d["tablice"] == 2 and d["roznica"] == 0.5
    assert d["lo"] == d["hi"] == 0.5
    same = benchmark.paired_difference(a, a)
    assert same["roznica"] == same["lo"] == same["hi"] == 0


def test_paired_difference_tylko_wspolne():
    a = plates([1, 1, 1, 1])
    b = plates([0, 0], plate_ids=("p2",))
    d = benchmark.paired_difference(a, b)
    assert d["wspolne"] == 2 and d["a"] == 1 and d["b"] == 0


def test_relabel():
    r = plates([0, 0, 1, 1]).assign(text_pred=["CD", "CD", "AB", "X"])
    gt = pd.DataFrame({"variant": ["vnone", "vblur"], "plate_id": ["p1", "p1"], "text": ["CD", "CD"]})
    out = benchmark.relabel(r, gt)
    assert out.text_true.tolist() == ["CD", "CD", "AB", "AB"] and out.text_ok.tolist() == [1, 1, 1, 0]


def test_compare_runs():
    a, b = plates([0, 0, 1, 0]), plates([1, 1, 1, 1])
    t = benchmark.compare_runs(a, b)
    assert t.wersje.tolist() == ["czyste", "zniekształcone", "wszystkie"]
    assert t.roznica.tolist() == [0.5, 1.0, 0.75]
    gt = pd.DataFrame({"variant": ["vnone"], "plate_id": ["p1"], "text": ["X"]})
    assert benchmark.compare_runs(a, b, gt).a.iloc[0] == 1.0


def test_detection_grid():
    configs = detection_grid({"n640": "a.pt", "s960": "b.pt"})
    assert [c.name for c in configs[:4]] == ["n640_b1", "n640_b8", "n640_b21", "n640_b32"]
    assert {c.imgsz for c in configs[4:]} == {960} and len(configs) == 8
