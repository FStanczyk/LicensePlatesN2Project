import json

import yaml

from alpr import config
from alpr.dataset import load_plates

TEMPLATE = config.ROOT / "alpr" / "build" / "template"


def test_sciezki_wewnatrz_repo():
    assert (config.ROOT / "pyproject.toml").exists()
    for path in (config.COMBINED, config.DOWNLOADS, config.EXPERIMENT):
        assert path.parent == config.DATA_DIR
    assert config.RESULTS.parent == config.RUNS.parent == config.ROOT


def test_rozszerzenia_obrazow_male_z_kropka():
    assert all(ext.startswith(".") and ext == ext.lower() for ext in config.IMG_EXTS)
    assert {".jpg", ".png"} <= config.IMG_EXTS


def test_kaggle_url_wskazuje_zbior():
    assert config.KAGGLE_URL.endswith(config.KAGGLE_DATASET)
    assert config.KAGGLE_INPUT.name == config.KAGGLE_DATASET.split("/")[1]


def test_default_dataset_lokalnie(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "KAGGLE_INPUT", tmp_path / "brak")
    assert config.default_dataset() == config.COMBINED


def test_default_dataset_na_kaggle(monkeypatch, tmp_path):
    (tmp_path / "plates.csv").write_text(
        "plate_id,sample_type,split,artificial_corruption,blurred\nd1_0,x,test,none,0\n"
    )
    monkeypatch.setattr(config, "KAGGLE_INPUT", tmp_path)
    assert config.default_dataset() == tmp_path
    assert load_plates().plate_id.tolist() == ["d1_0"]


def test_szablon_data_yaml():
    cfg = yaml.safe_load((TEMPLATE / "detection" / "data.yaml").read_text())
    assert cfg["names"] == {0: "plate"}
    assert {cfg[s] for s in ("train", "val", "test")} == {f"images/{s}" for s in ("train", "val", "test")}


def test_szablon_metadane_kaggle():
    meta = json.loads((TEMPLATE / "dataset-metadata.json").read_text())
    assert meta["id"] == config.KAGGLE_DATASET
    assert (TEMPLATE / "README.md").exists()
