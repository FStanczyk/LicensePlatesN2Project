import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("strhub", reason='wymaga extras [parseq]: pip install -e ".[parseq]"')

from alpr import parseq  # noqa: E402
from alpr.recognizer import Recognizer  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return parseq.create(pretrained=False, device="cpu").eval()


def test_konfiguracja_jak_w_strhub():
    assert parseq.CONFIG["img_size"] == list(parseq.SIZE) and parseq.CONFIG["embed_dim"] == 384
    assert parseq.WEIGHTS_URL.endswith("parseq-bb5792a6.pt")


def test_predict_format(model):
    x = torch.randint(0, 255, (3, 3, 32, 128), dtype=torch.uint8)
    out = parseq.predict(model, x, batch=2)
    assert len(out) == 3
    assert all(isinstance(t, str) and t == t.upper() and 0 <= c <= 1 for t, c in out)


def test_plate_accuracy(model, monkeypatch):
    monkeypatch.setattr(parseq, "predict", lambda m, x: [("AB1", 1.0), ("XY", 0.5)])
    assert parseq.plate_accuracy(model, torch.zeros(2, 3, 32, 128), ["AB1", "XY2"]) == 0.5


def synthetic(n):
    texts = [["AB12", "XY9", "K1"][i % 3] for i in range(n)]
    return torch.randint(0, 255, (n, 3, 32, 128), dtype=torch.uint8), texts


def test_train_parseq_i_odczyt(tmp_path):
    xt, tt = synthetic(8)
    cfg = parseq.ParseqConfig(epochs=2, batch=4, eval_every=2)
    best, history = parseq.train_parseq(tmp_path, tmp_path / "out", cfg, "cpu", (xt, tt, xt[:2], tt[:2]))
    assert best.exists() and list(history.epoch) == [1, 2]
    assert np.isnan(history.score.iloc[0]) and not np.isnan(history.score.iloc[1])
    ckpt = torch.load(best, weights_only=True)
    assert ckpt["name"] == "parseq" and ckpt["config"]["epochs"] == 2
    reader = Recognizer(best, "cpu")
    found = reader.read([np.zeros((20, 80, 3), np.uint8)] * 2)
    assert len(found) == 2 and reader.read([]) == []


def test_train_parseq_od_wag_i_z_wyborem(tmp_path):
    xt, tt = synthetic(4)
    start = tmp_path / "start" / "best.pt"
    start.parent.mkdir()
    torch.save({"name": "parseq", "state": parseq.create(False, "cpu").state_dict()}, start)
    scores = iter([0.5, 0.4, 0.9])
    cfg = parseq.ParseqConfig(epochs=2, batch=4, init=str(start))
    best, history = parseq.train_parseq(
        tmp_path, tmp_path / "out", cfg, "cpu", (xt, tt, xt, tt), lambda m: next(scores)
    )
    assert history.epoch.tolist() == [0, 1, 2] and history.score.tolist() == [0.5, 0.4, 0.9]
    assert torch.load(best, weights_only=True)["epoch"] == 2


def test_training_data_z_dekonwolucja(tmp_path, monkeypatch):
    import alpr.deblur

    monkeypatch.setattr(parseq, "load_split", lambda *a, **k: synthetic(3) if a[1] == "train" else synthetic(1))
    crop = np.zeros((10, 40, 3), np.uint8)
    monkeypatch.setattr(alpr.deblur, "deconv_training_crops", lambda *a: ([crop, crop], ["Q1", "Q2"]))
    xt, tt, xv, tv = parseq.training_data(tmp_path, parseq.ParseqConfig(deconv=str(tmp_path), deconv_repeat=2))
    assert len(xt) == 3 + 2 * 2 and tt[3:] == ["Q1", "Q2", "Q1", "Q2"] and len(tv) == 1
