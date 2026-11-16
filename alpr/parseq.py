"""PARSeq (Bautista, Atienza 2022) dotrenowany na tablicach; kod modelu z pakietu strhub (extras [parseq]).

Start z wag uczonych na dużych zbiorach tekstu ze zdjęć, dotrenowanie na tych samych danych co CRNN
(alpr.recognizer.load_split) z augmentacją na GPU. Opcjonalnie dochodzą wycinki po dekonwolucji ze znanym jądrem
(alpr.deblur.deconv_training_crops), żeby sieć znała obraz po Richardsonie-Lucy.

    path, history = train_parseq(COMBINED, RUNS / "ocr" / "parseq", ParseqConfig(detector="best.pt"))
    texts = Recognizer(path).read([crop])          # alpr.recognizer, silnik "parseq" w alpr.ocr
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alpr.config import SEED
from alpr.dataset import normalize_text
from alpr.recognizer import augment, load_split, normalize

SIZE = (32, 128)


CONFIG = {
    "charset_train": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    "charset_test": "0123456789abcdefghijklmnopqrstuvwxyz",
    "max_label_length": 25,
    "batch_size": 384,
    "lr": 7e-4,
    "warmup_pct": 0.075,
    "weight_decay": 0.0,
    "img_size": [32, 128],
    "patch_size": [4, 8],
    "embed_dim": 384,
    "enc_num_heads": 6,
    "enc_mlp_ratio": 4,
    "enc_depth": 12,
    "dec_num_heads": 12,
    "dec_mlp_ratio": 4,
    "dec_depth": 1,
    "perm_num": 6,
    "perm_forward": True,
    "perm_mirrored": True,
    "decode_ar": True,
    "refine_iters": 1,
    "dropout": 0.1,
}
WEIGHTS_URL = "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt"


def create(pretrained: bool = True, device: str | None = None):
    """PARSeq-S z konfiguracją z configs/ repozytorium strhub (create_model wymaga katalogu repozytorium).

    pretrained: wagi z wydań GitHub, plik sprawdzany skrótem z nazwy (jak w strhub).
    """
    from strhub.models.parseq.system import PARSeq

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PARSeq(**CONFIG)
    if pretrained:
        state = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu", check_hash=True)
        model.model.load_state_dict(state)
    return model.to(dev)


def load(weights: str | Path, device: str | None = None):
    model = create(pretrained=False, device=device)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["state"])
    return model.eval()


@torch.no_grad()
def predict(model, x: torch.Tensor, batch: int = 512) -> list[tuple[str, float]]:
    """Odczyty dla uint8 [N, 3, 32, 128]; pewność to iloczyn prawdopodobieństw znaków (z końcem napisu)."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(normalize(x[i : i + batch].to(dev)))
        texts, probs = model.tokenizer.decode(logits.float().softmax(-1))
        out += [(normalize_text(t), float(p.prod())) for t, p in zip(texts, probs, strict=True)]
    return out


def plate_accuracy(model, x: torch.Tensor, texts: list[str]) -> float:
    return float(np.mean([p == t for (p, _), t in zip(predict(model, x), texts, strict=True)]))


@dataclass(frozen=True)
class ParseqConfig:
    epochs: int = 20
    batch: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.01
    augment: float = 1.0
    blur: float = 0.3
    clip: float = 20.0
    seed: int = SEED
    jitter: int = 4
    detector: str | None = None
    extra: str | None = None
    align: bool = True
    init: str | None = None
    deconv: str | None = None
    deconv_repeat: int = 3
    deconv_iters: int = 30
    eval_every: int = 1


def training_data(dataset: Path, cfg: ParseqConfig) -> tuple[torch.Tensor, list[str], torch.Tensor, list[str]]:
    """(x_train, teksty, x_val, teksty); z cfg.deconv dochodzą wycinki po dekonwolucji (deconv_repeat razy)."""
    from alpr.recognizer import to_tensor

    xt, tt = load_split(dataset, "train", SIZE, cfg.jitter, cfg.detector, cfg.seed, cfg.extra, cfg.align)
    xv, tv = load_split(dataset, "val", SIZE)
    if cfg.deconv:
        from alpr.deblur import deconv_training_crops

        crops, texts = deconv_training_crops(Path(cfg.deconv), "train", cfg.deconv_iters, cfg.seed)
        xd = to_tensor(crops, SIZE)
        xt = torch.cat([xt, *[xd] * cfg.deconv_repeat])
        tt = tt + texts * cfg.deconv_repeat
    return xt, tt, xv, tv


def train_parseq(
    dataset: Path,
    out: Path,
    cfg: ParseqConfig = ParseqConfig(),
    device: str | None = None,
    data: tuple | None = None,
    select: Callable | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Dotrenowanie PARSeq (strata permutacyjna z oryginału); best.pt według select(model) albo plate_acc na val.

    cfg.init: wagi startowe zamiast wag z wydań strhub. data: (x_train, t_train, x_val, t_val) zamiast wczytywania.
    select: funkcja model -> wynik (większy lepszy), np. odczyt z dekonwolucją na zbiorze walidacyjnym eksperymentu.
    """
    torch.manual_seed(cfg.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    xt, tt, xv, tv = data or training_data(dataset, cfg)
    print(f"  {len(tt)} wycinków train, {len(tv)} val")
    model = load(cfg.init, dev) if cfg.init else create(pretrained=data is None, device=dev)
    xt = xt.to(dev)

    def score() -> float:
        return float(select(model)) if select else plate_accuracy(model, xv, tv)

    def save(epoch: int, value: float) -> None:
        torch.save(
            {"name": "parseq", "state": model.state_dict(), "epoch": epoch, "score": value, "config": asdict(cfg)},
            out / "best.pt",
        )

    out.mkdir(parents=True, exist_ok=True)
    best = score() if cfg.init else -1.0
    history = [{"epoch": 0, "loss": np.nan, "score": best}] if cfg.init else []
    if cfg.init:
        save(0, best)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(xt) / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    use_amp = dev.type == "cuda"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        order = torch.randperm(len(xt), device=dev, generator=gen)
        for i in range(0, len(xt), cfg.batch):
            idx = order[i : i + cfg.batch]
            x = augment(normalize(xt[idx]), gen, cfg.augment, cfg.blur)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*self.log.*")
                loss = model.training_step((x, [tt[j] for j in idx.tolist()]), 0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        row = {"epoch": epoch, "loss": total / len(xt), "score": np.nan}
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            row["score"] = score()
            if row["score"] > best:
                best = row["score"]
                save(epoch, best)
            print(f"  epoka {epoch}: strata {row['loss']:.3f}, wynik {row['score']:.3f}")
        history.append(row)
    history = pd.DataFrame(history)
    history.to_csv(out / "history.csv", index=False)
    return out / "best.pt", history


def deblur_select(experiment: Path, detector: str | Path, margin: float = 0.1) -> Callable:
    """Funkcja model -> odsetek bezbłędnych odczytów wersji zniekształconych przy odczycie z dekonwolucją (val)."""
    from alpr.benchmark import DeblurScorer
    from alpr.detection import PlateDetector
    from alpr.recognizer import to_tensor

    scorer = DeblurScorer(experiment, PlateDetector(detector))
    return lambda model: scorer.score(lambda imgs: predict(model, to_tensor(imgs, SIZE)), margin)
