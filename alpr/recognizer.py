"""Własne sieci OCR uczone od zera na wycinkach z plates.csv: CRNN i LPRNet z dekodowaniem CTC (torch).

Wycinki trzymane w pamięci GPU, augmentacja (przesunięcie, obrót, skala, jasność, kontrast, szum) liczona wsadowo
na GPU, trening w mixed precision. Klasa 0 to pusty znak CTC, pozostałe to ALPHABET.
Dane: wycinki plate_crop, tablice z całych zdjęć wycięte polygonem (opcjonalnie kopie z losowo przesuniętymi
narożnikami) i opcjonalnie wycinki z naszego detektora i PlateAligner, takie jak w eksperymencie.

    model = train_recognizer("crnn", COMBINED, RUNS / "ocr" / "crnn")
    texts = Recognizer(RUNS / "ocr" / "crnn" / "best.pt").read([crop1, crop2])
"""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from alpr.config import SEED
from alpr.dataset import crop_path, load_plates, parse_polygon, read_image
from alpr.geometry import warp

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SIZES = {"crnn": (32, 128), "lprnet": (24, 94)}


def encode(text: str) -> list[int]:
    return [ALPHABET.index(c) + 1 for c in text if c in ALPHABET]


def ctc_decode(logits: torch.Tensor) -> list[tuple[str, float]]:
    """Zachłanne CTC na logits [B, T, C]: sklejenie powtórzeń, usunięcie pustych; pewność to średnie max p znaków."""
    probs = logits.float().softmax(-1)
    best, idx = probs.max(-1)
    out = []
    for p, k in zip(best.cpu().numpy(), idx.cpu().numpy(), strict=True):
        keep = (k != 0) & np.r_[True, k[1:] != k[:-1]]
        text = "".join(ALPHABET[i - 1] for i in k[keep])
        out.append((text, float(p[keep].mean()) if keep.any() else 0.0))
    return out


class CRNN(nn.Module):
    """VGG z 5 poziomami splotów (wysokość 32 -> 1) i dwuwarstwowy BiLSTM; wyjście [B, 32, klasy] dla 32x128."""

    def __init__(self, classes: int = len(ALPHABET) + 1, hidden: int = 128):
        super().__init__()

        def block(cin, cout, pool):
            layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            return layers + ([nn.MaxPool2d(pool)] if pool else [])

        self.cnn = nn.Sequential(
            *block(3, 64, (2, 2)),
            *block(64, 128, (2, 2)),
            *block(128, 256, None),
            *block(256, 256, (2, 1)),
            *block(256, 256, None),
            *block(256, 256, (2, 1)),
            nn.Conv2d(256, 256, (2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.head = nn.Linear(2 * hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x).squeeze(2).transpose(1, 2)
        return self.head(self.rnn(f)[0])


class SmallBasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        mid = cout // 4
        self.block = nn.Sequential(
            nn.Conv2d(cin, mid, 1),
            nn.ReLU(),
            nn.Conv2d(mid, mid, (3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(mid, mid, (1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(mid, cout, 1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    """LPRNet (Zherzdev, Gruzdev 2018) dla wejścia 24x94; wyjście [B, 18, klasy].

    Normalizacja kontekstu globalnego liczona osobno dla każdej próbki (w oryginale po całym wsadzie),
    żeby odczyt nie zależał od tego, z jakimi wycinkami trafił do wsadu.
    """

    KEEP = (2, 6, 13, 22)

    def __init__(self, classes: int = len(ALPHABET) + 1, dropout: float = 0.5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 3, 3), (1, 1, 1)),
            SmallBasicBlock(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool3d((1, 3, 3), (2, 1, 2)),
            SmallBasicBlock(64, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SmallBasicBlock(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool3d((1, 3, 3), (4, 1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(64, 256, (1, 4)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(256, classes, (13, 1)),
            nn.BatchNorm2d(classes),
            nn.ReLU(),
        )
        self.container = nn.Conv2d(448 + classes, classes, 1)
        self.pools = [nn.AvgPool2d(5, 5), nn.AvgPool2d(5, 5), nn.AvgPool2d((4, 10), (4, 2)), nn.Identity()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kept = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.KEEP:
                kept.append(x)
        context = []
        for pool, f in zip(self.pools, kept, strict=True):
            f = pool(f)
            context.append(f / f.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6))
        return self.container(torch.cat(context, 1)).mean(2).transpose(1, 2)


MODELS = {"crnn": CRNN, "lprnet": LPRNet}


def to_tensor(crops: list[np.ndarray], size: tuple[int, int]) -> torch.Tensor:
    """Wycinki BGR -> uint8 [N, 3, H, W] RGB w rozmiarze sieci."""
    h, w = size
    arr = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1] for c in crops])
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x.float().div(127.5).sub(1.0)


def blur_kernels(n: int, gen: torch.Generator, device, size: int = 9) -> torch.Tensor:
    """Jądra [n, 1, size, size]: po połowie rozmycie ruchu (odcinek, losowa długość i kąt) i nieostrość (dysk)."""
    r = size // 2
    ys, xs = torch.meshgrid(
        torch.arange(-r, r + 1.0, device=device), torch.arange(-r, r + 1.0, device=device), indexing="ij"
    )
    rand = torch.rand(n, 3, device=device, generator=gen)
    angle = rand[:, 0, None, None] * math.pi
    length = 1 + rand[:, 1, None, None] * (size - 1)
    along = xs * torch.cos(angle) + ys * torch.sin(angle)
    across = -xs * torch.sin(angle) + ys * torch.cos(angle)
    motion = ((across.abs() <= 0.5) & (along.abs() <= length / 2)).float()
    radius = 0.5 + rand[:, 1, None, None] * r
    disk = (xs**2 + ys**2 <= radius**2).float()
    kernel = torch.where(rand[:, 2, None, None] < 0.5, motion, disk)
    return (kernel / kernel.sum((1, 2), keepdim=True))[:, None]


def blur(x: torch.Tensor, gen: torch.Generator, prob: float) -> torch.Tensor:
    """Z prawdopodobieństwem prob każda próbka dostaje losowe jądro rozmycia (splot grupowy, jedno jądro na próbkę)."""
    n, c, h, w = x.shape
    kernels = blur_kernels(n, gen, x.device)
    keep = torch.rand(n, device=x.device, generator=gen) >= prob
    identity = torch.zeros_like(kernels)
    identity[:, :, kernels.shape[-1] // 2, kernels.shape[-1] // 2] = 1
    kernels = torch.where(keep[:, None, None, None], identity, kernels).repeat_interleave(c, 0)
    pad = kernels.shape[-1] // 2
    out = F.conv2d(F.pad(x.reshape(1, n * c, h, w), (pad,) * 4, mode="replicate"), kernels, groups=n * c)
    return out.reshape(n, c, h, w)


def augment(x: torch.Tensor, gen: torch.Generator, strength: float = 1.0, blur_prob: float = 0.0) -> torch.Tensor:
    """Losowe przekształcenie afiniczne, rozmycie, jasność, kontrast i szum dla każdej próbki osobno; x po normalize."""
    n, dev = x.shape[0], x.device

    def u(lo, hi):
        return torch.rand(n, device=dev, generator=gen) * (hi - lo) + lo

    angle = u(-5, 5) * strength * math.pi / 180
    scale = 1 + u(-0.1, 0.1) * strength
    shear = u(-0.15, 0.15) * strength
    tx, ty = u(-0.06, 0.06) * strength, u(-0.1, 0.1) * strength
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack(
        [torch.stack([cos, -sin + shear, tx], 1), torch.stack([sin, cos, ty], 1)],
        1,
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    if blur_prob > 0:
        x = blur(x, gen, blur_prob)
    gain = 1 + u(-0.4, 0.4)[:, None, None, None] * strength
    bias = u(-0.3, 0.3)[:, None, None, None] * strength
    noise = torch.randn(x.shape, device=dev, generator=gen) * u(0, 0.08)[:, None, None, None] * strength
    return (x * gain + bias + noise).clamp(-1, 1)


def plate_samples(dataset: Path, split: str) -> pd.DataFrame:
    """Czytelne tablice z tekstem: wycinki plate_crop (data3) i tablice z całych zdjęć data1 (wycinane polygonem)."""
    crops = load_plates(dataset, sample_type="plate_crop", split=split, ocr_only=True)
    photos = load_plates(dataset, sample_type="full_photo", split=split, ocr_only=True)
    photos = photos[photos.polygon.notna()]
    return pd.concat([crops.assign(kind="crop"), photos.assign(kind="photo")], ignore_index=True)


def load_sample(dataset: Path, row) -> np.ndarray | None:
    if row.kind == "crop":
        return read_image(crop_path(dataset, row.crop))
    img = read_image(dataset / row.image)
    return None if img is None else warp(img, parse_polygon(row.polygon))


def jitter_poly(poly: np.ndarray, rng: np.random.Generator, amount: tuple[float, float] = (0.06, 0.15)) -> np.ndarray:
    """Narożniki przesunięte losowo o ułamek szerokości (x) i wysokości (y) tablicy, jak niedokładna ramka."""
    w = np.linalg.norm(poly[1] - poly[0])
    h = np.linalg.norm(poly[3] - poly[0])
    shift = rng.uniform(-1, 1, (4, 2)) * np.array(amount) * [w, h]
    return (poly + shift).astype(np.float32)


def jittered_samples(dataset: Path, rows: pd.DataFrame, copies: int, seed: int) -> tuple[list, list[str]]:
    """copies wycinków z przesuniętymi narożnikami dla każdej tablicy z całego zdjęcia."""
    rng = np.random.default_rng(seed)
    crops, texts = [], []
    for r in rows[rows.kind == "photo"].itertuples():
        img = read_image(dataset / r.image)
        if img is None:
            continue
        poly = parse_polygon(r.polygon)
        for _ in range(copies):
            crops.append(warp(img, jitter_poly(poly, rng)))
            texts.append(r.text)
    return crops, texts


def detector_samples(
    base: Path, photos: pd.DataFrame, weights: str | Path, align: bool = True
) -> tuple[list, list[str]]:
    """Wycinki z detektora i PlateAligner (jak w crop_plates) dopasowane do tablic photos (IoU >= 0.5).

    align=False: sama ramka detektora (jak detect_plates.py --no-align).

    photos: image (względem base), x1, y1, x2, y2, text; np. full_photo z plates.csv albo ground_truth.csv eksperymentu.
    """
    from alpr.detection import AlignedPlate, PlateAligner, PlateDetector
    from alpr.detection.metrics import best_match
    from alpr.detection.pipeline import detect_images

    groups = dict(tuple(photos.groupby("image")))
    detector, aligner = PlateDetector(weights), PlateAligner()
    crops, texts = [], []
    for rel, img, dets in detect_images(detector, sorted(groups), base, batch=8):
        boxes = np.array([d.box for d in dets], np.float32).reshape(-1, 4)
        for r in groups[rel].itertuples():
            i = best_match(boxes, np.array([r.x1, r.y1, r.x2, r.y2], np.float32))
            if i is None:
                continue
            det = dets[i]
            if det.corners is not None:
                plate = AlignedPlate.from_corners(img, det.corners)
            else:
                plate = aligner.align(img, det.box) if align else AlignedPlate.from_box(img, det.box)
            crops.append(plate.crop)
            texts.append(r.text)
    return crops, texts


def load_split(
    dataset: Path,
    split: str,
    size: tuple[int, int],
    jitter: int = 0,
    detector: str | Path | None = None,
    seed: int = SEED,
    extra: Path | None = None,
    align: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    """Wycinki podziału jako uint8 [N, 3, H, W] i teksty; wczytywanie w wątkach.

    jitter: tyle dodatkowych kopii z przesuniętymi narożnikami na tablicę z całego zdjęcia;
    detector: wagi YOLO, wtedy dochodzą wycinki z naszego detektora i PlateAligner;
    extra: katalog zbioru eksperymentu (build_experiment.py), z detectorem dochodzą wycinki wszystkich jego wersji
    zdjęć z tego podziału (zniekształcenia jak w eksperymencie, ale inne zdjęcia niż test);
    align=False: wycinki z detektora bez PlateAligner (sama ramka).
    """
    rows = plate_samples(dataset, split)
    with ThreadPoolExecutor(8) as pool:
        imgs = list(pool.map(lambda r: load_sample(dataset, r), rows.itertuples()))
    texts = rows.text.tolist()
    if jitter:
        found, found_texts = jittered_samples(dataset, rows, jitter, seed)
        imgs, texts = imgs + found, texts + found_texts
    if detector:
        photos = load_plates(dataset, sample_type="full_photo", split=split, ocr_only=True)
        found, found_texts = detector_samples(dataset, photos, detector, align)
        imgs, texts = imgs + found, texts + found_texts
    if detector and extra:
        gt = pd.read_csv(Path(extra) / "ground_truth.csv")
        found, found_texts = detector_samples(Path(extra), gt[gt.split == split], detector, align)
        imgs, texts = imgs + found, texts + found_texts
    keep = [i for i, img in enumerate(imgs) if img is not None and len(encode(texts[i]))]
    return to_tensor([imgs[i] for i in keep], size), [texts[i] for i in keep]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    augment: float = 1.0
    amp: bool = True
    seed: int = SEED
    jitter: int = 0
    detector: str | None = None
    blur: float = 0.0
    extra: str | None = None
    align: bool = True


def evaluate(model: nn.Module, x: torch.Tensor, texts: list[str], batch: int = 1024) -> dict:
    """Odsetek bezbłędnych odczytów i CAR (character_accuracy z alpr.ocr)."""
    from alpr.ocr import character_accuracy

    preds = predict(model, x, batch)
    return {
        "plate_acc": float(np.mean([p == t for (p, _), t in zip(preds, texts, strict=True)])),
        "car": float(np.mean([character_accuracy(p, t) for (p, _), t in zip(preds, texts, strict=True)])),
    }


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch: int = 1024) -> list[tuple[str, float]]:
    """Odczyty CTC dla uint8 [N, 3, H, W]."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out += ctc_decode(model(normalize(x[i : i + batch].to(dev))))
    return out


def train_recognizer(
    name: str,
    dataset: Path,
    out: Path,
    cfg: TrainConfig = TrainConfig(),
    device: str | None = None,
    data: tuple | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Uczy CRNN albo LPRNet stratą CTC; best.pt według plate_acc na val. Zwraca ścieżkę i historię epok.

    data: (x_train, t_train, x_val, t_val) zamiast wczytywania z dataset (testy).
    """
    torch.manual_seed(cfg.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    size = SIZES[name]
    xt, tt, xv, tv = data or (
        *load_split(dataset, "train", size, cfg.jitter, cfg.detector, cfg.seed, cfg.extra, cfg.align),
        *load_split(dataset, "val", size),
    )
    print(f"  {len(tt)} wycinków train, {len(tv)} val")
    xt, xv = xt.to(dev), xv.to(dev)
    labels = [torch.tensor(encode(t)) for t in tt]
    lengths = torch.tensor([len(lab) for lab in labels])
    targets = torch.zeros(len(labels), int(lengths.max()), dtype=torch.long)
    for i, lab in enumerate(labels):
        targets[i, : len(lab)] = lab
    targets, lengths = targets.to(dev), lengths.to(dev)

    model = MODELS[name]().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(xt) / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=steps, pct_start=0.15)
    use_amp = cfg.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(xt), device=dev, generator=gen)
        total = 0.0
        for i in range(0, len(xt), cfg.batch):
            idx = order[i : i + cfg.batch]
            x = augment(normalize(xt[idx]), gen, cfg.augment, cfg.blur)
            with torch.autocast(dev.type, enabled=use_amp):
                logits = model(x)
            logp = logits.float().log_softmax(-1).transpose(0, 1)
            steps_t = torch.full((len(idx),), logp.shape[0], dtype=torch.long, device=dev)
            tgt, tlen = targets[idx], lengths[idx]
            loss = ctc(logp, tgt, steps_t, tlen)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
        row = {"epoch": epoch, "loss": total / len(xt), **evaluate(model, xv, tv)}
        history.append(row)
        if row["plate_acc"] > best:
            best = row["plate_acc"]
            torch.save(
                {"name": name, "size": size, "state": model.state_dict(), "val": row, "config": cfg.__dict__},
                out / "best.pt",
            )
        if epoch % 5 == 0 or epoch == cfg.epochs:
            print(
                f"  epoka {epoch}: strata {row['loss']:.3f}, val plate_acc {row['plate_acc']:.3f}, CAR {row['car']:.3f}"
            )
    history = pd.DataFrame(history)
    history.to_csv(out / "history.csv", index=False)
    return out / "best.pt", history


class Recognizer:
    """Wytrenowana sieć z best.pt jako silnik wsadowy: wycinki BGR -> (tekst, pewność).

    Wagi PARSeq (alpr.parseq, "name": "parseq") czyta alpr.parseq.predict, wagi CRNN i LPRNet dekodowanie CTC.
    """

    def __init__(self, weights: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        self.parseq, self.model = None, None
        if ckpt["name"] == "parseq":
            from alpr import parseq

            self.parseq, self.size = parseq.load(weights, self.device), parseq.SIZE
        else:
            self.model = MODELS[ckpt["name"]]().to(self.device)
            self.model.load_state_dict(ckpt["state"])
            self.model.eval()
            self.size = tuple(ckpt["size"])

    def read(self, crops: list[np.ndarray], batch: int = 1024) -> list[tuple[str, float]]:
        if not crops:
            return []
        if self.parseq is not None:
            from alpr import parseq

            return parseq.predict(self.parseq, to_tensor(crops, self.size), min(batch, 512))
        return predict(self.model, to_tensor(crops, self.size), batch)
