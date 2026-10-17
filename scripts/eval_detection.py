#!/usr/bin/env python3
"""Ewaluacja detektora tablic: Precision, Recall, F1, AP50, AP50-95 w podziale na warunki zdjęcia.

Użycie:
    python3 scripts/eval_detection.py --weights runs/detect/plates/weights/best.pt
    python3 scripts/eval_detection.py --weights best.pt --split val --name val_baseline
    # test na zdjęciach po filtrze (Stańczyk zapisuje je z tą samą strukturą katalogów co data/combined):
    python3 scripts/eval_detection.py --weights best.pt --images-root data/filtered/clahe --name clahe
    # tylko przeliczenie metryk z zapisanych predykcji (bez modelu):
    python3 scripts/eval_detection.py --preds results/detection_eval/baseline/predictions.csv

Grupy: źródło (data1 = PL, data2 = UC3M-LP), lighting, exposure, artificial_corruption, siła
zniekształcenia oraz oryginał / kopia. Atrybuty tablicy przenosimy na zdjęcie; jeśli tablice na
jednym zdjęciu mają różne wartości, zdjęcie trafia do grupy "mixed". Predykcje liczone są z progiem
0.001 (potrzebne do AP), a P/R/F1 podajemy dla progu --conf. Metryki: alpr/detection/metrics.py.

Wynik w results/detection_eval/<name>/:
    predictions.csv    image, conf, x1, y1, x2, y2 (wszystkie predykcje)
    metrics.csv        group, value, images, plates, P, R, F1, AP50, AP50_95
    plates.csv         plate_id, image, detected (IoU >= 0.5 przy progu --conf), iou, conf
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # działa też bez pip install -e .

from alpr.config import RESULTS, default_dataset
from alpr.dataset import SPLITS, load_plates
from alpr.detection import detect, load_model
from alpr.detection.metrics import BOX, GROUPS, evaluate, image_groups


def predict_all(weights: str, images: list[str], base: Path, imgsz: int) -> pd.DataFrame:
    model = load_model(weights)
    rows = []
    for i, rel in enumerate(images, 1):
        img = cv2.imread(str(base / rel))
        if img is None:
            print(f"  BŁĄD: nie da się wczytać {base / rel}")
            continue
        for det in detect(model, img, conf=0.001, imgsz=imgsz):
            rows.append({"image": rel, "conf": det.conf, **dict(zip(BOX, det.box, strict=True))})
        if i % 100 == 0:
            print(f"  {i}/{len(images)}")
    return pd.DataFrame(rows, columns=["image", "conf", *BOX])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", help="wagi YOLO; niepotrzebne z --preds")
    parser.add_argument("--preds", type=Path, help="gotowy predictions.csv (pomija uruchamianie modelu)")
    parser.add_argument("--dataset", type=Path, default=default_dataset(), help="katalog zbioru (z plates.csv)")
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--images-root", type=Path, default=None,
                        help="katalog z obrazami o tych samych ścieżkach względnych co zbiór (np. po filtrze)")
    parser.add_argument("--conf", type=float, default=0.25, help="próg pewności dla P/R/F1")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()
    if not args.weights and not args.preds:
        parser.error("podaj --weights albo --preds")

    images_root = (args.images_root or args.dataset).resolve()
    gt = load_plates(args.dataset, sample_type="full_photo", split=args.split)
    images = sorted(gt.image.unique())
    out = RESULTS / "detection_eval" / args.name
    out.mkdir(parents=True, exist_ok=True)

    if args.preds:
        preds = pd.read_csv(args.preds)
    else:
        print(f"Predykcje dla {len(images)} zdjęć ({args.split}) z {images_root}")
        preds = predict_all(args.weights, images, images_root, args.imgsz)
        preds.to_csv(out / "predictions.csv", index=False)

    groups = image_groups(gt)
    overall, all_plates = evaluate(gt, preds, args.conf)
    results = [{"group": "all", "value": "all", **overall}]
    for col in GROUPS:
        for value, imgs in groups.groupby(col).groups.items():
            subset = gt[gt.image.isin(imgs)]
            metrics, _ = evaluate(subset, preds[preds.image.isin(imgs)], args.conf)
            results.append({"group": col, "value": value, **metrics})

    table = pd.DataFrame(results)
    table.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    all_plates.to_csv(out / "plates.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Wynik: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
