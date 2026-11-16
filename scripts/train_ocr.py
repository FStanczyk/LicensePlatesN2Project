#!/usr/bin/env python3
"""Uczy sieć OCR na czytelnych tablicach z plates.csv: CRNN albo LPRNet od zera (CTC) albo dotrenowuje PARSeq.

Dane: wycinki plate_crop (data3) i tablice z całych zdjęć data1 wycięte polygonem, train do uczenia, val do wyboru
epoki (bez blurred, z kopiami zniekształconymi). Augmentacja i trening na GPU w mixed precision, ziarno SEED.
PARSeq wymaga pip install -e ".[parseq]"; startuje z wag strhub (albo --init) i domyślnie ma przepis z eksperymentu
(20 epok, lr 1e-4, rozmycie w augmentacji 0.3, 4 kopie z przesuniętymi narożnikami).

Użycie:
    python3 scripts/train_ocr.py --model crnn
    python3 scripts/train_ocr.py --model lprnet --epochs 100 --lr 3e-3
    python3 scripts/train_ocr.py --model crnn --epochs 2 --device cpu                # szybki test
    python3 scripts/train_ocr.py --model crnn --jitter 4 --detector best.pt --blur 0.3 --extra data/experiment_tune
    python3 scripts/train_ocr.py --model parseq --detector best.pt --extra data/experiment_tune --out runs/ocr/parseq_s0
    python3 scripts/train_ocr.py --model parseq --init runs/ocr/parseq_s0/best.pt --deconv data/experiment_tune \\
        --detector best.pt --select data/experiment_val --epochs 6 --lr 3e-5 --out runs/ocr/parseq_dec_s0

--deconv: wycinki rozmytych wersji zdjęć train po dekonwolucji prawdziwym jądrem (alpr.deblur), 3 razy częściej.
--select: epoka wybierana po odczycie z dekonwolucją na tym zbiorze eksperymentu (zamiast plate_acc na val).

Wynik w runs/ocr/<model>/: best.pt i history.csv. Silnik w run_experiment.py i ocr_plates.py: --engine crnn / parseq
(inne wagi: --param weights=ścieżka/best.pt).
"""

import argparse
import sys
from pathlib import Path

from alpr.config import RUNS, default_dataset
from alpr.recognizer import MODELS, TrainConfig, train_recognizer

PARSEQ = {"epochs": 20, "batch": 128, "lr": 1e-4, "jitter": 4, "blur": 0.3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=[*sorted(MODELS), "parseq"], required=True)
    parser.add_argument("--dataset", type=Path, default=default_dataset())
    parser.add_argument("--epochs", type=int, help=f"domyślnie {TrainConfig.epochs}, PARSeq {PARSEQ['epochs']}")
    parser.add_argument("--batch", type=int, help=f"domyślnie {TrainConfig.batch}, PARSeq {PARSEQ['batch']}")
    parser.add_argument("--lr", type=float, help=f"domyślnie {TrainConfig.lr}, PARSeq {PARSEQ['lr']}")
    parser.add_argument("--augment", type=float, default=TrainConfig.augment, help="siła augmentacji, 0 = bez")
    parser.add_argument("--jitter", type=int, help="kopie z przesuniętymi narożnikami na tablicę z data1 (PARSeq: 4)")
    parser.add_argument("--detector", help="wagi YOLO: dodaj wycinki z naszego detektora i PlateAligner")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--blur", type=float, help="prawdopodobieństwo augmentacji rozmyciem (PARSeq: 0.3)")
    parser.add_argument("--extra", help="z --detector: zbiór eksperymentu z train (np. data/experiment_tune)")
    parser.add_argument("--no-align", action="store_true", help="z --detector: wycinki z samej ramki, bez PlateAligner")
    parser.add_argument("--init", help="PARSeq: wagi startowe zamiast wag strhub")
    parser.add_argument("--deconv", help="PARSeq: zbiór eksperymentu, z którego dochodzą wycinki po dekonwolucji")
    parser.add_argument("--select", type=Path, help="PARSeq z --detector: zbiór eksperymentu do wyboru epoki")
    parser.add_argument("--eval-every", type=int, default=1, help="PARSeq: ocena co tyle epok")
    parser.add_argument("--device", help="np. cuda, cpu; domyślnie GPU, jeśli jest")
    parser.add_argument("--out", type=Path, help="domyślnie runs/ocr/<model>")
    args = parser.parse_args()

    if not (args.dataset / "plates.csv").exists():
        print(f"Brak {args.dataset / 'plates.csv'}. Pobierz zbiór: python3 scripts/get_data.py")
        return 1
    defaults = PARSEQ if args.model == "parseq" else {"epochs": TrainConfig.epochs, "batch": TrainConfig.batch}
    defaults = {"lr": TrainConfig.lr, "jitter": 0, "blur": 0.0} | defaults
    common = {k: getattr(args, k) if getattr(args, k) is not None else defaults[k] for k in defaults}
    common |= {"augment": args.augment, "seed": args.seed, "detector": args.detector, "extra": args.extra}
    out = args.out or RUNS / "ocr" / args.model
    if args.model == "parseq":
        from alpr.parseq import ParseqConfig, deblur_select, train_parseq

        cfg = ParseqConfig(
            **common, align=not args.no_align, init=args.init, deconv=args.deconv, eval_every=args.eval_every
        )
        select = deblur_select(args.select, args.detector) if args.select and args.detector else None
        best, history = train_parseq(args.dataset.resolve(), out, cfg, args.device, select=select)
        top = history.loc[history.score.idxmax()]
        print(f"Najlepsza epoka {int(top.epoch)}: wynik {top.score:.3f}")
    else:
        cfg = TrainConfig(**common, align=not args.no_align)
        best, history = train_recognizer(args.model, args.dataset.resolve(), out, cfg, args.device)
        top = history.loc[history.plate_acc.idxmax()]
        print(f"Najlepsza epoka {int(top.epoch)}: val plate_acc {top.plate_acc:.3f}, CAR {top.car:.3f}")
    print(f"Wagi: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
