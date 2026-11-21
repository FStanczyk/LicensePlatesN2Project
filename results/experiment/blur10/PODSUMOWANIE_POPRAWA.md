# Pełny eksperyment: 10 poziomów rozmycia, 5 silników OCR, z poprawą odczytu i bez

Przebieg z 21.11.2026 (Golba, RTX 4080). Ten sam zbiór, detektor i silniki co w `PODSUMOWANIE.md` (bez poprawy);
tutaj dochodzą metody poprawy odczytu rozmytych tablic zaimplementowane przez Włodarczyka (`alpr/deblur.py`).
Wyniki dla 3 ziaren, pewność odczytu i wnioski: `PODSUMOWANIE_FINAL.md`.

## Metody poprawy

| Metoda | Opis | Gdzie |
|---|---|---|
| dekonwolucja (`run_experiment.py --deblur`) | kąt ruchu szacowany z widma mocy całego zdjęcia; każdy wycinek dostaje 16 widoków (bez zmian, 5 dysków, 5 sił ruchu w 2 kierunkach) po dekonwolucji Richardsona-Lucy na GPU; widok po dekonwolucji wygrywa tylko, gdy podnosi pewność odczytu o więcej niż 0.1 | wszystkie silniki |
| PARSeq x2 (`--ocr-weights parseq_s0 parseq_dec_s0`) | dekonwolucja jak wyżej i drugi PARSeq dotrenowany na wycinkach po dekonwolucji; wygrywa pewniejszy z dwóch odczytów | tylko PARSeq (pełny układ Włodarczyka) |

Wycinek jak w przebiegu bez poprawy: EasyOCR i Tesseract z `PlateAligner`, sieci z samej ramki (`--no-align`).
Detekcja jest wspólna (dekonwolucja działa na wycinkach), więc odsetek wykrytych tablic się nie zmienia (97.5-100%).

## Średnio na poziomach 1-10 (odczyt bezbłędny)

| Silnik | Czyste: bez / z | Ruch 1-10: bez / z | Ostrość 1-10: bez / z | Zysk ruch | Zysk ostrość |
|---|---|---|---|---|---|
| EasyOCR | 0.625 / 0.650 | 0.085 / 0.258 | 0.122 / 0.125 | +17.2 p.p. | +0.3 p.p. |
| Tesseract | 0.375 / 0.225 | 0.043 / 0.028 | 0.065 / 0.050 | -1.5 p.p. | -1.5 p.p. |
| LPRNet | 0.975 / 0.975 | 0.417 / 0.677 | 0.420 / 0.465 | +26.0 p.p. | +4.5 p.p. |
| CRNN | 0.975 / 0.975 | 0.795 / 0.830 | 0.622 / 0.640 | +3.5 p.p. | +1.7 p.p. |
| PARSeq | 1.000 / 1.000 | 0.865 / 0.930 | 0.690 / 0.740 | +6.5 p.p. | +5.0 p.p. |
| PARSeq x2 (pełny układ) | 1.000 / 1.000 | 0.865 / 0.935 | 0.690 / 0.757 | +7.0 p.p. | +6.8 p.p. |

## Odczyt bezbłędny: rozmycie ruchu (bez poprawy / z dekonwolucją)

| Poziom | EasyOCR | Tesseract | LPRNet | CRNN | PARSeq | PARSeq x2 |
|---|---|---|---|---|---|---|
| 0 (czyste) | 0.625 / 0.650 | 0.375 / 0.225 | 0.975 / 0.975 | 0.975 / 0.975 | 1.000 / 1.000 | 1.000 |
| 1 | 0.400 / 0.500 | 0.225 / 0.150 | 0.975 / 0.975 | 0.975 / 0.975 | 0.975 / 1.000 | 1.000 |
| 2 | 0.200 / 0.500 | 0.100 / 0.050 | 0.875 / 0.900 | 0.950 / 0.950 | 0.975 / 0.975 | 0.975 |
| 3 | 0.125 / 0.375 | 0.100 / 0.075 | 0.750 / 0.825 | 0.950 / 0.950 | 0.925 / 0.975 | 0.975 |
| 4 | 0.025 / 0.350 | 0.000 / 0.000 | 0.550 / 0.725 | 0.900 / 0.925 | 0.950 / 0.950 | 0.950 |
| 5 | 0.100 / 0.275 | 0.000 / 0.000 | 0.525 / 0.625 | 0.850 / 0.875 | 0.975 / 0.975 | 0.975 |
| 6 | 0.000 / 0.250 | 0.000 / 0.000 | 0.375 / 0.650 | 0.825 / 0.850 | 0.925 / 0.950 | 0.950 |
| 7 | 0.000 / 0.125 | 0.000 / 0.000 | 0.050 / 0.525 | 0.750 / 0.750 | 0.825 / 0.950 | 0.950 |
| 8 | 0.000 / 0.075 | 0.000 / 0.000 | 0.025 / 0.525 | 0.650 / 0.725 | 0.650 / 0.900 | 0.900 |
| 9 | 0.000 / 0.050 | 0.000 / 0.000 | 0.050 / 0.525 | 0.600 / 0.700 | 0.775 / 0.875 | 0.925 |
| 10 | 0.000 / 0.075 | 0.000 / 0.000 | 0.000 / 0.500 | 0.500 / 0.600 | 0.675 / 0.750 | 0.750 |

## Odczyt bezbłędny: rozmycie ostrości (bez poprawy / z dekonwolucją)

| Poziom | EasyOCR | Tesseract | LPRNet | CRNN | PARSeq | PARSeq x2 |
|---|---|---|---|---|---|---|
| 0 (czyste) | 0.625 / 0.650 | 0.375 / 0.225 | 0.975 / 0.975 | 0.975 / 0.975 | 1.000 / 1.000 | 1.000 |
| 1 | 0.475 / 0.475 | 0.225 / 0.150 | 0.925 / 0.925 | 0.975 / 0.975 | 0.975 / 0.975 | 0.975 |
| 2 | 0.225 / 0.225 | 0.100 / 0.075 | 0.750 / 0.775 | 0.875 / 0.900 | 0.975 / 0.975 | 0.950 |
| 3 | 0.275 / 0.275 | 0.125 / 0.100 | 0.750 / 0.750 | 0.825 / 0.825 | 0.925 / 0.975 | 0.950 |
| 4 | 0.175 / 0.175 | 0.100 / 0.075 | 0.650 / 0.675 | 0.825 / 0.850 | 0.850 / 0.900 | 0.900 |
| 5 | 0.075 / 0.075 | 0.025 / 0.025 | 0.400 / 0.400 | 0.700 / 0.700 | 0.750 / 0.700 | 0.750 |
| 6 | 0.000 / 0.025 | 0.025 / 0.025 | 0.325 / 0.400 | 0.625 / 0.625 | 0.700 / 0.775 | 0.800 |
| 7 | 0.000 / 0.000 | 0.050 / 0.050 | 0.275 / 0.300 | 0.475 / 0.500 | 0.550 / 0.625 | 0.625 |
| 8 | 0.000 / 0.000 | 0.000 / 0.000 | 0.100 / 0.150 | 0.425 / 0.450 | 0.475 / 0.550 | 0.625 |
| 9 | 0.000 / 0.000 | 0.000 / 0.000 | 0.025 / 0.150 | 0.325 / 0.325 | 0.400 / 0.525 | 0.575 |
| 10 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.125 | 0.175 / 0.250 | 0.300 / 0.400 | 0.425 |

## CAR: rozmycie ruchu (bez poprawy / z dekonwolucją)

| Poziom | EasyOCR | Tesseract | LPRNet | CRNN | PARSeq | PARSeq x2 |
|---|---|---|---|---|---|---|
| 0 (czyste) | 0.923 / 0.905 | 0.660 / 0.530 | 0.993 / 0.986 | 0.996 / 0.996 | 1.000 / 1.000 | 1.000 |
| 1 | 0.808 / 0.785 | 0.682 / 0.538 | 0.989 / 0.989 | 0.993 / 0.993 | 0.996 / 1.000 | 1.000 |
| 2 | 0.634 / 0.772 | 0.601 / 0.486 | 0.976 / 0.979 | 0.989 / 0.989 | 0.989 / 0.996 | 0.996 |
| 3 | 0.511 / 0.700 | 0.500 / 0.436 | 0.939 / 0.963 | 0.989 / 0.989 | 0.990 / 0.989 | 0.989 |
| 4 | 0.339 / 0.685 | 0.334 / 0.311 | 0.874 / 0.950 | 0.979 / 0.982 | 0.990 / 0.993 | 0.993 |
| 5 | 0.256 / 0.565 | 0.267 / 0.218 | 0.818 / 0.914 | 0.965 / 0.975 | 0.993 / 0.997 | 0.997 |
| 6 | 0.154 / 0.507 | 0.167 / 0.184 | 0.707 / 0.886 | 0.954 / 0.968 | 0.979 / 0.986 | 0.993 |
| 7 | 0.077 / 0.412 | 0.069 / 0.144 | 0.612 / 0.837 | 0.928 / 0.948 | 0.959 / 0.993 | 0.972 |
| 8 | 0.037 / 0.360 | 0.044 / 0.106 | 0.447 / 0.740 | 0.870 / 0.916 | 0.870 / 0.946 | 0.946 |
| 9 | 0.017 / 0.336 | 0.034 / 0.065 | 0.496 / 0.825 | 0.869 / 0.915 | 0.931 / 0.983 | 0.972 |
| 10 | 0.004 / 0.296 | 0.028 / 0.075 | 0.396 / 0.712 | 0.825 / 0.891 | 0.853 / 0.934 | 0.912 |

## CAR: rozmycie ostrości (bez poprawy / z dekonwolucją)

| Poziom | EasyOCR | Tesseract | LPRNet | CRNN | PARSeq | PARSeq x2 |
|---|---|---|---|---|---|---|
| 0 (czyste) | 0.923 / 0.905 | 0.660 / 0.530 | 0.993 / 0.986 | 0.996 / 0.996 | 1.000 / 1.000 | 1.000 |
| 1 | 0.745 / 0.688 | 0.629 / 0.433 | 0.983 / 0.983 | 0.993 / 0.993 | 0.993 / 0.993 | 0.993 |
| 2 | 0.535 / 0.534 | 0.429 / 0.400 | 0.873 / 0.894 | 0.961 / 0.964 | 0.996 / 0.982 | 0.986 |
| 3 | 0.495 / 0.506 | 0.442 / 0.396 | 0.817 / 0.861 | 0.944 / 0.955 | 0.983 / 0.979 | 0.975 |
| 4 | 0.375 / 0.370 | 0.342 / 0.277 | 0.750 / 0.800 | 0.930 / 0.930 | 0.969 / 0.983 | 0.972 |
| 5 | 0.230 / 0.239 | 0.193 / 0.182 | 0.617 / 0.594 | 0.895 / 0.885 | 0.905 / 0.902 | 0.891 |
| 6 | 0.139 / 0.210 | 0.166 / 0.158 | 0.508 / 0.596 | 0.821 / 0.832 | 0.862 / 0.891 | 0.893 |
| 7 | 0.057 / 0.091 | 0.094 / 0.073 | 0.473 / 0.571 | 0.772 / 0.773 | 0.765 / 0.796 | 0.808 |
| 8 | 0.000 / 0.042 | 0.021 / 0.049 | 0.358 / 0.460 | 0.710 / 0.723 | 0.723 / 0.745 | 0.792 |
| 9 | 0.004 / 0.047 | 0.024 / 0.027 | 0.280 / 0.464 | 0.651 / 0.671 | 0.681 / 0.757 | 0.771 |
| 10 | 0.007 / 0.056 | 0.007 / 0.030 | 0.204 / 0.376 | 0.569 / 0.612 | 0.599 / 0.692 | 0.683 |

## Wnioski

* Dekonwolucja pomaga głównie przy rozmyciu ruchu: PARSeq na poziomach 7-10 z 0.65-0.83 do 0.75-0.95, LPRNet
  z 0.00-0.05 do 0.50-0.53, EasyOCR średnio z 0.085 do 0.258. Kąt ruchu da się dobrze oszacować z widma zdjęcia.
* Przy rozmyciu ostrości zysk jest mniejszy (PARSeq +5.0 p.p., pełny układ +6.8 p.p., EasyOCR +0.3 p.p.).
  Od poziomu 7 nawet najlepszy układ odczytuje bezbłędnie poniżej 65% tablic: dysk o promieniu 7-10 px usuwa
  informację, której dekonwolucja nie odtworzy.
* Najlepszy układ (PARSeq x2) daje 0.935 na ruchu i 0.757 na ostrości; drugi model po dekonwolucji dodaje przede
  wszystkim przy silnej ostrości (poziomy 8-10: +2.5-7.5 p.p. ponad samą dekonwolucję).
* Na czystych zdjęciach dekonwolucja niczego nie psuje dla sieci (wybór widoku po pewności), a EasyOCR zyskuje
  jedną tablicę.
* Tesseract z dekonwolucją jest gorszy (czyste 0.375 -> 0.225). Z listą dozwolonych znaków zgłasza pewność 0 w 92%
  odczytów, więc wybór widoku po pewności faworyzuje dowolny widok po dekonwolucji z niezerową pewnością. Dla
  Tesseracta ta metoda wymaga innej miary pewności (np. bez `whitelist`).
* Zysk w CAR jest większy niż w odczycie bezbłędnym tam, gdzie silnik jest słaby (EasyOCR ruch 10: CAR 0.004 -> 0.296):
  dekonwolucja przywraca część znaków, ale rzadko całą tablicę.

## Zastrzeżenia

* Dekonwolucja zna rodzinę jąder z `imagecorruptions` (dyski i ruch o promieniach 1-5 biblioteki), więc na
  nieparzystych poziomach (parametry między siłami biblioteki) jądro jest tylko przybliżone. Na prawdziwym rozmyciu
  zysk może być mniejszy.
* Sieci (CRNN, LPRNet, PARSeq) uczyły się na rozmyciu tych samych rodzin; `parseq_dec_s0` także na wycinkach po
  dekonwolucji prawdziwym jądrem.
* 40 tablic na poziom: jedna tablica to 2.5 p.p., różnice poniżej ok. 2 tablic (5 p.p.) to szum.
* Jeden przebieg (ziarno 0) na model; Włodarczyk mierzył średnio z 3 ziaren.

## Pliki

| Plik | Zawartość |
|---|---|
| `tabela_pelna.csv` | silnik, metoda (brak, dekonwolucja, dekonwolucja + PARSeq po dekonwolucji), zniekształcenie, poziom, liczba tablic, detected, text_ok, car |
| `../blur10_<silnik>/` | przebiegi bez poprawy |
| `../blur10_deblur_<silnik>/`, `../blur10_deblur_parseq2/` | przebiegi z dekonwolucją i pełny układ PARSeq |

Polecenia (po `build_experiment.py --levels 10 --out data/experiment_blur10`):

```bash
python3 scripts/run_experiment.py --experiment data/experiment_blur10 --weights runs/detect/plates_ref/weights/best.pt     --batch 8 --no-align --deblur --engine parseq     --ocr-weights runs/ocr/parseq_s0/best.pt runs/ocr/parseq_dec_s0/best.pt --name blur10_deblur_parseq2
```
