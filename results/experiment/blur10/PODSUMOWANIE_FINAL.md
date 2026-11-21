# Podsumowanie końcowe: odczyt tablic przy 10 poziomach rozmycia, z poprawą i bez

Stan na 21.11.2026 (gałąź `rozmycie_10_mati`). Plik zbiera w jednym miejscu: dane, detekcję, silniki OCR, sposób
zniekształcania zdjęć i wyniki dla 3 ziaren. Wcześniejsze, częściowe zestawienia: `PODSUMOWANIE.md` (bez poprawy,
ziarno 0) i `PODSUMOWANIE_POPRAWA.md` (z poprawą, ziarno 0). Wszystkie liczby z tego pliku są w `tabela_final.csv`.

## 1. Zbiór danych

`data/combined` (przebudowa 21.11 z poprawionymi etykietami), trzy źródła. Liczby dotyczą oryginałów; każda tablica ma
dodatkowo jedną kopię ze sztucznym zniekształceniem (`artificial_corruption`), co podwaja liczbę wierszy do 16584.

| Źródło | Typ | train | val | test | Razem |
|---|---|---|---|---|---|
| data1 (Polska) | całe zdjęcia | 210 | 41 | 39 | 290 |
| data2 (UC3M-LP, zamazane znaki) | całe zdjęcia | 539 | 101 | 144 | 784 |
| data3 (AUTO.RIA) | gotowe wycinki tablic | 6317 | 311 | 372 | 7000 |
| Razem | | 7066 | 453 | 555 | 8074 |

* Całe zdjęcia (1074) mają razem 1292 tablice (train 905, val 175, test 212).
* Detekcja (`data/combined/detection`, oryginały i kopie zniekształcone): 1496 zdjęć train, 284 val, 368 test.
* OCR: tablice z czytelnym tekstem, czyli data1 i data3; data2 ma zamazane znaki (`blurred == 1`) i służy tylko detekcji.
* Podziały są rozłączne po numerze tablicy: ten sam numer nie występuje w dwóch podziałach (audyt etykiet Włodarczyka,
  `data1_205` i `data1_253` to to samo auto, oba w `test`).

Zbiory eksperymentu (`scripts/build_experiment.py`, całe zdjęcia data1, wycinanie i odczyt robi cały potok):

| Zbiór | Podział | Tablic | Wersji na tablicę | Zdjęć | Do czego |
|---|---|---|---|---|---|
| `data/experiment_blur10` | test | 40 (39 zdjęć) | 21: czyste + 2 rozmycia x 10 poziomów | 840 | **ten eksperyment** |
| `data/experiment` | test | 40 | 21: czyste + 4 zniekształcenia x 5 sił | 840 | wcześniejsze eksperymenty (v3) |
| `data/experiment_tune` | train + val | 250 | 21 | 5250 | trening OCR (`--extra`, `--deconv`) |
| `data/experiment_val` | val | 41 | 21 | 861 | wybór epoki PARSeq po dekonwolucji (`--select`) |

## 2. Detekcja tablic (YOLO)

* Model: `yolo11n.pt` (ultralytics, wagi z COCO) dotrenowany na `data/combined/detection` jako detektor jednej klasy
  (tablica), `scripts/train_yolo.py --name plates_ref`, wagi `runs/detect/plates_ref/weights/best.pt`.
* Parametry: 100 epok, wczesne zatrzymanie po 20 epokach bez poprawy, wsad 16, obraz 640 px, `optimizer=auto`
  (lr0 0.01), augmentacja domyślna ultralytics (mozaika, odbicie poziome, zmiany HSV), ziarno 0, `deterministic=True`.
  Epoka wybierana po metrykach na `val`. Ok. 10 min na RTX 4080.
* Wynik na `val`: precyzja 0.968, czułość 0.955, mAP50 0.977, mAP50-95 0.799.
* W eksperymencie wycinek to sama ramka YOLO (`--no-align`) dla sieci (LPRNet, CRNN, PARSeq), a dla EasyOCR i
  Tesseracta ramka wyprostowana przez `PlateAligner` (najlepsze ustawienie każdego silnika według pomiarów Włodarczyka).
  Tablica liczy się jako wykryta przy IoU >= 0.5 z ramką z ground truth.
* Detektor nie był trenowany dla różnych ziaren: we wszystkich przebiegach jest ten sam `plates_ref` (ziarno 0).

## 3. Silniki OCR

| Silnik | Co to jest | Uczenie na naszych danych | Confidence odczytu |
|---|---|---|---|
| EasyOCR | gotowy model (CRAFT + CRNN), strojone parametry `mag_ratio=1.5`, `low_text=0.5`, `add_margin=0.05` | brak, tylko strojenie parametrów na `experiment_tune` | średnia confidence fragmentów ważona długością |
| Tesseract 5.4 | gotowy silnik (LSTM uczony na dokumentach), binaryzacja Otsu, lista dozwolonych znaków | brak | średnia `conf` słów; z listą znaków prawie zawsze 0 |
| LPRNet | lekka sieć CTC (Zherzdev, Gruzdev 2018), wejście 24x94 | od zera: `train_ocr.py --model lprnet --epochs 100 --lr 3e-3` (bez rozmycia w augmentacji) | średnia max p znaków (CTC) |
| CRNN | VGG + dwuwarstwowy BiLSTM, CTC, wejście 32x128 | od zera: `train_ocr.py --model crnn --jitter 4 --detector plates_ref --blur 0.3 --extra data/experiment_tune` (60 epok) | średnia max p znaków (CTC) |
| PARSeq | transformer PARSeq-S (Bautista, Atienza 2022) z wagami strhub | dotrenowany: `train_ocr.py --model parseq --detector plates_ref --extra data/experiment_tune` (20 epok, lr 1e-4) | iloczyn prawdopodobieństw znaków |
| PARSeq x2 | dwa modele PARSeq: `parseq_s<n>` i `parseq_dec_s<n>`, wygrywa pewniejszy odczyt | drugi model to `parseq_s<n>` dotrenowany dalej na wycinkach po dekonwolucji (niżej) | confidence zwycięskiego odczytu |

Wspólne dla sieci (LPRNet, CRNN, PARSeq):
* Dane: `train` z `data/combined` (wycinki data3 i tablice z całych zdjęć data1 wycięte poligonem, z kopiami
  zniekształconymi), wybór epoki na `val`. CRNN i PARSeq dostają też wycinki z naszego detektora (`--detector`), 4 kopie
  z przesuniętymi narożnikami (`--jitter 4`) i zdjęcia z `experiment_tune` (`--extra`, 21 wersji 250 tablic z train i val).
* Augmentacja na GPU: losowe przekształcenie afiniczne, jasność, kontrast, szum; CRNN i PARSeq dodatkowo w 30% próbek
  losowe jądro rozmycia (motion blur albo dysk nieostrości).
* Ziarna: każdy model trenowany 3 razy (`--seed 0`, `1`, `2`), wagi w `runs/ocr/<model>` (ziarno 0) i
  `runs/ocr/<model>_s1`, `_s2`.

PARSeq x2, czyli pełny układ Włodarczyka (`run_experiment.py --deblur --ocr-weights parseq_s<n> parseq_dec_s<n>`):
1. `parseq_dec_s<n>`: start z wag `parseq_s<n>`, 6 epok, lr 3e-5, do danych dochodzą 3 razy częściej wycinki rozmytych
   wersji zdjęć `train` po dekonwolucji Richardsona-Lucy **prawdziwym jądrem** (`--deconv data/experiment_tune`).
   Epoka wybierana nie po `val`, tylko po odczycie z dekonwolucją na `experiment_val` (`--select`).
2. Przy odczycie każdy wycinek ma 16 widoków (bez zmian + 15 po dekonwolucji), oba modele czytają wszystkie, a wygrywa
   najpewniejszy odczyt (widok po dekonwolucji tylko, gdy jest pewniejszy o ponad 0.1 od wycinka bez zmian).

## 4. Zniekształcenia: 10 poziomów rozmycia

`alpr.build.corruptions.apply_blur`, `BLUR_LEVELS`; zbiór `build_experiment.py --levels 10 --out data/experiment_blur10`.
Rozmycie liczone na całym zdjęciu (przed detekcją), a nie na wycinku, więc pogarsza też detekcję.

* **Motion blur** (`motion_blur`): splot z odcinkiem o promieniu `r` i wagach Gaussa o `sigma` (funkcja
  `_motion_blur` z `imagecorruptions`), kąt losowy z przedziału -45..45 stopni, powtarzalny (ziarno z nazwy zdjęcia).
* **Gaussian blur** (`defocus_blur`): splot z dyskiem o promieniu `r` wygładzonym filtrem Gaussa o `alias`
  (`imagecorruptions.disk`). Nazwa „Gaussian blur” jest umowna: jądro to dysk nieostrości wygładzony Gaussem, a nie czysty filtr Gaussa.
* Poziom 2k jest bit w bit siłą k z `imagecorruptions` (sprawdza to test), poziomy nieparzyste leżą między sąsiednimi
  siłami, poziom 1 jest słabszy od siły 1 biblioteki.

| Poziom | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Motion blur: promień, sigma | 5, 2 | 10, 3 | 12, 4 | 15, 5 | 15, 6 | 15, 8 | 15, 10 | 15, 12 | 18, 13 | 20, 15 |
| Gaussian blur: promień, alias | 2, 0.1 | 3, 0.1 | 3.5, 0.3 | 4, 0.5 | 5, 0.5 | 6, 0.5 | 7, 0.5 | 8, 0.5 | 9, 0.5 | 10, 0.5 |

Poprawa odczytu (`--deblur`, metoda Włodarczyka, `alpr/deblur.py`): kąt ruchu szacowany z uśrednionego widma mocy
całego zdjęcia, 16 widoków wycinka (bez zmian, 5 dysków, 5 sił ruchu w obu kierunkach) po dekonwolucji
Richardsona-Lucy na GPU, wybór widoku po confidence OCR z marginesem 0.1. Jądra kandydujące to 5 sił z
`imagecorruptions`, więc na poziomach nieparzystych jądro jest tylko przybliżone.

## 5. Wyniki (test: 40 tablic x 21 wersji, 3 ziarna)

Metryka główna: odczyt bezbłędny (cały numer zgodny z etykietą; niewykryta tablica liczy się jako błąd). CAR
(dokładność znaków) i confidence dla każdego przebiegu są w `tabela_final.csv` (kolumny `car`, `conf`, `conf_ok`,
`conf_err`). Ziarno dotyczy treningu sieci OCR; detektor, zbiór i rozmycie są te same we wszystkich przebiegach.

### 5.1. Średnio na poziomach 1-10 i na czystych zdjęciach

Odczyt bezbłędny; dla sieci wartości ziaren 0 / 1 / 2 i średnia, EasyOCR i Tesseract są deterministyczne (jeden przebieg).

| Układ | Czyste | Motion blur 1-10 | Gaussian blur 1-10 | Średnia Motion blur | Średnia Gaussian blur |
|---|---|---|---|---|---|
| EasyOCR | 0.625 | 0.085 | 0.122 | **0.085** | **0.122** |
| EasyOCR + dekonwolucja | 0.650 | 0.258 | 0.125 | **0.258** | **0.125** |
| Tesseract | 0.375 | 0.043 | 0.065 | **0.043** | **0.065** |
| Tesseract + dekonwolucja | 0.225 | 0.028 | 0.050 | **0.028** | **0.050** |
| LPRNet | 0.975 / 0.975 / 0.975 (śr. 0.975) | 0.417 / 0.435 / 0.410 | 0.420 / 0.410 / 0.418 | **0.421** | **0.416** |
| LPRNet + dekonwolucja | 0.975 / 0.975 / 0.975 (śr. 0.975) | 0.677 / 0.733 / 0.677 | 0.465 / 0.465 / 0.450 | **0.696** | **0.460** |
| CRNN | 0.975 / 1.000 / 0.975 (śr. 0.983) | 0.795 / 0.823 / 0.828 | 0.622 / 0.655 / 0.635 | **0.815** | **0.637** |
| CRNN + dekonwolucja | 0.975 / 1.000 / 0.975 (śr. 0.983) | 0.830 / 0.872 / 0.857 | 0.640 / 0.675 / 0.662 | **0.853** | **0.659** |
| PARSeq | 1.000 / 1.000 / 1.000 (śr. 1.000) | 0.865 / 0.860 / 0.875 | 0.690 / 0.695 / 0.688 | **0.867** | **0.691** |
| PARSeq + dekonwolucja | 1.000 / 1.000 / 1.000 (śr. 1.000) | 0.930 / 0.935 / 0.927 | 0.740 / 0.748 / 0.743 | **0.931** | **0.743** |
| PARSeq x2 | 1.000 / 1.000 / 1.000 (śr. 1.000) | 0.935 / 0.933 / 0.933 | 0.757 / 0.758 / 0.753 | **0.933** | **0.756** |

### 5.2. Wyniki na każdym poziomie

Odczyt bezbłędny dla ziaren 0 / 1 / 2, średnia i średnia confidence odczytu (wszystkie wykryte tablice, średnio po ziarnach). Poziom 0 to czyste zdjęcia.

#### EasyOCR

| Poziom | Motion blur | Confidence | Gaussian blur | Confidence |
|---|---|---|---|---|
| 0 | 0.625 | 0.752 | 0.625 | 0.752 |
| 1 | 0.400 | 0.589 | 0.475 | 0.602 |
| 2 | 0.200 | 0.428 | 0.225 | 0.406 |
| 3 | 0.125 | 0.314 | 0.275 | 0.408 |
| 4 | 0.025 | 0.248 | 0.175 | 0.293 |
| 5 | 0.100 | 0.197 | 0.075 | 0.146 |
| 6 | 0.000 | 0.094 | 0.000 | 0.103 |
| 7 | 0.000 | 0.057 | 0.000 | 0.049 |
| 8 | 0.000 | 0.030 | 0.000 | 0.024 |
| 9 | 0.000 | 0.014 | 0.000 | 0.000 |
| 10 | 0.000 | 0.045 | 0.000 | 0.001 |

#### EasyOCR + dekonwolucja

| Poziom | Motion blur | Confidence | Gaussian blur | Confidence |
|---|---|---|---|---|
| 0 | 0.650 | 0.871 | 0.650 | 0.871 |
| 1 | 0.500 | 0.803 | 0.475 | 0.696 |
| 2 | 0.500 | 0.761 | 0.225 | 0.577 |
| 3 | 0.375 | 0.685 | 0.275 | 0.544 |
| 4 | 0.350 | 0.674 | 0.175 | 0.523 |
| 5 | 0.275 | 0.605 | 0.075 | 0.352 |
| 6 | 0.250 | 0.530 | 0.025 | 0.285 |
| 7 | 0.125 | 0.485 | 0.000 | 0.191 |
| 8 | 0.075 | 0.437 | 0.000 | 0.135 |
| 9 | 0.050 | 0.354 | 0.000 | 0.123 |
| 10 | 0.075 | 0.280 | 0.000 | 0.089 |

#### Tesseract

| Poziom | Motion blur | Confidence | Gaussian blur | Confidence |
|---|---|---|---|---|
| 0 | 0.375 | 0.006 | 0.375 | 0.006 |
| 1 | 0.225 | 0.015 | 0.225 | 0.021 |
| 2 | 0.100 | 0.015 | 0.100 | 0.048 |
| 3 | 0.100 | 0.031 | 0.125 | 0.027 |
| 4 | 0.000 | 0.024 | 0.100 | 0.004 |
| 5 | 0.000 | 0.019 | 0.025 | 0.024 |
| 6 | 0.000 | 0.042 | 0.025 | 0.024 |
| 7 | 0.000 | 0.040 | 0.050 | 0.019 |
| 8 | 0.000 | 0.056 | 0.000 | 0.021 |
| 9 | 0.000 | 0.030 | 0.000 | 0.015 |
| 10 | 0.000 | 0.023 | 0.000 | 0.035 |

#### Tesseract + dekonwolucja

| Poziom | Motion blur | Confidence | Gaussian blur | Confidence |
|---|---|---|---|---|
| 0 | 0.225 | 0.184 | 0.225 | 0.184 |
| 1 | 0.150 | 0.149 | 0.150 | 0.214 |
| 2 | 0.050 | 0.182 | 0.075 | 0.165 |
| 3 | 0.075 | 0.209 | 0.100 | 0.230 |
| 4 | 0.000 | 0.201 | 0.075 | 0.270 |
| 5 | 0.000 | 0.191 | 0.025 | 0.272 |
| 6 | 0.000 | 0.210 | 0.025 | 0.254 |
| 7 | 0.000 | 0.265 | 0.050 | 0.234 |
| 8 | 0.000 | 0.264 | 0.000 | 0.177 |
| 9 | 0.000 | 0.256 | 0.000 | 0.189 |
| 10 | 0.000 | 0.206 | 0.000 | 0.189 |

#### LPRNet

| Poziom | Motion blur s0 | s1 | s2 | śr. | Confidence | Gaussian blur s0 | s1 | s2 | śr. | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.975 | 0.975 | 0.975 | **0.975** | 0.975 | 0.975 | 0.975 | 0.975 | **0.975** | 0.975 |
| 1 | 0.975 | 0.925 | 0.925 | **0.942** | 0.973 | 0.925 | 0.925 | 0.925 | **0.925** | 0.969 |
| 2 | 0.875 | 0.900 | 0.875 | **0.883** | 0.963 | 0.750 | 0.750 | 0.750 | **0.750** | 0.897 |
| 3 | 0.750 | 0.750 | 0.700 | **0.733** | 0.934 | 0.750 | 0.750 | 0.725 | **0.742** | 0.857 |
| 4 | 0.550 | 0.650 | 0.650 | **0.617** | 0.885 | 0.650 | 0.625 | 0.650 | **0.642** | 0.808 |
| 5 | 0.525 | 0.550 | 0.550 | **0.542** | 0.844 | 0.400 | 0.375 | 0.425 | **0.400** | 0.691 |
| 6 | 0.375 | 0.350 | 0.250 | **0.325** | 0.778 | 0.325 | 0.325 | 0.325 | **0.325** | 0.628 |
| 7 | 0.050 | 0.125 | 0.125 | **0.100** | 0.705 | 0.275 | 0.200 | 0.200 | **0.225** | 0.576 |
| 8 | 0.025 | 0.050 | 0.025 | **0.033** | 0.668 | 0.100 | 0.100 | 0.150 | **0.117** | 0.504 |
| 9 | 0.050 | 0.025 | 0.000 | **0.025** | 0.627 | 0.025 | 0.025 | 0.025 | **0.025** | 0.416 |
| 10 | 0.000 | 0.025 | 0.000 | **0.008** | 0.590 | 0.000 | 0.025 | 0.000 | **0.008** | 0.396 |

#### LPRNet + dekonwolucja

| Poziom | Motion blur s0 | s1 | s2 | śr. | Confidence | Gaussian blur s0 | s1 | s2 | śr. | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.975 | 0.975 | 0.975 | **0.975** | 0.977 | 0.975 | 0.975 | 0.975 | **0.975** | 0.977 |
| 1 | 0.975 | 0.925 | 0.925 | **0.942** | 0.973 | 0.925 | 0.925 | 0.925 | **0.925** | 0.971 |
| 2 | 0.900 | 0.925 | 0.875 | **0.900** | 0.968 | 0.775 | 0.775 | 0.775 | **0.775** | 0.935 |
| 3 | 0.825 | 0.850 | 0.800 | **0.825** | 0.967 | 0.750 | 0.775 | 0.725 | **0.750** | 0.921 |
| 4 | 0.725 | 0.800 | 0.800 | **0.775** | 0.963 | 0.675 | 0.675 | 0.675 | **0.675** | 0.888 |
| 5 | 0.625 | 0.800 | 0.700 | **0.708** | 0.958 | 0.400 | 0.375 | 0.450 | **0.408** | 0.834 |
| 6 | 0.650 | 0.725 | 0.550 | **0.642** | 0.945 | 0.400 | 0.375 | 0.350 | **0.375** | 0.805 |
| 7 | 0.525 | 0.600 | 0.575 | **0.567** | 0.925 | 0.300 | 0.250 | 0.225 | **0.258** | 0.780 |
| 8 | 0.525 | 0.650 | 0.575 | **0.583** | 0.925 | 0.150 | 0.175 | 0.225 | **0.183** | 0.751 |
| 9 | 0.525 | 0.550 | 0.550 | **0.542** | 0.907 | 0.150 | 0.175 | 0.125 | **0.150** | 0.737 |
| 10 | 0.500 | 0.500 | 0.425 | **0.475** | 0.883 | 0.125 | 0.150 | 0.025 | **0.100** | 0.684 |

#### CRNN

| Poziom | Motion blur s0 | s1 | s2 | śr. | Confidence | Gaussian blur s0 | s1 | s2 | śr. | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.975 | 1.000 | 0.975 | **0.983** | 0.988 | 0.975 | 1.000 | 0.975 | **0.983** | 0.988 |
| 1 | 0.975 | 0.975 | 0.975 | **0.975** | 0.989 | 0.975 | 0.975 | 0.975 | **0.975** | 0.987 |
| 2 | 0.950 | 0.975 | 0.975 | **0.967** | 0.990 | 0.875 | 0.950 | 0.875 | **0.900** | 0.970 |
| 3 | 0.950 | 0.950 | 0.950 | **0.950** | 0.984 | 0.825 | 0.850 | 0.875 | **0.850** | 0.964 |
| 4 | 0.900 | 0.875 | 0.950 | **0.908** | 0.979 | 0.825 | 0.825 | 0.825 | **0.825** | 0.948 |
| 5 | 0.850 | 0.925 | 0.900 | **0.892** | 0.970 | 0.700 | 0.750 | 0.725 | **0.725** | 0.923 |
| 6 | 0.825 | 0.875 | 0.875 | **0.858** | 0.960 | 0.625 | 0.600 | 0.525 | **0.583** | 0.898 |
| 7 | 0.750 | 0.725 | 0.800 | **0.758** | 0.944 | 0.475 | 0.550 | 0.550 | **0.525** | 0.866 |
| 8 | 0.650 | 0.675 | 0.675 | **0.667** | 0.918 | 0.425 | 0.500 | 0.450 | **0.458** | 0.835 |
| 9 | 0.600 | 0.625 | 0.650 | **0.625** | 0.922 | 0.325 | 0.375 | 0.375 | **0.358** | 0.804 |
| 10 | 0.500 | 0.625 | 0.525 | **0.550** | 0.894 | 0.175 | 0.175 | 0.175 | **0.175** | 0.764 |

#### CRNN + dekonwolucja

| Poziom | Motion blur s0 | s1 | s2 | śr. | Confidence | Gaussian blur s0 | s1 | s2 | śr. | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.975 | 1.000 | 0.975 | **0.983** | 0.988 | 0.975 | 1.000 | 0.975 | **0.983** | 0.988 |
| 1 | 0.975 | 0.975 | 0.975 | **0.975** | 0.990 | 0.975 | 0.975 | 0.975 | **0.975** | 0.987 |
| 2 | 0.950 | 0.975 | 0.975 | **0.967** | 0.990 | 0.900 | 0.950 | 0.900 | **0.917** | 0.977 |
| 3 | 0.950 | 0.950 | 0.950 | **0.950** | 0.987 | 0.825 | 0.925 | 0.900 | **0.883** | 0.974 |
| 4 | 0.925 | 0.900 | 0.950 | **0.925** | 0.984 | 0.850 | 0.825 | 0.875 | **0.850** | 0.969 |
| 5 | 0.875 | 0.950 | 0.925 | **0.917** | 0.984 | 0.700 | 0.775 | 0.775 | **0.750** | 0.950 |
| 6 | 0.850 | 0.950 | 0.925 | **0.908** | 0.979 | 0.625 | 0.625 | 0.550 | **0.600** | 0.927 |
| 7 | 0.750 | 0.850 | 0.850 | **0.817** | 0.969 | 0.500 | 0.550 | 0.550 | **0.533** | 0.904 |
| 8 | 0.725 | 0.775 | 0.675 | **0.725** | 0.970 | 0.450 | 0.525 | 0.450 | **0.475** | 0.887 |
| 9 | 0.700 | 0.675 | 0.700 | **0.692** | 0.959 | 0.325 | 0.350 | 0.400 | **0.358** | 0.861 |
| 10 | 0.600 | 0.725 | 0.650 | **0.658** | 0.954 | 0.250 | 0.250 | 0.250 | **0.250** | 0.844 |

#### PARSeq

| Poziom | Motion blur s0 | s1 | s2 | śr. | Confidence | Gaussian blur s0 | s1 | s2 | śr. | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | **1.000** | 0.997 | 1.000 | 1.000 | 1.000 | **1.000** | 0.997 |
| 1 | 0.975 | 0.975 | 0.975 | **0.975** | 0.980 | 0.975 | 0.975 | 0.975 | **0.975** | 0.986 |
| 2 | 0.975 | 0.975 | 0.975 | **0.975** | 0.981 | 0.975 | 0.975 | 0.975 | **0.975** | 0.963 |
| 3 | 0.925 | 0.950 | 0.975 | **0.950** | 0.967 | 0.925 | 0.950 | 0.950 | **0.942** | 0.931 |
| 4 | 0.950 | 0.950 | 0.950 | **0.950** | 0.939 | 0.850 | 0.850 | 0.875 | **0.858** | 0.881 |
| 5 | 0.975 | 0.975 | 0.975 | **0.975** | 0.957 | 0.750 | 0.775 | 0.750 | **0.758** | 0.786 |
| 6 | 0.925 | 0.925 | 0.925 | **0.925** | 0.893 | 0.700 | 0.700 | 0.675 | **0.692** | 0.709 |
| 7 | 0.825 | 0.775 | 0.825 | **0.808** | 0.818 | 0.550 | 0.550 | 0.525 | **0.542** | 0.603 |
| 8 | 0.650 | 0.650 | 0.725 | **0.675** | 0.738 | 0.475 | 0.475 | 0.500 | **0.483** | 0.529 |
| 9 | 0.775 | 0.750 | 0.750 | **0.758** | 0.732 | 0.400 | 0.375 | 0.375 | **0.383** | 0.438 |
| 10 | 0.675 | 0.675 | 0.675 | **0.675** | 0.691 | 0.300 | 0.325 | 0.275 | **0.300** | 0.366 |

#### PARSeq + dekonwolucja

| Poziom | Motion blur s0 | s1 | s2 | śr. | Confidence | Gaussian blur s0 | s1 | s2 | śr. | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | **1.000** | 0.997 | 1.000 | 1.000 | 1.000 | **1.000** | 0.997 |
| 1 | 1.000 | 1.000 | 1.000 | **1.000** | 0.996 | 0.975 | 0.975 | 0.975 | **0.975** | 0.987 |
| 2 | 0.975 | 1.000 | 0.975 | **0.983** | 0.988 | 0.975 | 0.975 | 0.975 | **0.975** | 0.980 |
| 3 | 0.975 | 0.975 | 0.975 | **0.975** | 0.988 | 0.975 | 0.975 | 0.975 | **0.975** | 0.954 |
| 4 | 0.950 | 0.950 | 0.950 | **0.950** | 0.988 | 0.900 | 0.900 | 0.900 | **0.900** | 0.945 |
| 5 | 0.975 | 0.950 | 0.950 | **0.958** | 0.987 | 0.700 | 0.800 | 0.775 | **0.758** | 0.827 |
| 6 | 0.950 | 0.950 | 0.950 | **0.950** | 0.974 | 0.775 | 0.750 | 0.775 | **0.767** | 0.813 |
| 7 | 0.950 | 0.950 | 0.950 | **0.950** | 0.962 | 0.625 | 0.650 | 0.625 | **0.633** | 0.786 |
| 8 | 0.900 | 0.900 | 0.900 | **0.900** | 0.960 | 0.550 | 0.550 | 0.550 | **0.550** | 0.668 |
| 9 | 0.875 | 0.900 | 0.875 | **0.883** | 0.948 | 0.525 | 0.525 | 0.550 | **0.533** | 0.644 |
| 10 | 0.750 | 0.775 | 0.750 | **0.758** | 0.905 | 0.400 | 0.375 | 0.325 | **0.367** | 0.598 |

#### PARSeq x2

| Poziom | Motion blur s0 | s1 | s2 | śr. | Confidence | Gaussian blur s0 | s1 | s2 | śr. | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | **1.000** | 0.998 | 1.000 | 1.000 | 1.000 | **1.000** | 0.998 |
| 1 | 1.000 | 1.000 | 1.000 | **1.000** | 0.997 | 0.975 | 1.000 | 0.975 | **0.983** | 0.989 |
| 2 | 0.975 | 1.000 | 0.975 | **0.983** | 0.991 | 0.950 | 0.975 | 0.925 | **0.950** | 0.984 |
| 3 | 0.975 | 0.975 | 0.975 | **0.975** | 0.992 | 0.950 | 0.950 | 0.925 | **0.942** | 0.969 |
| 4 | 0.950 | 0.950 | 0.950 | **0.950** | 0.990 | 0.900 | 0.900 | 0.900 | **0.900** | 0.956 |
| 5 | 0.975 | 0.950 | 0.950 | **0.958** | 0.988 | 0.750 | 0.750 | 0.775 | **0.758** | 0.861 |
| 6 | 0.950 | 0.950 | 0.950 | **0.950** | 0.982 | 0.800 | 0.775 | 0.800 | **0.792** | 0.868 |
| 7 | 0.950 | 0.925 | 0.925 | **0.933** | 0.976 | 0.625 | 0.625 | 0.625 | **0.625** | 0.829 |
| 8 | 0.900 | 0.900 | 0.900 | **0.900** | 0.973 | 0.625 | 0.600 | 0.625 | **0.617** | 0.787 |
| 9 | 0.925 | 0.900 | 0.925 | **0.917** | 0.965 | 0.575 | 0.575 | 0.600 | **0.583** | 0.741 |
| 10 | 0.750 | 0.775 | 0.775 | **0.767** | 0.935 | 0.425 | 0.425 | 0.375 | **0.408** | 0.714 |

### 5.3. Confidence odczytu

Wszystkie wykryte tablice ze wszystkich poziomów i ziaren. AUROC: prawdopodobieństwo, że losowy trafny odczyt ma wyższą confidence niż losowy błędny (0.5 = confidence nic nie mówi, 1 = idealnie rozdziela).

| Układ | Średnia confidence | Trafne | Błędne | AUROC | Odczyty z confidence 0 |
|---|---|---|---|---|---|
| EasyOCR | 0.229 | 0.873 | 0.134 | 0.980 | 51% |
| EasyOCR + dekonwolucja | 0.476 | 0.936 | 0.352 | 0.935 | 31% |
| Tesseract | 0.026 | 0.000 | 0.028 | 0.451 | 89% |
| Tesseract + dekonwolucja | 0.215 | 0.067 | 0.222 | 0.268 | 36% |
| LPRNet | 0.747 | 0.969 | 0.569 | 0.981 | 3% |
| LPRNet + dekonwolucja | 0.890 | 0.973 | 0.767 | 0.933 | 0% |
| CRNN | 0.929 | 0.983 | 0.774 | 0.972 | 0% |
| CRNN + dekonwolucja | 0.954 | 0.985 | 0.852 | 0.950 | 0% |
| PARSeq | 0.804 | 0.954 | 0.240 | 0.988 | 0% |
| PARSeq + dekonwolucja | 0.900 | 0.979 | 0.465 | 0.975 | 0% |
| PARSeq x2 | 0.928 | 0.986 | 0.589 | 0.976 | 0% |

### 5.4. Detekcja

Odsetek wykrytych tablic (IoU >= 0.5), ten sam detektor we wszystkich przebiegach.

| Poziom | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Motion blur | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.975 | 1.000 | 1.000 |
| Gaussian blur | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## 6. Wstępne wnioski

1. **Ranking silników się nie zmienia**: PARSeq > CRNN > LPRNet >> EasyOCR > Tesseract, z poprawą i bez. Rozrzut między
   ziarnami to zwykle 0-5 p.p. na poziomie (najwyżej 12.5 p.p.) i do 5.6 p.p. w średniej 1-10 (LPRNet z dekonwolucją),
   czyli dużo mniej niż różnice między silnikami.
2. **Dekonwolucja pomaga głównie przy Motion blur**. Średnio na poziomach 1-10 zysk wynosi: LPRNet +27.5 p.p.,
   EasyOCR +17.3, PARSeq +6.4, CRNN +3.8. Przy Gaussian blur zysk to tylko 0-5 p.p. (PARSeq x2 względem PARSeq bez poprawy 6.5 p.p.), a od poziomu 7 nawet
   PARSeq x2 odczytuje bezbłędnie mniej niż 63% tablic. Kąt ruchu da się dobrze oszacować z widma, a silny dysk
   nieostrości wymazuje informację.
3. **Tesseract z poprawą wypada gorzej** (czyste 0.375 -> 0.225, Motion blur 0.043 -> 0.028). Hipoteza zespołu była taka, że
   Tesseract „dopasował się za mocno”. Pomiary jej nie potwierdzają: Tesseracta nie uczyliśmy na naszych danych,
   używamy gotowego modelu. Przyczyną jest jego miara confidence. Z listą dozwolonych znaków w 89% odczytów zgłasza
   confidence 0, a trafne odczyty mają średnio niższą confidence niż błędne (AUROC 0.45 bez poprawy, 0.27 z poprawą, czyli
   gorzej niż losowo). Wybór widoku po confidence przepuszcza więc widoki po dekonwolucji także wtedy, gdy psują odczyt.
   Ten sam problem dotyczyłby każdego silnika, którego confidence nie mówi nic o trafności.
4. **PARSeq x2 daje najlepsze wyniki** (Motion blur 0.933, Gaussian blur 0.756), ale przewaga nad samym PARSeq z dekonwolucją jest
   mała (+0.2 i +1.3 p.p.). Prawie cały zysk daje dekonwolucja, a nie drugi model.
5. **Wysoka skuteczność dotyczy naszych danych i naszego rozmycia**. Wszystkie sieci (LPRNet, CRNN, oba PARSeq) są
   uczone albo dotrenowane na `train` z naszego zbioru, a CRNN i PARSeq także na `experiment_tune`, czyli tych samych
   rodzajach zniekształceń z `imagecorruptions`. Drugi PARSeq poszedł dalej: uczył się na wycinkach po dekonwolucji
   jądrami z tej samej rodziny, a epokę wybrano na `experiment_val` z tym samym rozmyciem. Jądra kandydujące w
   dekonwolucji to też dokładnie siły z `imagecorruptions`. Tablice testowe są rozłączne z uczącymi, ale pochodzą z
   tego samego źródła (data1, polskie tablice) i są rozmyte tym samym generatorem. Wynik 0.93 / 0.76 jest więc górną
   granicą dla tego układu. Na prawdziwym rozmyciu (inny kształt jądra, ruch nie po linii prostej), innych kamerach
   albo tablicach z innych krajów będzie niższy, o ile, tego nie zmierzyliśmy. Bez uczenia na naszych danych pracują
   tylko EasyOCR i Tesseract.
6. **Detekcja nie jest wąskim gardłem**: YOLO znajduje 97.5-100% tablic na każdym poziomie obu rozmyć. Spadek wyniku to
   prawie wyłącznie błędy OCR.

## 7. TODO

* **Kiedy w ogóle wołać dekonwolucję?** Dziś liczymy 16 widoków dla każdego wycinka. Czy wszystkie silniki zwracają
  confidence? Tak, każdy silnik w `alpr.ocr` ją zwraca, ale nie każda się do tego nadaje (tabela 5.3):
  * PARSeq, CRNN, LPRNet i EasyOCR mają confidence dobrze rozdzielającą trafne i błędne odczyty (AUROC 0.97-0.99 bez
    poprawy). Przykład PARSeq bez poprawy, 3 ziarna: 70% odczytów ma confidence >= 0.9 i z nich 99.6% jest trafnych.
    Poniżej 0.9 trafnych jest tylko 31%. CRNN przy tym samym progu: 78% odczytów i 92.5% trafnych powyżej, 9% poniżej.
    Warto sprawdzić regułę „dekonwolucja tylko, gdy confidence odczytu bez poprawy < próg”, która oszczędza większość
    obliczeń, i dobrać próg na `experiment_val` dla każdego silnika osobno.
  * Tesseract: confidence bezużyteczna (AUROC < 0.5). Potrzebna inna miara (np. odczyt bez listy znaków, zgodność odczytów
    z kilku widoków) albo dekonwolucja wyłączona dla Tesseracta.
  * Inny wyzwalacz, niezależny od OCR: siła rozmycia oszacowana z widma (`alpr.deblur.estimate_blur` i tak ją liczy).
* Sprawdzić uogólnienie: zdjęcia z prawdziwym rozmyciem (np. własne zdjęcia z ruchu) i tablice spoza data1, bez
  dotrenowania na nich.
* Poprawa dla silnego Gaussian blur (poziomy 7-10).
* Detektor trenowany tylko z ziarnem 0; dla pełnej powtarzalności można powtórzyć YOLO dla 3 ziaren.

## 8. Zastrzeżenia

* 40 tablic na poziom: jedna tablica to 2.5 p.p., więc różnice poniżej ok. 5 p.p. na pojedynczym poziomie są w granicach
  szumu. Średnie po 10 poziomach i 3 ziarnach (1200 odczytów) są pewniejsze.
* Na poziomach nieparzystych jądro rozmycia nie należy do zestawu kandydatów dekonwolucji (jest tylko przybliżone).
* Kąt motion blur jest ten sam w każdym przebiegu (zależy od nazwy zdjęcia, nie od ziarna treningu).

## 9. Odtworzenie

```bash
python3 scripts/build_experiment.py --levels 10 --out data/experiment_blur10
python3 scripts/train_ocr.py --model crnn --jitter 4 --detector runs/detect/plates_ref/weights/best.pt --blur 0.3 \
    --extra data/experiment_tune --seed 1 --out runs/ocr/crnn_s1
python3 scripts/train_ocr.py --model lprnet --epochs 100 --lr 3e-3 --seed 1 --out runs/ocr/lprnet_s1
python3 scripts/train_ocr.py --model parseq --detector runs/detect/plates_ref/weights/best.pt --extra data/experiment_tune \
    --seed 1 --out runs/ocr/parseq_s1
python3 scripts/train_ocr.py --model parseq --init runs/ocr/parseq_s1/best.pt --deconv data/experiment_tune \
    --detector runs/detect/plates_ref/weights/best.pt --select data/experiment_val --epochs 6 --lr 3e-5 --eval-every 2 \
    --seed 1 --out runs/ocr/parseq_dec_s1
python3 scripts/run_experiment.py --experiment data/experiment_blur10 --weights runs/detect/plates_ref/weights/best.pt \
    --batch 8 --no-align --deblur --engine parseq \
    --ocr-weights runs/ocr/parseq_s1/best.pt runs/ocr/parseq_dec_s1/best.pt --name blur10_deblur_parseq2_s1
```

Pozostałe przebiegi analogicznie: `--name blur10_<silnik>[_s1|_s2]` bez poprawy i `blur10_deblur_<silnik>[_s1|_s2]` z
`--deblur`; EasyOCR z `--param batch_size=16 mag_ratio=1.5 low_text=0.5 add_margin=0.05` i bez `--no-align`,
Tesseract z `--engine tesseract --param binarize=true` i bez `--no-align`. Wyniki w `results/experiment/blur10_*`.
