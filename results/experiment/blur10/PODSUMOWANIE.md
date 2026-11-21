# Duży eksperyment: 10 poziomów rozmycia, 5 silników OCR

Przebieg z 21.11.2026 (Golba, RTX 4080). Bez poprawy zniekształconych zdjęć (bez dekonwolucji).
Te same zdjęcia z poprawą odczytu (dekonwolucja Włodarczyka): `PODSUMOWANIE_POPRAWA.md`.

## Konfiguracja

| Element | Ustawienie |
|---|---|
| Zbiór | `data/experiment_blur10` (`build_experiment.py --levels 10 --out data/experiment_blur10`): 40 tablic z data1 `test` (poprawione etykiety) x 21 wersji (840 zdjęć) |
| Zniekształcenia | tylko `motion_blur` i `defocus_blur`, poziomy 1-10 (`alpr.build.corruptions.BLUR_LEVELS`); `brightness` i `darken` pominięte |
| Poziomy | poziom 2k to siła k z `imagecorruptions` (obraz identyczny bit w bit), nieparzyste leżą między sąsiednimi siłami, poziom 1 jest słabszy od siły 1 |
| Detektor | YOLO `plates_ref` (`runs/detect/plates_ref`, wydanie `modele-2026-11-21`), `conf` 0.25, `--batch 8`, dopasowanie IoU >= 0.5 |
| EasyOCR | model `en`, strojony: `mag_ratio=1.5 low_text=0.5 add_margin=0.05`, wycinek z `PlateAligner` |
| Tesseract | 5.4 (UB Mannheim), `--psm 7`, binaryzacja Otsu (najlepszy wariant z pomiarów 16.11), wycinek z `PlateAligner` |
| LPRNet | od zera, `train_ocr.py --model lprnet --epochs 100 --lr 3e-3` (val plate_acc 0.895), wycinek z ramki (`--no-align`) |
| CRNN | `runs/ocr/crnn` (val plate_acc 0.943), wycinek z ramki |
| PARSeq | `runs/ocr/parseq_s0` (walidacja 0.972), jeden model, bez dekonwolucji, wycinek z ramki |

Ruch: (promień, sigma) `(5, 2) (10, 3) (12, 4) (15, 5) (15, 6) (15, 8) (15, 10) (15, 12) (18, 13) (20, 15)`.
Ostrość: (promień dysku, alias) `(2, 0.1) (3, 0.1) (3.5, 0.3) (4, 0.5) (5, 0.5) (6, 0.5) (7, 0.5) (8, 0.5) (9, 0.5) (10, 0.5)`.

## Odczyt bezbłędny: rozmycie ruchu

| Poziom | Detekcja | EasyOCR | Tesseract | LPRNet | CRNN | PARSeq |
|---|---|---|---|---|---|---|
| 0 (czyste) | 1.000 | 0.625 | 0.375 | 0.975 | 0.975 | 1.000 |
| 1 | 1.000 | 0.400 | 0.225 | 0.975 | 0.975 | 0.975 |
| 2 | 1.000 | 0.200 | 0.100 | 0.875 | 0.950 | 0.975 |
| 3 | 1.000 | 0.125 | 0.100 | 0.750 | 0.950 | 0.925 |
| 4 | 1.000 | 0.025 | 0.000 | 0.550 | 0.900 | 0.950 |
| 5 | 1.000 | 0.100 | 0.000 | 0.525 | 0.850 | 0.975 |
| 6 | 1.000 | 0.000 | 0.000 | 0.375 | 0.825 | 0.925 |
| 7 | 1.000 | 0.000 | 0.000 | 0.050 | 0.750 | 0.825 |
| 8 | 0.975 | 0.000 | 0.000 | 0.025 | 0.650 | 0.650 |
| 9 | 1.000 | 0.000 | 0.000 | 0.050 | 0.600 | 0.775 |
| 10 | 1.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.675 |

## Odczyt bezbłędny: rozmycie ostrości

| Poziom | Detekcja | EasyOCR | Tesseract | LPRNet | CRNN | PARSeq |
|---|---|---|---|---|---|---|
| 0 (czyste) | 1.000 | 0.625 | 0.375 | 0.975 | 0.975 | 1.000 |
| 1 | 1.000 | 0.475 | 0.225 | 0.925 | 0.975 | 0.975 |
| 2 | 1.000 | 0.225 | 0.100 | 0.750 | 0.875 | 0.975 |
| 3 | 1.000 | 0.275 | 0.125 | 0.750 | 0.825 | 0.925 |
| 4 | 1.000 | 0.175 | 0.100 | 0.650 | 0.825 | 0.850 |
| 5 | 1.000 | 0.075 | 0.025 | 0.400 | 0.700 | 0.750 |
| 6 | 1.000 | 0.000 | 0.025 | 0.325 | 0.625 | 0.700 |
| 7 | 1.000 | 0.000 | 0.050 | 0.275 | 0.475 | 0.550 |
| 8 | 1.000 | 0.000 | 0.000 | 0.100 | 0.425 | 0.475 |
| 9 | 1.000 | 0.000 | 0.000 | 0.025 | 0.325 | 0.400 |
| 10 | 1.000 | 0.000 | 0.000 | 0.000 | 0.175 | 0.300 |

## CAR (1 - odległość edycyjna / długość numeru): rozmycie ruchu

| Poziom | Detekcja | EasyOCR | Tesseract | LPRNet | CRNN | PARSeq |
|---|---|---|---|---|---|---|
| 0 (czyste) | 1.000 | 0.923 | 0.660 | 0.993 | 0.996 | 1.000 |
| 1 | 1.000 | 0.808 | 0.682 | 0.989 | 0.993 | 0.996 |
| 2 | 1.000 | 0.634 | 0.601 | 0.976 | 0.989 | 0.989 |
| 3 | 1.000 | 0.511 | 0.500 | 0.939 | 0.989 | 0.990 |
| 4 | 1.000 | 0.339 | 0.334 | 0.874 | 0.979 | 0.990 |
| 5 | 1.000 | 0.256 | 0.267 | 0.818 | 0.965 | 0.993 |
| 6 | 1.000 | 0.154 | 0.167 | 0.707 | 0.954 | 0.979 |
| 7 | 1.000 | 0.077 | 0.069 | 0.612 | 0.928 | 0.958 |
| 8 | 0.975 | 0.037 | 0.044 | 0.447 | 0.870 | 0.870 |
| 9 | 1.000 | 0.017 | 0.033 | 0.496 | 0.869 | 0.931 |
| 10 | 1.000 | 0.004 | 0.028 | 0.396 | 0.825 | 0.853 |

## CAR: rozmycie ostrości

| Poziom | Detekcja | EasyOCR | Tesseract | LPRNet | CRNN | PARSeq |
|---|---|---|---|---|---|---|
| 0 (czyste) | 1.000 | 0.923 | 0.660 | 0.993 | 0.996 | 1.000 |
| 1 | 1.000 | 0.745 | 0.629 | 0.983 | 0.993 | 0.993 |
| 2 | 1.000 | 0.535 | 0.429 | 0.873 | 0.961 | 0.996 |
| 3 | 1.000 | 0.495 | 0.442 | 0.817 | 0.944 | 0.983 |
| 4 | 1.000 | 0.375 | 0.342 | 0.750 | 0.930 | 0.969 |
| 5 | 1.000 | 0.230 | 0.193 | 0.617 | 0.895 | 0.905 |
| 6 | 1.000 | 0.139 | 0.166 | 0.508 | 0.821 | 0.862 |
| 7 | 1.000 | 0.057 | 0.094 | 0.473 | 0.772 | 0.765 |
| 8 | 1.000 | 0.000 | 0.021 | 0.358 | 0.710 | 0.723 |
| 9 | 1.000 | 0.004 | 0.024 | 0.279 | 0.651 | 0.681 |
| 10 | 1.000 | 0.007 | 0.007 | 0.204 | 0.569 | 0.599 |

## Średnio na wszystkich poziomach (odczyt bezbłędny)

| Silnik | Czyste | Rozmycie ruchu 1-10 | Rozmycie ostrości 1-10 |
|---|---|---|---|
| EasyOCR | 0.625 | 0.085 | 0.123 |
| Tesseract | 0.375 | 0.042 | 0.065 |
| LPRNet | 0.975 | 0.418 | 0.420 |
| CRNN | 0.975 | 0.795 | 0.622 |
| PARSeq | 1.000 | 0.865 | 0.690 |

## Wnioski

* Detekcja jest odporna na całą skalę: 97.5-100% tablic wykrytych na każdym poziomie. Cały spadek pochodzi z OCR.
* Silniki ogólne (EasyOCR, Tesseract) tracą połowę skuteczności już na poziomie 1-2, a od poziomu 6 nie odczytują
  bezbłędnie żadnej tablicy.
* Sieci uczone na tablicach z rozmyciem w augmentacji (CRNN, PARSeq) trzymają ponad 80% do poziomu 5-6 ruchu.
  LPRNet (bez rozmycia w augmentacji, bez wycinków z detektora) spada szybciej: poniżej 50% od poziomu 6 ruchu i 5 ostrości.
* Rozmycie ostrości jest trudniejsze niż ruch na tych samych poziomach: PARSeq na poziomie 10 ma 0.300 przy ostrości
  i 0.675 przy ruchu. Dysk o promieniu 8-10 px wymazuje kreski znaków, ruch zostawia je czytelne wzdłuż jednej osi.
* CAR spada dużo łagodniej niż odczyt bezbłędny: PARSeq przy ostrości 10 myli średnio 40% znaków, a całą tablicę
  odczytuje dobrze w 30% przypadków. Tu jest miejsce na poprawę obrazu (`PODSUMOWANIE_POPRAWA.md`).

## Zastrzeżenia

* 40 tablic na poziom: jedna tablica to 2.5 p.p., przedział ufności ok. +-5 p.p.; niemonotoniczność między sąsiednimi
  poziomami (np. PARSeq, ruch 8 i 9) to szum.
* CRNN, LPRNet i PARSeq uczyły się na zdjęciach `train` z rozmyciem tych samych rodzin (`imagecorruptions`), więc
  ich odporność na ten konkretny typ rozmycia może być większa niż na prawdziwe rozmycie.
* Wersje z poziomami parzystymi mają inny losowy kąt ruchu niż zbiór z 5 siłami (inny klucz losowości), więc nie
  są identyczne z `data/experiment`.
* Kąt ruchu losowany z zakresu -45 do 45 stopni, jak w `imagecorruptions`.

## Pliki

| Plik | Zawartość |
|---|---|
| `tabela.csv` | silnik, zniekształcenie, poziom, liczba tablic, detected, text_ok, car |
| `../blur10_<silnik>/` | `plates.csv`, `summary.csv`, `detections.csv`, `ocr_<silnik>.csv` dla każdego silnika |
