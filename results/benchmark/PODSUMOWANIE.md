# Pomiary 15-16.11.2026: przyspieszenie, silniki OCR, odczyt rozmytych tablic

Wszystkie pomiary na RTX 4070 12 GB. Metryka: odsetek tablic odczytanych bezbłędnie (`text_ok`), osobno dla czystych
zdjęć i 20 zniekształconych wersji (rozjaśnienie, przyciemnienie, rozmycie ruchu i ostrości, siła 1-5).
Pełna tabela każdej zmierzonej konfiguracji: `pomiary.csv` (kolumna `zbior` mówi, na jakim zbiorze).

Zbiory:
* **test przed poprawką**: 39 tablic z data1 `test` x 21 wersji (etykieta `SBE674C` błędna, auto z `train` w teście),
* **test po poprawce**: 40 tablic x 21 wersji (poprawione etykiety, `data1_253` przeniesione z `train`),
* **train+val**: 251 tablic, do strojenia; **walidacja eksperymentu**: 41 tablic z `val`, do wyboru przepisów.

Ustawienia wybierano na train+val albo walidacji, `test` liczony raz na każdy wybrany przepis.

## Przyspieszenie (test przed poprawką, 819 zdjęć)

| Ustawienie | Detekcja | OCR | Uwagi |
|---|---|---|---|
| obecne (`main`): YOLO wsad 1, EasyOCR | 5.4 s | 4.1 s | cały `run_experiment.py` 12.8 s |
| YOLO `--batch 8` | 3.5 s | | ramki różnią się o 1-2 px |
| YOLO fp16 | 6.2 s | | wolniej, odrzucone |
| TensorRT fp16, wsad 21 | 3.3 s | | eksport 2.5 min, plik pod jedną kartę, odrzucone |
| EasyOCR `batch_size=16` | | 3.8 s | wynik bez zmian |
| EasyOCR bez CRAFT, wsadowo | | 0.6 s | -5 tablic na czystych, odrzucone |
| Richardson-Lucy, 16 jąder (FFT, GPU) | | 41 ms na wycinek | zamiast 510 ms (pojedyncze jądra) |

`build_experiment.py --workers 8`: 180 s -> 58 s, pliki bajt w bajt te same.

## Strojenie EasyOCR (train+val, potem test przed poprawką)

| Parametry | Czyste (train+val) | Zniekształcone (train+val) | Czyste (test) | Wszystkie (test) |
|---|---|---|---|---|
| domyślne | 0.450 | 0.238 | 0.564 | 0.259 |
| `low_text=0.5` | 0.542 | 0.269 | | |
| `mag_ratio=1.5 low_text=0.5 add_margin=0.05` | 0.606 | 0.302 | 0.615 | 0.295 |

## Silniki OCR (test przed poprawką, te same wycinki)

| Silnik | Czyste | Wszystkie wersje | OCR na wycinek |
|---|---|---|---|
| EasyOCR | 0.564 | 0.259 | 5.0 ms |
| EasyOCR strojony | 0.615 | 0.293 | 5.8 ms |
| Tesseract (najlepszy wariant, Otsu) | 0.436 | 0.189 | 7.1 ms |
| LPRNet od zera | 0.667 | 0.504 | 0.14 ms |
| CRNN od zera | 0.897 | 0.631 | 0.5 ms |
| CRNN + wycinki potoku + rozmycie w augmentacji (3 ziarna) | 0.932 | 0.769 | 0.5 ms |
| PARSeq dotrenowany, wycinek z wyrównaniem (3 ziarna) | 0.949 | 0.834 | |
| PARSeq dotrenowany, wycinek z samej ramki (3 ziarna) | 0.974 | 0.860 | |

TrOCR-small dotrenowany: na walidacji 0.839 na zniekształconych, tyle co CRNN (0.833) przy 25x większym modelu.

## Odczyt rozmytych tablic (walidacja eksperymentu, PARSeq, zniekształcone)

| Układ | Wynik | Decyzja |
|---|---|---|
| PARSeq, wycinek z wyrównaniem | 0.883 | |
| PARSeq, wycinek z samej ramki | 0.907 | przyjęte |
| U-Net przed OCR (L1 / L1 + strata odczytu) | 0.891 / 0.909 | odrzucone |
| Wiener, CLAHE, inna interpolacja, sama luminancja | 0.86-0.91 | odrzucone |
| dekonwolucja: 5 dysków | 0.922 | |
| dekonwolucja: 5 dysków + ruch z kątem z widma | 0.938 | przyjęte |
| RL z regularyzacją TV | 0.937-0.939 | odrzucone |
| + PARSeq dotrenowany na dekonwolucji, pewniejszy z dwóch | 0.954 | przyjęte |
| głosowanie, TTA, ponowna ocena hipotez | +0.2 p.p. lub mniej | odrzucone |

Wyrocznia (prawdziwe jądro ruchu): rozmycie ruchu siły 5 z 0.76 do 0.95. Estymacja kąta z widma: mediana błędu
1.0-1.2 stopnia.

## Wynik końcowy (test po poprawce, wszystko przetrenowane, średnio 3 ziarna)

| Układ | Czyste | Zniekształcone | Rozmycie ostrości | Rozmycie ruchu |
|---|---|---|---|---|
| EasyOCR | 0.600 | 0.228 | | |
| CRNN | 0.958 | | | |
| PARSeq, wycinek z ramki | 1.000 | 0.867 | 0.643 | 0.852 |
| + dekonwolucja | 1.000 | 0.917 | 0.727 | 0.970 |
| + PARSeq po dekonwolucji, pewniejszy z dwóch | 1.000 | 0.922 (0.916-0.926) | 0.745 | 0.970 |

Poprawne odczyty na 40 tablic w końcowym układzie (średnio 3 ziarna):

| | siła 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| rozjaśnienie | 40 | 40 | 39 | 39 | 39 |
| przyciemnienie | 40 | 39.7 | 39.3 | 39.7 | 39 |
| rozmycie ruchu | 39.3 | 40 | 38.7 | 39 | 37 |
| rozmycie ostrości | 38.3 | 36 | 33 | 24 | 17.7 |

Przebiegi na teście po poprawce: `results/experiment/ref_poprawka/` (EasyOCR) i `results/experiment/parseq_deblur_s0/`
(końcowy układ, ziarno 0: 0.916 na zniekształconych). Zastrzeżenia: modele widziały w treningu zniekształcenia tego
samego rodzaju (inne zdjęcia), dekonwolucja zna rodzinę jąder `imagecorruptions`, przedział ufności przy 40 tablicach
to ok. +-5 p.p.
