# Podsumowanie wyników bloku 2 po poprawkach (v2)

Przebieg z 15.11.2026 na zbiorze przebudowanym lokalnie z `main` (`9513e37`), czyli z poprawkami Włodarczyka
z 08.11.2026. Wynik sprzed poprawek: `results/experiment/baseline/PODSUMOWANIE.md`.

## Konfiguracja

| Element | Ustawienie |
|---|---|
| Zbiór | przebudowany lokalnie (`download_data.py`, `build_dataset.py`), 16584 wierszy; wersja na Kaggle jeszcze stara |
| Zmiana podziału | 8 zdjęć data1 (to samo auto zawsze w jednym podziale); brak wspólnych numerów między `train` a `val`/`test` |
| Detektor | YOLO `yolo11n.pt`, 640 px, 100 epok, `batch` 16, ziarno 0, RTX 4080 (ok. 13 min) |
| Wagi | `runs/detect/plates_v2/weights/best.pt` (poza gitem) |
| Wyrównanie | `PlateAligner` z CLAHE (domyślnie od 08.11.2026) |
| OCR | EasyOCR, model `en`, znaki A-Z i 0-9, z filtrem `plate_fragments` (pomija napisy niższe niż 0.6 najwyższego) |
| Próg detekcji | `conf` 0.25, dopasowanie do ground truth przy IoU >= 0.5 |
| Ground truth | 39 oryginalnych zdjęć z data1 `test` (Polska, po 1 tablicy) |
| Wersje | `clean` (JPEG 95 jak pozostałe) + `brightness`, `darken`, `motion_blur`, `defocus_blur` w sile 1-5 (819 zdjęć) |

## Detektor (walidacja ultralytics, `val`, cały zbiór)

| Wagi | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| `plates_v2` (nowy zbiór) | 0.976 | 0.933 | 0.973 | 0.781 |
| `plates` (stary zbiór) | 0.969 | 0.943 | 0.969 | 0.786 |

## Wyniki eksperymentu (39 tablic na wersję)

`wykryte`: tablica dopasowana do detekcji. `IoU`: średnie IoU najlepszej ramki. `bezbłędny`: odczyt równy
prawdziwemu numerowi. `CAR`: 1 - odległość edycyjna / długość numeru (średnio, nieodczytana tablica = 0).
`pusty`: brak odczytu. `conf OCR`: średnia pewność EasyOCR.

| Wersja | Wykryte | IoU | Bezbłędny | CAR | Pusty | conf OCR |
|---|---|---|---|---|---|---|
| clean | 1.000 | 0.898 | 0.615 | 0.887 | 0.026 | 0.730 |
| brightness 1 | 1.000 | 0.900 | 0.641 | 0.894 | 0.026 | 0.782 |
| brightness 2 | 1.000 | 0.899 | 0.641 | 0.909 | 0.026 | 0.792 |
| brightness 3 | 1.000 | 0.895 | 0.718 | 0.905 | 0.026 | 0.765 |
| brightness 4 | 1.000 | 0.891 | 0.641 | 0.873 | 0.000 | 0.769 |
| brightness 5 | 1.000 | 0.877 | 0.513 | 0.821 | 0.026 | 0.662 |
| darken 1 | 1.000 | 0.899 | 0.462 | 0.821 | 0.026 | 0.641 |
| darken 2 | 1.000 | 0.897 | 0.308 | 0.752 | 0.026 | 0.580 |
| darken 3 | 1.000 | 0.894 | 0.282 | 0.649 | 0.051 | 0.471 |
| darken 4 | 1.000 | 0.889 | 0.154 | 0.481 | 0.051 | 0.341 |
| darken 5 | 1.000 | 0.875 | 0.103 | 0.353 | 0.179 | 0.208 |
| motion_blur 1 | 1.000 | 0.885 | 0.256 | 0.595 | 0.154 | 0.483 |
| motion_blur 2 | 1.000 | 0.870 | 0.077 | 0.294 | 0.333 | 0.168 |
| motion_blur 3 | 1.000 | 0.843 | 0.000 | 0.114 | 0.538 | 0.082 |
| motion_blur 4 | 0.974 | 0.828 | 0.000 | 0.038 | 0.718 | 0.058 |
| motion_blur 5 | 0.974 | 0.812 | 0.000 | 0.000 | 0.949 | 0.001 |
| defocus_blur 1 | 1.000 | 0.899 | 0.103 | 0.506 | 0.256 | 0.344 |
| defocus_blur 2 | 1.000 | 0.901 | 0.077 | 0.333 | 0.436 | 0.209 |
| defocus_blur 3 | 1.000 | 0.891 | 0.000 | 0.075 | 0.718 | 0.055 |
| defocus_blur 4 | 1.000 | 0.876 | 0.000 | 0.000 | 1.000 | 0.000 |
| defocus_blur 5 | 1.000 | 0.871 | 0.000 | 0.000 | 0.949 | 0.002 |

## Porównanie ze starym przebiegiem

Stary przebieg ograniczony do tablic z `test` (39 tablic, ale skład różni się o 3 zdjęcia przeniesione przez poprawkę
podziału), odczyt bezbłędny:

| Wersja | Stary (`baseline`, `test`) | Nowy (`v2`) |
|---|---|---|
| clean | 0.487 | 0.615 |
| brightness 3 | 0.513 | 0.718 |
| darken 3 / 5 | 0.205 / 0.026 | 0.282 / 0.103 |
| motion_blur 1 / 3 | 0.231 / 0.000 | 0.256 / 0.000 |
| defocus_blur 1 / 3 | 0.205 / 0.000 | 0.103 / 0.000 |

Na czystych zdjęciach błędne odczyty spadły z 20 do 15. Za długich odczytów jest 3 zamiast 7 (filtr
`plate_fragments` usuwa m.in. `SBL58012BRUTSCH` i doklejone litery z paska UE).

## Wnioski

* Detekcja jest odporna na badane zniekształcenia: wykryte 97.4-100% tablic, IoU spada najwyżej z 0.90 do 0.81
  (rozmycie ruchem w sile 5). YOLO uczyło się na losowo zniekształconych kopiach tych samych typów.
* Cały spadek skuteczności potoku pochodzi z OCR. Na czystych zdjęciach odczyt bezbłędny to 61.5% (CAR 0.887).
* Rozmycie niszczy odczyt najszybciej: od siły 3 nie ma ani jednego bezbłędnego odczytu, a EasyOCR przeważnie nic
  nie zwraca (54-100% pustych odczytów). Rozmycie ostrości szkodzi już przy sile 1 (10.3%).
* Przyciemnienie obniża skuteczność stopniowo, już od siły 1 (61.5% -> 46.2% -> ... -> 10.3%).
* Rozjaśnienie w sile 1-4 nie szkodzi (64-72%), dopiero siła 5 obniża wynik (51%). Różnice w sile 1-4 to 1-4 tablice,
  czyli w granicach szumu.
* Kierunek dla bloków 3-6: metody poprawy obrazu mają sens głównie dla przyciemnienia i lekkiego rozmycia
  (siła 1-2); przy silnym rozmyciu nie ma już czego odzyskiwać.

## Błędy OCR na czystych zdjęciach (15 z 39)

3 odczyty za długie, 4 za krótkie (w tym 1 pusty), 8 tej samej długości z podmienionymi znakami. Mylone pary:
1/I, I/L, G/C, 6/G, 0/O, W/1, E/F, 7/Z, S/0, O/2. Przykłady: `SK179VE` -> `SKI79VE`, `SRB6UK4` -> `SRBGUK4`,
`SG542CK` -> `GS42CK`, `SO9606N` -> `029606N`.

## Zastrzeżenia

* 39 tablic na wersję: jedna tablica to 2.6 punktu procentowego, więc różnice poniżej ok. 3 tablic to szum.
* Zbiór jest prawie wyłącznie dzienny; wnioski o oświetleniu dotyczą tylko sztucznych `brightness` i `darken`.
* Zbiór przebudowany lokalnie na Windows (OpenCV 5.0). Golden testy `test_warp` i `test_zbior_identyczny`
  nie przechodzą w tym środowisku, więc wycinki mogą minimalnie różnić się od przyszłej wersji na Kaggle.
* Detekcja na `test` wykrywa 17 nadmiarowych ramek na 819 zdjęciach (836 wycinków); nie wpływają na tę tabelę,
  bo liczymy tylko tablice z ground truth.

## Pliki

| Plik | Zawartość |
|---|---|
| `summary.csv` | wykryte i odczyt bezbłędny na wersję |
| `plates.csv` | jedna linia na (wersja, tablica): ground truth, dopasowana detekcja i odczyt |
| `detections.csv`, `ocr_easyocr.csv` | wszystkie detekcje (836 wycinków) i ich odczyty |
