# Dokumentacja pracy: Włodarczyk (detekcja tablic)

## 17.10.2026: narzędzia do detekcji, ewaluacji i wyrównania perspektywy

### Punkt wyjścia

Zbiór `data/combined` (Stańczyk) ma 1074 całych zdjęć z etykietami YOLO i 4 narożnikami każdej
tablicy. Celem było przygotowanie całej mojej części potoku tak, żeby po pobraniu zbioru wystarczyło
uruchomić trening na GPU: trening YOLO, ewaluacja detekcji w podziale na warunki oświetleniowe,
przycinanie i wyrównanie perspektywy oraz zapis wycinków w formacie dla OCR.

Lokalnie nie było tokenu Kaggle ani GPU, więc skrypty sprawdziłem na syntetycznym zbiorze w tym samym
formacie co `data/combined` (260 zdjęć 640x480, tablice z paskiem UE wklejone z losową perspektywą
i obrotem do 12 stopni, 30% zdjęć "nocnych", każde z kopią `darken`). Liczby z syntetyku służą tylko
do sprawdzenia kodu, nie są wynikami projektu.

### Moduł wspólny (`scripts/plate_detection.py`)

Funkcje: `load_model`, `detect`, `align_plate`, `warp`, `box_iou`, `polygon_iou`. Importują je wszystkie
skrypty detekcji i docelowo potok end-to-end (blok 7).

| Kwestia | Decyzja | Powód |
|---|---|---|
| Wyrównanie | szukanie 4 narożników w ramce powiększonej o 15%, potem `warpPerspective` | YOLO daje tylko prostokąt, a tablica na zdjęciu jest czworokątem (perspektywa, obrót) |
| Maski konturu | krawędzie Canny oraz jasne tło (Otsu) z dołożonym niebieskim paskiem UE (HSV) | bez paska lewa krawędź tablicy wypadała na granicy paska: błąd lewych narożników ok. 0.4-0.55 wysokości tablicy, po zmianie ok. 0.08 |
| Narożniki | `approxPolyDP`, potem dopasowanie prostych do punktów wzdłuż boków (bez okolic rogów) i przecięcia | wierzchołki z `approxPolyDP` bywały przesunięte o kilka pikseli, wycinek wychodził "pochylony" |
| Zabezpieczenia | IoU ramki czworokąta z ramką YOLO >= 0.6, proporcje 1-7, w razie porażki zwykły wycinek (`box`) | odrzuca kontury zderzaka, maski auta itp.; gorszy wycinek jest lepszy niż zły |
| Prostowanie | ta sama funkcja `warp` co w `build_dataset.py` | wycinki z detekcji wyglądają jak wycinki, na których Golba testuje OCR |

### Trening (`scripts/train_yolo.py`)

* Fine-tuning z wag COCO, domyślnie `yolo11n.pt`, 640 px, 100 epok, wczesne zatrzymanie po 20 epokach.
* Trening na train (oryginały i zniekształcone kopie), wybór epoki na val, test tylko do ewaluacji.
* `seed=0`, `deterministic=True`.
* `data.yaml` ma `path: .`, a ultralytics rozwiązuje względną ścieżkę względem swojego katalogu
  `datasets`, nie pliku yaml. Skrypt zapisuje kopię yaml z bezwzględną ścieżką; dzięki temu działa
  też na Kaggle (`/kaggle/input/...` jest tylko do odczytu).
* Na CPU (8 rdzeni) epoka na 360 zdjęciach trwała ok. 100 s, czyli na prawdziwym zbiorze ok. 8 min.
  Pełny trening robimy na GPU: `notebooks/detekcja_kaggle.ipynb` (trening, ewaluacja, wycinki, zip z wynikami).

### Ewaluacja detekcji (`scripts/eval_detection.py`)

| Kwestia | Decyzja | Powód |
|---|---|---|
| Metryki | P, R, F1 przy progu 0.25, AP50 i AP50-95 (interpolacja 101-punktowa jak w ultralytics) | `model.val` z ultralytics daje wynik tylko dla całego zbioru, a potrzebujemy podzbiorów |
| Grupy | źródło, `lighting`, `exposure`, `artificial_corruption`, siła, oryginał / kopia | pytanie badawcze: wpływ oświetlenia i zniekształceń na detekcję |
| Atrybuty tablicy a zdjęcie | wartość tablicy przenoszona na zdjęcie, przy różnych wartościach grupa `mixed` | fałszywe alarmy należą do zdjęcia, nie do tablicy |
| Test filtrów (blok 4) | `--images-root`: obrazy po filtrze z tą samą strukturą katalogów co zbiór | Stańczyk zapisuje przefiltrowane zdjęcia, ja liczę na nich te same metryki |
| Dopasowanie | zachłanne, po malejącej pewności, IoU >= próg | standard COCO/VOC |

Sprawdzenie na predykcjach o znanym wyniku: idealne ramki dają P = R = 1, AP50 = 0.995 (jak ultralytics);
połowa tablic plus jeden fałszywy alarm na zdjęcie daje P = 47/127 = 0.370, R = 0.5, zgodnie z rachunkiem.

### Wycinki dla OCR (`scripts/detect_plates.py`)

Wejście: dowolny folder zdjęć (potok) albo podział z `plates.csv`. Wynik: `detections.csv`
(`image, det_id, conf, x1, y1, x2, y2, polygon, align, crop`) i `crops/*.png`. Zdjęcie bez detekcji ma
jeden wiersz z pustym `det_id`, żeby potok liczył je jako brak odczytu. `--no-align` zapisuje zwykły
wycinek ramki (do porównania wpływu wyrównania na OCR).

### Ocena wyrównania (`scripts/eval_alignment.py`)

Porównuje zwykły wycinek ramki z `align_plate` względem prawdziwych narożników: IoU wielokątów i
średni błąd narożnika w wysokościach tablicy. Ramki z etykiet (sama jakość wyrównania) albo z
detektora (`--preds`). Na prawdziwym zbiorze sensowne będzie tylko data2 (UC3M-LP), bo w data1
etykiety są prostokątami.

### Wynik

Tylko syntetyk (sprawdzenie kodu, nie wynik projektu):

| | IoU z prawdziwym czworokątem | błąd narożnika (wys. tablicy) |
|---|---|---|
| wycinek ramki (z etykiet) | 0.686 | 0.302 |
| `align_plate`, pierwsza wersja | 0.861 | 0.208 |
| + doprecyzowanie boków | 0.876 | 0.198 |
| + niebieski pasek UE w masce | 0.888 | 0.133 |
| `align_plate` na ramkach z detektora (test) | 0.876 | 0.143 |

Detektor po 1 epoce na syntetyku: AP50 0.995, AP50-95 0.855, R 1.0, P 0.59 (za krótki trening,
dużo fałszywych alarmów przy progu 0.25). Cały łańcuch trening -> ewaluacja -> wyrównanie -> wycinki
przeszedł bez błędów.

### Stan na koniec dnia

* [x] moduł detekcji, wycinania i wyrównania (`plate_detection.py`)
* [x] skrypt treningu (`train_yolo.py`) i notebook na Kaggle (`notebooks/detekcja_kaggle.ipynb`)
* [x] ewaluacja detekcji w grupach (`eval_detection.py`) i wyrównania (`eval_alignment.py`)
* [x] wycinki i CSV dla OCR (`detect_plates.py`)
* [x] README (sekcja "Detekcja tablic"), `requirements.txt` (ultralytics), `.gitignore` (`runs/`, `*.pt`, wycinki)
* [ ] pobranie prawdziwego zbioru (brak tokenu Kaggle lokalnie)
* [ ] trening na prawdziwym zbiorze i liczby baseline

### Do zrobienia (moja część)

* token Kaggle, `get_data.py`, uruchomienie notebooka na GPU: trening `yolo11n` 640 px (blok 5),
* porównanie `yolov8n` / `yolo11n` / `yolo11s` i 640 / 960 px, tablice są na zdjęciach małe (blok 5-6),
* ewaluacja na test w grupach i ocena wyrównania na data2 (blok 6),
* test detekcji na zdjęciach po filtrach Stańczyka przez `--images-root` (blok 4),
* przekazanie Golbie `detections.csv` z test i interfejsu `align_plate` do potoku (blok 7),
* analiza wpływu oświetlenia na metryki detekcji, noc / dzień i under / over (bloki 8-9).
