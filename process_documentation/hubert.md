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

## 17.10.2026: refaktoryzacja repozytorium (pakiet `alpr`)

### Punkt wyjścia

Cały kod leżał płasko w `scripts/`. Obok siebie były skrypty budowy zbioru, moduł biblioteczny
`plate_detection.py` i skrypty CLI. Te same funkcje powtarzały się w kilku plikach: `warp`
w `build_dataset.py` i `plate_detection.py`, wczytywanie i filtrowanie `plates.csv`, źródło z `plate_id`
i flaga oryginał / kopia w trzech skryptach. `requirements.txt` nie miał numpy, pyyaml ani pakietów do
budowy zbioru. Brakowało testów i miejsca na kod Stańczyka (filtry) i Golby (OCR) przed potokiem
end-to-end (blok 7). Warunek: `build_dataset.py` ma dawać ten sam zbiór co na Kaggle.

### Struktura

| Kwestia | Decyzja | Powód |
|---|---|---|
| Organizacja kodu | pakiet `alpr/` (`pip install -e .`, `pyproject.toml`), skrypty w `scripts/` jako cienkie CLI pod tymi samymi nazwami i flagami | jeden import dla skryptów, notebooków i potoku; komendy w README i notebooku bez zmian |
| Moduły | `config`, `dataset`, `geometry`, `build/` (Stańczyk), `preprocessing/`, `detection/` (moja część), `ocr/` | podział jak role w zespole |
| Skrypty bez instalacji | `sys.path.insert(0, ROOT)` przed importem `alpr` | `python3 scripts/...` działa też u kogoś, kto nie przeinstalował pakietu |
| Ścieżka zbioru | `config.default_dataset()`: Kaggle input, jeśli jest, inaczej `data/combined` | w notebooku Kaggle nie trzeba podawać `--dataset` |
| Wycinki na Kaggle | `dataset.crop_path(dataset, crop)` | na Kaggle wycinki są w `ocr/ocr/<split>/`, łatwo o błąd w OCR |
| `exposure` | w `alpr/dataset.py`, nie w `build/` | definiuje kolumnę i przyda się przy filtrach, a nie wymaga imagecorruptions |
| `four_corners` i `order_corners` | wspólny kod, ale `four_corners` zostawia 4 punkty w kolejności z adnotacji | tak działał `build_dataset.py`; inna kolejność zmieniłaby wycinki w zbiorze |
| Szkielety | `preprocessing.enhance(img, method)` z rejestrem `METHODS` (działa tylko `none`), `ocr.read_plate(crop, engine)` z normalizacją tekstu, sygnatury metryk OCR | ustalony interfejs do potoku; implementacje to zadania Stańczyka i Golby (blok 3) |
| Styl | ruff (`E, F, I, UP, B`, linia 120), `ruff format` | jednolity styl całego repo |

### Poprawki przy okazji

* `detect_plates.py --images`: zdjęcia o tej samej nazwie w różnych podfolderach nadpisywały sobie wycinki.
  Teraz nazwa zawiera podfolder względem wspólnego katalogu (`a/b.jpg` -> `a__b_0.png`).
  Przy `--split` nazwy się nie zmieniły.
* `eval_alignment.py`: predykcje grupowane raz po zdjęciu, a nie filtrowane od nowa dla każdej tablicy.
* `build_dataset.py`: `argparse` z `--out`, zgodnie z konwencją skryptów.

### Weryfikacja

Przed przeniesieniem zapisałem wyniki starego kodu i porównałem je z nowym:

| Test | Wynik |
|---|---|
| zniekształcenia (wszystkie 9 typów), `darken`, `exposure`, `warp`, `align_plate`, `evaluate`, `image_groups` | identyczne hashe |
| `build_dataset.py` na sztucznych źródłach (12 zdjęć data1, 10 data2 z wielokątami 5-punktowymi, 15 wycinków data3) | 174 pliki identyczne bit w bit |
| `eval_alignment.py` (ramki z etykiet i z predykcji), `eval_detection.py --preds` | identyczne CSV |
| `train_yolo.py` (1 epoka, CPU), potem `eval_detection.py --weights` i `detect_plates.py --split` (10800 wycinków) | identyczne predykcje, CSV i wycinki |
| `pytest` | 58 testów, wszystkie przechodzą |

Przy pisaniu testów wyszło, że nasz AP liczy się dokładnie jak w ultralytics: idealna detekcja daje 0.995,
a przy niepełnym recall wartość jest nieco zawyżona (interpolacja liniowa do punktu recall 1, precision 0).
Zostawiłem tak, bo dzięki temu liczby są porównywalne z `model.val()`.

### Stan na koniec dnia

* [x] pakiet `alpr`, cienkie skrypty, `pyproject.toml`, pełne zależności
* [x] testy (`tests/`), w tym `test_build_golden.py` pilnujący powtarzalności zbioru
* [x] szkielety `preprocessing` i `ocr`
* [x] ruff, README (sekcja "Struktura kodu"), CLAUDE.md, notebook (`pip install -e .`)
* [ ] sprawdzenie na prawdziwym zbiorze (brak tokenu Kaggle lokalnie)

### Do zrobienia (moja część)

* przekazać zespołowi: importy z `alpr`, `pip install -r requirements.txt` po pullu, interfejsy `enhance` i `read_plate` (blok 3),
* po pobraniu zbioru: `eval_alignment.py --split test` na prawdziwych danych przed i po refaktoryzacji (blok 6),
* reszta bez zmian względem poprzedniego wpisu (trening na GPU, bloki 5-6).

## 17.10.2026: porządki w kodzie i szybsze pobieranie zbioru

### Punkt wyjścia
Po pierwszej refaktoryzacji kod był nadal przegadany: hack `sys.path.insert` w każdym skrypcie, po 60-80 linii
logiki w `main()` skryptów detekcji, konfiguracja źródeł jako słowniki z kluczami-flagami, logika jednego źródła
rozrzucona po trzech plikach, komentarze w kodzie, notebook wywołujący tylko 4 skrypty. Cel: czytelna struktura,
klasy tam, gdzie jest stan, kod bez komentarzy, wynik bit w bit ten sam.

### Struktura
| Było | Jest | Powód |
|---|---|---|
| `build/download.py` (`DATASETS`), `sources.py` (`select_*`), `combine.py` (`build_dataN`) | `build/sources.py`: `SourceDataset` i klasy `PolishPlates`, `Uc3mLp`, `AutoRia` | całe źródło w jednym miejscu, bez rozgałęzień na kluczach słownika |
| `add_photo(out, rows, ...)`, pętla oryginał + kopia skopiowana dwa razy | `build/writer.py`: `CombinedWriter` (`add_photo`, `add_crop`, `save`, `summary`) | wiersze w obiekcie, jedna metoda `_variants` |
| `detection/yolo.py`: `load_model`, `detect` | `detection/detector.py`: `PlateDetector` | model i parametry w jednym obiekcie |
| `align_plate` i stałe modułu | `PlateAligner` (parametry) i `AlignedPlate` (crop, poly, method) | parametry konfigurowalne, wynik nazwany |
| pętle po zdjęciach w skryptach | `detection/pipeline.py`, `evaluate_by_group`, `summarize_alignment`, `training.py` | skrypty tylko parsują argumenty |
| `preprocessing/__init__.py`, `ocr/__init__.py` | `preprocessing.py`, `ocr.py` | pakiet z jednym plikiem to zbędny folder |
| `scripts/combined_template/` | `alpr/build/template/` | to dane budowy zbioru |
| `notebooks/detekcja_kaggle.ipynb` | komenda w README, sekcja "Detekcja tablic" | notebook tylko wołał skrypty |

Dodatkowo: `.vscode/settings.json` ukrywa `.venv`, `__pycache__`, cache i `data/data2/annotations`;
`requires-python >= 3.11` (`hashlib.file_digest`).

### Pobieranie (`scripts/get_data.py`, `scripts/download_data.py`)
| Zmiana | Powód |
|---|---|
| archiwum Kaggle w `data/_downloads/`, klient Kaggle wznawia przerwane pobieranie | wcześniej zip leżał w `data/combined`, a ponowne uruchomienie kończyło się błędem "nie jest pusty" |
| rozpakowanie do katalogu tymczasowego i podmiana dopiero po sukcesie | `--force` kasował stary zbiór przed pobraniem nowego |
| rozpakowanie równoległe (8 wątków) z od razu poprawionym `ocr/ocr/` | jeden przebieg zamiast rozpakowania i przenoszenia |
| sprawdzanie plików jednym `os.walk` zamiast `exists()` na każdy plik | mniej wywołań systemowych |
| SHA-256 zdjęć liczone równolegle, MD5 archiwum zapamiętywane w `<zip>.md5` | MD5 4.5 GB liczyło się przy każdym uruchomieniu |

### Wynik
Pomiar na danych syntetycznych (lokalnie brak tokenu Kaggle i zdjęć źródłowych):
| Operacja | Przed | Po |
|---|---|---|
| weryfikacja SHA-256, 3000 plików po 400 kB | 1.38 s | 0.48 s |
| rozpakowanie zipa, 2000 plików po 400 kB | 0.80 s | 0.45 s |

Zgodność ze starym kodem (wzorce zapisane przed zmianami):
| Test | Wynik |
|---|---|
| budowa zbioru na syntetycznych źródłach data1..3 (24 wiersze, 53 pliki) | identyczne bit w bit, utrwalone w `tests/test_build_combine_golden.py` |
| `eval_detection.py --preds` (train, test), `eval_alignment.py` (etykiety i predykcje, przykłady) | identyczne pliki |
| `detect_plates.py` z atrapą YOLO (`--split`, `--images --no-align`) | identyczne CSV i wycinki |
| `pytest`, `ruff check`, `ruff format --check` | 76 testów przechodzi, lint czysty |

### Stan na koniec dnia
* [x] klasy źródeł, `CombinedWriter`, `PlateDetector`, `PlateAligner`, `pipeline`, `training`
* [x] skrypty bez `sys.path`, kod bez komentarzy, notebook usunięty
* [x] szybsze i wznawialne pobieranie zbioru
* [x] README, CLAUDE.md
* [ ] sprawdzenie `get_data.py` na prawdziwym Kaggle (brak tokenu lokalnie)

### Do zrobienia (moja część)
* po pullu: `pip install -e .` (skrypty nie działają już bez instalacji pakietu), przekazać to zespołowi (blok 3),
* uruchomić `get_data.py` z tokenem i sprawdzić wznawianie po przerwaniu (blok 3),
* trening na GPU komendą z README (bloki 5-6).
