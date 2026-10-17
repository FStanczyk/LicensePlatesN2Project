# Rozpoznawanie tablic rejestracyjnych (ALPR)

**Temat:** Wpływ warunków oświetleniowych i metod poprawy jakości obrazu na skuteczność rozpoznawania tablic rejestracyjnych.  
**Problem badawczy:** Jak warunki oświetleniowe wpływają na skuteczność detekcji i rozpoznawania tablic rejestracyjnych oraz które techniki poprawy jakości obrazu pozwalają ograniczyć ten wpływ?

## Zespół i podział ról

**Stańczyk: dane i przetwarzanie obrazu**
* Pozyskanie, czyszczenie i podział baz danych na podzbiory testowe (dzień, noc, prześwietlenie, niedoświetlenie).
* Implementacja algorytmów poprawy jakości obrazu (CLAHE, korekcja gamma, wyrównanie histogramu, odszumianie).
* Przygotowanie potoku augmentacji danych do treningu.

**Włodarczyk: detekcja tablic**
* Wybór, trening i ewaluacja modeli do wykrywania ramek tablic rejestracyjnych (np. YOLOv8, YOLOv11).
* Ocena skuteczności samej detekcji (metryki: mAP, Precision, Recall) w zależności od oświetlenia i zastosowanych metod poprawy (testowanie we współpracy ze Stańczykiem).
* Przycinanie (cropping) i perspektywiczne wyrównywanie wykrytych tablic do dalszego przetwarzania.

**Golba: OCR i złożenie całości**
* Implementacja i porównanie silników OCR (np. EasyOCR, Tesseract, LPRNet, CRNN).
* Ewaluacja skuteczności odczytu znaków (metryki: Character Accuracy Rate, Word/Plate Accuracy).
* Integracja wszystkich modułów w jeden spójny system i zmierzenie ostatecznej skuteczności (F1-score) dla całego potoku.

---

## Harmonogram (30 godzin, 10 bloków po 3h)

| Blok | Faza | Zadania przypisane do członków zespołu |
| :--- | :--- | :--- |
| **1** | **Setup & Dane** | **Wszyscy:** Założenie repozytorium (GitHub), wybór środowiska (np. Google Colab).<br>**Stańczyk:** Pobranie bazy i weryfikacja podzbiorów (dzień/noc).<br>**Włodarczyk:** Załadowanie pre-trenowanego modelu YOLO.<br>**Golba:** Wybór i instalacja narzędzi OCR. |
| **2** | **Baseline (Test 0)** | **Stańczyk:** Skrypt ładujący zdjęcia i iterujący po katalogach.<br>**Włodarczyk:** Puszczenie surowych zdjęć przez YOLO i zapisanie wyciętych ramek.<br>**Golba:** Przepuszczenie ramek przez OCR i zapisanie odczytów do CSV. |
| **3** | **Preprocessing** | **Stańczyk:** Napisanie funkcji do CLAHE, korekcji gamma i wyostrzania.<br>**Włodarczyk:** Konwersja etykiet bazy pod przyszły trening.<br>**Golba:** Napisanie funkcji obliczających metryki dla OCR. |
| **4** | **Test 1: Wpływ filtrów** | **Stańczyk:** Zapisanie zdjęć po nałożeniu filtrów poprawiających obraz.<br>**Włodarczyk:** Ponowny test detekcji na poprawionych zdjęciach.<br>**Golba:** Test OCR na tablicach wyciętych z poprawionych zdjęć. |
| **5** | **Trening Detektora** | **Stańczyk:** Przygotowanie skryptu augmentacji danych do treningu.<br>**Włodarczyk:** Konfiguracja potoku treningowego YOLO i odpalenie uczenia (po bloku).<br>**Golba:** Analiza błędów OCR z Bloku 4 (np. mylone znaki). |
| **6** | **Integracja OCR** | **Stańczyk:** Eksperymenty z binaryzacją obrazu wyciętej tablicy.<br>**Włodarczyk:** Ewaluacja wytrenowanego YOLO, przycinanie i wyrównanie perspektywy.<br>**Golba:** Konfiguracja alternatywnego modelu OCR / dostrajanie obecnego. |
| **7** | **Potok End-to-End** | **Wszyscy:** Połączenie skryptów w jeden działający pipeline: *Wejście (folder zdjęć) -> Przetwarzanie (Stańczyk) -> Detekcja i wyrównanie (Włodarczyk) -> OCR (Golba) -> Wyniki.* |
| **8** | **Wielki Eksperyment** | **Stańczyk:** Odpalenie pełnego potoku dla zbioru "Dzień".<br>**Włodarczyk:** Odpalenie pełnego potoku dla zbiorów "Noc" i "Zmierzch".<br>**Golba:** Odpalenie pełnego potoku dla zdjęć z prześwietleniem/odblaskami. |
| **9** | **Analiza Wyników** | **Stańczyk:** Wykresy porównawcze skuteczności przy różnych filtrach.<br>**Włodarczyk:** Analiza wpływu oświetlenia na metryki samej detekcji.<br>**Golba:** Zliczanie ostatecznych metryk całego systemu, analiza przypadków brzegowych. |
| **10** | **Raportowanie** | **Wszyscy:** Podział sekcji, zebranie wniosków, sprzątanie kodu, uzupełnienie dokumentacji oraz napisanie ostatecznego raportu z badań. |

---

## Możliwe eksperymenty
* analiza obrazów wykonanych w dzień, w nocy i o zmierzchu,
* różne poziomy oświetlenia: niedoświetlenie i prześwietlenie,
* porównanie skuteczności detekcji tablicy i rozpoznawania znaków,
* zastosowanie metod poprawy obrazu, np. CLAHE, korekcja gamma, odszumianie, wyostrzanie,
* porównanie różnych modeli OCR,
* analiza wpływu jakości obrazu na skuteczność całego systemu.

---

## Metryki
* **Detekcja:** mAP (mean Average Precision), Precision, Recall.
* **Rozpoznawanie znaków (OCR):** Character Accuracy Rate, Word Accuracy, F1-score.
* **Cały system:** Skuteczność End-to-End (czy system poprawnie wykrył tablicę i bezbłędnie odczytał wszystkie znaki).

---

## Przegląd literatury

**1. "An Improvement of License Plate Detection Under Low Light Condition Using CLAHE and Unsharp Masking" (2025)**
* Cel: sprawdzenie, jak wstępne przetwarzanie obrazu wpływa na skuteczność YOLOv11 na nocnych zdjęciach z kamer miejskich.
* Metoda: 4 warianty: samo YOLOv11 (baseline), YOLOv11 + CLAHE, YOLOv11 + Unsharp Masking (USM), YOLOv11 + CLAHE + USM. CLAHE nakładano osobno na każdy kanał R, G, B, żeby nie wzmacniać szumu.
* Wyniki: najlepiej wypadło połączenie CLAHE (lokalny kontrast) i USM (wyostrzenie krawędzi): Precision 0.945, Recall 0.977, mAP 0.830. Sam model nie wystarcza na ciemnych zdjęciach, preprocessing wyraźnie pomaga.

**2. "Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline" (Z. Xu i in., ECCV 2018)**
* Cel: udostępnienie dużego zbioru CCPD (ponad 250 tys. zdjęć) i model, który nie wymaga dzielenia tablicy na pojedyncze znaki.
* Metoda: zamiast klasycznego potoku (detekcja -> cięcie na litery -> odczyt liter) jeden model end-to-end łączący CNN (cechy obrazu) z RNN (odczyt sekwencji).
* Wyniki: na trudnych podzbiorach (CCPD-Weather, CCPD-Challenge) skuteczność systemów spada. OCR czytający całą tablicę naraz (segmentation-free) jest dużo bardziej odporny na cienie i niedoświetlenie niż podejście z segmentacją znaków.

**3. "Design of a Rapid License Plate Localization Algorithm Utilizing Adaptive Image Enhancement" (2024/2026)**
* Cel: system, który radzi sobie ze złym oświetleniem w czasie rzeczywistym, przy wysokim FPS.
* Metoda: zamiast poprawiać obraz w RGB, konwersja do przestrzeni LAB i CLAHE tylko na kanale L (jasność), potem złożenie obrazu z powrotem.
* Wyniki: dobrze działa na zdjęciach niedoświetlonych i robionych pod słońce. Poprawia precyzję lokalizacji tablic i mocno skraca czas przetwarzania jednej klatki.

**Wnioski dla naszego projektu**
1. Zamiast globalnego wyrównania histogramu (HE) używamy CLAHE. Warto porównać dwa warianty: CLAHE na kanałach RGB oraz CLAHE tylko na kanale jasności (V w HSV albo L w LAB).
2. Samo rozjaśnienie nie wystarczy, liczą się też ostre krawędzie znaków. Unsharp Masking po CLAHE powinien ograniczyć mylenie znaków przez OCR na nocnych zdjęciach (B/8, O/0). Zadanie dla Stańczyka i Golby.
3. Do OCR lepiej wybrać modele czytające cały ciąg znaków (EasyOCR, LPRNet, CRNN), żeby uniknąć błędów cięcia pojedynczych liter przy mocnych odblaskach.

---

## Dane

Gotowy, znormalizowany zbiór jest na Kaggle jako prywatny dataset
[`filipstanczyk/eu-license-plates-alpr`](https://www.kaggle.com/datasets/filipstanczyk/eu-license-plates-alpr)
(dostęp nadaje Stańczyk). Pobierasz go raz do `data/combined/`:

```bash
pip install -r requirements.txt     # instaluje pakiet alpr (pip install -e .) z zależnościami
python3 scripts/get_data.py
```

Potrzebny jest token API Kaggle: kaggle.com -> Settings -> API -> Create New Token, plik zapisz
w `~/.kaggle/`. Skrypt pobiera ok. 4.5 GB do `data/_downloads/` (przerwane pobieranie wznawia się
po ponownym uruchomieniu), rozpakowuje równolegle i sprawdza, czy są wszystkie pliki z `plates.csv`.
Przy kolejnym uruchomieniu nic nie pobiera; po nowej wersji na Kaggle użyj `--force` (stary zbiór
jest podmieniany dopiero po udanym pobraniu).

Na Kaggle (notebook) wystarczy dodać dataset przez "Add Input"; pliki są w
`/kaggle/input/eu-license-plates-alpr/`. Uwaga: tam wycinki leżą o poziom głębiej, w `ocr/ocr/<split>/`,
więc ścieżka do wycinka to `"/kaggle/input/eu-license-plates-alpr/ocr/" + crop`
(`get_data.py` lokalnie to wyrównuje, więc lokalnie wystarczy `"data/combined/" + crop`).
`alpr.dataset.crop_path(dataset, crop)` obsługuje oba przypadki.

### Zawartość

| Ścieżka | Co to jest |
|---|---|
| `plates.csv` | jedna linia na tablicę, opis kolumn niżej |
| `detection/images/<split>/`, `detection/labels/<split>/`, `detection/data.yaml` | całe zdjęcia + etykiety YOLO (klasa 0 = tablica), do detekcji |
| `ocr/<split>/` | wycięte tablice ze wszystkich źródeł, do OCR |

8292 tablic, każda w dwóch wersjach (oryginał + kopia z losowym sztucznym zniekształceniem),
razem 16584 wierszy. Z tego 1074 zdjęć (1292 tablic) to całe zdjęcia, reszta to same wycinki.

| Źródło | Tablice | Typ | Uwagi |
|---|---|---|---|
| [Poland Vehicle License Plate Dataset](https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset) | 290 (PL) | całe zdjęcia | brak podziału, losujemy 70/15/15 |
| [UC3M-LP](https://github.com/ramajoballester/uc3m-lp) | 1002 (ES), 784 zdjęć | całe zdjęcia | wszystkie nocne (362) + losowe dzienne; **każda** tablica ma zamazane znaki (`*`), więc nadaje się do detekcji i testów nocnych, ale nie do metryk OCR |
| [AUTO.RIA Numberplate OCR EU](https://github.com/ria-com/nomeroff-net) | 7000 (1710 PL + po ok. 204 z 26 krajów UE) | same wycinki | CC BY 4.0 |

### Kolumny `plates.csv`

| Kolumna | Opis |
|---|---|
| `plate_id` | identyfikator wiersza; prefiks `data1_` / `data2_` / `data3_` mówi, z którego źródła |
| `sample_type` | `full_photo` (całe zdjęcie, do detekcji) albo `plate_crop` (tylko wycinek, do OCR) |
| `split` | `train` / `val` / `test`; zniekształcona kopia jest zawsze w tym samym podziale co oryginał |
| `text` | numer tablicy, tylko A-Z i 0-9; `*` = znak zamazany |
| `text_raw` | numer w oryginalnym zapisie źródła |
| `blurred` | 1 = tablica ma zamazane znaki, pomijamy przy metrykach OCR |
| `country` | kod kraju (`pl`, `es`, `de`, ...) |
| `lighting` | pora zdjęcia według źródła: `day` / `night` / `unknown` (znana tylko dla UC3M-LP) |
| `exposure` | naświetlenie tablicy zmierzone na wycinku: `under` / `normal` / `over` |
| `lines` | liczba wierszy znaków na tablicy (1 lub 2) |
| `artificial_corruption` | `none` albo dodane przez nas zniekształcenie: `brightness`, `darken`, `contrast`, `fog`, `gaussian_noise`, `shot_noise`, `motion_blur`, `defocus_blur`, `jpeg_compression` |
| `artificial_severity` | 0 dla oryginału, 1-5 dla kopii |
| `image` | ścieżka do całego zdjęcia (puste dla `plate_crop`) |
| `x1`, `y1`, `x2`, `y2`, `polygon` | ramka i 4 narożniki tablicy na zdjęciu (puste dla `plate_crop`) |
| `crop` | ścieżka do wyciętej tablicy |

Przykłady (`load_plates` czyta `data/combined`, a w notebooku Kaggle podpięty dataset; dodaje kolumny
`source` = `data1` / `data2` / `data3` i `corrupted` = `original` / `corrupted`):

```python
from alpr.dataset import load_plates
df = load_plates()
detekcja = load_plates(sample_type="full_photo", split="test")
ocr_test = load_plates(split="test", ocr_only=True)          # bez zamazanych tablic (blurred == 1)
oryginaly = load_plates(split="test", originals_only=True)   # bez sztucznych zniekształceń
noc = df[df.lighting == "night"]
niedoswietlone = df[df.exposure == "under"]
```

### Jak zbiór został zbudowany

Do codziennej pracy niepotrzebne (wystarczy `get_data.py`). Tylko do odtworzenia albo zmiany zbioru:

```bash
pip install -e ".[build]"          # imagecorruptions==1.1.2, scikit-image
python3 scripts/download_data.py    # pobiera źródła do data/data1..3 (ok. 6 GB, w tym archiwum 4.5 GB)
python3 scripts/build_dataset.py    # składa data/combined/
kaggle datasets version -p data/combined -m "opis zmian" -r zip   # nowa wersja na Kaggle
```

* `download_data.py` z archiwów UC3M-LP i AUTO.RIA bierze tylko wybrane pliki. Wybór jest zapisany
  w `data/<zbiór>/labels.csv` (w gicie), więc za każdym razem wychodzi to samo. Sumy kontrolne
  w `data/<zbiór>/SHA256SUMS` (sprawdzane równolegle). Pobieranie można przerwać i wznowić (`--force`).
* `build_dataset.py` normalizuje etykiety, zmniejsza zdjęcia (dłuższy bok 1920 px), wycina i prostuje
  tablice, dodaje zniekształcenia i liczy `exposure`. Szczegóły decyzji: `process_documentation/filip.md`.
* Logika obu skryptów jest w `alpr/build/` (każde źródło to klasa w `sources.py`, zapis w `writer.py`).
  `tests/test_build_golden.py` i `tests/test_build_combine_golden.py` pilnują, żeby po zmianach w kodzie
  wychodził ten sam zbiór co na Kaggle.

---

## Detekcja tablic

Skrypty w `scripts/`, wspólne funkcje (detekcja, wycinanie, wyrównanie, metryki) w `alpr/detection/`.
Szczegóły decyzji: `process_documentation/hubert.md`.

```bash
python3 scripts/train_yolo.py                                   # trening YOLO (GPU: Kaggle/Colab)
python3 scripts/eval_detection.py --weights runs/detect/plates/weights/best.pt
python3 scripts/eval_alignment.py                               # jakość wyrównania perspektywy
python3 scripts/detect_plates.py --weights best.pt --images zdjecia/   # wycinki dla OCR
```

| Skrypt | Co robi | Wynik |
|---|---|---|
| `train_yolo.py` | fine-tuning YOLO (domyślnie `yolo11n.pt`, 640 px) na `detection/data.yaml`, stałe ziarno | `runs/detect/<name>/weights/best.pt` |
| `eval_detection.py` | P, R, F1, AP50, AP50-95 dla całego podziału i w grupach: źródło, `lighting`, `exposure`, `artificial_corruption`, siła, oryginał / kopia; `--images-root` testuje zdjęcia po filtrze | `results/detection_eval/<name>/metrics.csv` |
| `eval_alignment.py` | porównuje wycinek ramki z `PlateAligner` względem prawdziwych 4 narożników (IoU, błąd narożnika) | `results/alignment_eval/<name>/summary.csv` |
| `detect_plates.py` | folder zdjęć albo podział z `plates.csv` -> wyprostowane wycinki + CSV detekcji | `results/detections/<name>/detections.csv`, `crops/` |

Wszystkie skrypty przyjmują `--dataset`. Domyślnie to `data/combined`, a w notebooku Kaggle
z podpiętym datasetem `/kaggle/input/eu-license-plates-alpr`. Trening na GPU: nowy notebook Kaggle
(Accelerator GPU, Internet włączony, Add Input: `filipstanczyk/eu-license-plates-alpr`) i komórka:

```bash
!git clone -q https://github.com/FStanczyk/LicensePlatesN2Project.git && cd LicensePlatesN2Project && pip install -q -e . \
  && python3 scripts/train_yolo.py --project /kaggle/working/runs --name yolo11n_640 \
  && python3 scripts/eval_detection.py --weights /kaggle/working/runs/yolo11n_640/weights/best.pt --name yolo11n_640 \
  && python3 scripts/eval_alignment.py --preds results/detection_eval/yolo11n_640/predictions.csv --name yolo11n_640 \
  && python3 scripts/detect_plates.py --weights /kaggle/working/runs/yolo11n_640/weights/best.pt --split test
```

Wyrównanie (`PlateAligner`): YOLO daje prostokątną ramkę, a tablica jest czworokątem. W ramce
powiększonej o 15% szukamy konturu tablicy (krawędzie Canny oraz jasne tło z progiem Otsu razem
z niebieskim paskiem UE), przybliżamy go 4 narożnikami, doprecyzowujemy je dopasowaniem prostych
do boków i prostujemy wycinek tą samą metodą co `build_dataset.py`. Gdy kontur nie pasuje do ramki
(IoU < 0.6) albo ma nietypowe proporcje, zostaje zwykły wycinek ramki (`align = box`).

Format `detections.csv` (wejście dla OCR): `image, det_id, conf, x1, y1, x2, y2, polygon, align, crop`;
zdjęcie bez detekcji ma jeden wiersz z pustym `det_id`.

---

## Struktura kodu

Logika jest w pakiecie `alpr/`, a skrypty w `scripts/` to cienkie CLI (argparse + wywołania `alpr`).
Skrypty wymagają zainstalowanego pakietu (`pip install -e .`). Najczęściej używane rzeczy są w `alpr`:
`from alpr import load_plates, PlateDetector, PlateAligner`.

| Moduł | Co jest w środku | Kto |
|---|---|---|
| `alpr/config.py` | ścieżki, `SEED`, nazwa zbioru Kaggle, `default_dataset()` | wszyscy |
| `alpr/dataset.py` | `load_plates`, `crop_path`, `normalize_text`, `exposure`, format `polygon`, `write_csv` | wszyscy |
| `alpr/geometry.py` | narożniki, `warp` (prostowanie), IoU ramek i wielokątów | wszyscy |
| `alpr/download.py` | pobieranie HTTP z wznawianiem, zdalne archiwa zip, zbiór z Kaggle, sumy kontrolne | wszyscy |
| `alpr/build/` | `sources.py` (klasa na każde źródło: pobieranie, wybór podzbioru, składanie), `writer.py` (`CombinedWriter`), `corruptions.py` | Stańczyk |
| `alpr/preprocessing.py` | `enhance(img, method)`: szkielet, metody do uzupełnienia | Stańczyk |
| `alpr/detection/` | `PlateDetector`, `PlateAligner`, `training.py`, `pipeline.py` (przebiegi po zdjęciach), `metrics.py` | Włodarczyk |
| `alpr/ocr.py` | `read_plate(crop, engine)`, metryki OCR: szkielet do uzupełnienia | Golba |

Nowa funkcja trafia do `alpr/`, a skrypt tylko ją wywołuje. Testy i lint:

```bash
pip install -e ".[dev]"            # pytest, ruff (".[build,dev]" razem z testami budowy zbioru)
pytest                             # testy w tests/, bez danych i GPU
ruff check . && ruff format .
```
