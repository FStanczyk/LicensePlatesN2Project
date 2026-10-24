# Dokumentacja pracy: Golba (OCR i złożenie całości)

## 24.10.2026: OCR wycinków i zbiór eksperymentu

### Punkt wyjścia
Blok 2: przepuścić ramki z YOLO przez OCR i zapisać odczyty do CSV. `alpr/ocr.py` był szkieletem,
wag YOLO lokalnie nie było. Do badania wpływu poprawy obrazu potrzebny był też zbiór z ground truth
(ramka i tekst) oraz jego zniekształcone wersje.

### Środowisko
| Kwestia | Decyzja | Powód |
|---|---|---|
| Python | `.venv` z Pythonem 3.11, torch 2.11 + CUDA 12.8 | ultralytics i EasyOCR nie wspierają jeszcze 3.14/3.15 |
| Trening YOLO | lokalnie na RTX 4080 (`scripts/train_yolo.py --device 0`, domyślne parametry) | ok. 10 min, bez Kaggle |
| Zbiór | `scripts/get_data.py` z tokenem `KAGGLE_API_TOKEN` | zbiór prywatny |

### OCR (`scripts/ocr_plates.py`)
| Kwestia | Decyzja | Powód |
|---|---|---|
| Silnik | EasyOCR, model `en`, znaki tylko A-Z i 0-9 | gotowy, czyta cały ciąg (wnioski z przeglądu literatury) |
| Kilka fragmentów tekstu | sklejane w kolejności czytania (wiersze z góry, w wierszu od lewej) | tablice dwuwierszowe |
| Pewność | średnia pewności fragmentów ważona długością | jedna liczba na tablicę |
| Ponowne uruchomienie | istniejący `ocr_<silnik>.csv` nie jest liczony ponownie bez `--force` | OCR trwa kilka minut |

### Zbiór eksperymentu (`scripts/build_experiment.py`)
| Kwestia | Decyzja | Powód |
|---|---|---|
| Typ zdjęć | tylko `full_photo` | detekcja jest częścią potoku, gotowe wycinki (AUTO.RIA) odpadają |
| Tekst | tylko `blurred == 0` | UC3M-LP ma zamazane znaki, brak ground truth dla OCR |
| Podziały | `val` + `test` (81 zdjęć z data1) | `train` uczył YOLO; `val` służył tylko do wyboru epoki; sam `test` to 39 zdjęć |
| Zniekształcenia | `brightness`, `darken`, `motion_blur`, `defocus_blur`, siła 1-5 | rozjaśnienie, przyciemnienie i rozmycie; ta sama implementacja co w zbiorze Kaggle |
| Kopie z Kaggle | nieużyte | jedno losowe zniekształcenie na zdjęcie, po kilka zdjęć na typ w `test` |
| Etykiety | wspólne dla wszystkich wersji | zniekształcenia nie zmieniają geometrii |

`alpr/build/corruptions.py`: wydzielone `apply_corruption(img, name, severity, key)`, `add_corruption` je woła
(golden testy zniekształceń przechodzą, zbiór Kaggle bez zmian).

### Wynik (`scripts/run_experiment.py`)
YOLO: val P 0.969, R 0.943, mAP50 0.969, mAP50-95 0.786 (`runs/detect/plates/weights/best.pt`).
Na 100 losowych oryginalnych wycinkach `test` z `plates.csv` EasyOCR odczytał bezbłędnie 17.

Odsetek bezbłędnie odczytanych tablic (81 na wersję, `results/experiment/baseline/summary.csv`):

| Zniekształcenie | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `brightness` | 0.358 | 0.407 | 0.432 | 0.420 | 0.444 | 0.444 |
| `darken` | 0.358 | 0.395 | 0.333 | 0.198 | 0.136 | 0.074 |
| `motion_blur` | 0.358 | 0.210 | 0.086 | 0.025 | 0.000 | 0.000 |
| `defocus_blur` | 0.358 | 0.198 | 0.111 | 0.012 | 0.000 | 0.000 |

Detekcja: wykryte 98.8-100% tablic w każdej wersji (YOLO widziało w treningu zniekształcone kopie).

Przy okazji poprawione: `crop_plates` na Windows (`os.path.commonpath` zwraca `\`, przez co `detect_plates.py` się wywalał).
`test_warp` i `test_zbior_identyczny` nie przechodzą w tym środowisku także bez moich zmian (OpenCV 5.0).

### Stan na koniec dnia
* [x] EasyOCR w `alpr/ocr.py`, `scripts/ocr_plates.py`
* [x] zbiór eksperymentu z ground truth i 20 zniekształconymi wersjami
* [x] detekcja i OCR na wszystkich wersjach, zestawienie z ground truth
* [ ] metryki OCR (`character_accuracy`, `plate_accuracy`)

### Do zrobienia (moja część)
* metryki OCR: CAR, Plate Accuracy, F1 całego potoku (blok 3),
* OCR na wycinkach z poprawionych zdjęć (CLAHE, gamma) po filtrach Stańczyka (blok 4),
* analiza błędów EasyOCR: pasek UE czytany jako litera, I/L/1, 5/S (blok 5).
