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

## 15.11.2026: blok 2 ponownie, po poprawkach Włodarczyka

### Punkt wyjścia
Po scaleniu `main` (`9513e37`) README oznaczało wyniki z 24.10 jako do policzenia ponownie: przeciek w data1
(7 numerów w `train` i w `val`/`test`), `val` w eksperymencie, ponowna kompresja tylko zniekształconych wersji,
OCR bez `plate_fragments`. Nowej wersji zbioru na Kaggle jeszcze nie ma.

### Przebieg
| Krok | Decyzja | Powód |
|---|---|---|
| Stare wyniki | opisane w `results/experiment/baseline/PODSUMOWANIE.md` (IoU, CAR, puste odczyty, podział na `val`/`test`, błędy OCR) | punkt odniesienia |
| Zbiór (`download_data.py`, `build_dataset.py`) | przebudowany lokalnie (najpierw do `data/combined_v2`, bo Windows blokował zmianę nazwy), potem podmieniony: nowy w `data/combined`, stary w `data/combined_kaggle_v1` | brak nowej wersji na Kaggle |
| Wczytywanie zdjęć | `alpr.dataset.read_image` (`np.fromfile` + `cv2.imdecode`) w `alpr/build/sources.py` | `cv2.imread` na Windows nie otwiera ścieżek spoza ASCII (`Huracán` w data3), budowa przerywała się |
| Kontrola zbioru | 8 zdjęć data1 zmieniło podział, brak wspólnych numerów `train`/`val`/`test` | zgodne z opisem poprawki |
| Trening (`train_yolo.py --name plates_v2 --data data/combined_v2/...`) | te same parametry co 24.10 | porównywalność |
| Eksperyment (`build_experiment.py --dataset data/combined_v2`, `run_experiment.py --name v2`) | tylko `test` (domyślnie) | `val` wybiera `best.pt` |

### Wynik
Nowy zbiór (`data/combined`), tylko oryginały (bez kopii zniekształconych, które podwajają liczbę wierszy do 16584):

| Źródło | Typ | train | val | test | Razem |
|---|---|---|---|---|---|
| data1 (Polska) | całe zdjęcia | 210 | 41 | 39 | 290 |
| data2 (UC3M-LP, zamazane znaki) | całe zdjęcia | 539 | 101 | 144 | 784 |
| data3 (AUTO.RIA) | gotowe wycinki | 6317 | 311 | 372 | 7000 |
| Razem | | 7066 | 453 | 555 | 8074 |

Całe zdjęcia (1074) mają razem 1292 tablice (train 905, val 175, test 212).

YOLO `plates_v2`: val P 0.976, R 0.933, mAP50 0.973, mAP50-95 0.781.

Odczyt bezbłędny (39 tablic na wersję, `results/experiment/v2/summary.csv`):

| Zniekształcenie | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `brightness` | 0.615 | 0.641 | 0.641 | 0.718 | 0.641 | 0.513 |
| `darken` | 0.615 | 0.462 | 0.308 | 0.282 | 0.154 | 0.103 |
| `motion_blur` | 0.615 | 0.256 | 0.077 | 0.000 | 0.000 | 0.000 |
| `defocus_blur` | 0.615 | 0.103 | 0.077 | 0.000 | 0.000 | 0.000 |

Wykryte 97.4-100% tablic w każdej wersji. CAR na czystych 0.887 (stary przebieg na `test`: 0.864).
Szczegóły: `results/experiment/v2/PODSUMOWANIE.md`.

Testy: 432 przechodzą; 4 nie przechodzą także bez moich zmian (2 golden testy przez OpenCV 5.0, 2 testy YOLO-pose
przez brak uprawnień do symlinków na Windows). 12 testów wymaga `PYTHONUTF8=1` (pliki otwierane bez `encoding`).

### Stan na koniec dnia
* [x] podsumowanie starych wyników
* [x] zbiór przebudowany lokalnie, YOLO przetrenowane, eksperyment policzony ponownie
* [ ] nowa wersja zbioru na Kaggle (do uzgodnienia ze Stańczykiem)
* [ ] metryki OCR w `alpr/ocr.py` (`character_accuracy`, `plate_accuracy`)

### Kolejne kroki (ustalone 15.11.2026)
Po poprawkach potok odczytuje bezbłędnie ponad 60% tablic na czystych zdjęciach (detekcja 97-100%). Dalsze
badania skupiają się na rozmyciu, które niszczy odczyt najszybciej.

1. Zbiór eksperymentu bez `brightness` i `darken`. `motion_blur` i `defocus_blur` po 10 poziomów siły zamiast 5
   (`imagecorruptions` ma tylko 5, więc poziomy trzeba zdefiniować własnymi parametrami, np. przez interpolację
   między poziomami biblioteki).
2. Drugi silnik OCR do porównania z EasyOCR (np. Tesseract). W razie potrzeby dotrenowanie go na wycinkach `train`.
3. Poprawa zdjęć zniszczonych rozmyciem (np. CLAHE, a także metody wyostrzające: unsharp masking, dekonwolucja)
   i pomiar, ile utraconej skuteczności odczytu udaje się odzyskać.

Do rozważenia: usunięcie ze zbioru zdjęć UV/IR (z kamer w podczerwieni), bo mogą pogarszać detekcję. `plates.csv`
nie ma takiej etykiety; kandydaci to nocne zdjęcia UC3M-LP (`lighting == "night"`: 212 train, 36 val, 52 test),
które trzeba przejrzeć i oznaczyć ręcznie, a potem porównać YOLO z nimi i bez nich.

Tabela bloków (harmonogram w README) jest nieaktualna: trzeba ją przepisać pod powyższe plany.

### Do zrobienia (moja część)
* metryki OCR: CAR, Plate Accuracy, F1 całego potoku w `alpr/ocr.py` (blok 3),
* analiza błędów EasyOCR: 1/I, I/L, G/C, 0/O (blok 5).

## 21.11.2026: przebudowa zbioru i trening OCR po zmianach Włodarczyka

### Punkt wyjścia
Po scaleniu PR #6 i #7 (`430f496`) w `main` są: 3 poprawione etykiety (`data1_253` przeniesione z `train` do `test`),
silniki CRNN, LPRNet i PARSeq, dekonwolucja rozmytych wycinków. Wagi Włodarczyka są tylko u niego (`runs/`, poza
gitem). Punkty 2 i 3 z "Kolejnych kroków" (drugi silnik OCR, poprawa rozmytych zdjęć) zrobił Włodarczyk.

### Przebieg
| Krok | Decyzja | Powód |
|---|---|---|
| Zbiór (`build_dataset.py`) | przebudowany do `data/combined`, poprzedni w `data/combined_v2` | poprawki etykiet; brak nowej wersji na Kaggle |
| YOLO (`train_yolo.py --name plates_ref`) | przetrenowane | stare wagi widziały `data1_253`, które jest teraz w `test` |
| Zbiory eksperymentu | `data/experiment` (test, 40 tablic), `data/experiment_tune` (train + val), `data/experiment_val` (val, 41 tablic); poprzedni test w `data/experiment_v2` | przepisy z README, sekcja "Eksperyment" |
| Wieloprocesowość | `alpr.config.MP_START`: `forkserver`, a na Windows `spawn` | `get_context("forkserver")` rzucał `ValueError` na Windows |
| OCR (`train_ocr.py`) | CRNN, PARSeq (`parseq_s0`), PARSeq po dekonwolucji (`parseq_dec_s0`), ziarno 0, przepisy z README | odtworzenie najlepszego układu |

### Wynik
YOLO `plates_ref`: val P 0.968, R 0.955, mAP50 0.977, mAP50-95 0.799. Walidacja OCR: CRNN plate_acc 0.943,
PARSeq 0.972, PARSeq po dekonwolucji 0.955 (wybór epoki na `experiment_val`).

Poprawne odczyty na `test` (40 tablic na wersję):

| Wersja | EasyOCR strojony | CRNN | PARSeq + dekonwolucja |
|---|---|---|---|
| clean | 25 | 39 | 40 |
| brightness 1 / 3 / 5 | 26 / 26 / 23 | 39 / 39 / 37 | 40 / 39 / 39 |
| darken 1 / 3 / 5 | 25 / 11 / 5 | 38 / 38 / 37 | 40 / 39 / 38 |
| motion_blur 1 / 3 / 5 | 11 / 0 / 0 | 39 / 31 / 20 | 39 / 38 / 35 |
| defocus_blur 1 / 3 / 5 | 9 / 0 / 0 | 35 / 25 / 7 | 38 / 32 / 17 |
| razem zniekształcone | 0.278 | 0.815 | 0.912 |

Wyniki w `results/experiment/v3_easyocr/`, `v3_crnn/`, `v3_parseq_deblur/`, wagi w `runs/detect/plates_ref/` i `runs/ocr/`.
PARSeq zgodny z przebiegiem Włodarczyka (ziarno 0: 0.916). Zostaje rozmycie ostrości siły 4-5 (25 i 17 na 40).

### Udostępnienie wag
| Kwestia | Decyzja | Powód |
|---|---|---|
| Miejsce | wydanie GitHub `modele-2026-11-21` (YOLO, CRNN, 2x PARSeq, `SHA256SUMS`) | `*.pt` są w `.gitignore`, PARSeq ma 91 MB; tak samo jak wydanie Włodarczyka `modele-2026-11-16` |
| Pobieranie | `scripts/get_models.py` (`alpr.download.fetch_release`) do `runs/modele/<tag>/` | pobiera tylko brakujące albo zmienione pliki i sprawdza sumy |
| Instrukcja | README, sekcja "Gotowe modele" | polecenia sprawdzone: CRNN z wag z `runs/modele/` daje ten sam wynik co `v3_crnn` |

### Stan na koniec dnia
* [x] zbiór z poprawionymi etykietami, YOLO i trzy modele OCR przetrenowane lokalnie
* [x] `get_models.py` z testami, instrukcja w README
* [x] `MP_START` z testem, równoległa budowa i `--deblur` działają na Windows
* [ ] nowa wersja zbioru na Kaggle (Stańczyk)
* [ ] 10 poziomów rozmycia bez `brightness` i `darken` (punkt 1 "Kolejnych kroków")
* [ ] przepisanie tabeli bloków w README
