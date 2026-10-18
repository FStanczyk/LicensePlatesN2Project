# CLAUDE.md

Projekt zespołowy (studia): rozpoznawanie tablic rejestracyjnych (ALPR) i wpływ oświetlenia oraz
metod poprawy obrazu (CLAHE, gamma, odszumianie, unsharp masking) na detekcję i OCR.
Temat, podział ról, harmonogram, metryki i opis danych są w [README.md](README.md); czytaj go,
zanim zaczniesz większe zadanie.

## Zespół

* Stańczyk: dane i przetwarzanie obrazu (filtry, augmentacja).
* Włodarczyk: detekcja tablic (YOLO), przycinanie i wyrównanie perspektywy.
* Golba: OCR (EasyOCR, Tesseract, LPRNet, CRNN) i złożenie całego potoku.

Potok docelowy: folder zdjęć -> przetwarzanie -> detekcja i wyrównanie -> OCR -> wyniki (CSV, metryki).

## Struktura

* `alpr/`: pakiet z całą logiką (`pip install -e .`), opis modułów w README, sekcja "Struktura kodu":
  `config` (ścieżki, `SEED`), `dataset` (`load_plates`, `crop_path`, `normalize_text`, `write_csv`),
  `geometry` (`warp`, IoU), `download` (HTTP, zip, Kaggle), `build/` (`sources.py`: klasa na źródło,
  `writer.py`: `CombinedWriter`), `preprocessing.py` i `ocr.py` (szkielety),
  `detection/` (`PlateDetector`, `PlateAligner`, `training`, `pipeline`, `metrics`).
* `scripts/`: cienkie CLI wywołujące `alpr`:
  * `get_data.py`: pobiera gotowy zbiór z Kaggle do `data/combined/`. **Zwykle wystarczy tylko to.**
  * `download_data.py`, `build_dataset.py`: tylko do przebudowy zbioru (źródła w `data/data1..3/photos/`).
  * `train_yolo.py`, `eval_detection.py`, `eval_alignment.py`, `detect_plates.py`: detekcja
    (opis w README, sekcja "Detekcja tablic", tam też komenda do notebooka Kaggle na GPU).
* `alpr/build/template/`: pliki kopiowane do `data/combined/` (README, data.yaml, metadane Kaggle).
* `tests/`: pytest, bez danych i GPU; `test_build_golden.py` i `test_build_combine_golden.py` pilnują powtarzalności zbioru.
* `runs/` (wagi) i `results/` (metryki, wycinki): wyniki skryptów, duże pliki poza gitem.
* `data/dataN/`: w gicie tylko adnotacje, `labels.csv` i `SHA256SUMS`; zdjęcia są poza gitem.
* `process_documentation/<imię>.md`: dziennik pracy każdej osoby (format: patrz skill `dokumentacja`).

## Dane

* Pracujemy na `data/combined/plates.csv` (jedna linia na tablicę). Kolumny opisuje README, sekcja "Dane".
* Ścieżki w `image` i `crop` są względne do `data/combined/`.
* `blurred == 1` (cały UC3M-LP, `data2_`): znaki zamazane, **pomijaj przy metrykach OCR**.
* `lighting` jest znane tylko dla UC3M-LP; do analizy naświetlenia używaj `exposure` (under/normal/over).
* Każda tablica ma kopię z `artificial_corruption != "none"`; przy porównaniach rozdzielaj oryginały i kopie.
* Detekcja: `sample_type == "full_photo"`, `data/combined/detection/data.yaml`, klasa 0 = tablica.
* Na Kaggle wycinki leżą w `ocr/ocr/<split>/` (lokalnie `get_data.py` to wyrównuje); `crop_path` obsługuje oba układy.
* Wczytuj przez `alpr.dataset.load_plates(...)` (filtry `sample_type`, `split`, `originals_only`, `ocr_only`),
  a nie ręcznie przez `pd.read_csv`.
* Nie commituj zdjęć ani `data/combined/` (są w `.gitignore`).

## Uruchamianie

```bash
pip install -r requirements.txt             # = pip install -e . (pakiet alpr z zależnościami)
python3 scripts/get_data.py                 # gotowy zbiór z Kaggle (token w ~/.kaggle/)
python3 scripts/download_data.py            # tylko do przebudowy zbioru
python3 scripts/build_dataset.py            # wymaga pip install -e ".[build]"
pip install -e ".[dev]" && pytest && ruff check . && ruff format --check .
```

Duże obliczenia (trening YOLO) robimy w Google Colab / notebookach Kaggle, dataset
`filipstanczyk/eu-license-plates-alpr`.

## Konwencje

* Wszystko po polsku: kod (komentarze, docstringi, komunikaty), README, dokumentacja, commity.
* W tekstach bez em dashów (używaj dwukropka, przecinka albo nawiasu) i bez emotek.
* Logika w `alpr/`, skrypt tylko ją wywołuje. Nie duplikuj funkcji (np. `warp`, IoU, filtrów `plates.csv`):
  importuj z `alpr`. Każda nowa funkcja w `alpr/` ma test w `tests/`.
* Skrypty: `#!/usr/bin/env python3`, docstring z sekcją "Użycie:" (to tekst `--help`), `argparse`,
  `main() -> int` i `sys.exit(main())`. Wymagają `pip install -e .` (bez `sys.path.insert`).
  `main()` tylko parsuje argumenty, wywołuje `alpr` i drukuje wynik. Ścieżki i stałe z `alpr.config`.
* Kod bez komentarzy `#` (poza `# noqa` itp.) i lakoniczny: docstring jednolinijkowy i tylko wtedy,
  gdy nazwa nie wystarcza. Klasy tam, gdzie jest stan albo konfiguracja; czyste funkcje zostają funkcjami.
* Styl: `ruff format` i `ruff check` (konfiguracja w `pyproject.toml`) muszą przechodzić.
* Zmiany w `alpr/build/` zmieniają zbiór na Kaggle: `tests/test_build_golden.py` i `test_build_combine_golden.py` muszą przechodzić
  albo zmiana jest świadoma i idzie razem z nową wersją zbioru.
* Losowość zawsze ze stałym ziarnem (`SEED`), żeby wyniki były powtarzalne w całym zespole.
* Commity: krótki tytuł w trybie rozkazującym ("Dodaj ...", "Przenieś ..."), w treści lista zmian per plik.
* Po zmianie danych, skryptów lub wyników zaktualizuj README i swój plik w `process_documentation/`.
