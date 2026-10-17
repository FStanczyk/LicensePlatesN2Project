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

* `scripts/download_data.py`: pobiera zbiory źródłowe do `data/data1..3/photos/` (tylko gdy trzeba przebudować zbiór).
* `scripts/build_dataset.py`: składa `data/combined/` (plates.csv, YOLO, wycinki OCR, sztuczne zniekształcenia).
* `scripts/get_data.py`: pobiera gotowy zbiór z Kaggle do `data/combined/`. **Zwykle wystarczy tylko to.**
* `scripts/combined_template/`: pliki kopiowane do `data/combined/` (README, data.yaml, metadane Kaggle).
* `scripts/plate_detection.py`: moduł (nie skrypt) z detekcją YOLO, wycinaniem i wyrównaniem perspektywy.
* `scripts/train_yolo.py`, `eval_detection.py`, `eval_alignment.py`, `detect_plates.py`: detekcja
  (opis w README, sekcja "Detekcja tablic"); `notebooks/detekcja_kaggle.ipynb` uruchamia je na GPU.
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
* Na Kaggle wycinki leżą w `ocr/ocr/<split>/` (lokalnie `get_data.py` to wyrównuje).
* Nie commituj zdjęć ani `data/combined/` (są w `.gitignore`).

## Uruchamianie

```bash
pip install -r requirements.txt
python3 scripts/get_data.py                 # gotowy zbiór z Kaggle (token w ~/.kaggle/)
python3 scripts/download_data.py            # tylko do przebudowy zbioru
python3 scripts/build_dataset.py            # wymaga imagecorruptions==1.1.2 i opencv-python
```

Duże obliczenia (trening YOLO) robimy w Google Colab / notebookach Kaggle, dataset
`filipstanczyk/eu-license-plates-alpr`.

## Konwencje

* Wszystko po polsku: kod (komentarze, docstringi, komunikaty), README, dokumentacja, commity.
* W tekstach bez em dashów (używaj dwukropka, przecinka albo nawiasu) i bez emotek.
* Skrypty: `#!/usr/bin/env python3`, docstring z sekcją "Użycie:", `argparse`, ścieżki przez
  `ROOT = Path(__file__).resolve().parent.parent`, `main() -> int` i `sys.exit(main())`.
* Losowość zawsze ze stałym ziarnem (`SEED`), żeby wyniki były powtarzalne w całym zespole.
* Commity: krótki tytuł w trybie rozkazującym ("Dodaj ...", "Przenieś ..."), w treści lista zmian per plik.
* Po zmianie danych, skryptów lub wyników zaktualizuj README i swój plik w `process_documentation/`.
