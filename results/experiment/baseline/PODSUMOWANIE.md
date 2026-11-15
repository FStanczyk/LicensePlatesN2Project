# Podsumowanie wyników bloku 2 (stary zbiór)

Przebieg z 24.10.2026: zbiór z Kaggle w wersji z 11.10.2026, przed poprawkami Włodarczyka z 08.11.2026.
Wyniki zostają jako punkt odniesienia. Nie są miarodajne z powodów wymienionych w sekcji "Zastrzeżenia".

## Konfiguracja

| Element | Ustawienie |
|---|---|
| Zbiór | `filipstanczyk/eu-license-plates-alpr`, wersja z 11.10.2026 (16584 wierszy, 2584 tablic na całych zdjęciach) |
| Detektor | YOLO `yolo11n.pt`, 640 px, 100 epok, `batch` 16, ziarno 0, trening lokalnie na RTX 4080 (ok. 11.5 min) |
| Wagi | `runs/detect/plates/weights/best.pt` (poza gitem) |
| Wyrównanie | `PlateAligner` bez CLAHE |
| OCR | EasyOCR, model `en`, znaki A-Z i 0-9, bez filtra `plate_fragments` |
| Próg detekcji | `conf` 0.25, dopasowanie do ground truth przy IoU >= 0.5 |
| Ground truth | 81 oryginalnych zdjęć z data1 (Polska, po 1 tablicy): 42 z `val`, 39 z `test` |
| Wersje | `clean` + `brightness`, `darken`, `motion_blur`, `defocus_blur` w sile 1-5 (21 wersji, 1701 zdjęć) |

## Detektor (walidacja ultralytics, `val`, cały zbiór)

| P | R | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.969 | 0.943 | 0.969 | 0.786 |

## Wyniki eksperymentu (81 tablic na wersję)

`wykryte`: tablica dopasowana do detekcji. `IoU`: średnie IoU najlepszej ramki. `bezbłędny`: odczyt równy
prawdziwemu numerowi. `CAR`: 1 - odległość edycyjna / długość numeru (średnio, nieodczytana tablica = 0).
`pusty`: brak odczytu. `conf OCR`: średnia pewność EasyOCR.

| Wersja | Wykryte | IoU | Bezbłędny | CAR | Pusty | conf OCR |
|---|---|---|---|---|---|---|
| clean | 1.000 | 0.899 | 0.358 | 0.775 | 0.037 | 0.677 |
| brightness 1 | 1.000 | 0.902 | 0.407 | 0.781 | 0.025 | 0.712 |
| brightness 2 | 1.000 | 0.901 | 0.432 | 0.798 | 0.037 | 0.704 |
| brightness 3 | 1.000 | 0.900 | 0.420 | 0.802 | 0.049 | 0.704 |
| brightness 4 | 1.000 | 0.899 | 0.444 | 0.808 | 0.049 | 0.683 |
| brightness 5 | 0.988 | 0.878 | 0.444 | 0.779 | 0.086 | 0.656 |
| darken 1 | 1.000 | 0.901 | 0.395 | 0.780 | 0.025 | 0.686 |
| darken 2 | 1.000 | 0.902 | 0.333 | 0.667 | 0.074 | 0.571 |
| darken 3 | 1.000 | 0.897 | 0.198 | 0.619 | 0.049 | 0.457 |
| darken 4 | 1.000 | 0.890 | 0.136 | 0.500 | 0.086 | 0.353 |
| darken 5 | 0.988 | 0.873 | 0.074 | 0.334 | 0.160 | 0.206 |
| motion_blur 1 | 1.000 | 0.893 | 0.210 | 0.605 | 0.123 | 0.487 |
| motion_blur 2 | 1.000 | 0.879 | 0.086 | 0.356 | 0.198 | 0.301 |
| motion_blur 3 | 1.000 | 0.858 | 0.025 | 0.147 | 0.457 | 0.119 |
| motion_blur 4 | 1.000 | 0.839 | 0.000 | 0.027 | 0.765 | 0.048 |
| motion_blur 5 | 1.000 | 0.831 | 0.000 | 0.009 | 0.926 | 0.011 |
| defocus_blur 1 | 1.000 | 0.904 | 0.198 | 0.570 | 0.173 | 0.400 |
| defocus_blur 2 | 1.000 | 0.899 | 0.111 | 0.364 | 0.333 | 0.303 |
| defocus_blur 3 | 1.000 | 0.894 | 0.012 | 0.128 | 0.630 | 0.083 |
| defocus_blur 4 | 1.000 | 0.881 | 0.000 | 0.028 | 0.852 | 0.042 |
| defocus_blur 5 | 0.988 | 0.865 | 0.000 | 0.007 | 0.914 | 0.009 |

Odczyt bezbłędny osobno dla podziałów (39 tablic `test`, 42 tablice `val`):

| Wersja | test | val |
|---|---|---|
| clean | 0.487 | 0.238 |
| brightness 3 / 5 | 0.513 / 0.564 | 0.333 / 0.333 |
| darken 3 / 5 | 0.205 / 0.026 | 0.190 / 0.119 |
| motion_blur 1 / 3 | 0.231 / 0.000 | 0.190 / 0.048 |
| defocus_blur 1 / 3 | 0.205 / 0.000 | 0.190 / 0.024 |

## Wnioski

* Detekcja prawie nie reaguje na badane zniekształcenia: wykryte 98.8-100% tablic w każdej wersji, IoU spada
  najwyżej z 0.90 do 0.83 (rozmycie ruchem w sile 5). YOLO uczyło się na losowo zniekształconych kopiach tych
  samych typów, więc to wynik dla odpornego detektora, a nie dla detektora z COCO.
* Skuteczność całego potoku zależy prawie wyłącznie od OCR. Rozmycie (ruchem i ostrości) niszczy odczyt
  najszybciej: przy sile 3 bezbłędnie odczytane jest 1-2.5% tablic, a przy sile 4-5 EasyOCR zwykle nic nie zwraca.
* Przyciemnienie działa stopniowo: siła 1 nie szkodzi, od siły 3 odczyt bezbłędny spada poniżej 20%.
* Rozjaśnienie lekko poprawia odczyt (0.36 -> 0.44). Różnica to 7 tablic na 81, więc może być szumem.
* Już na czystych zdjęciach EasyOCR odczytuje bezbłędnie tylko 36% tablic, choć średnio myli się w mniej niż
  1/4 znaków (CAR 0.775). Na 52 błędnych odczytach: 18 za długich (doklejone znaki, np. "PL" z paska UE albo napis
  z ramki dealera), 12 za krótkich, 22 tej samej długości z podmienionymi znakami (V/1, S/5, S/C, O/0, G/6, 9/Q).
* Wyrównanie tablicy (`quad` 66, `rect` 7, `box` 8 na czystych zdjęciach) nie zmienia wyraźnie skuteczności OCR.

## Zastrzeżenia

* Przeciek w data1: 7 z 81 tablic (5 w `test`, 2 w `val`) ma ten sam numer co zdjęcie w `train` (to samo auto).
  Dotyczy to detekcji. OCR nie był uczony, więc jego wynik przeciek zmienia tylko przez lepsze ramki.
* `val` wybierał `best.pt` i wczesne zatrzymanie, więc 42 z 81 tablic nie są w pełni niezależne od detektora.
* Wersja `clean` to kopia oryginalnego pliku, a pozostałe wersje są zapisane ponownie jako JPEG 95, więc
  porównanie zawiera też różnicę kompresji.
* OCR bez `plate_fragments`: według pomiaru Włodarczyka filtr podnosi odczyt bezbłędny na wycinkach z `test`
  z 19.0% do 36.3%, więc obecne liczby OCR są zaniżone.
* Zbiór jest prawie wyłącznie dzienny (282 z 290 tablic data1 ma `exposure == normal`). Wnioski o oświetleniu
  dotyczą tylko sztucznych `brightness` i `darken`.
* 81 tablic na wersję: jedna tablica to 1.2 punktu procentowego, więc różnice poniżej kilku punktów to szum.

## Pozostałe pomiary z tego dnia

* EasyOCR na 100 losowych oryginalnych wycinkach `test` z `plates.csv` (data1 + data3): 17 odczytów bezbłędnych.
* `detect_plates.py` na całym `test` (366 zdjęć, z UC3M-LP): 439 wycinków, 6 zdjęć bez detekcji, wyrównanie
  `quad` 312, `rect` 57, `box` 70; EasyOCR zwrócił niepusty odczyt dla 324 wycinków.

## Pliki

| Plik | Zawartość |
|---|---|
| `summary.csv` | wykryte i odczyt bezbłędny na wersję |
| `plates.csv` | jedna linia na (wersja, tablica): ground truth, dopasowana detekcja i odczyt |
| `detections.csv`, `ocr_easyocr.csv` | wszystkie detekcje (1708 wycinków, w tym 7 nadmiarowych) i ich odczyty |
