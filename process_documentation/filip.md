# Dokumentacja pracy: Stańczyk (dane i przetwarzanie obrazu)

## 11.10.2026: pozyskanie i przygotowanie danych

### Punkt wyjścia

Na start mieliśmy jeden zbiór, `data1` ([Poland Vehicle License Plate Dataset](https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset)):
290 zdjęć polskich samochodów, adnotacje CVAT (ramka tablicy + numer). To za mało do treningu
detektora i OCR, więc celem było dojście do ok. 5-10 tys. oznaczonych tablic.

Wymagania wobec nowych zbiorów:
* etykiety muszą zawierać **numer tablicy** (nie tylko ramkę), bo bez tego nie policzymy metryk OCR,
* tablice w alfabecie łacińskim (bez znaków chińskich, cyrylicy itp.),
* najlepiej tablice europejskie, najlepiej polskie,
* najlepiej zróżnicowane oświetlenie (dzień / noc), bo to temat projektu,
* najlepiej do pobrania bez rejestracji i bez wniosku.

### Przegląd zbiorów

| Zbiór | Co ma | Decyzja | Powód |
|---|---|---|---|
| UC3M-LP (Hiszpania) | 1975 zdjęć, 2547 tablic, wielokąt + numer + znacznik dzień/noc | **wzięty** (`data2`) | jedyny darmowy zbiór europejski z całymi zdjęciami, numerami i oznaczeniem nocy |
| AUTO.RIA Numberplate OCR EU (Nomeroff Net) | 74 tys. wyciętych tablic z całej Europy, numer + kraj | **wzięty** (`data3`) | darmowy (CC BY 4.0), ma etykiety tekstowe i kraj, zawiera ok. 1700 polskich tablic |
| RodoSol-ALPR (Brazylia) | 20 tys. zdjęć, dzień i noc, numer + narożniki | odłożony | wymaga wysłania umowy licencyjnej z maila uczelnianego; zostaje jako opcja, jeśli zabraknie zdjęć nocnych |
| UFPR-ALPR (Brazylia) | 4500 zdjęć | odłożony | jak wyżej, na wniosek |
| CCPD (Chiny) | 300 tys. zdjęć, podzbiór ciemnych zdjęć | odrzucony | tablice zawierają chińskie znaki |
| AOLP (Tajwan) | 2049 zdjęć | odrzucony | nie udało się potwierdzić, że link do pobrania działa |
| AUTO.RIA Numberplate Dataset (detekcja) | ok. 23 tys. całych zdjęć | odrzucony na razie | ma tylko ramki tablic, bez numerów |
| OpenALPR benchmark EU | 108 zdjęć z numerami | odrzucony | za mały |
| Roboflow "eu-license-plate" | 89 zdjęć | odrzucony | za mały, prawdopodobnie bez numerów |
| Hugging Face 0xnu/european-licence-plate | zdjęcia tablic | odrzucony | brak etykiet (tylko nazwy plików) |
| Unidata (Polska, 196 tys.) | numery tablic | odrzucony | płatny |

### Dobór podzbiorów

* **data2 (UC3M-LP)**: zmniejszony do ok. 1000 tablic, żeby tablice hiszpańskie nie dominowały
  zbioru. Wzięte są **wszystkie zdjęcia nocne** (temat projektu), resztę dopełniają losowe zdjęcia dzienne.
  Zachowany oficjalny podział train/test.
* **data3 (AUTO.RIA OCR EU)**: 7000 tablic. Wszystkie polskie (1710), resztę po równo z pozostałych
  26 krajów UE (ok. 204 na kraj). Tablice bez kraju i spoza UE pominięte. Zachowany oficjalny podział
  train/val/test.

Wybór jest losowy, ale powtarzalny (stałe ziarno) i zapisany w `data/<zbiór>/labels.csv`, więc każdy
w zespole dostaje te same pliki.

### Różnica: całe zdjęcia a wycięte tablice

* `data1` i `data2` to **całe zdjęcia** z zaznaczoną tablicą. Nadają się do detekcji (YOLO) i do testu
  całego potoku.
* `data3` to **same wycięte tablice**. Nadają się tylko do OCR. W archiwum nie ma całych zdjęć.

Dlatego w zbiorze końcowym jest kolumna `sample_type` (`full_photo` / `plate_crop`).

### Pobieranie (`scripts/download_data.py`)

* Archiwum AUTO.RIA ma 12 GB, a potrzebujemy ok. 10% plików. Skrypt czyta spis archiwum zdalnie
  (zapytania HTTP Range) i pobiera tylko wybrane pliki, ok. 1.2 GB zamiast 12 GB.
* Zenodo (UC3M-LP) blokował pobieranie pojedynczych plików (HTTP 429, za dużo zapytań) i ograniczał
  prędkość jednego połączenia do ok. 1 MB/s. Rozwiązanie: archiwum (4.5 GB) pobierane w całości,
  kawałkami w kilku połączeniach naraz, a podzbiór wybierany lokalnie.
* Sumy SHA-256 pobranych plików są w gicie (`SHA256SUMS`), skrypt je sprawdza.

### Normalizacja: jeden zbiór (`scripts/build_dataset.py`)

Każde źródło miało inny format (XML z CVAT, JSON na zdjęcie, CSV + PNG). Skrypt składa wszystko
w `data/combined/`:

* `plates.csv`: jedna linia na tablicę, wspólne kolumny dla wszystkich źródeł,
* `detection/`: całe zdjęcia (dłuższy bok najwyżej 1920 px) + etykiety YOLO + `data.yaml`,
* `ocr/`: wycięte tablice ze wszystkich źródeł; z całych zdjęć wycinane i prostowane perspektywicznie
  na podstawie 4 narożników.

Decyzje:
* **Tekst tablicy**: wielkie litery, tylko A-Z i 0-9. W data2 część znaków jest zamazana przez autorów
  (anonimizacja), oznaczamy je `*` i kolumną `blurred`; takie tablice pomijamy przy metrykach OCR.
  Oryginalny zapis zostaje w `text_raw`.
* **Podział**: data1 nie miał podziału, losujemy 70/15/15. W data2 oficjalny test zostaje testem,
  z treningu wydzielamy 15% na walidację. data3 ma własny podział.
* **Wielokąty z więcej niż 4 punktami**: w data2 48 z 1002 tablic ma 5-6 punktów (powtórzone punkty
  albo ucięty róg). Sprowadzamy je do 4 narożników (lewy górny = min x+y, prawy dolny = max x+y,
  prawy górny = max x-y, lewy dolny = min x-y); sprawdzone wzrokowo.
* **Wycinki z pomniejszonego zdjęcia**: tablice z data1 i data2 wycinamy z zapisanego zdjęcia
  (najwyżej 1920 px), żeby oryginał i jego zniekształcona kopia miały tę samą rozdzielczość.
* Kolumny `source` i `original_id` celowo pominięte jako zbędne; źródło widać w prefiksie nazwy pliku
  (`data1_`, `data2_`, `data3_`).

### Sztuczne zniekształcenia (pakiet `imagecorruptions` 1.1.2)

Każde zdjęcie (i każdy wycinek z data3) ma jedną kopię z losowym zniekształceniem o losowej sile 1-5,
więc zbiór jest 2 razy większy. Kopia zawsze trafia do tego samego podziału co oryginał.

* Użyte: `brightness`, `contrast`, `fog`, `gaussian_noise`, `shot_noise`, `motion_blur`,
  `defocus_blur`, `jpeg_compression` + własne `darken`.
* `darken` (gamma + spadek jasności) dopisaliśmy sami, bo pakiet nie ma żadnego przyciemnienia
  (`brightness` tylko rozjaśnia), a niedoświetlenie to główny przypadek w naszym temacie.
* Pominięte: `elastic_transform` i `zoom_blur` (przesuwają piksele, ramki tablic przestałyby się
  zgadzać), `frost`, `snow`, `spatter`, `saturate` (mało związane z oświetleniem).
* Pakiet nie działa z numpy 2 i nowym scikit-image (`fog`, `gaussian_blur`, `glass_blur` się
  wywalają). Naprawione kilkoma liniami w `build_dataset.py`, bez zmiany wersji bibliotek.
* Kolumny: `artificial_corruption` (`none` albo nazwa) i `artificial_severity` (0-5). Nazwa
  "artificial", bo oryginały też mogą mieć zakłócenia (noc, odblaski, brud), których nie oznaczamy.

### Oświetlenie i naświetlenie

* `lighting` (`day` / `night` / `unknown`): pora zdjęcia według źródła; znana tylko w data2.
* `exposure` (`under` / `normal` / `over`): mierzone przez nas na wycinku tablicy. Pomiar na całym
  zdjęciu nie działał (wszystkie zdjęcia z data1 wychodziły "normalne"), bo odblask albo ciemna tablica
  to problem lokalny. Reguła: `under`, gdy nawet jasne tło tablicy jest ciemne (95. percentyl jasności
  < 100), `over`, gdy nawet znaki są jasne (5. percentyl > 100). Progi dobrane na data1 i data3 i
  sprawdzone wzrokowo na najciemniejszych i najjaśniejszych tablicach. Wśród oryginałów wychodzi ok.
  3% `under` i 4% `over`.

Dzięki temu mamy podzbiory testowe z planu: dzień / noc (`lighting`) oraz niedoświetlenie /
prześwietlenie (`exposure`), zarówno prawdziwe, jak i sztuczne (`artificial_corruption`).

### Wynik

| | Tablice | Wiersze (x2) |
|---|---|---|
| data1 (PL, całe zdjęcia) | 290 | 580 |
| data2 (ES, całe zdjęcia, 784 zdjęć, 362 tablice nocne) | 1002 | 2004 |
| data3 (UE, wycinki, 1710 PL) | 7000 | 14000 |
| **razem** | **8292** | **16584** |

* do detekcji: 1074 całych zdjęć (2148 z kopiami), podział 904 / 176 / 212 tablic (train / val / test),
* `exposure` wśród oryginałów: 7231 normal, 644 over, 417 under,
* rozmiar: 4.4 GB (detekcja 1.5 GB, OCR 3.0 GB).

Ważne: w UC3M-LP **każda** tablica ma kilka zamazanych znaków (anonimizacja autorów), więc
wszystkie 1002 tablice mają `blurred = 1`. Ten zbiór służy do detekcji i testów nocnych, ale nie do
metryk OCR. OCR liczymy na data1 i data3.

Sprawdzone wzrokowo: wycinki (w tym z wielokątów 5-6 punktowych), ramki YOLO na zdjęciach czystych,
zniekształconych, dziennych i nocnych. Zielonkawy kolor niektórych aut w data2 pochodzi ze źródła
(autorzy zmienili kolory pojazdów).

### Udostępnienie zespołowi

Zbiór końcowy idzie na Kaggle jako prywatny dataset `filipstanczyk/eu-license-plates-alpr`
(CLI `kaggle`), żeby reszta zespołu nie musiała uruchamiać skryptów pobierania. Dostęp dla zespołu
nadawany ręcznie w ustawieniach datasetu (Settings, Sharing).

Pobranie: `python3 scripts/get_data.py` (pakiet `kaggle`, raz, do `data/combined/`).

Problemy przy wrzucaniu:
* pierwsza próba przez CLI przeszła bez błędu, ale Kaggle po cichu odrzucił wersję; dopiero upload
  przez stronę pokazał przyczynę: **Kaggle nie przyjmuje nazw plików z `&`** (a oryginalne nazwy
  z AUTO.RIA miały też spacje, cyrylicę, nawiasy). Rozwiązanie: wycinki z data3 mają teraz nazwy
  `data3_00000.png`, `data3_00000_fog3.png` itd.,
* po utworzeniu wersji Kaggle przez jakiś czas zwraca 404 przy pobieraniu całego zbioru (buduje
  archiwum do pobrania), pojedyncze pliki działają od razu.

Wersja 4 na Kaggle: 4.7 GB, 20.9 tys. plików (detection + ocr + plates.csv + README).

### Stan na koniec dnia

* [x] wybór źródeł danych
* [x] skrypt pobierania z wyborem podzbiorów (`download_data.py`)
* [x] skrypt normalizacji, zniekształceń i kolumny `exposure` (`build_dataset.py`)
* [x] zbudowanie zbioru (`data/combined/`)
* [x] wrzucenie na Kaggle (wersja 4) i skrypt `get_data.py` do pobierania
* [ ] nadanie dostępu zespołowi (Settings, Sharing)
* [x] aktualizacja sekcji "Dane" w README

### Do zrobienia (moja część)

* funkcje poprawy jakości obrazu: CLAHE (na RGB i na kanale jasności L/V), korekcja gamma,
  wyrównanie histogramu, odszumianie, unsharp masking (blok 3),
* zapis zdjęć po filtrach i test wpływu filtrów (blok 4),
* potok augmentacji do treningu detektora (blok 5),
* eksperymenty z binaryzacją wyciętych tablic (blok 6),
* uruchomienie potoku na zbiorze "dzień" (blok 8) i wykresy porównawcze (blok 9).
* Brak danych o zmierzchu: żadne źródło tego nie oznacza.
