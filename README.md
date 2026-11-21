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

## Stan prac

**24.10.2026, blok 2 (Golba)**: baseline całego potoku (detekcja + OCR) na zdjęciach czystych i zniekształconych.

* Zbiór eksperymentu z ground truth (ramka i tekst): 81 polskich zdjęć z data1 (`val` + `test`, nieużyte do uczenia
  YOLO), każde w wersji czystej i 20 zniekształconych (rozjaśnienie, przyciemnienie, rozmycie ruchem i ostrości,
  siła 1-5). Szczegóły w sekcji "Eksperyment".
* YOLO `yolo11n` (640 px) wytrenowane lokalnie: val P 0.969, R 0.943, mAP50 0.969, mAP50-95 0.786.
* OCR: EasyOCR bez dostrajania, odczyty zapisywane do CSV (`scripts/ocr_plates.py`, `scripts/run_experiment.py`).

Najważniejsze wyniki (81 tablic na wersję):

| | Czyste | Siła 3 | Siła 5 |
|---|---|---|---|
| Wykryte tablice, każde zniekształcenie | 100% | 100% | 98.8-100% |
| Odczyt bezbłędny, przyciemnienie | 36% | 20% | 7% |
| Odczyt bezbłędny, rozmycie ruchem | 36% | 2.5% | 0% |
| Odczyt bezbłędny, rozmycie ostrości | 36% | 1.2% | 0% |
| Odczyt bezbłędny, rozjaśnienie | 36% | 42% | 44% |

* Detektor jest odporny na wszystkie badane zniekształcenia (w treningu widział zniekształcone kopie),
  więc spadek skuteczności całego potoku wynika z OCR.
* Najbardziej szkodzi rozmycie: już przy sile 3 odczyt praktycznie nie działa. Przyciemnienie obniża skuteczność
  stopniowo, a lekkie rozjaśnienie nawet ją poprawia.
* Już na czystych zdjęciach EasyOCR odczytuje bezbłędnie tylko 36% tablic. Typowe błędy: niebieski pasek UE
  czytany jako litera oraz mylenie I/L/1 i 5/S. To pierwsze miejsce do poprawy (bloki 5-6).

**08.11.2026 (Włodarczyk)**: poprawki błędów, które zniekształcały powyższe wyniki. Wyniki bloku 2 trzeba
policzyć ponownie po zbudowaniu nowej wersji zbioru i przetrenowaniu YOLO.

* Przeciek w data1: zdjęcia tego samego auta (kilka ujęć z jednego miejsca) trafiały losowo do różnych podziałów
  (7 numerów tablic było jednocześnie w `train` i w `val`/`test`). Teraz auto dziedziczy podział pierwszego
  zdjęcia; zmienia to podział 8 z 290 zdjęć, reszta zbioru bez zmian. Wymaga nowej wersji zbioru na Kaggle.
* Eksperyment liczony był też na `val`, którym YOLO wybiera `best.pt` i wczesne zatrzymanie (42 z 81 zdjęć).
  Domyślnie jest teraz tylko `test`; czysta wersja zdjęć jest zapisywana tak samo jak zniekształcone (JPEG 95),
  więc różnice między wersjami nie wynikają z ponownej kompresji.
* Eksperyment nie mierzy prawdziwego złego oświetlenia: UC3M-LP (całe nocne zdjęcia) ma zamazane znaki, więc
  zostaje tylko data1 (dzienne, 282 z 290 tablic `exposure == normal`). Wnioski o oświetleniu dotyczą tylko
  sztucznych `brightness` i `darken`.
* `PlateAligner` przy niskim kontraście tablicy z tłem zwracał obrys całego okna z marginesem. Po poprawce na
  data2 (2004 tablice, ramki z etykiet): IoU 0.774 -> 0.814, błąd narożnika 0.295 -> 0.202 (sama ramka: 0.769, 0.232).
* Ocena wyrównania obejmowała data1, gdzie "prawdziwe narożniki" to prostokąt ramki (wyrównanie mogło tam tylko
  przegrać); teraz liczona jest tylko na źródłach z narożnikami (`alpr.dataset.with_corners`).
* OCR doklejał napisy z ramki dealera i "PL" z paska UE (np. `SBL58012BRUTSCH`); fragmenty niższe niż 0.6
  najwyższego są teraz pomijane (`alpr.ocr.plate_fragments`). Na wycinkach z `test` (411 czytelnych oryginałów,
  data1 + data3) odczyt bezbłędny rośnie z 19.0% do 36.3%.
* 16 tablic UC3M-LP ma narożniki zapisane przeciwnie do ruchu zegara, więc ich wycinki były transponowane, a błąd
  narożnika w ocenie wyrównania zawyżony. `four_corners` porządkuje teraz narożniki (TL, TR, BR, BL) przy dowolnym
  obrocie poniżej 90 stopni. Wymaga nowej wersji zbioru.
* 20 wycinków AUTO.RIA miało `lines == 0` (brak informacji w źródle, wszystkie jednowierszowe); teraz `1`.
* Wycinki z `detect_plates.py` dla `x.jpg` i `x.png` w jednym folderze nadpisywały się; nazwa zawiera teraz
  rozszerzenie (`crops/<ścieżka>.jpg_<k>.png`).
* Sprawdzone bez błędów: AP z `evaluate` zgodne z walidacją ultralytics na tych samych wagach (AP50 0.361 wobec
  0.358, AP50-95 0.169 wobec 0.163), spójność `plates.csv` z etykietami YOLO, kopiami i wycinkami.

**15.11.2026, blok 2 ponownie (Golba)**: wyniki po poprawkach z 08.11.2026. Zbiór przebudowany lokalnie
(`download_data.py`, `build_dataset.py`; na Kaggle jest jeszcze stara wersja), YOLO wytrenowane od nowa
(`plates_v2`: val P 0.976, R 0.933, mAP50 0.973, mAP50-95 0.781), eksperyment tylko na `test` (39 tablic).

| | Czyste | Siła 1 | Siła 3 | Siła 5 |
|---|---|---|---|---|
| Wykryte tablice, każde zniekształcenie | 100% | 100% | 100% | 97.4-100% |
| Odczyt bezbłędny, przyciemnienie | 61.5% | 46.2% | 28.2% | 10.3% |
| Odczyt bezbłędny, rozmycie ruchem | 61.5% | 25.6% | 0% | 0% |
| Odczyt bezbłędny, rozmycie ostrości | 61.5% | 10.3% | 0% | 0% |
| Odczyt bezbłędny, rozjaśnienie | 61.5% | 64.1% | 71.8% | 51.3% |

* Odczyt bezbłędny na czystych zdjęciach wzrósł z 48.7% (stary przebieg, tablice z `test`) do 61.5%,
  głównie dzięki `plate_fragments` (mniej doklejonych znaków).
* Wnioski z 24.10 się utrzymują: detekcja jest odporna, rozmycie od siły 3 uniemożliwia odczyt, przyciemnienie
  szkodzi stopniowo już od siły 1, rozjaśnienie w sile 1-4 nie szkodzi.
* Przy 39 tablicach jedna tablica to 2.6 punktu procentowego, więc różnice poniżej ok. 3 tablic to szum.
* Przy budowie zbioru na Windows poprawione wczytywanie zdjęć z nazwami spoza ASCII (`alpr.dataset.read_image`).

**15-16.11.2026 (Włodarczyk)**: przyspieszenie potoku, nowe silniki OCR i odczyt rozmytych tablic.
Decyzje: `process_documentation/hubert.md`, wszystkie pomiary: `results/benchmark/PODSUMOWANIE.md` i `pomiary.csv`. Wyniki na poprawionym zbiorze (3 etykiety, niżej),
`test` eksperymentu (40 tablic x 21 wersji), średnio z 3 ziaren, ustawienia wybrane na osobnym zbiorze walidacyjnym:

| Układ | Czyste | Zniekształcone | Rozmycie ostrości | Rozmycie ruchu |
|---|---|---|---|---|
| EasyOCR (jak wyżej) | 0.600 | 0.228 | | |
| CRNN (`train_ocr.py --model crnn`) | 0.958 | | | |
| PARSeq dotrenowany, wycinek z samej ramki | 1.000 | 0.867 | 0.643 | 0.852 |
| + dekonwolucja z jądrem z widma (`--deblur`) | 1.000 | 0.917 | 0.727 | 0.970 |
| + drugi PARSeq uczony na wycinkach po dekonwolucji | 1.000 | 0.922 | 0.745 | 0.970 |

* Zostaje głównie rozmycie ostrości siły 4-5 (24 i 17.7 na 40): dysk o promieniu 8-10 px wymazuje informację.
* Zastrzeżenia: modele widziały w treningu zniekształcenia tego samego rodzaju (inne zdjęcia), dekonwolucja zna
  rodzinę jąder `imagecorruptions` (kąt i siłę szacuje), przedział ufności przy 40 tablicach to ok. +-5 p.p.
* Poprawione etykiety (sprawdzone ręcznie po zgodnym odczycie 3 silników, `results/label_audit/`): `data1_205`
  `SBE674C` -> `SBE5754C` (to samo auto co `data1_253` w train, więc po poprawce oba zdjęcia są w test),
  `data3_00244` `O2KP10` -> `02KP10`, `data3_04171` `HCC722` -> `HCC772`. Zbiór na Kaggle wymaga nowej wersji.

**21.11.2026 (Golba)**: odtworzenie wyników Włodarczyka na drugim komputerze (RTX 4080, Windows). Zbiór przebudowany
z poprawionymi etykietami, YOLO `plates_ref` (val mAP50 0.977, mAP50-95 0.799), CRNN, PARSeq i PARSeq po
dekonwolucji przetrenowane według przepisów z sekcji "OCR" (ziarno 0). Odczyt bezbłędny na `test` (40 tablic):

| Układ | Czyste | Zniekształcone | Rozmycie ostrości | Rozmycie ruchu |
|---|---|---|---|---|
| EasyOCR strojony (`results/experiment/v3_easyocr/`) | 0.625 | 0.278 | 0.080 | 0.065 |
| CRNN, wycinek z ramki (`v3_crnn/`) | 0.975 | 0.815 | 0.585 | 0.775 |
| PARSeq + dekonwolucja, pewniejszy z dwóch (`v3_parseq_deblur/`) | 1.000 | 0.912 | 0.740 | 0.940 |

* Wynik PARSeq zgodny z przebiegiem Włodarczyka dla ziarna 0 (0.916) w granicach jednej-dwóch tablic.
* `build_experiment.py --workers` i `run_experiment.py --deblur` używały metody `forkserver`, której nie ma na
  Windows; teraz `alpr.config.MP_START` wybiera `spawn`, gdy `forkserver` nie istnieje.

**21.11.2026 (Golba), duży eksperyment**: 10 poziomów rozmycia ruchu i ostrości (bez `brightness` i `darken`), 5 silników
OCR, bez poprawy obrazu. 40 tablic `test` x 21 wersji; pełne tabele (też CAR): `results/experiment/blur10/PODSUMOWANIE.md`.
Odczyt bezbłędny, wybrane poziomy (2k = siła k z `imagecorruptions`):

| Silnik | Czyste | Ruch 2 / 6 / 10 | Ostrość 2 / 6 / 10 |
|---|---|---|---|
| EasyOCR strojony | 0.625 | 0.200 / 0.000 / 0.000 | 0.225 / 0.000 / 0.000 |
| Tesseract (Otsu) | 0.375 | 0.100 / 0.000 / 0.000 | 0.100 / 0.025 / 0.000 |
| LPRNet | 0.975 | 0.875 / 0.375 / 0.000 | 0.750 / 0.325 / 0.000 |
| CRNN | 0.975 | 0.950 / 0.825 / 0.500 | 0.875 / 0.625 / 0.175 |
| PARSeq | 1.000 | 0.975 / 0.925 / 0.675 | 0.975 / 0.700 / 0.300 |

* Detekcja wykrywa 97.5-100% tablic na każdym poziomie; cały spadek to OCR.
* EasyOCR i Tesseract od poziomu 6 nie odczytują bezbłędnie żadnej tablicy; CRNN i PARSeq trzymają ponad 80% do
  poziomu 5-6 ruchu. Rozmycie ostrości jest trudniejsze niż ruch na tym samym poziomie.
* CAR spada łagodniej niż odczyt bezbłędny (PARSeq, ostrość 10: CAR 0.599, odczyt bezbłędny 0.300), więc jest co
  odzyskiwać poprawą obrazu.

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

Zbiór jest prywatny, więc potrzebny jest token API Kaggle konta, które Stańczyk dodał do zbioru
(Settings -> Sharing): kaggle.com -> Settings -> API -> Create New Token, plik zapisz w `~/.kaggle/`.
Skrypt najpierw próbuje pobrać archiwum zwykłym HTTP (równolegle, kawałkami, działa bez tokenu, jeśli
zbiór stanie się publiczny), a po odpowiedzi 403 przechodzi na API Kaggle z tokenem.
Skrypt pobiera ok. 4.5 GB do `data/_downloads/` (przerwane pobieranie wznawia się
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
| [Poland Vehicle License Plate Dataset](https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset) | 290 (PL) | całe zdjęcia | brak podziału, losujemy 70/15/15; zdjęcia tego samego auta (ten sam numer) zawsze w jednym podziale |
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
* Poprawki etykiet (16.11.2026, Włodarczyk, za zgodą Stańczyka): 3 numery sprawdzone ręcznie po tym, jak trzy
  silniki OCR zgodnie odczytały inny tekst (lista 7 podejrzanych i werdykty: `results/label_audit/` na gałęzi
  `przyspieszenie`). `data/data1/annotations.xml`: `SBE674C` -> `SBE5754C` (zdjęcie 205; to samo auto co zdjęcie 253,
  więc oba trafiają teraz do `test`), `data/data3/labels.csv`: `O2KP10` -> `02KP10`, `HCC722` -> `HCC772`.
  Liczba wierszy `plates.csv` bez zmian (16584), zmieniają się 3 numery i podział 2 wierszy (`data1_253`).
  Golden testy przechodzą (sprawdzają kod na danych syntetycznych); zbiór na Kaggle wymaga nowej wersji.

---

## Gotowe modele

Wytrenowane wagi są w wydaniach GitHub (nie w gicie). Zamiast trenować od nowa:

```bash
python3 scripts/get_models.py                      # runs/modele/modele-2026-11-21/, sprawdza SHA256
```

| Plik | Co to jest | Przepis | Wynik |
|---|---|---|---|
| `yolo11n_plates.pt` | detektor tablic YOLO11n, 640 px | `train_yolo.py --device 0 --name plates_ref` | val P 0.968, R 0.955, mAP50 0.977, mAP50-95 0.799 |
| `crnn.pt` | CRNN od zera (CTC) | `train_ocr.py --model crnn --jitter 4 --detector <yolo> --blur 0.3 --extra data/experiment_tune` | val plate_acc 0.943 |
| `parseq_s0.pt` | PARSeq dotrenowany na tablicach | `train_ocr.py --model parseq --detector <yolo> --extra data/experiment_tune` | walidacja 0.972 |
| `parseq_dec_s0.pt` | PARSeq dotrenowany dalej na wycinkach po dekonwolucji | `train_ocr.py --model parseq --init parseq_s0.pt --deconv ...` (sekcja "OCR") | walidacja 0.955 |

Uczone 21.11.2026 (Golba, RTX 4080) na zbiorze z poprawionymi etykietami, ziarno 0. Starsze wydanie
`modele-2026-11-16` (Włodarczyk, ten sam przepis, bez CRNN): `get_models.py --tag modele-2026-11-16`.
PARSeq wymaga `pip install -e ".[parseq]"`. Zbiór eksperymentu: `build_experiment.py` (sekcja "Eksperyment").

```bash
M=runs/modele/modele-2026-11-21
python3 scripts/run_experiment.py --weights $M/yolo11n_plates.pt --batch 8 --no-align --deblur \
    --engine parseq --ocr-weights $M/parseq_s0.pt $M/parseq_dec_s0.pt --name parseq_deblur
python3 scripts/run_experiment.py --weights $M/yolo11n_plates.pt --batch 8 --no-align \
    --engine crnn --ocr-weights $M/crnn.pt --name crnn
python3 scripts/detect_plates.py --weights $M/yolo11n_plates.pt --split test
python3 scripts/ocr_plates.py --detections results/detections/test/detections.csv --engine parseq \
    --param weights=$M/parseq_s0.pt
```

Wynik na `test` eksperymentu (40 tablic x 21 wersji): PARSeq z dekonwolucją czyste 1.000, zniekształcone 0.912;
CRNN 0.975 i 0.815 (`results/experiment/v3_*`).

Nowe wydanie (wymaga `gh` i uprawnień do repozytorium): pliki do jednego katalogu, `sha256sum` w formacie
`<suma>  <plik>` do `SHA256SUMS`, `gh release create <tag> <pliki>`, potem `MODELS_TAG` w `alpr/config.py`.

---

## Detekcja tablic

Skrypty w `scripts/`, wspólne funkcje (detekcja, wycinanie, wyrównanie, metryki) w `alpr/detection/`.
Szczegóły decyzji: `process_documentation/hubert.md`.

```bash
python3 scripts/train_yolo.py                                   # trening YOLO (GPU: Kaggle/Colab)
python3 scripts/train_yolo.py --pose --imgsz 960                # YOLO-pose: ramka + 4 narożniki tablicy
python3 scripts/eval_detection.py --weights runs/detect/plates/weights/best.pt
python3 scripts/eval_detection.py --classical --name classical  # klasyczny detektor (punkt odniesienia)
python3 scripts/eval_alignment.py                               # jakość wyrównania perspektywy
python3 scripts/detect_plates.py --weights best.pt --images zdjecia/   # wycinki dla OCR
```

`--batch N` w `detect_plates.py`, `eval_detection.py` i `run_experiment.py`: N zdjęć na przebieg modelu, wsad tylko
ze zdjęć tego samego kształtu (w eksperymencie 21 wersji zdjęcia), około 1.5x szybciej; ramki różnią się o 1-2 px.
fp16 i TensorRT sprawdzone i odrzucone (`process_documentation/hubert.md`, 15-16.11.2026). `--no-align` (wycinek
z samej ramki) jest przy rozmytych zdjęciach lepszy dla OCR niż `PlateAligner` (+2.6 p.p. odczytów w eksperymencie).

| Skrypt | Co robi | Wynik |
|---|---|---|
| `train_yolo.py` | fine-tuning YOLO (domyślnie `yolo11n.pt`, 640 px) na `detection/data.yaml`, stałe ziarno; `--pose`: `yolo11n-pose.pt` na zbiorze z narożnikami (eksport do `data/pose/`) | `runs/detect/<name>/weights/best.pt` |
| `eval_detection.py` | P, R, F1, AP50, AP50-95 dla całego podziału i w grupach: źródło, `lighting`, `exposure`, `artificial_corruption`, siła, oryginał / kopia; czas na zdjęcie i próg `conf` maksymalizujący F1 (dobierz na `val`, podaj jako `--conf` na `test`); `--classical` zamiast wag, `--images-root` testuje zdjęcia po filtrze | `results/detection_eval/<name>/metrics.csv`, `predictions.csv` (`image, conf, x1-y2, polygon`; `polygon` tylko z YOLO-pose) |
| `eval_alignment.py` | porównuje wycinek ramki z `PlateAligner` (albo z narożnikami YOLO-pose z `--preds`) względem prawdziwych 4 narożników (IoU, błąd narożnika) | `results/alignment_eval/<name>/summary.csv` |
| `detect_plates.py` | folder zdjęć albo podział z `plates.csv` -> wyprostowane wycinki + CSV detekcji | `results/detections/<name>/detections.csv`, `crops/` |

Wszystkie skrypty przyjmują `--dataset`. Domyślnie to `data/combined`, a w notebooku Kaggle
z podpiętym datasetem `/kaggle/input/eu-license-plates-alpr`.

Wydajność (od 08.11.2026, wyniki bez zmian, sprawdzone na prawdziwym zbiorze i w `tests/test_rownowaznosc.py`):
zdjęcia są wczytywane w tle w 8 wątkach (`read_images`), a metryki liczą macierz IoU i dopasowania raz na zdjęcie
dla wszystkich progów i grup. Na CPU: `evaluate_by_group` 3.4 s -> 0.08 s, `evaluate_alignment` (2004 tablice)
27 s -> 11 s, `crop_plates` bez modelu 7.2 s -> 3.5 s. Czas na zdjęcie z `eval_detection.py` obejmuje teraz
wczytywanie w tle, więc jest niższy niż w pomiarach sprzed tej zmiany (np. klasyczny detektor ok. 40 ms).

**Trening na GPU: `notebooks/detekcja.ipynb`.** Na Kaggle: nowy notebook, File -> Import Notebook,
Accelerator GPU, Internet włączony, Add Input `filipstanczyk/eu-license-plates-alpr`, Run All.
W Colab: Runtime GPU i token Kaggle w Secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`), zbiór pobiera się do `data/combined/`.
Pierwsza komórka klonuje repozytorium (gałąź `main`) i instaluje `alpr`, druga to parametry
(`MODEL`, `IMGSZ`, `EPOCHS`, `NAME`, `WEIGHTS`). Notatnik wywołuje te same funkcje co skrypty
i pokazuje po drodze: liczności zbioru, przykłady z etykietami, krzywe treningu, tabelę i wykres
metryk w grupach, przeoczone tablice, ocenę wyrównania z przykładami i wycinki dla OCR.
Gdy wagi `runs/detect/<NAME>/weights/best.pt` już są, trening jest pomijany. Na koniec pakuje
wagi i wyniki do `<NAME>.zip` (Kaggle: Output, Colab: pobranie).

Bez notatnika, jedną komórką:

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
(IoU < 0.6), ma nietypowe proporcje albo sięga obu przeciwległych brzegów okna (to tło, nie tablica),
zostaje zwykły wycinek ramki (`align = box`). `eval_alignment.py` pomija data1, bo jej narożniki to prostokąty ramek.
Przed szukaniem konturu wycinek przechodzi przez CLAHE na kanale L (`PlateAligner(clahe=True)`, domyślnie).
Na data2 (2004 tablice, ramki z etykiet) CLAHE podnosi IoU z 0.814 do 0.820, a przy niedoświetleniu z 0.810 do 0.828
(sama ramka 0.817); w nocy aligner nadal ma większy błąd narożnika niż ramka (0.179 wobec 0.148).

YOLO-pose (`--pose`): ten sam detektor uczy się też 4 narożników tablicy (punkty kluczowe TL, TR, BR, BL, odbicie
poziome zamienia lewe z prawymi). Narożniki są tylko w data2; tablice z data1 mają widoczność 0, więc uczą tylko ramki.
`PlateDetector` z takimi wagami zwraca narożniki, a `detect_plates.py` i `run_experiment.py` prostują nimi wycinki
(`align = pose`) zamiast `PlateAligner`. Wymaga treningu na GPU (Kaggle/Colab). Cała ścieżka (eksport, trening,
`eval_detection.py` z kolumną `polygon`, `eval_alignment.py --preds` z `method = pose`) sprawdzona na CPU
(2 epoki, 30% danych, 320 px); narożniki uczą się wolniej niż ramki, więc miarodajny wynik da dopiero pełny trening.
Na CPU `--workers 4 --cache ram` skraca trening kilkukrotnie (2 epoki w ok. 1 min zamiast kilku).

Klasyczny detektor (`alpr.detection.ClassicalDetector`, bez uczenia): black-hat i pionowe krawędzie Sobela,
domknięcie morfologiczne w 3 skalach, filtr proporcji i weryfikacja znaków (plamy po Otsu w jednym rzędzie).
Parametry dobrane na `val`. Wynik na `test` (próg 0.49 dobrany na `val`, CPU, ok. 40 ms na zdjęcie):

| Grupa | P | R | F1 | AP50 | AP50-95 |
|---|---|---|---|---|---|
| wszystkie | 0.706 | 0.340 | 0.459 | 0.344 | 0.171 |
| data1 (dzień, PL) | 0.824 | 0.718 | 0.767 | 0.691 | 0.386 |
| data2, dzień | 0.693 | 0.307 | 0.426 | 0.301 | 0.138 |
| data2, noc | 0.514 | 0.153 | 0.235 | 0.166 | 0.067 |
| oryginały / kopie zniekształcone (AP50) | | | | 0.436 / 0.266 | |

Dla porównania YOLO `yolo11n` na `val`: P 0.969, R 0.943, mAP50 0.969. Klasyczna metoda działa tylko na prostych
zdjęciach z bliska (data1) i traci najwięcej w nocy i przy rozmyciu.

Format `detections.csv` (wejście dla OCR): `image, det_id, conf, x1, y1, x2, y2, polygon, align, crop`;
zdjęcie bez detekcji ma jeden wiersz z pustym `det_id`.


---

## OCR

Wejście to `detections.csv` z `detect_plates.py`. Silniki (`--engine`, parametry przez `--param klucz=wartość`):

| Silnik | Co to jest | Uwagi |
|---|---|---|
| `easyocr` (domyślny) | EasyOCR, model angielski, znaki A-Z i 0-9 | fragmenty niższe niż 0.6 najwyższego są pomijane; strojenie: `--param batch_size=16 mag_ratio=1.5 low_text=0.5 add_margin=0.05` (+3.6 p.p. w eksperymencie) |
| `tesseract` | Tesseract 5 (`--psm 7`) | wymaga programu `tesseract` w systemie; najsłabszy |
| `crnn`, `lprnet` | własne sieci CTC uczone od zera (`alpr/recognizer.py`) | `train_ocr.py --model crnn` |
| `parseq` | PARSeq (strhub) dotrenowany na tablicach (`alpr/parseq.py`) | `pip install -e ".[parseq]"`, najlepszy |

Wagi sieci: `--param weights=runs/ocr/<model>/best.pt` (domyślnie `runs/ocr/<silnik>/best.pt`). Modele EasyOCR
pobierają się raz do `~/.EasyOCR/`, wagi startowe PARSeq do `~/.cache/torch/hub/`.

```bash
python3 scripts/train_ocr.py --model crnn --jitter 4 --detector best.pt --blur 0.3 --extra data/experiment_tune
python3 scripts/train_ocr.py --model parseq --detector best.pt --extra data/experiment_tune --out runs/ocr/parseq_s0
python3 scripts/train_ocr.py --model parseq --init runs/ocr/parseq_s0/best.pt --deconv data/experiment_tune \
    --detector best.pt --select data/experiment_val --epochs 6 --lr 3e-5 --eval-every 2 --out runs/ocr/parseq_dec_s0
```

Dane do uczenia: wycinki data3, tablice data1 z polygonu (`--jitter`: kopie z przesuniętymi narożnikami), wycinki
z naszego YOLO na wszystkich wersjach zdjęć `train` (`--extra`), augmentacja na GPU (`--blur`: rozmycie).
`--deconv`: wycinki rozmytych wersji po dekonwolucji prawdziwym jądrem (`alpr.deblur.deconv_training_crops`).
CRNN uczy się ok. 2 min, PARSeq ok. 6 min na RTX 4070.

Odczyt rozmytych tablic (`alpr/deblur.py`, `run_experiment.py --deblur`): kąt ruchu szacujemy z uśrednionego widma
mocy całego zdjęcia (mediana błędu 1 stopień), każdy wycinek dostaje 16 widoków (bez zmian, 5 dysków, 5 sił ruchu
w 2 kierunkach) po dekonwolucji Richardsona-Lucy na GPU (splot przez FFT, ok. 40 ms na wycinek), a widok po
dekonwolucji wygrywa tylko, gdy podnosi pewność odczytu o `--margin` (0.1). `--ocr-weights a b`: kilka wag,
wygrywa pewniejszy odczyt.

```bash
python3 scripts/train_yolo.py --device 0                                             # raz: runs/detect/plates/weights/best.pt
python3 scripts/detect_plates.py --weights runs/detect/plates/weights/best.pt --split test
python3 scripts/ocr_plates.py --detections results/detections/test/detections.csv    # ocr_easyocr.csv obok
```

| Skrypt | Co robi | Wynik |
|---|---|---|
| `ocr_plates.py` | odczyt każdego wycinka z `detections.csv`; istniejący wynik nie jest liczony ponownie bez `--force` | `results/detections/<name>/ocr_<silnik>.csv` |

Format `ocr_<silnik>.csv`: kolumny `detections.csv` oraz `text` (znormalizowany jak `text` w `plates.csv`)
i `conf_ocr` (0-1); zdjęcie bez detekcji ma puste `text`. Tablice UC3M-LP (`data2_`) mają zamazane znaki,
więc ich odczyty nie nadają się do metryk.

---

## Eksperyment: wpływ zniekształceń na detekcję i OCR

Osobny zbiór, na którym mierzymy cały potok. Ground truth to zdjęcia, które jednocześnie:
są całymi zdjęciami z ramką tablicy (nie gotowymi wycinkami, więc odpada AUTO.RIA), mają czytelny tekst
(`blurred == 0`, więc odpada UC3M-LP), są oryginałami (bez kopii z `plates.csv`) i nie były użyte przy uczeniu
wag YOLO (domyślnie tylko `test`; `val` wybiera `best.pt` i wczesne zatrzymanie, `--splits val test` dokłada go
świadomie). Zostaje ok. 40 zdjęć z data1 (Polska, po 1 tablicy, prawie wyłącznie dzienne), więc różnice poniżej
kilku tablic to szum, a prawdziwego złego oświetlenia ten zbiór nie zawiera. Czysta wersja jest zapisywana tak
samo jak zniekształcone (JPEG 95). Każde zdjęcie dostaje 20 zniekształconych wersji: `brightness` (rozjaśnienie),
`darken` (przyciemnienie), `motion_blur` i `defocus_blur` (rozmycie), każde w sile 1-5. Zniekształcenia
nie zmieniają geometrii, więc ramki i tekst z ground truth obowiązują we wszystkich wersjach.

```bash
pip install -e ".[build]"                                                    # imagecorruptions
python3 scripts/build_experiment.py                                          # raz, ok. 9 min
python3 scripts/run_experiment.py --weights runs/detect/plates/weights/best.pt
```

| Skrypt | Co robi | Wynik |
|---|---|---|
| `build_experiment.py` | wybiera ground truth i zapisuje wersje zdjęć (`--splits`, `--corruptions`, `--workers`: procesy, wynik bajt w bajt ten sam, 3x szybciej) | `data/experiment/images/<wersja>/`, `ground_truth.csv` |
| `run_experiment.py` | detekcja i wyrównanie, OCR, dopasowanie do ground truth (IoU >= 0.5, jak w metrykach detekcji); `--no-align`, `--deblur`, `--ocr-weights` jak w sekcji OCR | `results/experiment/<name>/` |
| `compare_results.py` | dwa `plates.csv` na wspólnych wersjach i tablicach: różnica odsetka bezbłędnych odczytów i jej 95% przedział (bootstrap sparowany po tablicach); `--ground-truth` liczy od nowa według aktualnych etykiet | tabela w terminalu |
| `benchmark.py` | czas i jakość siatek ustawień (`--grid detection`, `ocr_sweep`, `engines`) z 95% przedziałami bootstrap po tablicach | `results/benchmark/<name>/timings.csv` |

Do strojenia i wyboru epok służy osobny zbiór z `train` i `val`, żeby nie dopasowywać ustawień do `test`:
`build_experiment.py --splits train val --out data/experiment_tune`, a z niego same tablice z `val` jako
`data/experiment_val` (ten sam `ground_truth.csv` przefiltrowany po `split`).

10 poziomów rozmycia (tylko `motion_blur` i `defocus_blur`, `alpr.build.corruptions.BLUR_LEVELS`; poziom 2k to siła k
z `imagecorruptions`, nieparzyste leżą między sąsiednimi siłami):

```bash
python3 scripts/build_experiment.py --levels 10 --out data/experiment_blur10
python3 scripts/run_experiment.py --experiment data/experiment_blur10 --weights runs/detect/plates_ref/weights/best.pt     --batch 8 --no-align --engine parseq --ocr-weights runs/ocr/parseq_s0/best.pt --name blur10_parseq
```

Najlepszy układ (wynik w sekcji "Stan prac", 15-16.11.2026), ok. 1 min na 840 zdjęć:

```bash
python3 scripts/run_experiment.py --weights runs/detect/plates_ref/weights/best.pt --batch 8 --no-align --deblur \
    --engine parseq --ocr-weights runs/ocr/parseq_s0/best.pt runs/ocr/parseq_dec_s0/best.pt --name parseq_deblur
```

Każdy krok z gotowym wynikiem jest pomijany (`--force`, aby policzyć od nowa).
Wersja to `clean` albo `<zniekształcenie>_<siła>`.

| Plik | Zawartość |
|---|---|
| `data/experiment/ground_truth.csv` | jedna linia na (wersja, tablica): `variant`, `artificial_corruption`, `artificial_severity`, `image`, `source_image` (zdjęcie w `data/combined`), `plate_id`, `split`, `text`, `country`, `exposure`, `x1`-`y2`, `polygon` |
| `results/experiment/<name>/detections.csv`, `crops/` | detekcja na wszystkich wersjach, format jak w `detect_plates.py` |
| `results/experiment/<name>/ocr_easyocr.csv` | to samo z odczytem `text`, `conf_ocr` |
| `results/experiment/<name>/plates.csv` | tablica z ground truth obok dopasowanej detekcji: `text_true`, `detected`, `iou`, `det_conf`, `align`, `crop`, `text_pred`, `conf_ocr`, `text_ok` |
| `results/experiment/<name>/summary.csv` | odsetek wykrytych (`detected`) i bezbłędnie odczytanych (`text_ok`) tablic na wersję |

Wyniki po poprawkach z 08.11.2026 (zbiór przebudowany lokalnie, `yolo11n` 640 px `plates_v2`, EasyOCR
z `plate_fragments`), 39 tablic z `test` na wersję. Pełne tabele (IoU, CAR, puste odczyty, pewność OCR) w
`results/experiment/v2/PODSUMOWANIE.md`, stary przebieg w `results/experiment/baseline/PODSUMOWANIE.md`.

| Zniekształcenie | Siła 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| wykryte, każde zniekształcenie | 1.000 | 1.000 | 1.000 | 1.000 | 0.974-1.000 | 0.974-1.000 |
| odczyt bezbłędny, `brightness` | 0.615 | 0.641 | 0.641 | 0.718 | 0.641 | 0.513 |
| odczyt bezbłędny, `darken` | 0.615 | 0.462 | 0.308 | 0.282 | 0.154 | 0.103 |
| odczyt bezbłędny, `motion_blur` | 0.615 | 0.256 | 0.077 | 0.000 | 0.000 | 0.000 |
| odczyt bezbłędny, `defocus_blur` | 0.615 | 0.103 | 0.077 | 0.000 | 0.000 | 0.000 |

Detektor jest prawie niewrażliwy na te zniekształcenia, bo w treningu widział losowo zniekształcone kopie
zdjęć (te same typy). Spadek skuteczności całego potoku wynika więc głównie z OCR.

---

## Struktura kodu

Logika jest w pakiecie `alpr/`, a skrypty w `scripts/` to cienkie CLI (argparse + wywołania `alpr`).
Skrypty wymagają zainstalowanego pakietu (`pip install -e .`). Najczęściej używane rzeczy są w `alpr`:
`from alpr import load_plates, PlateDetector, PlateAligner`.

| Moduł | Co jest w środku | Kto |
|---|---|---|
| `alpr/config.py` | ścieżki, `SEED`, nazwa zbioru Kaggle, `default_dataset()` | wszyscy |
| `alpr/dataset.py` | `load_plates`, `crop_path`, `normalize_text`, `exposure`, format `polygon`, `write_csv` | wszyscy |
| `alpr/geometry.py` | narożniki, `warp` (prostowanie), IoU ramek (też macierzowo: `box_iou_matrix`) i wielokątów | wszyscy |
| `alpr/download.py` | pobieranie HTTP z wznawianiem, zdalne archiwa zip, zbiór z Kaggle, sumy kontrolne | wszyscy |
| `alpr/build/` | `sources.py` (klasa na każde źródło: pobieranie, wybór podzbioru, składanie), `writer.py` (`CombinedWriter`), `corruptions.py` | Stańczyk |
| `alpr/preprocessing.py` | `enhance(img, method)`: szkielet, metody do uzupełnienia | Stańczyk |
| `alpr/detection/` | `PlateDetector`, `PlateAligner`, `training.py`, `pipeline.py` (przebiegi po zdjęciach), `metrics.py` | Włodarczyk |
| `alpr/experiment.py` | zbiór eksperymentu: `ground_truth`, `build_experiment` (`workers`), `images_by_source`, `match_results`, `summary` | Golba |
| `alpr/ocr.py` | silniki (`easyocr`, `tesseract`, `crnn`, `lprnet`, `parseq`), `read_batch`, `read_crops`, odczyt z dekonwolucją (`gate`, `read_deblurred`, `read_crops_deblurred`), metryki `character_accuracy`, `plate_accuracy` | Golba |
| `alpr/recognizer.py` | CRNN i LPRNet (CTC) uczone od zera, dane do uczenia OCR (`load_split`), augmentacja na GPU, `Recognizer` (wagi jako silnik) | Włodarczyk |
| `alpr/parseq.py` | PARSeq: `create`, `predict`, `train_parseq`, `deblur_select` | Włodarczyk |
| `alpr/deblur.py` | jądra rozmycia (jak `imagecorruptions`), `estimate_blur` (jądro z widma zdjęcia), `richardson_lucy` (GPU, FFT), `deconv_views` | Włodarczyk |
| `alpr/benchmark.py` | pomiar czasu i jakości ustawień, `bootstrap_ci`, `DeblurScorer` | Włodarczyk |

Nowa funkcja trafia do `alpr/`, a skrypt tylko ją wywołuje. Testy i lint:

```bash
pip install -e ".[dev]"            # pytest, ruff (".[build,dev]" razem z testami budowy zbioru)
pip install -e ".[parseq]"         # PARSeq (strhub z przejrzanego commita 1902db0)
pytest                             # testy w tests/, bez danych i GPU
ruff check . && ruff format .
```
