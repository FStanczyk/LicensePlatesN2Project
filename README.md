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

Zdjęcia nie są trzymane w gicie. Żeby je pobrać:

```bash
python3 scripts/download_data.py
```

Skrypt pobiera archiwum z Kaggle (bez logowania), rozpakowuje do `data/<zbiór>/photos/`
i sprawdza sumy SHA-256 z `data/<zbiór>/SHA256SUMS`. Jeśli archiwum masz już lokalnie:

```bash
python3 scripts/download_data.py data1 --zip ~/Downloads/archive.zip
```

| Zbiór | Źródło | Zawartość |
|---|---|---|
| `data1` | [Poland Vehicle License Plate Dataset](https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset) | 290 zdjęć, adnotacje CVAT (`annotations.xml`: bbox + numer tablicy) |

W gicie zostają tylko małe pliki: adnotacje i sumy kontrolne.

Nowy zbiór dodajesz tak:
1. Dopisz wpis do `DATASETS` w `scripts/download_data.py`.
2. Wygeneruj sumy: `cd data/<zbiór>/photos && sha256sum * > ../SHA256SUMS`.
