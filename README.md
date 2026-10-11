# Rozpoznawanie tablic rejestracyjnych (ALPR)

**Temat:** Wpływ warunków oświetleniowych i metod poprawy jakości obrazu na skuteczność rozpoznawania tablic rejestracyjnych.  
**Problem badawczy:** Jak warunki oświetleniowe wpływają na skuteczność detekcji i rozpoznawania tablic rejestracyjnych oraz które techniki poprawy jakości obrazu pozwalają ograniczyć ten wpływ?

## 👥 Zespół i podział ról

**1. Stańczyk — Inżynier Danych i Przetwarzania Obrazu (Data & Preprocessing)**
* Pozyskanie, czyszczenie i podział baz danych na podzbiory testowe (dzień, noc, prześwietlenie, niedoświetlenie).
* Implementacja algorytmów poprawy jakości obrazu (CLAHE, korekcja gamma, wyrównanie histogramu, odszumianie).
* Przygotowanie potoku augmentacji danych do treningu.

**2. Włodarczyk — Specjalista ds. Detekcji Obiektów (Detection Engineer)**
* Wybór, trening i ewaluacja modeli do wykrywania ramek tablic rejestracyjnych (np. YOLOv8, YOLOv11).
* Ocena skuteczności samej detekcji (metryki: mAP, Precision, Recall) w zależności od oświetlenia i zastosowanych metod poprawy (testowanie we współpracy ze Stańczykiem).
* Przycinanie (cropping) i perspektywiczne wyrównywanie wykrytych tablic do dalszego przetwarzania.

**3. Golba — Specjalista ds. Rozpoznawania Znaków i Integracji (OCR & Integration)**
* Implementacja i porównanie silników OCR (np. EasyOCR, Tesseract, LPRNet, CRNN).
* Ewaluacja skuteczności odczytu znaków (metryki: Character Accuracy Rate, Word/Plate Accuracy).
* Integracja wszystkich modułów w jeden spójny system i zmierzenie ostatecznej skuteczności (F1-score) dla całego potoku.

---

## 📅 Harmonogram prac (30 godzin / 10 bloków x 3h)

| Blok | Faza | Zadania przypisane do członków zespołu |
| :--- | :--- | :--- |
| **1** | **Setup & Dane** | **Wszyscy:** Założenie repozytorium (GitHub), wybór środowiska (np. Google Colab).<br>**Stańczyk:** Pobranie bazy i weryfikacja podzbiorów (dzień/noc).<br>**Włodarczyk:** Załadowanie pre-trenowanego modelu YOLO.<br>**Golba:** Wybór i instalacja narzędzi OCR. |
| **2** | **Baseline (Test 0)** | **Stańczyk:** Skrypt ładujący zdjęcia i iterujący po katalogach.<br>**Włodarczyk:** Puszczenie surowych zdjęć przez YOLO i zapisanie wyciętych ramek.<br>**Golba:** Przepuszczenie ramek przez OCR i zapisanie odczytów do CSV. |
| **3** | **Preprocessing** | **Stańczyk:** Napisanie funkcji do CLAHE, korekcji gamma i wyostrzania.<br>**Włodarczyk:** Konwersja etykiet bazy pod przyszły trening.<br>**Golba:** Napisanie funkcji obliczających metryki dla OCR. |
| **4** | **Test 1: Wpływ filtrów** | **Stańczyk:** Zapisanie zdjęć po nałożeniu filtrów poprawiających obraz.<br>**Włodarczyk:** Ponowny test detekcji na poprawionych zdjęciach.<br>**Golba:** Test OCR na tablicach wyciętych z poprawionych zdjęć. |
| **5** | **Trening Detektora** | **Stańczyk:** Przygotowanie skryptu augmentacji danych do treningu.<br>**Włodarczyk:** Konfiguracja potoku treningowego YOLO i odpalenie uczenia (po bloku).<br>**Golba:** Analiza błędów OCR z Bloku 4 (np. mylone znaki). |
| **6** | **Integracja OCR** | **Stańczyk:** Eksperymenty z binaryzacją obrazu wyciętej tablicy.<br>**Włodarczyk:** Ewaluacja wytrenowanego YOLO, przycinanie i wyrównanie perspektywy.<br>**Golba:** Konfiguracja alternatywnego modelu OCR / dostrajanie obecnego. |
| **7** | **Potok End-to-End** | **Wszyscy:** Połączenie skryptów w jeden działający pipeline: *Wejście (folder zdjęć) $\rightarrow$ Przetwarzanie (Stańczyk) $\rightarrow$ Detekcja i wyrównanie (Włodarczyk) $\rightarrow$ OCR (Golba) $\rightarrow$ Wyniki.* |
| **8** | **Wielki Eksperyment** | **Stańczyk:** Odpalenie pełnego potoku dla zbioru "Dzień".<br>**Włodarczyk:** Odpalenie pełnego potoku dla zbiorów "Noc" i "Zmierzch".<br>**Golba:** Odpalenie pełnego potoku dla zdjęć z prześwietleniem/odblaskami. |
| **9** | **Analiza Wyników** | **Stańczyk:** Wykresy porównawcze skuteczności przy różnych filtrach.<br>**Włodarczyk:** Analiza wpływu oświetlenia na metryki samej detekcji.<br>**Golba:** Zliczanie ostatecznych metryk całego systemu, analiza przypadków brzegowych. |
| **10** | **Raportowanie** | **Wszyscy:** Podział sekcji, zebranie wniosków, sprzątanie kodu, uzupełnienie dokumentacji oraz napisanie ostatecznego raportu z badań. |

---

## 🧪 Możliwe eksperymenty
* analiza obrazów wykonanych w dzień, w nocy i o zmierzchu,
* różne poziomy oświetlenia: niedoświetlenie i prześwietlenie,
* porównanie skuteczności detekcji tablicy i rozpoznawania znaków,
* zastosowanie metod poprawy obrazu, np. CLAHE, korekcja gamma, odszumianie, wyostrzanie,
* porównanie różnych modeli OCR,
* analiza wpływu jakości obrazu na skuteczność całego systemu.

---

## 🛠️ Metryki ewaluacyjne
* **Detekcja:** mAP (mean Average Precision), Precision, Recall.
* **Rozpoznawanie znaków (OCR):** Character Accuracy Rate, Word Accuracy, F1-score.
* **Cały system:** Skuteczność End-to-End (czy system poprawnie wykrył tablicę i bezbłędnie odczytał wszystkie znaki).

---

## 📦 Dane

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
