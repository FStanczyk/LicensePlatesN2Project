# EU License Plates (PL, ES, EU) for ALPR

Zbiór złożony z trzech publicznych źródeł do projektu "Wpływ warunków oświetleniowych i metod
poprawy jakości obrazu na skuteczność rozpoznawania tablic rejestracyjnych".
Zbudowany skryptem `scripts/build_dataset.py` z repozytorium projektu.

## Zawartość

| Ścieżka | Co to jest |
|---|---|
| `plates.csv` | jedna linia na tablicę: tekst, kraj, oświetlenie, ramka, ścieżki do zdjęcia i wycinka |
| `detection/images/<split>/` | całe zdjęcia (data1, data2), dłuższy bok najwyżej 1920 px |
| `detection/labels/<split>/` | etykiety YOLO, klasa 0 = tablica |
| `detection/data.yaml` | konfiguracja dla Ultralytics YOLO (popraw `path` na Kaggle) |
| `ocr/<split>/` | wycięte tablice ze wszystkich źródeł (data1 i data2 wyprostowane perspektywicznie) |

`<split>` to `train`, `val` albo `test`.

Każde zdjęcie (data1, data2) i każdy wycinek (data3) występuje dwa razy: oryginał i kopia
z jednym losowym zniekształceniem z pakietu [imagecorruptions](https://github.com/bethgelab/imagecorruptions)
(siła 1-5) albo z naszym `darken` (niedoświetlenie: gamma + spadek jasności). Kopia jest zawsze
w tym samym podziale co oryginał. `artificial_corruption = none` nie znaczy, że zdjęcie jest
idealne: oryginały mogą mieć własne zakłócenia (noc, odblaski, brud), których nie oznaczamy.

| Do czego | Filtr |
|---|---|
| detekcja | `sample_type == "full_photo"` (albo od razu `detection/data.yaml`) |
| OCR | wszystkie wiersze, kolumny `crop` i `text`, metryki tylko dla `blurred == 0` |
| cały potok | `sample_type == "full_photo"`, porównanie po `lighting`, `exposure` i `artificial_corruption` x `artificial_severity` |

## Kolumny `plates.csv`

| Kolumna | Opis |
|---|---|
| `plate_id` | unikalny identyfikator wiersza |
| `sample_type` | `full_photo`: tablica na całym zdjęciu (data1, data2, nadaje się do detekcji); `plate_crop`: mamy tylko wyciętą tablicę (data3, tylko OCR) |
| `split` | `train` / `val` / `test` |
| `text` | numer tablicy: wielkie litery, tylko A-Z i 0-9; `*` = znak zamazany (tylko data2) |
| `text_raw` | numer w oryginalnym zapisie źródła |
| `blurred` | 1, jeśli tablica ma zamazane znaki; takie tablice pomijamy przy metrykach OCR (dotyczy wszystkich tablic z data2) |
| `country` | kod kraju (`pl`, `es`, `de`, ...) |
| `lighting` | pora zdjęcia według źródła: `day`, `night` albo `unknown` (znana tylko dla data2) |
| `exposure` | naświetlenie tablicy zmierzone na wycinku: `under` (95. percentyl jasności < 100), `over` (5. percentyl > 100) albo `normal`; liczone dla każdego wiersza, także dla zniekształconych kopii |
| `lines` | liczba wierszy na tablicy (1 lub 2) |
| `artificial_corruption` | `none` albo sztuczne zniekształcenie dodane przez nas: `brightness`, `darken`, `contrast`, `fog`, `gaussian_noise`, `shot_noise`, `motion_blur`, `defocus_blur`, `jpeg_compression` |
| `artificial_severity` | siła zniekształcenia: 0 dla oryginału, 1-5 dla kopii |
| `image` | ścieżka do całego zdjęcia (puste dla data3) |
| `x1, y1, x2, y2` | ramka tablicy w pikselach zapisanego zdjęcia (puste dla data3) |
| `polygon` | 4 narożniki `x y` (lewy górny, prawy górny, prawy dolny, lewy dolny) |
| `crop` | ścieżka do wyciętej tablicy |

## Źródła i licencje

| Źródło | Co wzięliśmy | Licencja |
|---|---|---|
| `data1`: [Poland Vehicle License Plate Dataset](https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset) (UE Katowice) | wszystkie 290 zdjęć | według opisu na Kaggle |
| `data2`: [UC3M-LP](https://github.com/ramajoballester/uc3m-lp) (Uniwersytet Karola III w Madrycie) | wszystkie zdjęcia nocne + losowe dzienne, ok. 1000 tablic | CC BY 4.0 ([Zenodo](https://zenodo.org/records/17152029)) |
| `data3`: [AUTO.RIA Numberplate OCR EU](https://github.com/ria-com/nomeroff-net) (Nomeroff Net, ARS Online OU) | wszystkie polskie tablice + po równo z innych krajów UE, 7000 tablic | CC BY 4.0 |

Zbiór jest prywatny, do użytku w projekcie studenckim.
