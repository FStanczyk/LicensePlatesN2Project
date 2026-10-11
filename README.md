# License plates project

Badanie wpływu warunków (oświetlenie, pogoda, rozmycie itp.) na jakość odczytu tablic rejestracyjnych.

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
