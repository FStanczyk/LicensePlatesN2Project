"""Rozpoznawanie tablic rejestracyjnych (ALPR): wspólny kod dla skryptów w scripts/ i notebooków.

Moduły:
    config          ścieżki, ziarno losowości, nazwa zbioru na Kaggle
    dataset         wczytywanie i filtrowanie plates.csv, normalizacja tekstu tablicy
    geometry        narożniki, prostowanie perspektywy, IoU
    build           pobieranie zbiorów źródłowych i składanie data/combined (Stańczyk)
    preprocessing   metody poprawy obrazu (Stańczyk)
    detection       detekcja YOLO, wyrównanie tablicy, metryki detekcji (Włodarczyk)
    ocr             odczyt znaków i metryki OCR (Golba)
"""
