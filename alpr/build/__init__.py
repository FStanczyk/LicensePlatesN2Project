"""Budowa zbioru data/combined (Stańczyk). Do codziennej pracy niepotrzebne, wystarczy get_data.py.

Moduły:
    download      pobieranie archiwów (całych albo pojedynczych plików przez HTTP Range), DATASETS
    sources       wybór podzbiorów z UC3M-LP i AUTO.RIA (zapisywany w data/<zbiór>/labels.csv)
    corruptions   sztuczne zniekształcenia (imagecorruptions + darken); wymaga extras [build]
    combine       składanie data1..3 w jeden zbiór: plates.csv, YOLO, wycinki OCR

Wynik zależy od SEED i kolejności wierszy: każda zmiana tutaj zmienia zbiór na Kaggle.
"""
