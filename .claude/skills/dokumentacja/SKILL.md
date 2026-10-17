---
name: dokumentacja
description: Dopisuje wpis do dziennika pracy w process_documentation/<imię>.md po zakończonym bloku pracy (dane, filtry, detekcja, OCR, eksperymenty). Używaj, gdy ktoś prosi o udokumentowanie pracy, podsumowanie dnia albo aktualizację dokumentacji procesu.
---

# Wpis w dokumentacji pracy

Każda osoba ma swój plik: `process_documentation/filip.md` (Stańczyk), `hubert.md` (Włodarczyk),
`<imię>.md` (Golba). Jeśli pliku nie ma, utwórz go z nagłówkiem
`# Dokumentacja pracy: <Nazwisko> (<rola z README>)`.

## Kroki

1. Ustal, czyj to plik (zapytaj, jeśli nie wynika z kontekstu) i przeczytaj go, żeby nie powtarzać treści.
2. Zbierz fakty: `git log` i `git diff` z danego dnia, uruchomione skrypty, liczby z ich wyjścia.
   Nie wymyślaj wyników: wpisuj tylko to, co zostało zmierzone albo jest w kodzie.
3. Dopisz na końcu pliku sekcję w formacie poniżej.
4. Jeśli zmieniły się dane, skrypty lub wyniki widoczne dla zespołu, zaktualizuj też README.

## Format sekcji

```markdown
## DD.MM.RRRR: <krótki temat>

### Punkt wyjścia
Co było na starcie i jaki był cel.

### <Podsekcje merytoryczne>
Decyzje z uzasadnieniem (najlepiej tabela: opcja | decyzja | powód), użyte skrypty
w nagłówkach w backtickach, np. ### Trening (`scripts/train_yolo.py`).

### Wynik
Liczby i metryki (tabela), ścieżki do plików wynikowych.

### Stan na koniec dnia
* [x] zrobione
* [ ] niezrobione

### Do zrobienia (moja część)
* kolejne kroki z numerem bloku harmonogramu, np. (blok 6)
```

## Styl

* Po polsku, rzeczowo, krótkie zdania.
* Bez em dashów (dwukropek, przecinek albo nawias) i bez emotek.
* Daty w formacie DD.MM.RRRR.
