"""Wybór podzbiorów z dużych archiwów (UC3M-LP, AUTO.RIA).

Funkcje dostają download.Source (otwarte archiwum) i opcje z DATASETS[...]["sample"], zwracają
(wiersze labels.csv, dodatkowe pliki {ścieżka w data/<zbiór>/: zawartość}).
"""

import json
import random


def select_uc3m(src, opts: dict) -> tuple[list[dict], dict[str, bytes]]:
    """UC3M-LP: wszystkie zdjęcia z nocnymi tablicami + losowe dzienne, aż będzie "plates" tablic."""
    split_of = {}
    for split in ("train", "test"):
        for stem in src.zf.read(f"UC3M-LP/{split}.txt").decode().split():
            split_of[stem] = split
    infos = [i for i in src.zf.infolist() if i.filename.endswith(".json")]
    images = []
    for info, raw in zip(infos, src.read_many(infos)):
        split, name = info.filename.split("/")[1:]
        stem = name[: -len(".json")]
        lps = json.loads(raw)["lps"]
        images.append({
            "file": f"{stem}.jpg",
            "member": f"UC3M-LP/{split}/{stem}.jpg",
            "split": split_of[stem],
            "plates": len(lps),
            "night": sum(lp["lp_id"][1] == "N" for lp in lps),
            "_ann": raw,
        })
    night = [r for r in images if r["night"]]
    day = [r for r in images if not r["night"]]
    random.Random(opts["seed"]).shuffle(day)
    chosen, total = list(night), sum(r["plates"] for r in night)
    for r in day:
        if total >= opts["plates"]:
            break
        chosen.append(r)
        total += r["plates"]
    print(f"  {len(images)} zdjęć w archiwum, wybrano {len(chosen)} ({total} tablic, "
          f"{sum(r['night'] for r in chosen)} nocnych)")
    extra = {f"annotations/{r['file'][:-4]}.json": r.pop("_ann") for r in chosen}
    return sorted(chosen, key=lambda r: r["file"]), extra


def select_autoria(src, opts: dict) -> tuple[list[dict], dict[str, bytes]]:
    """AUTO.RIA OCR: wszystkie tablice z "prefer", resztę po równo z krajów z "countries"."""
    prefix = src.zf.namelist()[0].split("/")[0]
    by_country = {}
    for info in src.zf.infolist():
        parts = info.filename.split("/")
        if len(parts) != 4 or parts[2] != "ann" or not parts[3].endswith(".json"):
            continue
        ann = json.loads(src.zf.read(info).decode("utf-8-sig"))
        stem = parts[3][: -len(".json")]
        country = ann.get("country", "")
        if country in opts["countries"]:
            by_country.setdefault(country, []).append({
                "file": f"{parts[1]}/{stem}.png",
                "member": f"{prefix}/{parts[1]}/img/{stem}.png",
                "text": ann["description"],
                "country": country,
                "lines": ann.get("count_lines", ""),
            })
    chosen = [r for c in opts["prefer"] for r in by_country.pop(c, [])]
    rng = random.Random(opts["seed"])
    queues = [by_country[c] for c in sorted(by_country)]
    for q in queues:
        rng.shuffle(q)
    # po jednej tablicy z każdego kraju na rundę, aż do "size" (małe kraje kończą się wcześniej)
    while len(chosen) < opts["size"] and any(queues):
        for q in queues:
            if q and len(chosen) < opts["size"]:
                chosen.append(q.pop())
    counts = {}
    for r in chosen:
        counts[r["country"]] = counts.get(r["country"], 0) + 1
    print(f"  wybrano {len(chosen)} tablic: " + ", ".join(f"{c} {n}" for c, n in sorted(counts.items())))
    return sorted(chosen, key=lambda r: r["file"]), {"LICENSE.txt": src.zf.read(f"{prefix}/license.txt")}
