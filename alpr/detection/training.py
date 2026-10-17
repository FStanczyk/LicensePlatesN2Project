"""Trening YOLO na data/combined/detection."""

from pathlib import Path

import yaml

from alpr.config import SEED


def resolved_yaml(data: Path, out_dir: Path) -> Path:
    """Kopia data.yaml z bezwzględnym path (ultralytics rozwiązuje "." względem swojego katalogu)."""
    cfg = yaml.safe_load(data.read_text())
    cfg["path"] = str(data.parent.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "data_resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return out


def train_detector(data: Path, model: str, project: Path, name: str, **params) -> Path:
    """Fine-tuning wag model; zwraca ścieżkę best.pt."""
    from ultralytics import YOLO

    project = project.resolve()
    YOLO(model).train(
        data=str(resolved_yaml(data, project)),
        project=str(project),
        name=name,
        exist_ok=True,
        seed=SEED,
        deterministic=True,
        **params,
    )
    return project / name / "weights" / "best.pt"
