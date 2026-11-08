"""Skrypty w scripts/: konwencje z CLAUDE.md i działające --help (import bez uruchamiania obliczeń)."""

import ast
import importlib.util
import sys

import pytest

from alpr.config import ROOT

SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def test_sa_skrypty():
    assert len(SCRIPTS) >= 10


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_konwencje(path):
    source = path.read_text()
    tree = ast.parse(source)
    assert source.startswith("#!/usr/bin/env python3\n")
    assert "Użycie:" in (ast.get_docstring(tree) or "")
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert ast.unparse(main.returns) == "int"
    assert source.rstrip().endswith("sys.exit(main())")
    assert "sys.path.insert" not in source
    assert "argparse" in source


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_help(path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(f"skrypt_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        if e.name in ("imagecorruptions", "skimage"):
            pytest.skip('wymaga extras [build]: pip install -e ".[build]"')
        raise
    monkeypatch.setattr(sys, "argv", [path.name, "--help"])
    with pytest.raises(SystemExit) as exit_info:
        module.main()
    assert exit_info.value.code == 0
    assert "Użycie:" in capsys.readouterr().out
