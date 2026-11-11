"""tools/route_check.py: usage handling (no template is checked here)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "route_check.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("route_check_under_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("args", [[], ["-h"], ["--help"], ["a.yaml", "--help"]])
def test_usage_without_templates_or_with_help(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    route_check = _load()
    assert route_check.main(["route_check.py", *args]) == 2
    out = capsys.readouterr().out
    assert out == route_check.USAGE + "\n"
    assert out.startswith("usage: uv run --frozen python tools/route_check.py ")
