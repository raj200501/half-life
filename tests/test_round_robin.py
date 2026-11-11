"""Tests for tools/round_robin.py with the experiment runner replaced by a recorder."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "round_robin", Path(__file__).resolve().parents[1] / "tools" / "round_robin.py"
)
assert SPEC is not None and SPEC.loader is not None
round_robin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(round_robin)


def _recorder(codes: dict[tuple[str, str], int]) -> tuple[list[list[str]], Any]:
    calls: list[list[str]] = []

    def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        key = (cmd[cmd.index("--index") + 1], cmd[cmd.index("--model") + 1])
        return subprocess.CompletedProcess(cmd, codes.get(key, 0))

    return calls, run


def test_round_robin_interleaves_models_per_index_and_passes_prereg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plans = tmp_path / "plans.jsonl"
    plans.write_text("{}\n{}\n\n{}\n", encoding="utf-8")
    calls, run = _recorder({("1", "b"): 1})
    monkeypatch.setattr(round_robin.subprocess, "run", run)
    argv = ["--plans", str(plans), "--prefix", "e1", "--models", "a", "b"]
    assert round_robin.main([*argv, "--prereg", "prereg/0001-e1.md"]) == 1
    order = [(c[c.index("--index") + 1], c[c.index("--model") + 1]) for c in calls]
    assert order == [("0", "a"), ("0", "b"), ("1", "a"), ("1", "b"), ("2", "a"), ("2", "b")]
    assert all(c[c.index("--prereg") + 1] == "prereg/0001-e1.md" for c in calls)
    assert calls[0][calls[0].index("--run-id") + 1] == "e1-a"
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {"index": 1, "model": "b", "exit": 1} in lines


def test_round_robin_stops_on_a_cap_and_omits_prereg_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = tmp_path / "plans.jsonl"
    plans.write_text("{}\n{}\n", encoding="utf-8")
    calls, run = _recorder({("0", "b"): 3})
    monkeypatch.setattr(round_robin.subprocess, "run", run)
    argv = ["--plans", str(plans), "--prefix", "p", "--models", "a", "b", "--indices", "0", "1"]
    assert round_robin.main(argv) == 3
    assert len(calls) == 2 and all("--prereg" not in c for c in calls)
