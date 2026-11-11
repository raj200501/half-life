"""Tests for the offline demo."""

from __future__ import annotations

from pathlib import Path

import pytest

from halflife import demo

GOLDEN = demo.REPO / "demo" / "golden_output.txt"


def test_demo_matches_golden_output() -> None:
    lines = demo.run_demo()
    assert "\n".join(lines) + "\n" == GOLDEN.read_text(encoding="utf-8")
    assert lines[1].startswith("Synthetic:")


def test_main_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "sub" / "report.txt"
    assert demo.main(["--out", str(out)]) == 0
    assert (
        out.read_text(encoding="utf-8")
        == capsys.readouterr().out
        == GOLDEN.read_text(encoding="utf-8")
    )


def test_replay_model_queues_and_leftovers() -> None:
    model = demo.ReplayModel([{"purpose": "step", "text": "a"}, {"purpose": "probe", "text": "b"}])
    assert model.chat([], max_tokens=1, temperature=0.0, seed=0) == "a"
    with pytest.raises(RuntimeError, match="no recorded reply"):
        model.chat([], max_tokens=1, temperature=0.0, seed=0)
    with pytest.raises(RuntimeError, match="left unused"):
        model.assert_consumed()
    assert model.count_tokens([{"role": "user", "content": "abcdefgh"}]) == 2
