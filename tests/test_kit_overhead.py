"""Tests for halflife.kit.overhead: guard latency and pin size measurement.

The measurements run on a synthetic temporary project and on hook inputs from
tests/fixtures/claude_hooks/ (authored from the documented format). The tests
check the shape and consistency of the records, never particular timings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from halflife.kit.overhead import (
    CHARS_PER_TOKEN,
    PLACEHOLDER,
    estimate_tokens,
    latency_summary,
    load_fixture,
    measure_guard_latency,
    measure_pin_latency,
    measure_pin_tokens,
    overhead_record,
)
from halflife.kit.pin import render_block
from halflife.kit.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "halflife.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "claude_hooks"
RULES = load_spec(EXAMPLE)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_core.py").write_text("def test():\n    assert True\n", encoding="utf-8")
    (root / "mypkg").mkdir()
    (root / "mypkg" / "core.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "requirements.txt").write_text("pyyaml==6.0.1\n", encoding="utf-8")
    return root


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == (0, "chars/4")
    assert estimate_tokens("abcde") == (2, "chars/4")
    assert estimate_tokens("x" * 8) == (8 // CHARS_PER_TOKEN, "chars/4")
    assert estimate_tokens("a b c", lambda text: len(text.split())) == (3, "tokenizer")
    assert estimate_tokens("a b", lambda text: text.split()) == (2, "tokenizer")


def test_latency_summary() -> None:
    summary = latency_summary([3.0, 1.0, 2.0, 10.0])
    assert summary == {
        "n": 4,
        "median_ms": 2.5,
        "p90_ms": 10.0,
        "mean_ms": 4.0,
        "min_ms": 1.0,
        "max_ms": 10.0,
    }
    assert latency_summary([5.0])["p90_ms"] == 5.0
    with pytest.raises(ValueError, match="no latency samples"):
        latency_summary([])


def test_load_fixture(workspace: Path) -> None:
    data = load_fixture(FIXTURES / "pretooluse_write_protected.json", workspace)
    assert data["cwd"] == str(workspace)
    assert data["tool_input"]["file_path"] == str(workspace / "tests" / "test_core.py")
    inline = load_fixture({"cwd": PLACEHOLDER, "tool_name": "Bash"}, workspace)
    assert inline["cwd"] == str(workspace)
    bad = workspace / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_fixture(bad, workspace)


def test_measure_guard_latency(workspace: Path) -> None:
    names = [
        "pretooluse_bash_sed_protected.json",
        "pretooluse_bash_pytest.json",
        "pretooluse_write_protected.json",
        "pretooluse_edit_requirements.json",
    ]
    fixtures = [FIXTURES / name for name in names]
    record = measure_guard_latency(
        fixtures, RULES, repeats=3, workspace=workspace, subprocess_repeats=1
    )
    assert record["decide"]["n"] == 3 * len(fixtures)
    assert record["hook_subprocess"]["n"] == len(fixtures)
    assert record["decide"]["min_ms"] >= 0 and record["hook_subprocess"]["min_ms"] > 0
    rows = {row["fixture"]: row for row in record["fixtures"]}
    assert set(rows) == {Path(name).stem for name in names}
    assert rows["pretooluse_bash_pytest"]["allow"] is True
    assert rows["pretooluse_bash_pytest"]["reason_tokens"] == 0
    assert rows["pretooluse_bash_sed_protected"]["constraint_id"] == "tests-frozen"
    assert rows["pretooluse_edit_requirements"]["constraint_id"] == "stdlib-only"
    assert all(row["hook_matches_decide"] for row in rows.values())
    assert all(row["hook_exit_codes"] == [0] for row in rows.values())
    denied = [row["reason_tokens"] for row in rows.values() if not row["allow"]]
    assert all(tokens > 0 for tokens in denied)
    assert record["deny_reason_tokens_mean"] == round(sum(denied) / len(denied), 2)
    assert record["token_method"] == "chars/4"
    assert record["repeats"] == 3 and record["subprocess_repeats"] == 1
    json.dumps(record)


def test_measure_guard_latency_defaults(tmp_path: Path) -> None:
    fixture = {"hook_event_name": "PreToolUse", "tool_name": "Read", "tool_input": "x"}
    record = measure_guard_latency(
        [fixture], RULES, repeats=1, tokenizer=lambda text: len(text.split())
    )
    (row,) = record["fixtures"]
    assert row["fixture"] == "fixture-0" and row["allow"] is True
    assert record["deny_reason_tokens_mean"] == 0.0 and record["token_method"] == "tokenizer"


def test_measure_guard_latency_rejects_zero_repeats() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        measure_guard_latency([], RULES, repeats=0)
    with pytest.raises(ValueError, match="at least 1"):
        measure_guard_latency([], RULES, repeats=1, subprocess_repeats=0)


def test_measure_pin_tokens() -> None:
    record = measure_pin_tokens(RULES)
    block = render_block(RULES)
    assert record["chars"] == len(block)
    assert record["tokens"] == estimate_tokens(block)[0]
    assert record["rules"] == len(RULES) and record["token_method"] == "chars/4"
    bigger = measure_pin_tokens(RULES, with_reasons=True, compact_instructions=True)
    assert bigger["chars"] > record["chars"]
    counted = measure_pin_tokens(RULES, lambda text: text.split(), fmt="agents")
    assert counted["token_method"] == "tokenizer" and counted["format"] == "agents"
    assert counted["tokens"] == len(render_block(RULES, fmt="agents").split())


def test_measure_pin_latency() -> None:
    record = measure_pin_latency(RULES, 5, fmt="agents")
    assert record["n"] == 5 and record["min_ms"] >= 0
    with pytest.raises(ValueError, match="at least 1"):
        measure_pin_latency(RULES, 0)


def test_overhead_record_layout(workspace: Path) -> None:
    guard = measure_guard_latency(
        [FIXTURES / "pretooluse_bash_sed_protected.json"], RULES, repeats=1, workspace=workspace
    )
    record = overhead_record(guard, measure_pin_tokens(RULES), measure_pin_latency(RULES, 2))
    for tool in ("halflife-guard", "halflife-pin"):
        assert isinstance(record[tool]["latency_ms"], int | float)
        assert isinstance(record[tool]["tokens"], int | float)
        assert record[tool]["latency_definition"] and record[tool]["tokens_definition"]
    assert record["halflife-guard"]["latency_ms"] == guard["hook_subprocess"]["median_ms"]
    json.dumps(record)
