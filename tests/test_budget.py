"""Tests for budget caps and the usage ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from halflife import budget

REPO_ROOT = Path(__file__).resolve().parents[1]


def _caps() -> dict[str, float]:
    return dict(budget.DEFAULT_CAPS)


def test_committed_budget_toml_is_valid() -> None:
    caps = budget.load_caps(REPO_ROOT / "budget.toml")
    assert caps["cash_usd"] == 0
    for key, value in caps.items():
        assert key in budget.DEFAULT_CAPS
        assert isinstance(value, int | float)


def test_lowering_a_cap_is_allowed() -> None:
    raw = _caps() | {"claude_agent_runs_per_day": 10}
    assert budget.validate_caps(raw)["claude_agent_runs_per_day"] == 10


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("claude_agent_runs_per_day", 31, "raises the default cap"),
        ("disk_floor_gb", 4, "lowers the default floor"),
        ("claude_review_runs_per_day", "6", "must be a number"),
        ("claude_review_runs_per_day", True, "must be a number"),
    ],
)
def test_loosening_or_malformed_caps_fail(key: str, value: object, message: str) -> None:
    raw: dict[str, object] = dict(_caps())
    raw[key] = value
    with pytest.raises(budget.BudgetError, match=message):
        budget.validate_caps(raw)


def test_missing_cap_fails() -> None:
    raw = _caps()
    del raw["claude_agent_max_turns"]
    with pytest.raises(budget.BudgetError, match="missing"):
        budget.validate_caps(raw)


def test_nonzero_cash_cap_fails() -> None:
    # A negative cash cap is "lower" than 0 but still not zero.
    with pytest.raises(budget.BudgetError, match="cash_usd must be 0"):
        budget.validate_caps(_caps() | {"cash_usd": -1})


@pytest.mark.parametrize(("raw", "expected"), [("1", 1.0), ("0.5", 0.5), ("3", 1.0)])
def test_cap_share_parsing(raw: str, expected: float) -> None:
    assert budget.cap_share({"CAP_SHARE": raw}) == expected


@pytest.mark.parametrize("raw", ["0", "-1", "abc", "nan"])
def test_cap_share_rejects_bad_values(raw: str) -> None:
    with pytest.raises(budget.BudgetError):
        budget.cap_share({"CAP_SHARE": raw})


def test_cap_share_defaults_to_one() -> None:
    assert budget.cap_share({}) == 1.0


def test_effective_caps_scale_only_claude_caps() -> None:
    scaled = budget.effective_caps(_caps(), 0.5)
    assert scaled["claude_agent_runs_per_day"] == 15
    assert scaled["claude_review_runs_per_day"] == 3
    assert scaled["local_model_max_params_b"] == 8
    assert scaled["disk_floor_gb"] == 8


def test_ledger_round_trip_and_daily_counts(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    records = [
        {"ts": "2026-10-23T01:00:00Z", "kind": "claude_agent_run", "turns": 12, "cash_usd": 0},
        {"ts": "2026-10-23T02:00:00Z", "kind": "claude_agent_run", "turns": 40, "cash_usd": 0},
        {"ts": "2026-10-24T02:00:00Z", "kind": "claude_agent_run", "turns": 3, "cash_usd": 0},
        {"ts": "2026-10-23T03:00:00Z", "kind": "local", "params_b": 7.6, "cash_usd": 0},
        {"ts": "2026-10-23T03:00:00Z", "kind": "claude_single_call", "plan_equivalent_usd": 0.25},
    ]
    ledger.write_text("\n".join(json.dumps(r) for r in records) + "\n\n", encoding="utf-8")
    report = budget.evaluate(_caps(), budget.read_ledger(ledger))
    assert report.ok
    assert report.max_daily["claude_agent_runs_per_day"] == 2
    assert report.max_daily["claude_single_calls_per_day"] == 1
    assert report.max_turns == 40
    assert report.max_params_b == pytest.approx(7.6)
    assert report.plan_equivalent_usd == pytest.approx(0.25)
    assert report.cash_usd == 0


def test_missing_ledger_means_no_usage(tmp_path: Path) -> None:
    assert budget.read_ledger(tmp_path / "absent.jsonl") == []


def test_ledger_rejects_non_objects(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(budget.BudgetError, match="line 1"):
        budget.read_ledger(ledger)


def test_violations_are_reported() -> None:
    caps = _caps() | {"claude_review_runs_per_day": 1}
    records = [
        {"ts": "2026-10-23T01:00:00Z", "kind": "claude_review_run"},
        {"ts": "2026-10-23T02:00:00Z", "kind": "claude_review_run"},
        {"ts": "2026-10-23T02:00:00Z", "kind": "claude_agent_run", "turns": 41},
        {"ts": "2026-10-23T02:00:00Z", "kind": "local", "params_b": 13},
        {"ts": "2026-10-23T02:00:00Z", "kind": "free_tier", "cash_usd": 0.01},
    ]
    report = budget.evaluate(caps, records)
    assert not report.ok
    text = "\n".join(report.violations)
    assert "claude_review_runs_per_day" in text
    assert "claude_agent_max_turns" in text
    assert "local_model_max_params_b" in text
    assert "cash_usd" in text
    lines = budget.render(caps, report, 1.0)
    assert lines[0] == "cash_usd = 0.01"
    assert lines[-1] == "BUDGET: FAIL"


def test_main_prints_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = budget.main(
        ["--budget", str(REPO_ROOT / "budget.toml"), "--ledger", str(tmp_path / "none.jsonl")]
    )
    out = capsys.readouterr().out.splitlines()
    assert code == 0
    assert out[0] == "cash_usd = 0"
    assert out[-1] == "BUDGET: OK"


def test_main_fails_on_bad_budget(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "budget.toml"
    bad.write_text("cash_usd = 5\n", encoding="utf-8")
    code = budget.main(["--budget", str(bad), "--ledger", str(tmp_path / "none.jsonl")])
    assert code == 1
    assert capsys.readouterr().out.startswith("BUDGET: FAIL")


def test_main_fails_on_violation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"kind": "local", "cash_usd": 1}) + "\n", encoding="utf-8")
    code = budget.main(["--budget", str(REPO_ROOT / "budget.toml"), "--ledger", str(ledger)])
    assert code == 1
    assert capsys.readouterr().out.splitlines()[-1] == "BUDGET: FAIL"


def test_file_cap_share_reads_budget_toml(tmp_path: Path) -> None:
    assert budget.file_cap_share(REPO_ROOT / "budget.toml") == 0.2
    plain = tmp_path / "budget.toml"
    plain.write_text("cash_usd = 0\n", encoding="utf-8")
    assert budget.file_cap_share(plain) == 1.0


@pytest.mark.parametrize("value", ["0", "1.5", "true", "'x'"])
def test_file_cap_share_rejects_bad_values(tmp_path: Path, value: str) -> None:
    path = tmp_path / "budget.toml"
    path.write_text(f"cap_share = {value}\n", encoding="utf-8")
    with pytest.raises(budget.BudgetError):
        budget.file_cap_share(path)


def test_environment_can_only_lower_file_share() -> None:
    assert budget.cap_share({"CAP_SHARE": "0.5"}, file_share=0.2) == 0.2
    assert budget.cap_share({"CAP_SHARE": "0.1"}, file_share=0.2) == 0.1
    assert budget.cap_share({}, file_share=0.2) == 0.2


def test_main_applies_recorded_share(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = budget.main(
        ["--budget", str(REPO_ROOT / "budget.toml"), "--ledger", str(tmp_path / "none.jsonl")]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "claude_agent_runs_per_day = 6 " in out
    assert "claude_review_runs_per_day = 1 " in out
    assert "cap_share = 0.2" in out
