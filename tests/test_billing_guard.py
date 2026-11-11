"""Tests for the Claude billing guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from halflife import billing_guard


def test_forbidden_env_vars_finds_exact_and_prefixed_names() -> None:
    env = {
        "ANTHROPIC_API_KEY": "x",
        "ANTHROPIC_BASE_URL": "",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CODE_ENTRYPOINT": "cli",
        "PATH": "/usr/bin",
    }
    assert billing_guard.forbidden_env_vars(env) == (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
    )


def test_clean_environment_passes(tmp_path: Path) -> None:
    report = billing_guard.check(
        env={"PATH": "/usr/bin"}, settings_paths=[tmp_path / "missing.json"]
    )
    assert report.clean
    assert report.problems() == []


def test_api_key_helper_detected_without_reading_value(tmp_path: Path) -> None:
    with_helper = tmp_path / "a.json"
    with_helper.write_text(json.dumps({"apiKeyHelper": "/bin/echo hidden"}), encoding="utf-8")
    without = tmp_path / "b.json"
    without.write_text(json.dumps({"model": "sonnet"}), encoding="utf-8")
    broken = tmp_path / "c.json"
    broken.write_text("{not json", encoding="utf-8")
    array = tmp_path / "d.json"
    array.write_text("[]", encoding="utf-8")

    report = billing_guard.check(env={}, settings_paths=[with_helper, without, broken, array])

    assert not report.clean
    assert report.api_key_helper_files == (str(with_helper),)
    problems = report.problems()
    assert problems == [f"apiKeyHelper is configured in {with_helper}"]
    assert "hidden" not in " ".join(problems)


def test_problems_name_variables_not_values() -> None:
    report = billing_guard.check(env={"ANTHROPIC_AUTH_TOKEN": "secret-value"}, settings_paths=[])
    assert report.problems() == ["environment variable ANTHROPIC_AUTH_TOKEN is set"]


def test_default_settings_paths_cover_user_and_project(tmp_path: Path) -> None:
    paths = billing_guard.default_settings_paths(tmp_path / "home", tmp_path / "proj")
    assert tmp_path / "home" / ".claude" / "settings.json" in paths
    assert tmp_path / "proj" / ".claude" / "settings.local.json" in paths


def test_check_defaults_to_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    report = billing_guard.check(settings_paths=[])
    assert "CLAUDE_CODE_USE_VERTEX" in report.env_vars
