"""Tests for halflife.kit.pin: rendering the rules into CLAUDE.md or AGENTS.md, idempotently."""

from __future__ import annotations

import io
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from halflife.kit import pin
from halflife.kit.compaction_prompt import COMPACT_HEADING
from halflife.kit.pin import (
    BEGIN,
    END,
    PinError,
    apply_block,
    main,
    render_block,
    render_rules,
    update_file,
)
from halflife.kit.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "halflife.yaml"
RULES = load_spec(EXAMPLE)


def _run(argv: list[str], env: dict[str, str] | None = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, stdout=out, stderr=err, env=env or {})
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- rendering


def test_render_rules_numbers_each_rule() -> None:
    lines = render_rules(RULES).splitlines()
    assert len(lines) == len(RULES)
    assert lines[0] == f"1. {RULES[0].text}"
    with_reasons = render_rules(RULES, with_reasons=True).splitlines()
    assert with_reasons[0] == f"1. {RULES[0].text} Reason: {RULES[0].reason}"


@pytest.mark.parametrize("fmt", ["claude", "agents"])
def test_render_block_has_markers_and_rules(fmt: str) -> None:
    block = render_block(RULES, fmt=fmt, source="team.yaml")
    lines = block.splitlines()
    assert lines[0] == BEGIN and lines[-1] == END and block.endswith(END + "\n")
    assert "Written by halflife-pin from team.yaml" in lines[1]
    assert "## Project rules" in block and "These rules come from team.yaml." in block
    assert render_rules(RULES) in block
    assert ("compacted" in block) == (fmt == "claude")
    assert COMPACT_HEADING not in block


def test_render_block_compact_instructions_for_claude_only() -> None:
    block = render_block(RULES, compact_instructions=True)
    assert COMPACT_HEADING in block and "Rules in force" in block
    assert block.index(COMPACT_HEADING) > block.index(render_rules(RULES))
    with pytest.raises(PinError, match=r"CLAUDE\.md only"):
        render_block(RULES, fmt="agents", compact_instructions=True)


def test_render_block_errors() -> None:
    with pytest.raises(PinError, match="unknown format"):
        render_block(RULES, fmt="cursor")
    with pytest.raises(PinError, match="no constraints"):
        render_block(())


def test_templates_ship_with_the_package() -> None:
    for fmt in pin.FORMATS:
        text = (pin.TEMPLATES_DIR / f"{fmt}.md").read_text(encoding="utf-8")
        assert "$rules" in text and "$source" in text


# --------------------------------------------------------------------------- applying


BLOCK = render_block(RULES)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", BLOCK),
        ("# Notes", "# Notes\n\n" + BLOCK),
        ("# Notes\n", "# Notes\n\n" + BLOCK),
        ("# Notes\n\n", "# Notes\n\n" + BLOCK),
    ],
)
def test_apply_block_appends(text: str, expected: str) -> None:
    assert apply_block(text, BLOCK) == expected


def test_apply_block_replaces_in_place_and_keeps_the_rest() -> None:
    before = "# Project\n\nIntro line.\n\n"
    after = "\n## Later section\n\nKeep me exactly.   \n"
    stale = f"{BEGIN}\nold rules\n{END}"
    updated = apply_block(before + stale + after, BLOCK)
    assert updated == before + BLOCK.rstrip("\n") + after
    assert apply_block(updated, BLOCK) == updated


def test_apply_block_is_idempotent() -> None:
    once = apply_block("# Title\nText", BLOCK)
    assert apply_block(once, BLOCK) == once
    assert once.count(BEGIN) == 1 and once.count(END) == 1


def test_apply_block_keeps_crlf_line_endings() -> None:
    text = "# Title\r\nLine\r\n"
    updated = apply_block(text, BLOCK)
    assert "\r\n" in updated and "\n" not in updated.replace("\r\n", "")
    assert apply_block(updated, BLOCK) == updated
    assert updated.startswith(text)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (f"{BEGIN}\nrules but no end\n", "without a matching"),
        (f"text\n{END}\n", "without a"),
        (f"{END}\n{BEGIN}\n", "without a"),
        (f"{BEGIN}\na\n{END}\n{BEGIN}\nb\n{END}\n", "more than one"),
        (f"{BEGIN}\na\n{END}\n{END}\n", "more than one"),
    ],
)
def test_apply_block_refuses_malformed_markers(text: str, message: str) -> None:
    with pytest.raises(PinError, match=message):
        apply_block(text, BLOCK)


# --------------------------------------------------------------------------- files


def test_update_file_creates_updates_and_stays_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "CLAUDE.md"
    assert update_file(target, BLOCK) is True
    assert target.read_text(encoding="utf-8") == BLOCK
    assert update_file(target, BLOCK) is False
    target.write_text("# Mine\n\n" + BLOCK + "\nAfter.\n", encoding="utf-8")
    assert update_file(target, BLOCK) is False
    other = render_block(RULES[:2])
    assert update_file(target, other, check=True) is True
    assert "After.\n" in target.read_text(encoding="utf-8")
    assert render_rules(RULES) in target.read_text(encoding="utf-8")
    assert update_file(target, other) is True
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# Mine\n\n") and text.endswith("\nAfter.\n")
    assert render_rules(RULES[:2]) in text and RULES[2].text not in text
    assert not list(tmp_path.glob("docs/.CLAUDE.md.*"))


def test_update_file_keeps_the_mode(tmp_path: Path) -> None:
    target = tmp_path / "CLAUDE.md"
    target.write_text("# x\n", encoding="utf-8")
    os.chmod(target, 0o640)
    update_file(target, BLOCK)
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_update_file_cleans_up_after_a_failed_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "CLAUDE.md"
    target.write_text("# x\n", encoding="utf-8")

    def fail(*_args: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pin.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        update_file(target, BLOCK)
    assert target.read_text(encoding="utf-8") == "# x\n"
    assert [p.name for p in tmp_path.iterdir()] == ["CLAUDE.md"]


def test_update_file_refuses_malformed_markers(tmp_path: Path) -> None:
    target = tmp_path / "CLAUDE.md"
    target.write_text(f"{BEGIN}\nunterminated\n", encoding="utf-8")
    with pytest.raises(PinError):
        update_file(target, BLOCK)
    assert target.read_text(encoding="utf-8") == f"{BEGIN}\nunterminated\n"


# --------------------------------------------------------------------------- command line


def test_cli_prints_the_block() -> None:
    code, out, err = _run(["--spec", str(EXAMPLE)])
    assert (code, err) == (0, "") and out == render_block(RULES)
    code, out, _err = _run(["--spec", str(EXAMPLE), "--with-reasons", "--format", "agents"])
    assert code == 0 and out == render_block(RULES, fmt="agents", with_reasons=True)


def test_cli_updates_a_file_in_place(tmp_path: Path) -> None:
    target = tmp_path / "CLAUDE.md"
    target.write_text("# My project\n\nOwn notes.\n", encoding="utf-8")
    code, out, _err = _run(["--spec", str(EXAMPLE), "--out", str(target)])
    assert code == 0 and out == f"halflife-pin: updated {target}\n"
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# My project\n\nOwn notes.\n\n" + BEGIN)
    code, out, _err = _run(["--spec", str(EXAMPLE), "--out", str(target)])
    assert code == 0 and out == f"halflife-pin: no change to {target}\n"
    assert target.read_text(encoding="utf-8") == text


def test_cli_check(tmp_path: Path) -> None:
    target = tmp_path / "CLAUDE.md"
    code, out, _err = _run(["--spec", str(EXAMPLE), "--out", str(target), "--check"])
    assert code == 1 and "is stale" in out and not target.exists()
    _run(["--spec", str(EXAMPLE), "--out", str(target)])
    code, out, _err = _run(["--spec", str(EXAMPLE), "--out", str(target), "--check"])
    assert code == 0 and "is up to date" in out


def test_cli_infers_agents_format(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    assert _run(["--spec", str(EXAMPLE), "--out", str(target)])[0] == 0
    assert target.read_text(encoding="utf-8") == render_block(RULES, fmt="agents")
    forced = tmp_path / "AGENTS.md"
    assert _run(["--spec", str(EXAMPLE), "--out", str(forced), "--format", "claude"])[0] == 0
    assert "compacted" in forced.read_text(encoding="utf-8")


def test_cli_reads_the_spec_from_the_environment(tmp_path: Path) -> None:
    code, out, _err = _run([], env={"HALFLIFE_SPEC": str(EXAMPLE)})
    assert code == 0 and out.startswith(BEGIN)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--check"], "--check needs --out"),
        (["--format", "cursor"], "invalid choice"),
        (["--spec", "missing.yaml"], "spec file not found"),
        (["--spec", "EXAMPLE", "--format", "agents", "--compact-instructions"], "CLAUDE.md only"),
    ],
)
def test_cli_errors(argv: list[str], message: str) -> None:
    argv = [str(EXAMPLE) if a == "EXAMPLE" else a for a in argv]
    code, out, err = _run(argv)
    assert (code, out) == (2, "") and err.startswith("halflife-pin: ") and message in err


def test_cli_without_a_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    code, _out, err = _run([])
    assert code == 2 and "no spec found" in err


def test_cli_reports_malformed_targets(tmp_path: Path) -> None:
    target = tmp_path / "CLAUDE.md"
    target.write_text(f"{END}\n", encoding="utf-8")
    code, _out, err = _run(["--spec", str(EXAMPLE), "--out", str(target)])
    assert code == 2 and "without a" in err


def test_pin_runs_as_a_module(tmp_path: Path) -> None:
    target = tmp_path / "CLAUDE.md"
    result = subprocess.run(
        [sys.executable, "-m", "halflife.kit.pin", "--spec", str(EXAMPLE), "--out", str(target)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8") == render_block(RULES)
