"""Tests for halflife.kit.reinject and halflife.kit.compaction_prompt (the compaction mitigations).

Hook inputs come from tests/fixtures/claude_hooks/, authored from the documented
format, not captured from Claude Code. The model in the compaction test is a
fake that records its prompt and returns fixed synthetic text.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from halflife.compaction import FULL_SUMMARY_PROMPT, compact
from halflife.kit.compaction_prompt import (
    COMPACT_HEADING,
    RULES_SECTION,
    claude_compact_instructions,
    compact_command,
    compact_instructions_section,
    preserve_prompt,
)
from halflife.kit.reinject import (
    REINJECT_INTRO,
    main,
    reinject_text,
    session_start_output,
)
from halflife.kit.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "halflife.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "claude_hooks"
RULES = load_spec(EXAMPLE)


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _run(argv: list[str], stdin: str, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, stdin=io.StringIO(stdin), stdout=out, stderr=err, env=env or {})
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- reinject_text


def test_reinject_text_lists_the_rules() -> None:
    text = reinject_text(["Never modify tests/.", "Do not use the network."])
    assert text.splitlines() == [
        REINJECT_INTRO,
        "1. Never modify tests/.",
        "2. Do not use the network.",
    ]
    assert reinject_text([]) == REINJECT_INTRO


def test_reinject_text_is_factual_not_a_system_command() -> None:
    assert "system" not in REINJECT_INTRO.lower()
    assert REINJECT_INTRO.endswith("still apply:")


def test_session_start_output_shape() -> None:
    assert session_start_output("x") == {
        "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "x"}
    }


# --------------------------------------------------------------------------- hook


def test_hook_adds_the_rules_after_compaction() -> None:
    code, out, err = _run(["--spec", str(EXAMPLE)], _fixture("session_start_compact.json"))
    assert (code, err) == (0, "")
    context = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert context == reinject_text([c.text for c in RULES])


def test_hook_with_reasons_and_plain_text() -> None:
    code, out, _err = _run(
        ["--spec", str(EXAMPLE), "--with-reasons", "--plain"],
        _fixture("session_start_compact.json"),
    )
    assert code == 0
    assert out == reinject_text([c.render("reason") for c in RULES]) + "\n"


@pytest.mark.parametrize(
    "name", ["session_start_startup.json", "pretooluse_bash_pytest.json", "post_compact_auto.json"]
)
def test_hook_is_silent_for_other_events_and_sources(name: str) -> None:
    assert _run(["--spec", str(EXAMPLE)], _fixture(name)) == (0, "", "")


def test_hook_any_source() -> None:
    code, out, _err = _run(
        ["--spec", str(EXAMPLE), "--any-source"], _fixture("session_start_startup.json")
    )
    assert code == 0 and "additionalContext" in out


def test_hook_without_stdin_prints_the_rules() -> None:
    code, out, _err = _run(["--spec", str(EXAMPLE), "--plain"], "")
    assert code == 0 and out.startswith(REINJECT_INTRO)


@pytest.mark.parametrize(
    ("stdin", "message"),
    [("not json", "hook input is not JSON"), ("[1]", "hook input is not a JSON object")],
)
def test_hook_reports_malformed_input(stdin: str, message: str) -> None:
    code, out, err = _run(["--spec", str(EXAMPLE)], stdin)
    assert (code, out) == (1, "") and err == f"halflife-reinject: {message}\n"


def test_hook_spec_problems(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stdin = _fixture("session_start_compact.json")
    code, out, err = _run(["--spec", str(tmp_path / "missing.yaml")], stdin)
    assert (code, out) == (1, "") and "spec file not found" in err
    code, out, _err = _run([], stdin, env={"CLAUDE_PROJECT_DIR": str(EXAMPLE.parent)})
    assert code == 0 and "additionalContext" in out
    monkeypatch.chdir(tmp_path)
    code, out, err = _run([], stdin)
    assert (code, out) == (0, "") and "nothing to add" in err


def test_hook_usage_error() -> None:
    code, out, err = _run(["--bogus"], "")
    assert (code, out) == (1, "") and err.startswith("halflife-reinject: ")


def test_hook_runs_as_a_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "halflife.kit.reinject", "--spec", str(EXAMPLE)],
        input=_fixture("session_start_compact.json"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"


# --------------------------------------------------------------------------- compaction prompt


def test_preserve_prompt_extends_the_default_prompt() -> None:
    prompt = preserve_prompt()
    first_sentence = FULL_SUMMARY_PROMPT.split(". ")[0]
    assert prompt.startswith(first_sentence)
    assert f'"{RULES_SECTION}"' in prompt and "word for word" in prompt
    assert prompt.index(RULES_SECTION) < prompt.index("the files changed")
    assert "the requests so far and their status, the current request" in prompt
    assert prompt != FULL_SUMMARY_PROMPT


def test_preserve_prompt_names_no_rule() -> None:
    prompt = preserve_prompt().lower()
    for constraint in RULES:
        assert constraint.text.lower() not in prompt


def test_claude_compact_instructions() -> None:
    text = claude_compact_instructions()
    assert RULES_SECTION in text and "word for word" in text
    section = compact_instructions_section()
    assert section.startswith(COMPACT_HEADING + "\n\n") and section.endswith(text + "\n")
    assert compact_command() == f"/compact {text}"


class _RecordingModel:
    """A fake model: records the prompts it is sent and returns synthetic text."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        json_schema: Mapping[str, Any] | None = None,
        purpose: str = "step",
    ) -> str:
        self.prompts.append(messages[-1]["content"])
        return f"{RULES_SECTION}:\n- Never modify files under tests/.\nWork so far: synthetic."

    def count_tokens(self, messages: Sequence[Mapping[str, str]]) -> int:
        return sum(len(m["content"]) // 4 for m in messages)


def test_preserve_prompt_drives_the_minimal_agent_compaction() -> None:
    model = _RecordingModel()
    rule = "Never modify files under tests/."
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": f"Rules:\n1. {rule}\n\nFirst request: fix the bug."},
        {"role": "assistant", "content": '{"tool": "done", "args": {"summary": "ok"}}'},
    ]
    after, event = compact(
        messages,
        strategy="full_summary",
        model=model,
        event_id="c1",
        tool_call_index=3,
        scheduled=True,
        rendered_constraints={"tests-frozen": rule},
        rules_message_index=1,
        continuation=None,
        seed=1,
        summary_prompt=preserve_prompt(),
    )
    assert model.prompts[0].startswith(preserve_prompt())
    assert event.prompt_variant == "preserve"
    assert event.constraint_text_retained == {"tests-frozen": True}
    assert rule in after[-1]["content"]
