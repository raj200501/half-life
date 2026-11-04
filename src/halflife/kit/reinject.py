"""Re-inject the rules after context compaction (the ``reinject`` mitigation).

:func:`reinject_text` is the message the minimal agent's ``reinject`` arm
appends right after a compaction, and the text the Claude Code hook adds.

For Claude Code, the documented place to re-add context after compaction is a
SessionStart hook with the ``compact`` matcher: it runs after auto or manual
compaction with ``source`` set to ``"compact"``, and Claude Code adds its
output to the compacted context, either plain stdout or
``hookSpecificOutput.additionalContext`` (https://code.claude.com/docs/en/hooks
and https://code.claude.com/docs/en/context-window, retrieved 2026-11-04).
PostCompact is not used: it has no decision control and cannot affect the
compacted context. The reference also advises writing such context as factual
statements rather than imperative system instructions, so the text states that
the session's rules still apply and lists them.

Run the hook as ``python -m halflife.kit.reinject`` (see docs/MITIGATIONS.md).
It reads the hook JSON on stdin and prints::

    {"hookSpecificOutput": {"hookEventName": "SessionStart",
                            "additionalContext": "<the rules>"}}

or the bare text with ``--plain``. It prints nothing for other events or other
SessionStart sources, unless ``--any-source`` is given. Errors print one
``halflife-reinject: ...`` line on stderr and exit 1, which Claude Code treats
as a non-blocking error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn, TextIO

from halflife.kit.spec import SpecError, find_spec, load_spec
from halflife.schema import Phrasing

__all__ = ["REINJECT_INTRO", "main", "reinject_text", "session_start_output"]

REINJECT_INTRO = (
    "Context compaction replaced the earlier conversation with a summary. "
    "The project rules stated at the start of the session still apply:"
)
"""First line of the re-injected message."""

EXIT_OK = 0
EXIT_ERROR = 1


def reinject_text(rules: Sequence[str]) -> str:
    """The re-injection message: :data:`REINJECT_INTRO` and the rules as a numbered list.

    Args:
        rules: Rule texts as the agent was shown them (with reasons under the
            ``reason`` phrasing).
    """
    lines = [f"{number}. {rule}" for number, rule in enumerate(rules, start=1)]
    return "\n".join([REINJECT_INTRO, *lines])


def session_start_output(text: str) -> dict[str, Any]:
    """SessionStart hook output that adds ``text`` to Claude's context."""
    return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m halflife.kit.reinject",
        description=(
            "Claude Code SessionStart hook (matcher: compact): adds the rules of a "
            "halflife.yaml spec to the context after compaction."
        ),
    )
    parser.add_argument(
        "--spec",
        help="spec file (default: $HALFLIFE_SPEC, ./halflife.yaml, or "
        "$CLAUDE_PROJECT_DIR/halflife.yaml)",
    )
    parser.add_argument("--with-reasons", action="store_true", help="append each rule's reason")
    parser.add_argument("--plain", action="store_true", help="print plain text instead of JSON")
    parser.add_argument(
        "--any-source",
        action="store_true",
        help="also print for SessionStart sources other than compact",
    )
    return parser


def _wanted(raw: str, any_source: bool) -> bool | str:
    """Whether this hook input calls for the rules; a string explains malformed input."""
    if not raw.strip():
        return True
    try:
        data = json.loads(raw)
    except ValueError:
        return "hook input is not JSON"
    if not isinstance(data, dict):
        return "hook input is not a JSON object"
    if data.get("hook_event_name", "SessionStart") != "SessionStart":
        return False
    return any_source or data.get("source", "compact") == "compact"


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run the hook.

    Returns:
        0 after printing the rules or deciding they are not needed; 1 on an
        error (Claude Code treats it as non-blocking).
    """
    inp = sys.stdin if stdin is None else stdin
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        args = _parser().parse_args(argv)
        wanted = _wanted(inp.read(), args.any_source)
        if isinstance(wanted, str):
            raise _UsageError(wanted)
        if not wanted:
            return EXIT_OK
        spec_path = find_spec(args.spec, env=env)
        if spec_path is None:
            err.write("halflife-reinject: no halflife.yaml found; nothing to add\n")
            return EXIT_OK
        constraints = load_spec(spec_path)
    except (_UsageError, SpecError) as exc:
        err.write(f"halflife-reinject: {exc}\n")
        return EXIT_ERROR
    phrasing: Phrasing = "reason" if args.with_reasons else "bare"
    text = reinject_text([c.render(phrasing) for c in constraints])
    if args.plain:
        out.write(text + "\n")
    else:
        out.write(json.dumps(session_start_output(text)) + "\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
