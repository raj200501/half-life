"""Overhead of the mitigation kit: guard latency and pin size (for artifacts/e4/overhead.json).

Nothing here writes an artifact; callers bank the returned records with their
provenance. Rule 15 asks for every mitigation's cost beside its benefit, and
these are the per-call and per-session costs of the two shipped tools:

* :func:`measure_guard_latency`: wall time (``time.perf_counter``) of
  :func:`halflife.kit.guard.decide` in process, and of the hook as Claude Code
  runs it: a fresh ``python -m halflife.kit.guard`` process per tool call, fed
  one hook-input fixture on stdin. The subprocess time includes interpreter
  start-up and imports, which dominate in practice.
* :func:`measure_pin_tokens`: size of the pinned CLAUDE.md or AGENTS.md block,
  which sits in the context for the whole session. Without a tokenizer the
  token count is an estimate, ``ceil(characters / 4)``, a rough rule for
  English text under common subword tokenizers; pass a tokenizer callable
  (returning a count or a token list) for an exact count.
* :func:`measure_pin_latency`: wall time to render the block and apply it to a
  memory file, paid once per spec change rather than per tool call.
* :func:`overhead_record`: the ``{"halflife-guard": {latency_ms, tokens},
  "halflife-pin": {latency_ms, tokens}}`` layout docs/ARTIFACTS.md requires.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from halflife.kit.guard import decide
from halflife.kit.pin import apply_block, render_block
from halflife.kit.spec import dump_spec
from halflife.schema import Constraint

__all__ = [
    "CHARS_PER_TOKEN",
    "PLACEHOLDER",
    "estimate_tokens",
    "latency_summary",
    "load_fixture",
    "measure_guard_latency",
    "measure_pin_latency",
    "measure_pin_tokens",
    "overhead_record",
]

PLACEHOLDER = "/workspace/project"
"""Project directory placeholder in the hook-input fixtures (tests/fixtures/claude_hooks/)."""
CHARS_PER_TOKEN = 4
"""Characters per token in the documented estimate."""
SAMPLE_MEMORY = "# Project notes\n\nBuild with make; run the tests with pytest.\n"
"""Stand-in for a user's existing CLAUDE.md in :func:`measure_pin_latency`."""

Tokenizer = Callable[[str], int | Sequence[object]]
FixtureSource = Mapping[str, Any] | str | os.PathLike[str]


def estimate_tokens(text: str, tokenizer: Tokenizer | None = None) -> tuple[int, str]:
    """Token count of ``text`` and how it was obtained (``chars/4`` or ``tokenizer``)."""
    if tokenizer is None:
        return math.ceil(len(text) / CHARS_PER_TOKEN), "chars/4"
    result = tokenizer(text)
    return (result if isinstance(result, int) else len(result)), "tokenizer"


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float | int]:
    """Count, median, nearest-rank 90th percentile, mean, minimum, and maximum, in ms."""
    if not samples_ms:
        raise ValueError("no latency samples")
    ordered = sorted(samples_ms)
    rank = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "n": len(ordered),
        "median_ms": round(statistics.median(ordered), 4),
        "p90_ms": round(ordered[rank], 4),
        "mean_ms": round(statistics.fmean(ordered), 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
    }


def load_fixture(fixture: FixtureSource, workspace: str | os.PathLike[str]) -> dict[str, Any]:
    """A hook-input fixture with :data:`PLACEHOLDER` replaced by ``workspace``.

    Args:
        fixture: The parsed hook input, or the path of a JSON file holding it.
        workspace: Project directory the fixture's paths should point into.

    Raises:
        ValueError: if the fixture is not a JSON object.
    """
    if isinstance(fixture, Mapping):
        text = json.dumps(fixture)
    else:
        text = Path(fixture).read_text(encoding="utf-8")
    data = json.loads(text.replace(PLACEHOLDER, os.fspath(workspace)))
    if not isinstance(data, dict):
        raise ValueError("a hook-input fixture must be a JSON object")
    return data


def _name(fixture: FixtureSource, index: int) -> str:
    if isinstance(fixture, Mapping):
        return f"fixture-{index}"
    return Path(fixture).stem


def measure_guard_latency(
    fixtures: Sequence[FixtureSource],
    constraints: Sequence[Constraint],
    repeats: int,
    *,
    workspace: str | os.PathLike[str] | None = None,
    subprocess_repeats: int | None = None,
    python: str | None = None,
    tokenizer: Tokenizer | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Time the guard on hook-input fixtures, in process and as a hook subprocess.

    Args:
        fixtures: Hook inputs (parsed, or JSON file paths) using :data:`PLACEHOLDER`.
        constraints: Rules in force; written to a temporary spec for the hook.
        repeats: Timed :func:`~halflife.kit.guard.decide` calls per fixture.
        workspace: Project directory the fixtures point into (default: an
            empty temporary directory).
        subprocess_repeats: Hook runs per fixture (default: ``repeats``).
        python: Interpreter for the hook (default: this one).
        tokenizer: Counts the tokens of each refusal reason (default: estimate).
        timeout_s: Time limit for one hook run.

    Returns:
        Latency summaries (``decide`` and ``hook_subprocess``), one row per
        fixture (decision, whether the hook agreed with ``decide``, tokens of
        the refusal reason Claude would see), the mean refusal-reason tokens,
        and the environment (Python version, platform, repeat counts).

    Raises:
        ValueError: if ``repeats`` or ``subprocess_repeats`` is below 1.
    """
    runs = repeats if subprocess_repeats is None else subprocess_repeats
    if repeats < 1 or runs < 1:
        raise ValueError("repeats must be at least 1")
    env = {k: v for k, v in os.environ.items() if k not in ("HALFLIFE_SPEC", "CLAUDE_PROJECT_DIR")}
    in_process: list[float] = []
    hook: list[float] = []
    rows: list[dict[str, Any]] = []
    method = estimate_tokens("", tokenizer)[1]
    with tempfile.TemporaryDirectory(prefix="halflife-overhead-") as scratch:
        root = os.fspath(workspace) if workspace is not None else scratch
        spec = Path(scratch) / "halflife.yaml"
        spec.write_text(dump_spec(constraints), encoding="utf-8")
        command = [python or sys.executable, "-m", "halflife.kit.guard", "--spec", str(spec)]
        for index, fixture in enumerate(fixtures):
            data = load_fixture(fixture, root)
            tool_name = str(data.get("tool_name") or "")
            raw_input = data.get("tool_input")
            tool_input: dict[str, Any] = raw_input if isinstance(raw_input, dict) else {}
            cwd = str(data.get("cwd") or root)
            decision = decide(tool_name, tool_input, constraints, cwd, root=root)  # warm-up
            for _ in range(repeats):
                start = time.perf_counter()
                decide(tool_name, tool_input, constraints, cwd, root=root)
                in_process.append((time.perf_counter() - start) * 1000.0)
            payload = json.dumps(data)
            outputs: set[tuple[int, str]] = set()
            for _ in range(runs):
                start = time.perf_counter()
                result = subprocess.run(
                    [*command, "--root", root],
                    input=payload,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=timeout_s,
                    check=False,
                )
                hook.append((time.perf_counter() - start) * 1000.0)
                outputs.add((result.returncode, result.stdout.strip()))
            reason = ""
            hook_denied = []
            for code, stdout in outputs:
                if code == 0 and stdout:
                    output = json.loads(stdout)["hookSpecificOutput"]
                    reason = str(output["permissionDecisionReason"])
                hook_denied.append(bool(stdout))
            rows.append(
                {
                    "fixture": _name(fixture, index),
                    "tool_name": tool_name,
                    "allow": decision.allow,
                    "constraint_id": decision.constraint_id,
                    "hook_matches_decide": all(d == (not decision.allow) for d in hook_denied),
                    "hook_exit_codes": sorted({code for code, _ in outputs}),
                    "reason_tokens": estimate_tokens(reason, tokenizer)[0] if reason else 0,
                }
            )
    denied = [row["reason_tokens"] for row in rows if not row["allow"]]
    return {
        "decide": latency_summary(in_process),
        "hook_subprocess": latency_summary(hook),
        "fixtures": rows,
        "deny_reason_tokens_mean": round(statistics.fmean(denied), 2) if denied else 0.0,
        "token_method": method,
        "repeats": repeats,
        "subprocess_repeats": runs,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def measure_pin_tokens(
    constraints: Sequence[Constraint],
    tokenizer: Tokenizer | None = None,
    *,
    fmt: str = "claude",
    with_reasons: bool = False,
    compact_instructions: bool = False,
) -> dict[str, Any]:
    """Characters and tokens of the pinned block (see the module docstring for the estimate)."""
    block = render_block(
        constraints, fmt=fmt, with_reasons=with_reasons, compact_instructions=compact_instructions
    )
    tokens, method = estimate_tokens(block, tokenizer)
    return {
        "format": fmt,
        "with_reasons": with_reasons,
        "compact_instructions": compact_instructions,
        "rules": len(constraints),
        "chars": len(block),
        "tokens": tokens,
        "token_method": method,
    }


def measure_pin_latency(
    constraints: Sequence[Constraint],
    repeats: int,
    *,
    fmt: str = "claude",
    base_text: str = SAMPLE_MEMORY,
) -> dict[str, float | int]:
    """Wall time to render the block and apply it to ``base_text`` (in memory), in ms."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        apply_block(base_text, render_block(constraints, fmt=fmt))
        samples.append((time.perf_counter() - start) * 1000.0)
    return latency_summary(samples)


def overhead_record(
    guard: Mapping[str, Any], pin_tokens: Mapping[str, Any], pin_latency: Mapping[str, Any]
) -> dict[str, Any]:
    """The artifacts/e4/overhead.json layout from the three measurements above.

    ``halflife-guard.latency_ms`` is the median hook run as a subprocess (paid
    on every guarded tool call) and ``tokens`` the mean refusal reason (added
    to the context only when a call is refused). ``halflife-pin.latency_ms`` is
    the median render-and-apply time (paid once per spec change) and
    ``tokens`` the block size (in the context for the whole session).
    """
    return {
        "halflife-guard": {
            "latency_ms": guard["hook_subprocess"]["median_ms"],
            "tokens": guard["deny_reason_tokens_mean"],
            "latency_definition": "median wall time of one hook run as a subprocess",
            "tokens_definition": "mean tokens of a refusal reason; allowed calls add none",
            "decide_median_ms": guard["decide"]["median_ms"],
            "token_method": guard["token_method"],
        },
        "halflife-pin": {
            "latency_ms": pin_latency["median_ms"],
            "tokens": pin_tokens["tokens"],
            "latency_definition": "median wall time to render and apply the block",
            "tokens_definition": "tokens of the pinned block, in context for the whole session",
            "chars": pin_tokens["chars"],
            "token_method": pin_tokens["token_method"],
        },
    }
