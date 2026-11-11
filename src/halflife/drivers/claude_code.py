"""Headless Claude Code driver for E2, the ecological arm (sections 5 and 8 of the brief).

One session plan is one Claude Code conversation in a disposable workspace
outside the repository. The first subtask message starts it with
``claude -p --session-id <uuid>``; every later subtask message continues it with
``--resume <uuid>``, one launch per message. The prompt goes on stdin, never on
the command line.

Every launch is a clean child (section 8): ``--restricted --strict-mcp-config``,
an explicit ``--tools`` list with matching ``--allowedTools``,
``--permission-mode dontAsk``, ``--output-format stream-json --verbose
--include-hook-events``, ``--model <tier>``, ``--autocompact 100000`` (the
smallest auto-compaction window), and ``--settings`` with Claude Code's Bash
sandbox (``failIfUnavailable``, no unsandboxed commands, network blocked) plus
the :mod:`halflife.drivers.snapshot_hook` hook. It never uses ``--bare``,
``--dangerously-skip-permissions``, or ``bypassPermissions``. The child
environment is an allowlist: tokens, API keys, cloud credentials,
``SSH_AUTH_SOCK``, and the parent-session markers ``CLAUDECODE`` and
``CLAUDE_CODE_ENTRYPOINT`` never reach it, and git reads no system or global
configuration.

Before any launch, :func:`halflife.billing_guard.check` must be clean; if it is
not, :class:`~halflife.client.BillingGuardRefused` names the variables and
nothing else happens. ``claude --help`` does not list ``--max-turns`` (the CLI
reference documents it for print mode), so the driver does not rely on it and
counts assistant turns per subtask message in the stream and stops the child
(SIGINT, then SIGTERM, then SIGKILL as a last resort) at
``claude_agent_max_turns``, after the last turn's tool results arrive. The
per-subtask and per-session tool-call caps of the plan stop it the same way.
``total_cost_usd`` is logged as ``plan_equivalent_usd`` and never as cash. A
usage limit, a billing or authentication error, or any sign that the account
could draw on paid extra usage (overage) stops the run before the next launch.
Auto memory is switched off in the child, so no rule carrier exists besides
the placement under test and nothing is written to the user's memory folders.

The system prompt is recorded on the conversation's first request
(``--system-prompt-snapshot on``), so every launch passes identical
``--append-system-prompt`` text (only under ``system`` placement) and rules
reach the agent otherwise as user-message text (``first_user``, ``reminder``) or
as a real ``CLAUDE.md`` committed into the workspace (``pinned_file``). The
prompt actually sent is copied from the transcript's ``prompt_snapshot`` entries
into ``system_prompt.json``.

Session and opportunity rows carry the model tier (for example ``haiku``) as
``model`` and the exact model id the CLI reported as ``model_id``; every row is
labeled ``ecological``. Instruction files loaded from outside the workspace
(such as a user-level ``CLAUDE.md``) are known contaminants: the smoke run
reports them, and ``run`` refuses until they are acknowledged with
``--accept-contaminants`` (after a DECISIONS.md entry records them, section 8 of
the brief), then records them in every session's provenance.

At each decision point, a recall probe forks the conversation
(``--resume <uuid> --fork-session --tools ""``), asks
:data:`halflife.probe.PROBE_QUESTION`, and scores the answer with
:func:`halflife.probe.recalled_in`; the main transcript is hashed before and
after to show that the fork left it untouched.

Outputs go to ``runs/<run_id>/<session_id>/`` in the layout of
:func:`halflife.runner.save_session` (``plan.json``, ``action_log.json``,
``probes.json``, ``violations.json``, ``result.json``, ``compactions/``, and
``usage.json`` as the minimal-agent runner writes it), plus ``stream.jsonl``
(every stream event), ``commands.jsonl`` (exact command lines), ``hooks.jsonl``,
``system_prompt.json``, and the settings files, so
:func:`halflife.experiment.bank_runs` banks them unchanged. A session with
``result.json`` is skipped on a rerun.

Command line (run on a machine with a clean billing guard; see docs/E2_RUNBOOK.md)::

    python -m halflife.drivers.claude_code smoke [--model haiku]
    python -m halflife.drivers.claude_code run --plans <plans.jsonl> --run-id <id> --model <tier>

Only one driver process runs at a time (a lock under the output root), so two
runs can never race each other past the daily caps.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, Any, Protocol

from halflife import billing_guard
from halflife.agent import FIRST_REQUEST, REMINDER_INTRO, RULES_INTRO, SYSTEM_RULES_INTRO, numbered
from halflife.analysis import RunFacts, opportunity_rows, session_row
from halflife.budget import cap_share, effective_caps, file_cap_share, load_caps, read_ledger
from halflife.client import NO_MODEL_CALLS_ENV, BillingGuardRefused, CapExceeded, env_flag
from halflife.client import append_jsonl as _append_jsonl
from halflife.detect import detect_all
from halflife.drivers import snapshot_hook
from halflife.probe import PROBE_QUESTION, ProbeResult, recalled_in
from halflife.sandbox import (
    DEFAULT_OUTPUT_LIMIT,
    SUBJECT_BIN_DEFAULT,
    SandboxOptions,
    is_secret_name,
    shims_dir,
    subject_bin,
    truncate_output,
)
from halflife.schema import (
    ActionLog,
    HarnessCheck,
    NetAttempt,
    SessionPlan,
    Subtask,
    SubtaskSpan,
    Template,
    ToolEvent,
    ToolName,
)
from halflife.scripted import BASELINE_SUBTASK, DEFAULT_CHECK_TIMEOUT_S, run_check
from halflife.secrets_scan import env_secret_values
from halflife.workspace import GIT_IDENTITY, Workspace, diff_snapshots, remove_tree

# --------------------------------------------------------------------------- constants

DRIVER_CONTRACT = "claude-code-driver/1"
"""Version of the flags, settings, and output layout; a smoke result is valid only for it."""

CLAUDE_TOOLS: tuple[str, ...] = ("Read", "Write", "Edit", "Bash", "Glob", "Grep")
"""The only tools a subject child gets (``--tools`` and ``--allowedTools``)."""

AUTOCOMPACT_TOKENS = 100_000
"""Smallest auto-compaction window Claude Code accepts (``--autocompact``)."""

PERMISSION_MODE = "dontAsk"
HOOK_TIMEOUT_S = 60
SANDBOX_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "failIfUnavailable": True,
    "allowUnsandboxedCommands": False,
}
"""Claude Code's Bash sandbox: required, no escape hatch, no network domains allowed."""

COMPACTION_STRATEGY = "claude_code_auto"
"""Label for Claude Code's own auto-compaction in rows and compaction logs."""

FAMILY = "claude"
LABEL = "ecological"
LEDGER_BACKEND = "claude-code"
SYNTHETIC_MODEL = "<synthetic>"
PINNED_FILE = "CLAUDE.md"
NETLOG_NAME = "halflife-netlog.jsonl"
LOCK_NAME = ".claude-code-driver.lock"
"""Lock file under the output root, held while a smoke or run command is active."""
SYSTEM_PROMPT_FILE = "system_prompt.json"
"""Per-session copy of the system prompt the CLI recorded (``prompt_snapshot`` entries)."""

REQUIRED_HELP_FLAGS: tuple[str, ...] = (
    "--print",
    "--output-format",
    "--verbose",
    "--include-hook-events",
    "--model",
    "--restricted",
    "--strict-mcp-config",
    "--tools",
    "--allowedTools",
    "--permission-mode",
    "dontAsk",
    "--settings",
    "--session-id",
    "--resume",
    "--fork-session",
    "--append-system-prompt",
    "--autocompact",
    "--system-prompt-snapshot",
)
"""Everything the driver passes must appear in ``claude --help``, or it refuses to launch."""

REQUIRED_ARGS: tuple[str, ...] = (
    "-p",
    "--restricted",
    "--strict-mcp-config",
    "--tools",
    "--permission-mode",
    "--output-format",
    "--verbose",
    "--model",
    "--settings",
)
FORBIDDEN_ARGS: frozenset[str] = frozenset(
    {
        "--bare",
        "--dangerously-skip-permissions",
        "--allow-dangerously-skip-permissions",
        "bypassPermissions",
        "--max-budget-usd",
        "--mcp-config",
        "--plugin-dir",
        "--plugin-url",
        "--add-dir",
        "--agents",
        "--agent",
        "--setting-sources",
        "--continue",
        "-c",
        "--chrome",
        "--ide",
        "--remote-control",
        "--cloud",
        "--bg",
        "--background",
        "--worktree",
        "--teleport",
        "--from-pr",
        "--file",
    }
)
"""Flags no child may get: auth bypasses, permission bypasses, and extra configuration."""

CHILD_ENV_KEEP: frozenset[str] = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "TERM",
        "TZ",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
        "XDG_RUNTIME_DIR",
    }
)
"""Host variables a child keeps (plus non-secret ``LC_*``); everything else is dropped."""

CHILD_ENV_CONFIG: frozenset[str] = frozenset({"CLAUDE_CONFIG_DIR"})
"""Kept so the child finds the Max-plan login; a path, never a credential."""

PARENT_SESSION_MARKERS: tuple[str, ...] = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")

ACCOUNT_ERRORS: dict[str, str] = {
    "billing_error": "billing_error",
    "rate_limit": "usage_limit",
    "authentication_failed": "auth_error",
    "oauth_org_not_allowed": "auth_error",
    "account_on_hold": "auth_error",
    "verification_required": "auth_error",
    "cloud_credential_error": "auth_error",
}
"""Assistant-message ``error`` values after which no more children are launched."""

MODEL_ERRORS: frozenset[str] = frozenset({"max_output_tokens"})
"""Assistant-message ``error`` values that describe the reply, not the service (warnings).

Every other ``error`` value (``overloaded``, ``server_error``, ``invalid_request``,
``model_not_found``, ``unknown``, or one this list does not know) is an API
error that ends the session, because the subtask was not really attempted."""

ACCOUNT_STOPS: frozenset[str] = frozenset({"usage_limit", "billing_error", "auth_error", "overage"})
"""Stop reasons that end the run (exit status :data:`EXIT_LIMIT`)."""

OVERAGE_STATUSES: frozenset[str] = frozenset({"allowed", "allowed_warning"})
"""``overageStatus`` values meaning paid extra usage would cover calls past the plan limit."""

BILLING_TEXT = re.compile(
    r"credit balance|billing|payment|insufficient (?:credit|funds)|spend limit", re.I
)
USAGE_TEXT = re.compile(
    r"usage limit|hit your (?:\w+ )?limit|limit reached|rate.?limit|out of (?:extra )?usage",
    re.I,
)
AUTH_TEXT = re.compile(
    r"not logged in|run /login|invalid api key|authentication|unauthori[sz]ed|oauth token", re.I
)
EXIT_CODE = re.compile(r"Exit code (\d+)")
COMPACT_SUMMARY_PREFIX = "This session is being continued from a previous conversation"
_TIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\[\]-]{0,63}$")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2
EXIT_LIMIT = 3
EXIT_SAFETY = 4

SMOKE_MARKER = "halflife-smoke-ok"
SMOKE_END = "halflife-smoke-end"
SMOKE_NET_TARGET = "http://example.invalid/"
SMOKE_CLAUDE_MD = (
    "# Harness self-test\n\n"
    "This disposable workspace checks the half-life harness. There are no project rules.\n"
)


class DriverError(RuntimeError):
    """Base class for refusals and stops of the Claude Code driver."""


class PreflightError(DriverError):
    """The CLI is missing, fails, or lacks a flag the driver needs."""


class UnsupportedPlan(DriverError):
    """A plan asks for a condition this driver cannot apply faithfully."""


class SafetyStop(DriverError):
    """A child was not clean or not instrumented; nothing it produced is banked."""


# --------------------------------------------------------------------------- small helpers


def utc_now() -> dt.datetime:
    """Current UTC time."""
    return dt.datetime.now(dt.UTC)


def utc_stamp() -> str:
    """Current UTC time, seconds precision."""
    return utc_now().isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    """SHA-256 of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path | None) -> str | None:
    """SHA-256 of a file's bytes, or None if it cannot be read."""
    if path is None:
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _epoch_iso(value: object) -> str | None:
    seconds = _as_int(value)
    if seconds is None:
        return None
    return dt.datetime.fromtimestamp(seconds, dt.UTC).isoformat(timespec="seconds")


def secret_values(env: Mapping[str, str]) -> list[str]:
    """Values of secret-looking variables, longest first, for redaction."""
    return sorted(env_secret_values(env).values(), key=len, reverse=True)


def classify_text(text: str) -> str | None:
    """Account problem named by an error text: billing, usage limit, auth, or None."""
    if BILLING_TEXT.search(text):
        return "billing_error"
    if USAGE_TEXT.search(text):
        return "usage_limit"
    if AUTH_TEXT.search(text):
        return "auth_error"
    return None


def overage_signals(info: Mapping[str, Any]) -> list[str]:
    """Fields of a ``rate_limit_info`` showing that paid extra usage is on or in use.

    Any of them stops the run: past the plan limit such an account is billed,
    and the $0 rule (section 8 of the brief) allows no billed call.
    """
    found = [key for key in ("isUsingOverage", "overageInUse", "overageEnabled") if info.get(key)]
    if info.get("rateLimitType") == "overage":
        found.append("rateLimitType=overage")
    if info.get("overageStatus") in OVERAGE_STATUSES:
        found.append(f"overageStatus={info['overageStatus']}")
    return found


def read_transcript(path: Path | None) -> list[dict[str, Any]]:
    """Every JSON object in a transcript file, in order (missing or unreadable: none)."""
    if path is None:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [entry for entry in (parse_line(line) for line in lines) if entry is not None]


def prompt_snapshots(entries: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """System prompts the CLI recorded (``prompt_snapshot`` entries), in transcript order."""
    found = []
    for entry in entries:
        blocks = entry.get("systemPrompt")
        if entry.get("type") == "prompt_snapshot" and isinstance(blocks, list):
            found.append([str(block) for block in blocks])
    return found


def strings_in(value: object) -> Iterator[str]:
    """Every string inside JSON-like data (decoded, so escaping cannot hide a match)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from strings_in(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from strings_in(item)


def transcript_leaks(entries: Sequence[Mapping[str, Any]], answers: Sequence[str]) -> list[str]:
    """Probe material in a main transcript, as :func:`halflife.probe.leaked` checks contexts.

    Looks for the probe question and for any probe answer longer than 20
    characters in the decoded strings of every transcript entry.
    """
    texts = list(strings_in(list(entries)))
    found = []
    if any(PROBE_QUESTION in text for text in texts):
        found.append("the probe question")
    for answer in answers:
        if len(answer) > 20 and any(answer in text for text in texts):
            found.append("a probe answer")
            break
    return found


def compact_summaries(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """Texts of the compaction summary messages (``isCompactSummary``), in transcript order."""
    found = []
    for entry in entries:
        if entry.get("isCompactSummary") is True:
            found.append(result_text(_as_dict(entry.get("message")).get("content")))
    return found


def result_text(content: object) -> str:
    """Text of a ``tool_result`` content field (a string or a list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, Mapping):
                parts.append(f"[{block.get('type', 'block')}]")
        return "\n".join(parts)
    return "" if content is None else str(content)


def unfence(answer: str) -> str:
    """The JSON object inside a probe answer wrapped in a code fence or prose, if any."""
    text = answer.strip()
    with suppress(ValueError):
        json.loads(text)
        return text
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidate = text[start : end + 1]
        with suppress(ValueError):
            if isinstance(json.loads(candidate), dict):
                return candidate
    return text


def canonical(value: object) -> str:
    """Canonical JSON text, for comparing tool inputs."""
    return json.dumps(value, sort_keys=True, default=str)


# --------------------------------------------------------------------------- prompts and placements


def rules_text(plan: SessionPlan, template: Template) -> str:
    """The numbered rules as rendered under the plan's phrasing."""
    return numbered([c.render(plan.phrasing) for c in template.constraints])


def append_system_prompt(placement: str, rules: str) -> str | None:
    """Text passed with ``--append-system-prompt`` on every launch (``system`` only)."""
    return f"{SYSTEM_RULES_INTRO}\n{rules}" if placement == "system" else None


def pinned_file_text(rules: str) -> str:
    """Content of the workspace ``CLAUDE.md`` under ``pinned_file`` placement."""
    return f"# Project rules\n\n{SYSTEM_RULES_INTRO}\n{rules}\n"


def first_message(placement: str, rules: str, prompt: str) -> str:
    """The first subtask message; rules open it under ``first_user`` and ``reminder``."""
    if placement in {"first_user", "reminder"}:
        return f"{RULES_INTRO}\n{rules}\n\n{FIRST_REQUEST} {prompt}"
    return prompt


def reminder_text(rules: str) -> str:
    """Rules appended to a subtask message under ``reminder`` placement."""
    return f"{REMINDER_INTRO}\n{rules}"


def smoke_prompt(probe_file: Path) -> str:
    """The smoke session's only request: one shell command with known effects."""
    target = shlex.quote(str(probe_file))
    command = (
        f"python -c \"print('{SMOKE_MARKER}')\"; command -v python curl; "
        f"printenv GIT_CONFIG_GLOBAL; curl -sS {SMOKE_NET_TARGET}; "
        f"touch {target}; echo {SMOKE_END}"
    )
    return (
        "This is a short self-test of the harness that runs you. Use the Bash tool once to run "
        "exactly this command, without changing it:\n\n"
        f"{command}\n\n"
        "Some parts are expected to fail (there is no network, and writes outside this "
        "workspace are blocked); that is fine, so do not retry or work around them. Then reply "
        "with one short line saying the command ran."
    )


# --------------------------------------------------------------------------- command lines


def hook_command(python: str, out: Path, root: Path, netlog: Path | None) -> str:
    """Shell command Claude Code runs for the snapshot hook (``-I``: isolated mode)."""
    parts = [python, "-I", "-m", "halflife.drivers.snapshot_hook"]
    parts += ["--out", str(out), "--root", str(root)]
    if netlog is not None:
        parts += ["--netlog", str(netlog)]
    return shlex.join(parts)


def settings_for(hook: str | None) -> dict[str, Any]:
    """The ``--settings`` JSON: the sandbox, and the snapshot hook when ``hook`` is given.

    The hook runs after every tool call (``PostToolUse``, ``PostToolUseFailure``),
    when an instruction file loads (``InstructionsLoaded``), and at the start of
    every launch (``SessionStart``, which reports the transcript path). It prints
    nothing, so it adds nothing to the agent's context.
    """
    data: dict[str, Any] = {"sandbox": dict(SANDBOX_SETTINGS)}
    if hook is not None:
        entry = {"type": "command", "command": hook, "timeout": HOOK_TIMEOUT_S}
        data["hooks"] = {
            "PostToolUse": [{"matcher": "*", "hooks": [dict(entry)]}],
            "PostToolUseFailure": [{"matcher": "*", "hooks": [dict(entry)]}],
            "InstructionsLoaded": [{"hooks": [dict(entry)]}],
            "SessionStart": [{"hooks": [dict(entry)]}],
        }
    return data


def _common_args(model: str, settings_path: Path, append: str | None) -> list[str]:
    args = [
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-hook-events",
        "--model",
        model,
        "--restricted",
        "--strict-mcp-config",
        "--permission-mode",
        PERMISSION_MODE,
        "--settings",
        str(settings_path),
        "--autocompact",
        str(AUTOCOMPACT_TOKENS),
        "--system-prompt-snapshot",
        "on",
    ]
    if append is not None:
        args += ["--append-system-prompt", append]
    return args


def main_argv(
    claude_bin: str,
    *,
    model: str,
    settings_path: Path,
    append: str | None,
    session_uuid: str,
    first: bool,
) -> list[str]:
    """Command line of a subtask launch (the prompt goes on stdin).

    The variadic ``--tools`` and ``--allowedTools`` come last and nothing
    positional follows them, so no argument can be swallowed as a tool name.
    """
    argv = [claude_bin, *_common_args(model, settings_path, append)]
    argv += ["--session-id", session_uuid] if first else ["--resume", session_uuid]
    tools = ",".join(CLAUDE_TOOLS)
    return [*argv, "--tools", tools, "--allowedTools", tools]


def probe_argv(
    claude_bin: str, *, model: str, settings_path: Path, append: str | None, session_uuid: str
) -> list[str]:
    """Command line of a recall probe: a tool-less fork of the conversation."""
    argv = [claude_bin, *_common_args(model, settings_path, append)]
    return [*argv, "--resume", session_uuid, "--fork-session", "--tools", ""]


def check_argv(argv: Sequence[str], *, probe: bool = False) -> None:
    """Refuse a command line that is not a clean child (a defensive invariant).

    Raises:
        SafetyStop: if a required flag is missing, a forbidden one is present,
            or the permission mode, output format, or tool list is wrong.
    """
    present = set(argv)
    missing = [a for a in REQUIRED_ARGS if a not in present]
    forbidden = sorted(present & FORBIDDEN_ARGS)
    problems = []
    if missing:
        problems.append(f"missing {missing}")
    if forbidden:
        problems.append(f"forbidden {forbidden}")

    def value(flag: str) -> str | None:
        positions = [i for i, a in enumerate(argv) if a == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(argv):
            return None
        return argv[positions[0] + 1]

    if value("--permission-mode") != PERMISSION_MODE:
        problems.append("permission mode is not dontAsk")
    if value("--output-format") != "stream-json":
        problems.append("output format is not stream-json")
    tools = value("--tools")
    if probe:
        if tools != "" or "--fork-session" not in present:
            problems.append("a probe must fork the session and have no tools")
    elif tools != ",".join(CLAUDE_TOOLS) or value("--allowedTools") != tools:
        problems.append("tools and allowedTools must both be the explicit tool list")
    if problems:
        raise SafetyStop("unsafe child command line: " + "; ".join(problems))


def check_settings(path: Path, expected: Mapping[str, Any]) -> None:
    """Refuse to launch unless the ``--settings`` file is exactly what the driver wrote.

    Raises:
        SafetyStop: if the file is missing, unreadable, or changed, or lacks the sandbox.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SafetyStop(f"settings file {path.name} cannot be read: {exc}") from exc
    if not isinstance(data, dict) or data != dict(expected):
        raise SafetyStop(f"settings file {path.name} changed since the driver wrote it")
    if data.get("sandbox") != SANDBOX_SETTINGS:
        raise SafetyStop(f"settings file {path.name} does not hold the required sandbox")


def child_env(base: Mapping[str, str], *, netlog: Path) -> dict[str, str]:
    """The environment of a Claude child: an allowlist plus fixed settings.

    Keeps only :data:`CHILD_ENV_KEEP`, non-secret ``LC_*``, and
    ``CLAUDE_CONFIG_DIR``. Drops every token, API key, cloud credential,
    ``SSH_AUTH_SOCK``, proxy, ``ANTHROPIC_*`` and ``CLAUDE*`` variable
    (including the parent-session markers). Sets ``PATH`` (sandbox shims, then
    the subject Python, then the host ``PATH``), git isolation
    (``GIT_CONFIG_NOSYSTEM=1``, ``GIT_CONFIG_GLOBAL=/dev/null``, a fixed
    identity), the Python network logger, and switches that stop the CLI's
    auto-updater, non-essential traffic, and auto memory (which would be an
    extra carrier of the rules and would write into the user's memory folders).

    Raises:
        SafetyStop: if a billing-guard variable or a parent-session marker survived.
    """
    env = {
        key: value
        for key, value in base.items()
        if (key in CHILD_ENV_KEEP or key.startswith("LC_")) and not is_secret_name(key)
    }
    env.update({key: base[key] for key in CHILD_ENV_CONFIG if key in base})
    shims = shims_dir()
    entries = [str(shims), str(subject_bin(base))]
    for entry in base.get("PATH", "/usr/local/bin:/usr/bin:/bin").split(os.pathsep):
        if entry and entry not in entries:
            entries.append(entry)
    env.update(
        {
            "PATH": os.pathsep.join(entries),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONPATH": str(shims),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "HALFLIFE_NETLOG": str(netlog),
            "HALFLIFE_SANDBOX": "claude-code",
            "HALFLIFE_NO_MODEL_CALLS": "1",
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        }
    )
    env.update(GIT_IDENTITY)
    leaked = list(billing_guard.forbidden_env_vars(env))
    leaked += [m for m in PARENT_SESSION_MARKERS if m in env]
    if leaked:
        raise SafetyStop(f"child environment still has {sorted(leaked)}")
    return env


# --------------------------------------------------------------------------- preflight


@dataclass(frozen=True)
class CliInfo:
    """What ``claude --version`` and ``claude --help`` report (no model call)."""

    path: str
    version: str
    help_sha256: str
    has_max_turns: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (the home directory in the path is shown as ``~``)."""
        return dataclasses.asdict(self) | {"path": snapshot_hook.display_path(self.path)}


def parse_version(text: str) -> str:
    """The ``X.Y.Z`` version in ``claude --version`` output.

    Raises:
        PreflightError: if there is none.
    """
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", text)
    if match is None:
        raise PreflightError(f"cannot read a version from {text.strip()[:80]!r}")
    return match.group(1)


def missing_flags(help_text: str) -> list[str]:
    """Flags in :data:`REQUIRED_HELP_FLAGS` that ``claude --help`` does not list."""
    return [
        flag
        for flag in REQUIRED_HELP_FLAGS
        if not re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text)
    ]


def require_clean_guard(guard: Callable[[], billing_guard.GuardReport]) -> None:
    """Raise :class:`BillingGuardRefused` naming the problems if the guard is not clean."""
    report = guard()
    if not report.clean:
        raise BillingGuardRefused("Claude children are refused: " + "; ".join(report.problems()))


# --------------------------------------------------------------------------- launchers


class Child(Protocol):
    """A running child process whose stdout is read line by line."""

    def read_line(self, timeout: float) -> str | None:
        """The next stdout line, ``""`` at end of output, or None after ``timeout``."""
        ...

    def send(self, sig: int, *, group: bool) -> None:
        """Send a signal to the child (or to its whole process group)."""
        ...

    def wait(self, timeout: float) -> int | None:
        """Exit status, or None if the child is still running after ``timeout``."""
        ...


class Launcher(Protocol):
    """Starts children. Tests inject a fake; :class:`SubprocessLauncher` is the real one."""

    offline_safe: bool

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin_text: str,
        stderr_path: Path,
    ) -> Child:
        """Start ``argv`` with the prompt on stdin."""
        ...

    def run(
        self, argv: Sequence[str], *, env: Mapping[str, str], timeout: float
    ) -> tuple[int, str]:
        """Run a short command (``--version``, ``--help``) and return (status, stdout)."""
        ...


class PopenChild:
    """A :class:`subprocess.Popen` child with a reader thread for stdout."""

    def __init__(self, proc: subprocess.Popen[str], stdin_text: str) -> None:
        self.proc = proc
        self._lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._write, args=(stdin_text,), daemon=True).start()
        threading.Thread(target=self._read, daemon=True).start()

    def _write(self, text: str) -> None:
        stdin = self.proc.stdin
        if stdin is None:
            return
        with suppress(OSError, ValueError):
            stdin.write(text)
        with suppress(OSError, ValueError):
            stdin.close()

    def _read(self) -> None:
        try:
            if self.proc.stdout is not None:
                for line in self.proc.stdout:
                    self._lines.put(line)
        finally:
            self._lines.put("")

    def read_line(self, timeout: float) -> str | None:
        """See :meth:`Child.read_line`."""
        try:
            return self._lines.get(timeout=max(timeout, 0.0))
        except queue.Empty:
            return None

    def send(self, sig: int, *, group: bool) -> None:
        """See :meth:`Child.send`."""
        if self.proc.poll() is not None:
            return
        with suppress(ProcessLookupError, PermissionError):
            if group:
                os.killpg(self.proc.pid, sig)
            else:
                self.proc.send_signal(sig)

    def wait(self, timeout: float) -> int | None:
        """See :meth:`Child.wait`."""
        try:
            return self.proc.wait(timeout=max(timeout, 0.0))
        except subprocess.TimeoutExpired:
            return None


class SubprocessLauncher:
    """Starts real children in their own process group, stderr to a file."""

    offline_safe = False

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin_text: str,
        stderr_path: Path,
    ) -> PopenChild:
        """See :meth:`Launcher.start`."""
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with stderr_path.open("w", encoding="utf-8") as err:
            proc = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=err,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        return PopenChild(proc, stdin_text)

    def run(
        self, argv: Sequence[str], *, env: Mapping[str, str], timeout: float
    ) -> tuple[int, str]:
        """See :meth:`Launcher.run`."""
        try:
            proc = subprocess.run(
                list(argv),
                env=dict(env),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PreflightError(f"{argv[0]} {' '.join(argv[1:])} failed: {exc}") from exc
        return proc.returncode, proc.stdout


def preflight(launcher: Launcher, claude_bin: str, env: Mapping[str, str]) -> CliInfo:
    """Check the CLI version and that every flag the driver uses exists.

    Raises:
        PreflightError: if a command fails or a required flag is missing.
    """
    code, out = launcher.run([claude_bin, "--version"], env=env, timeout=30)
    if code != 0:
        raise PreflightError(f"claude --version exited {code}")
    version = parse_version(out)
    code, help_text = launcher.run([claude_bin, "--help"], env=env, timeout=30)
    if code != 0:
        raise PreflightError(f"claude --help exited {code}")
    missing = missing_flags(help_text)
    if missing:
        raise PreflightError(f"claude {version} lacks {', '.join(missing)}; update Claude Code")
    return CliInfo(
        path=claude_bin,
        version=version,
        help_sha256=sha256_text(help_text),
        has_max_turns="--max-turns" in help_text,
    )


# --------------------------------------------------------------------------- stream parsing


@dataclass
class ToolUse:
    """One ``tool_use`` block from the stream and its result."""

    tool_use_id: str
    name: str
    input: dict[str, Any]
    index: int
    turn: int
    started: float
    result_text: str | None = None
    is_error: bool = False
    finished: float | None = None
    denied: bool = False

    @property
    def has_result(self) -> bool:
        """True once a ``tool_result`` for this call arrived."""
        return self.result_text is not None


@dataclass
class CompactionSeen:
    """A compaction event in the stream, at the tool call index where it happened."""

    subtype: str
    tool_call_index: int
    trigger: str | None
    pre_tokens: int | None
    post_tokens: int | None
    event: dict[str, Any]
    summary: str | None = None
    summary_source: str | None = None
    """``stream`` or ``transcript``: where the summary text was found."""


@dataclass(frozen=True)
class Limits:
    """Caps for one launch.

    Attributes:
        turn_cap: Assistant turns allowed for this message (``claude_agent_max_turns``).
        subtask_calls: Tool calls allowed in this message (``max_calls_per_subtask``).
        session_calls: Tool calls left in the session (``max_tool_calls`` minus used).
    """

    turn_cap: int
    subtask_calls: int
    session_calls: int


class StreamState:
    """Everything one launch's stream-json output says, plus its stop decisions.

    :meth:`feed` takes one parsed event and returns a stop reason when the
    child must be stopped now: ``turn_cap``, ``subtask_tool_cap``, or
    ``session_tool_cap`` (only once every tool result of a turn that wants to
    continue has arrived), or an account problem (``usage_limit``,
    ``billing_error``, ``auth_error``, ``overage``). An API error that is not an
    account problem is kept in :attr:`api_error`; the CLI ends the launch itself.
    """

    def __init__(self, *, first_index: int, limits: Limits, clock: Callable[[], float]) -> None:
        self.first_index = first_index
        self.next_index = first_index
        self.limits = limits
        self.clock = clock
        self.turns = 0
        self._last_message_id: str | None = None
        self._turn_tools: dict[int, list[str]] = {}
        self.tool_uses: list[ToolUse] = []
        self.by_id: dict[str, ToolUse] = {}
        self.inits: list[dict[str, Any]] = []
        self.compactions: list[CompactionSeen] = []
        self.compaction_notes: list[dict[str, Any]] = []
        self.hook_events: list[dict[str, Any]] = []
        self.hook_failures: list[dict[str, Any]] = []
        self.rate_limits: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
        self.account_error: dict[str, Any] | None = None
        self.api_error: dict[str, Any] | None = None
        self.texts: list[str] = []
        self.message_usage: dict[str, dict[str, Any]] = {}
        self.warnings: list[str] = []
        self.overrun = False
        self.unknown = 0
        self.subagent_events = 0
        self.orphan_results = 0

    # ------------------------------------------------------------------ queries

    @property
    def calls(self) -> int:
        """Tool calls started in this launch."""
        return self.next_index - self.first_index

    def pending(self) -> int:
        """Tool calls still waiting for a result."""
        return sum(1 for t in self.tool_uses if not t.has_result)

    def final_text(self) -> str:
        """The final answer: the result event's text, else the assistant text."""
        if self.result is not None and isinstance(self.result.get("result"), str):
            return str(self.result["result"])
        return "\n".join(self.texts)

    # ------------------------------------------------------------------ feeding

    def feed(self, event: Mapping[str, Any]) -> str | None:
        """Process one stream event; see the class docstring for the return value."""
        kind = event.get("type")
        if kind == "system":
            return self._system(event)
        if kind == "assistant":
            return self._assistant(event)
        if kind == "user":
            return self._user(event)
        if kind == "result":
            self._result(event)
            return None
        if kind == "rate_limit_event":
            return self._rate_limit(event)
        self.unknown += 1
        return None

    def _account(self, kind: str, detail: str, resets_at: object = None) -> str:
        if self.account_error is None:
            self.account_error = {
                "kind": kind,
                "detail": detail[:300],
                "resets_at": _epoch_iso(resets_at),
            }
        return kind

    def _system(self, event: Mapping[str, Any]) -> str | None:
        subtype = str(event.get("subtype", ""))
        if subtype == "init":
            self.inits.append(dict(event))
        elif "compact" in subtype:
            meta = event.get("compact_metadata")
            if subtype == "compact_boundary" or isinstance(meta, Mapping):
                info = _as_dict(meta)
                trigger = info.get("trigger")
                self.compactions.append(
                    CompactionSeen(
                        subtype=subtype,
                        tool_call_index=self.next_index,
                        trigger=str(trigger) if trigger is not None else None,
                        pre_tokens=_as_int(info.get("pre_tokens")),
                        post_tokens=_as_int(info.get("post_tokens")),
                        event=dict(event),
                    )
                )
            else:
                self.compaction_notes.append(dict(event))
        elif subtype.startswith("hook_"):
            self.hook_events.append(dict(event))
            failed = _as_int(event.get("exit_code")) not in (None, 0)
            if subtype == "hook_response" and (failed or event.get("outcome") == "error"):
                self.hook_failures.append(dict(event))
        elif subtype == "api_retry":
            self.warnings.append(f"api retry: {str(event.get('error', ''))[:120]}")
        return None

    def _assistant(self, event: Mapping[str, Any]) -> str | None:
        if event.get("parent_tool_use_id"):
            self.subagent_events += 1
            return None
        message = _as_dict(event.get("message"))
        content = message.get("content")
        blocks = [b for b in content if isinstance(b, Mapping)] if isinstance(content, list) else []
        text = "\n".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")
        error = event.get("error")
        if isinstance(error, str) and error:
            kind = ACCOUNT_ERRORS.get(error)
            if kind is not None:
                return self._account(kind, f"{error}: {text}")
            self.warnings.append(f"assistant error {error}: {text[:120]}")
            if error not in MODEL_ERRORS and self.api_error is None:
                self.api_error = {"kind": error, "detail": text[:300]}
            return None
        if message.get("model") == SYNTHETIC_MODEL or event.get("is_meta"):
            if text:
                self.texts.append(text)
            return None
        message_id = message.get("id")
        message_id = str(message_id) if message_id is not None else None
        if message_id is None or message_id != self._last_message_id:
            self.turns += 1
            self._last_message_id = message_id
        if message_id is not None and isinstance(message.get("usage"), Mapping):
            self.message_usage[message_id] = dict(message["usage"])
        stop = None
        if self.turns > self.limits.turn_cap:
            self.overrun = True
            stop = "turn_cap"
        for block in blocks:
            if block.get("type") == "tool_use":
                use = ToolUse(
                    tool_use_id=str(block.get("id", f"missing-id-{self.next_index}")),
                    name=str(block.get("name", "")),
                    input=_as_dict(block.get("input")),
                    index=self.next_index,
                    turn=self.turns,
                    started=self.clock(),
                )
                self.next_index += 1
                self.tool_uses.append(use)
                self.by_id[use.tool_use_id] = use
                self._turn_tools.setdefault(self.turns, []).append(use.tool_use_id)
            elif block.get("type") == "text":
                self.texts.append(str(block.get("text", "")))
        return stop

    def _user(self, event: Mapping[str, Any]) -> str | None:
        message = _as_dict(event.get("message"))
        content = message.get("content")
        resolved = False
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "tool_result":
                    use = self.by_id.get(str(block.get("tool_use_id")))
                    if use is None:
                        self.orphan_results += 1
                        continue
                    use.result_text = result_text(block.get("content"))
                    use.is_error = bool(block.get("is_error"))
                    use.finished = self.clock()
                    resolved = True
                elif block.get("type") == "text":
                    texts.append(str(block.get("text", "")))
        summary = "\n".join(texts)
        is_summary = bool(event.get("isCompactSummary")) or summary.startswith(
            COMPACT_SUMMARY_PREFIX
        )
        if is_summary and self.compactions and self.compactions[-1].summary is None:
            self.compactions[-1].summary = summary
            self.compactions[-1].summary_source = "stream"
        return self._cap_reason() if resolved else None

    def _cap_reason(self) -> str | None:
        if self.pending() or not self._turn_tools.get(self.turns):
            return None
        if self.turns >= self.limits.turn_cap:
            return "turn_cap"
        if self.calls >= self.limits.subtask_calls:
            return "subtask_tool_cap"
        if self.calls >= self.limits.session_calls:
            return "session_tool_cap"
        return None

    def _result(self, event: Mapping[str, Any]) -> None:
        self.result = dict(event)
        for denial in event.get("permission_denials") or []:
            if isinstance(denial, Mapping):
                use = self.by_id.get(str(denial.get("tool_use_id")))
                if use is not None:
                    use.denied = True
        if event.get("is_error"):
            errors = event.get("errors")
            extra = " ".join(str(e) for e in errors) if isinstance(errors, list) else ""
            text = f"{event.get('result') or ''} {extra}".strip()
            kind = classify_text(text)
            if kind is not None:
                self._account(kind, text)

    def _rate_limit(self, event: Mapping[str, Any]) -> str | None:
        info = _as_dict(event.get("rate_limit_info"))
        self.rate_limits.append(info)
        overage = overage_signals(info)
        if overage:
            detail = "the account could draw on paid extra usage: " + ", ".join(overage)
            return self._account("overage", detail, info.get("resetsAt"))
        if info.get("status") == "rejected":
            detail = f"rate limit rejected ({info.get('rateLimitType')})"
            return self._account("usage_limit", detail, info.get("resetsAt"))
        return None

    # ------------------------------------------------------------------ usage

    def usage(self) -> dict[str, Any]:
        """Token use and plan-equivalent cost of the launch.

        From the result event when there is one; otherwise summed over the
        distinct assistant messages seen (``estimated: true``).
        """
        if self.result is not None and isinstance(self.result.get("usage"), Mapping):
            source = [dict(self.result["usage"])]
            estimated = False
        else:
            source = list(self.message_usage.values())
            estimated = True
        totals: dict[str, Any] = {
            k: sum(_as_int(u.get(k)) or 0 for u in source) for k in USAGE_TOKENS
        }
        cost = self.result.get("total_cost_usd") if self.result is not None else None
        totals["plan_equivalent_usd"] = float(cost) if isinstance(cost, int | float) else 0.0
        totals["estimated"] = estimated
        return totals


def parse_line(line: str) -> dict[str, Any] | None:
    """A stream line as a JSON object, or None if it is not one."""
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def terminate(child: Child, grace_s: float, signals: list[str]) -> int | None:
    """Stop a child: SIGINT, then SIGTERM, then SIGKILL to its process group."""
    for sig, name, group in (
        (signal.SIGINT, "SIGINT", False),
        (signal.SIGTERM, "SIGTERM", True),
        (signal.SIGKILL, "SIGKILL", True),
    ):
        code = child.wait(0)
        if code is not None:
            return code
        child.send(sig, group=group)
        signals.append(name)
        code = child.wait(grace_s)
        if code is not None:
            return code
    return child.wait(grace_s)


def consume(
    child: Child,
    state: StreamState,
    *,
    on_line: Callable[[str, dict[str, Any] | None], None],
    timeout_s: float,
    grace_s: float,
    poll_s: float,
    clock: Callable[[], float],
) -> tuple[int | None, str | None, list[str]]:
    """Read a child's stream to the end, stopping it when :class:`StreamState` asks.

    Returns:
        ``(exit status, stop reason, signals sent)``. The stop reason is None
        when the child ended on its own.
    """
    signals: list[str] = []
    reason: str | None = None
    stop_at = 0.0
    level = 0
    exited_at: float | None = None
    deadline = clock() + timeout_s
    escalation = (
        (signal.SIGTERM, "SIGTERM"),
        (signal.SIGKILL, "SIGKILL"),
    )
    try:
        while True:
            now = clock()
            if reason is None and now >= deadline:
                reason = "timeout"
                stop_at = now
                child.send(signal.SIGINT, group=False)
                signals.append("SIGINT")
            if reason is not None and now - stop_at >= grace_s * (level + 1):
                if level >= len(escalation):
                    break
                sig, name = escalation[level]
                child.send(sig, group=True)
                signals.append(name)
                level += 1
            line = child.read_line(poll_s)
            if line is None:
                # The child may have exited while a grandchild still holds the pipe.
                if child.wait(0) is not None:
                    exited_at = clock() if exited_at is None else exited_at
                    if clock() - exited_at >= grace_s:
                        break
                continue
            if line == "":
                break
            event = parse_line(line)
            on_line(line, event)
            if event is None:
                continue
            wanted = state.feed(event)
            if wanted is not None and reason is None:
                reason = wanted
                stop_at = clock()
                child.send(signal.SIGINT, group=False)
                signals.append("SIGINT")
    finally:
        code = child.wait(grace_s)
        if code is None:
            code = terminate(child, grace_s, signals)
    return code, reason, signals


# --------------------------------------------------------------------------- hook records, events


def align_records(
    uses: Sequence[ToolUse], records: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Match hook records to tool uses.

    First by ``tool_use_id``; then, for records without a usable id, in order
    by tool name and identical input. Each record matches at most one use.

    Returns:
        ``(tool_use_id -> record, records that matched nothing)``.
    """
    ids = {u.tool_use_id for u in uses}
    matched: dict[str, dict[str, Any]] = {}
    rest: list[dict[str, Any]] = []
    for record in records:
        rid = record.get("tool_use_id")
        if isinstance(rid, str) and rid in ids and rid not in matched:
            matched[rid] = dict(record)
        else:
            rest.append(dict(record))
    leftover: list[dict[str, Any]] = []
    for record in rest:
        for use in uses:
            if (
                use.tool_use_id not in matched
                and record.get("tool_name") == use.name
                and canonical(record.get("tool_input")) == canonical(use.input)
            ):
                matched[use.tool_use_id] = record
                break
        else:
            leftover.append(record)
    return matched, leftover


def instruction_load(record: Mapping[str, Any]) -> dict[str, Any]:
    """The kept fields of an ``instructions`` hook record (never the file's content)."""
    loaded = {k: record.get(k) for k in ("file_path", "memory_type", "load_reason")}
    return loaded | {"in_root": bool(record.get("in_root"))}


def foreign_instruction(loaded: Mapping[str, Any]) -> str:
    """Contaminant line for an instruction file loaded from outside the workspace."""
    kind = loaded.get("memory_type") or "unknown"
    return f"instruction file from outside the workspace: {kind}:{loaded.get('file_path')}"


def workspace_path(path: object, roots: Sequence[Path]) -> str:
    """A tool's file path as a workspace-relative POSIX path (absolute if outside)."""
    text = str(path) if path not in (None, "") else "."
    candidate = Path(text)
    if not candidate.is_absolute():
        return PurePosixPath(text).as_posix()
    for root in roots:
        for form in (candidate, candidate.resolve(strict=False)):
            with suppress(ValueError):
                return form.relative_to(root).as_posix() or "."
    return text


def map_tool(
    name: str, data: Mapping[str, Any], roots: Sequence[Path]
) -> tuple[ToolName, dict[str, Any]]:
    """Translate a Claude Code tool call into a :class:`~halflife.schema.ToolEvent` tool and args.

    Bash becomes ``run`` (``cmd``), Write ``write_file``, Edit and MultiEdit
    ``replace_in_file``, Read ``read_file``, and Glob, Grep, and LS
    ``list_files``; anything else is ``invalid``. ``claude_tool`` keeps the
    original name.
    """
    tool: ToolName
    args: dict[str, Any]
    if name == "Bash":
        tool, args = "run", {"cmd": str(data.get("command", ""))}
        if data.get("run_in_background"):
            args["background"] = True
    elif name == "Write":
        tool = "write_file"
        args = {
            "path": workspace_path(data.get("file_path"), roots),
            "content": str(data.get("content", "")),
        }
    elif name == "Edit":
        tool = "replace_in_file"
        args = {
            "path": workspace_path(data.get("file_path"), roots),
            "old": str(data.get("old_string", "")),
            "new": str(data.get("new_string", "")),
            "replace_all": bool(data.get("replace_all", False)),
        }
    elif name == "MultiEdit":
        edits = data.get("edits")
        tool = "replace_in_file"
        args = {
            "path": workspace_path(data.get("file_path"), roots),
            "edits": [
                {
                    "old": str(e.get("old_string", "")),
                    "new": str(e.get("new_string", "")),
                    "replace_all": bool(e.get("replace_all", False)),
                }
                for e in (edits if isinstance(edits, list) else [])
                if isinstance(e, Mapping)
            ],
        }
    elif name == "Read":
        tool, args = "read_file", {"path": workspace_path(data.get("file_path"), roots)}
        args.update({k: data[k] for k in ("offset", "limit") if k in data})
    elif name in {"Glob", "Grep", "LS"}:
        tool, args = "list_files", {"path": workspace_path(data.get("path"), roots)}
        args.update({k: data[k] for k in ("pattern", "glob") if k in data})
    else:
        tool, args = "invalid", {"raw": canonical(dict(data))[:500]}
    args["claude_tool"] = name
    return tool, args


def tool_events(
    uses: Sequence[ToolUse],
    matched: Mapping[str, Mapping[str, Any]],
    *,
    subtask_id: str,
    fs_before: Mapping[str, str],
    fs_end: Mapping[str, str],
    roots: Sequence[Path],
    redact: Callable[[str], str],
) -> tuple[list[ToolEvent], list[str]]:
    """Tool events for one launch, with ``fs_after`` from each call's hook record.

    A call without a hook record (denied, not executed, or a missing hook)
    keeps the previous state. If the workspace at the end of the launch differs
    from the last event's state, the last event takes the end state, so every
    change lands inside the subtask that made it; the changed paths are returned.
    """
    events: list[ToolEvent] = []
    fs: Mapping[str, str] = dict(fs_before)
    for use in uses:
        record = matched.get(use.tool_use_id)
        if record is not None and isinstance(record.get("fs"), Mapping):
            fs = dict(record["fs"])
        tool, args = map_tool(use.name, use.input, roots)
        if use.denied:
            output = f"denied: {use.result_text or ''}"
        elif use.has_result:
            output = use.result_text or ""
        else:
            output = "not executed: the harness stopped the child before this call ran"
        exit_code: int | None = None
        if tool == "run" and use.has_result and not use.denied:
            found = EXIT_CODE.search(output) if use.is_error else None
            exit_code = int(found.group(1)) if found else (None if use.is_error else 0)
        elapsed = 0.0
        if record is not None and isinstance(record.get("duration_ms"), int | float):
            elapsed = float(record["duration_ms"]) / 1000
        elif use.finished is not None:
            elapsed = max(0.0, use.finished - use.started)
        net: tuple[NetAttempt, ...] = ()
        if record is not None and isinstance(record.get("net"), list):
            net = tuple(
                NetAttempt(target=str(n.get("target", "")), via=str(n.get("via", "")))
                for n in record["net"]
                if isinstance(n, Mapping)
            )
        text, _ = truncate_output(redact(output), DEFAULT_OUTPUT_LIMIT)
        events.append(
            ToolEvent(
                index=use.index,
                subtask_id=subtask_id,
                tool=tool,
                args=args,
                output=text,
                exit_code=exit_code,
                fs_after=dict(fs),
                net=net,
                blocked=use.denied or not use.has_result,
                elapsed_s=elapsed,
            )
        )
    drift: list[str] = []
    if dict(fs_end) != dict(fs):
        drift = list(diff_snapshots(fs, fs_end).changed)
        if events:
            events[-1] = dataclasses.replace(events[-1], fs_after=dict(fs_end))
    return events, drift


def init_problems(init: Mapping[str, Any], expected_tools: Sequence[str]) -> list[str]:
    """Ways a child's ``init`` event shows configuration beyond what the driver set.

    Checks the tool list, MCP servers, plugins (built-in plugins are allowed),
    the permission mode, the API key source (a Max-plan login reports
    ``none``), and the output style.
    """
    problems = []
    tools = {str(t) for t in init.get("tools") or []}
    expected = set(expected_tools)
    if tools - expected:
        problems.append(f"unexpected tools {sorted(tools - expected)}")
    if expected - tools:
        problems.append(f"missing tools {sorted(expected - tools)}")
    servers = [_as_dict(s).get("name", s) for s in init.get("mcp_servers") or []]
    if servers:
        problems.append(f"MCP servers loaded {servers}")
    plugins = [
        _as_dict(p).get("name", p)
        for p in init.get("plugins") or []
        if not str(_as_dict(p).get("source", "")).endswith("@builtin")
    ]
    if plugins:
        problems.append(f"plugins loaded {plugins}")
    mode = init.get("permissionMode")
    if mode is not None and mode != PERMISSION_MODE:
        problems.append(f"permission mode {mode!r}")
    source = init.get("apiKeySource")
    if source is not None and source != "none":
        problems.append(f"apiKeySource {source!r} (a Max-plan login reports 'none')")
    style = init.get("output_style")
    if style not in (None, "default"):
        problems.append(f"output style {style!r}")
    return problems


def init_summary(init: Mapping[str, Any]) -> dict[str, Any]:
    """The parts of an ``init`` event worth keeping in provenance."""
    keep = (
        "session_id",
        "model",
        "tools",
        "mcp_servers",
        "plugins",
        "permissionMode",
        "apiKeySource",
        "output_style",
        "claude_code_version",
        "skills",
        "slash_commands",
        "agents",
    )
    summary = {k: init[k] for k in keep if k in init}
    if isinstance(summary.get("plugins"), list):
        summary["plugins"] = [_masked_plugin(p) for p in summary["plugins"]]
    return summary


def _masked_plugin(plugin: Any) -> Any:
    """A plugin entry of an ``init`` event with its ``path`` shown with ``~`` for home."""
    if isinstance(plugin, Mapping) and isinstance(plugin.get("path"), str):
        return {**plugin, "path": snapshot_hook.display_path(plugin["path"])}
    return plugin


# --------------------------------------------------------------------------- caps and ledger


@dataclass(frozen=True)
class Caps:
    """The ``claude_*`` caps in force, already scaled by ``cap_share``."""

    agent_runs_per_day: int
    max_turns: int
    single_calls_per_day: int
    share: float

    @classmethod
    def load(cls, budget_path: Path, env: Mapping[str, str]) -> Caps:
        """Read budget.toml and apply ``cap_share`` (budget.toml and ``CAP_SHARE``)."""
        share = cap_share(env, file_share=file_cap_share(budget_path))
        caps = effective_caps(load_caps(budget_path), share)
        loaded = cls(
            agent_runs_per_day=int(caps["claude_agent_runs_per_day"]),
            max_turns=int(caps["claude_agent_max_turns"]),
            single_calls_per_day=int(caps["claude_single_calls_per_day"]),
            share=share,
        )
        if loaded.max_turns < 1:
            raise CapExceeded("claude_agent_max_turns rounds to 0 at this cap_share")
        return loaded

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return dataclasses.asdict(self)


def used_today(ledger_path: Path, kind: str, today: str | None = None) -> int:
    """Ledger rows of ``kind`` on the current UTC day (cached rows excluded)."""
    day = today or utc_now().date().isoformat()
    return sum(
        1
        for record in read_ledger(ledger_path)
        if record.get("kind") == kind
        and str(record.get("ts", ""))[:10] == day
        and not record.get("cached", False)
    )


def append_ledger(path: Path, row: Mapping[str, Any]) -> None:
    """Append one usage row (``cash_usd`` is always 0)."""
    _append_jsonl(path, {**row, "cash_usd": 0, "cached": False})


USAGE_TOKENS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def empty_usage() -> dict[str, Any]:
    """Zero launches, token counts, wall time, and plan-equivalent cost."""
    return {
        "launches": 0,
        **dict.fromkeys(USAGE_TOKENS, 0),
        "wall_s": 0.0,
        "plan_equivalent_usd": 0.0,
        "estimated": False,
    }


def add_usage(total: dict[str, Any], usage: Mapping[str, Any], wall_s: float) -> None:
    """Add one launch's usage (from :meth:`StreamState.usage`) to a running total."""
    total["launches"] += 1
    for key in USAGE_TOKENS:
        total[key] += int(usage[key])
    total["wall_s"] += float(wall_s)
    total["plan_equivalent_usd"] += float(usage["plan_equivalent_usd"])
    total["estimated"] = bool(total["estimated"] or usage["estimated"])


def purpose_usage(total: Mapping[str, Any]) -> dict[str, Any]:
    """One purpose of ``usage.json``, in the layout the minimal-agent runner writes.

    The fields are those of :class:`halflife.llm.ClientModel` usage (``calls``,
    ``prompt_tokens``, ``completion_tokens``, ``elapsed_s``, ``cached``) plus the
    plan-equivalent cost, so :func:`halflife.experiment.bank_runs` banks E2
    costs the same way as other arms.
    """
    fields = usage_fields(total)
    return {
        "calls": int(total["launches"]),
        "prompt_tokens": fields["prompt_tokens"],
        "completion_tokens": fields["completion_tokens"],
        "elapsed_s": round(float(total["wall_s"]), 3),
        "cached": 0,
        "plan_equivalent_usd": fields["plan_equivalent_usd"],
        "usage_estimated": fields["usage_estimated"],
    }


def usage_fields(usage: Mapping[str, Any]) -> dict[str, Any]:
    """The token and cost fields of a ledger row (prompt tokens include cache reads and writes)."""
    return {
        "prompt_tokens": int(usage["input_tokens"])
        + int(usage["cache_creation_input_tokens"])
        + int(usage["cache_read_input_tokens"]),
        "completion_tokens": int(usage["output_tokens"]),
        "plan_equivalent_usd": round(float(usage["plan_equivalent_usd"]), 6),
        "usage_estimated": bool(usage["estimated"]),
    }


# --------------------------------------------------------------------------- configuration

Checker = Callable[[Workspace, str], int]


def sandbox_checker(method: str) -> Checker:
    """Harness checks on a scratch copy through :mod:`halflife.sandbox` with ``method``."""
    options = SandboxOptions(timeout_s=DEFAULT_CHECK_TIMEOUT_S, method=method)

    def check(workspace: Workspace, cmd: str) -> int:
        return run_check(workspace, cmd, options).exit_code

    return check


@dataclass
class DriverConfig:
    """Everything a run shares across sessions.

    Attributes:
        run_id: Run label (``runs/<run_id>/``).
        model: Model tier passed to ``--model`` (for example ``haiku``).
        claude_bin: Absolute path of the ``claude`` executable.
        cli: Preflight facts about the CLI.
        caps: Caps in force.
        out_root: Directory that holds ``<run_id>/``.
        ledger_path: Usage ledger.
        launcher: Starts children.
        python: Interpreter for the snapshot hook (the harness environment).
        prereg: Preregistration path for confirmatory sessions.
        experiment_label: Label written to ledger rows when a plan has none.
        sandbox_method: :mod:`halflife.sandbox` method for harness checks.
        checker: Runs a harness command on a scratch copy; defaults to the sandbox.
        guard: Billing guard; called before every launch.
        env: Host environment the child environment is built from.
        clock: Monotonic clock.
        launch_timeout_s: Wall-time limit of one subtask launch.
        probe_timeout_s: Wall-time limit of one probe.
        grace_s: Wait between stop signals.
        poll_s: Stream read poll interval.
        workspace_parent: Where workspaces are created (default: system temp).
        repo_root: The repository (workspaces must lie outside it).
        network_logging: Whether the smoke showed that network attempts are logged.
        contaminants: Known contaminants the smoke reported and the operator
            acknowledged (``--accept-contaminants``); copied into every session.
    """

    run_id: str
    model: str
    claude_bin: str
    cli: CliInfo
    caps: Caps
    out_root: Path
    ledger_path: Path
    launcher: Launcher
    python: str = sys.executable
    prereg: str | None = None
    experiment_label: str = "e2"
    sandbox_method: str = "unknown"
    checker: Checker | None = None
    guard: Callable[[], billing_guard.GuardReport] = billing_guard.check
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    clock: Callable[[], float] = time.monotonic
    launch_timeout_s: float = 1800.0
    probe_timeout_s: float = 600.0
    grace_s: float = 10.0
    poll_s: float = 0.5
    workspace_parent: Path | None = None
    repo_root: Path | None = None
    network_logging: bool = True
    contaminants: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not _TIER.match(self.model):
            raise UnsupportedPlan(f"model tier {self.model!r} is not a plain alias or id")

    def check(self, workspace: Workspace, cmd: str) -> int:
        """Run a harness check with the configured checker."""
        checker = self.checker or sandbox_checker(self.sandbox_method)
        return checker(workspace, cmd)


def plan_problems(plan: SessionPlan, template: Template, *, network_logging: bool) -> list[str]:
    """Why this driver cannot run a plan faithfully (empty when it can)."""
    problems = []
    if plan.template_id != template.id:
        problems.append(f"plan names template {plan.template_id}, got {template.id}")
    if plan.mitigation != "none":
        problems.append(f"mitigation {plan.mitigation!r} is not supported by this driver")
    if plan.compaction.strategy not in {"none", "full_summary"}:
        problems.append(
            f"compaction strategy {plan.compaction.strategy!r} cannot be applied to Claude Code"
        )
    names = {PurePosixPath(p).name for p in template.files}
    parts = {part for p in template.files for part in PurePosixPath(p).parts}
    if names & {"CLAUDE.md", "CLAUDE.local.md"} or ".claude" in parts:
        problems.append("template files include Claude Code instruction or settings files")
    kinds = {c.detector.kind for c in template.constraints}
    if "no_network" in kinds and not network_logging:
        problems.append("template has a no_network rule but the smoke run found no network log")
    return problems


# --------------------------------------------------------------------------- one conversation


@dataclass
class Launch:
    """One finished launch."""

    number: int
    purpose: str
    subtask_id: str | None
    checkpoint: str | None
    argv: list[str]
    state: StreamState
    returncode: int | None
    stop_reason: str | None
    signals: list[str]
    started: str
    wall_s: float

    @property
    def init(self) -> dict[str, Any] | None:
        """The launch's ``init`` event, if it reached one."""
        return self.state.inits[0] if self.state.inits else None


class Conversation:
    """One Claude Code conversation in one workspace, with its logs.

    Owns the session UUID, the settings files, the hook output, the stream
    and command logs, and the merged blob store.
    """

    def __init__(
        self,
        cfg: DriverConfig,
        *,
        workspace: Workspace,
        out_dir: Path,
        append: str | None,
        session_id: str,
        experiment: str,
    ) -> None:
        self.cfg = cfg
        self.ws = workspace
        self.out_dir = out_dir
        self.append = append
        self.session_id = session_id
        self.experiment = experiment
        self.uuid = str(uuid.uuid4())
        self.roots = [workspace.root]
        self.hooks_path = out_dir / "hooks.jsonl"
        self.stream_path = out_dir / "stream.jsonl"
        self.commands_path = out_dir / "commands.jsonl"
        git_dir = workspace.root / ".git"
        if not git_dir.is_dir():
            raise SafetyStop("workspace has no .git directory for the network log")
        self.netlog = git_dir / NETLOG_NAME
        self.netlog.write_text("", encoding="utf-8")
        hook = hook_command(cfg.python, self.hooks_path, workspace.root, self.netlog)
        self.settings_path = out_dir / "settings.json"
        self.probe_settings_path = out_dir / "probe-settings.json"
        self.settings = settings_for(hook)
        self.probe_settings = settings_for(None)
        _write_json(self.settings_path, self.settings)
        _write_json(self.probe_settings_path, self.probe_settings)
        self.env = child_env(cfg.env, netlog=self.netlog)
        self.redact_values = secret_values(cfg.env)
        self.started = 0
        self.main_launches = 0
        self.next_seq = 0
        self.blobs: dict[str, str] = dict(workspace.blobs)
        self.transcript: Path | None = None
        self.instructions: list[dict[str, Any]] = []
        self.hook_errors: list[str] = []
        self.model_ids: list[str] = []
        self.inits: list[dict[str, Any]] = []
        self.usage = empty_usage()
        """Usage of the conversation's own launches (the agent run in the ledger)."""
        self.probe_usage = empty_usage()
        """Usage of the forked probes (each also has its own single-call ledger row)."""
        snapshot_hook.write_baseline(
            self.hooks_path, workspace.baseline, workspace.blobs, workspace.skipped
        )
        self.next_seq = 1

    def redact(self, text: str) -> str:
        """Redact credential-shaped text and live secret values."""
        return snapshot_hook.redact_text(text, self.redact_values)

    def _log_line(
        self, handle: IO[str], number: int, purpose: str, subtask_id: str | None
    ) -> Callable[[str, dict[str, Any] | None], None]:
        def write(raw: str, event: dict[str, Any] | None) -> None:
            entry: dict[str, Any] = {"launch": number, "purpose": purpose, "subtask_id": subtask_id}
            if event is None:
                entry["raw"] = self.redact(raw.rstrip("\n"))[:2000]
            else:
                entry["event"] = snapshot_hook.redact_json(event, self.redact_values)
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()

        return write

    def launch(
        self,
        *,
        purpose: str,
        message: str,
        limits: Limits,
        first_index: int,
        subtask_id: str | None = None,
        checkpoint: str | None = None,
        ledger_purpose: str | None = None,
    ) -> Launch:
        """Run one child to the end and return what it did.

        ``purpose`` is ``subtask``, ``smoke``, or ``probe`` (a tool-less fork).
        With ``ledger_purpose``, one ``claude_single_call`` ledger row is written
        once the child has started, even if reading it fails.

        Raises:
            BillingGuardRefused: if the billing guard is not clean.
            SafetyStop: if the command line or the settings file is not what a
                clean child needs, the child's ``init`` event shows configuration
                the driver did not set, or the conversation id is wrong.
        """
        cfg = self.cfg
        require_clean_guard(cfg.guard)
        probe = purpose == "probe"
        if probe:
            argv = probe_argv(
                cfg.claude_bin,
                model=cfg.model,
                settings_path=self.probe_settings_path,
                append=self.append,
                session_uuid=self.uuid,
            )
        else:
            argv = main_argv(
                cfg.claude_bin,
                model=cfg.model,
                settings_path=self.settings_path,
                append=self.append,
                session_uuid=self.uuid,
                first=self.main_launches == 0,
            )
        check_argv(argv, probe=probe)
        if probe:
            check_settings(self.probe_settings_path, self.probe_settings)
        else:
            check_settings(self.settings_path, self.settings)
        number = self.started + 1
        state = StreamState(first_index=first_index, limits=limits, clock=cfg.clock)
        started = utc_stamp()
        t0 = cfg.clock()
        child = cfg.launcher.start(
            argv,
            cwd=self.ws.root,
            env=self.env,
            stdin_text=message,
            stderr_path=self.out_dir / "stderr" / f"{number:03d}-{purpose}.txt",
        )
        self.started = number
        code: int | None = None
        reason: str | None = None
        signals: list[str] = []
        try:
            with self.stream_path.open("a", encoding="utf-8") as handle:
                code, reason, signals = consume(
                    child,
                    state,
                    on_line=self._log_line(handle, number, purpose, subtask_id),
                    timeout_s=cfg.probe_timeout_s if probe else cfg.launch_timeout_s,
                    grace_s=cfg.grace_s,
                    poll_s=cfg.poll_s,
                    clock=cfg.clock,
                )
        finally:
            wall = cfg.clock() - t0
            if ledger_purpose is not None:
                self._single_call_row(state, wall, ledger_purpose, checkpoint)
        if not probe:
            self.main_launches += 1
        launch = Launch(
            number=number,
            purpose=purpose,
            subtask_id=subtask_id,
            checkpoint=checkpoint,
            argv=argv,
            state=state,
            returncode=code,
            stop_reason=reason,
            signals=signals,
            started=started,
            wall_s=wall,
        )
        self._record(launch, message)
        self._check_init(launch)
        return launch

    def _single_call_row(
        self, state: StreamState, wall: float, purpose: str, checkpoint: str | None
    ) -> None:
        append_ledger(
            self.cfg.ledger_path,
            {
                "ts": utc_stamp(),
                "kind": "claude_single_call",
                "backend": LEDGER_BACKEND,
                "model": self.cfg.model,
                "model_id": self.model_ids[-1] if self.model_ids else None,
                "run_id": self.cfg.run_id,
                "experiment": self.experiment,
                "session_id": self.session_id,
                "purpose": purpose,
                "checkpoint": checkpoint,
                **usage_fields(state.usage()),
                "turns": state.turns,
                "wall_s": round(wall, 3),
                "cli_version": self.cfg.cli.version,
            },
        )

    def _record(self, launch: Launch, message: str) -> None:
        state = launch.state
        usage = state.usage()
        target = self.probe_usage if launch.purpose == "probe" else self.usage
        add_usage(target, usage, launch.wall_s)
        result = state.result or {}
        stderr_file = self.out_dir / "stderr" / f"{launch.number:03d}-{launch.purpose}.txt"
        try:
            stderr_tail = stderr_file.read_text(encoding="utf-8", errors="replace")[-800:]
        except OSError:
            stderr_tail = ""
        entry = {
            "launch": launch.number,
            "purpose": launch.purpose,
            "subtask_id": launch.subtask_id,
            "checkpoint": launch.checkpoint,
            "argv": launch.argv,
            "stdin_text": self.redact(message),
            "stdin_sha256": sha256_text(message),
            "cwd": str(self.ws.root),
            "env_names": sorted(self.env),
            "settings_sha256": file_sha256(
                self.probe_settings_path if launch.purpose == "probe" else self.settings_path
            ),
            "started": launch.started,
            "wall_s": round(launch.wall_s, 3),
            "returncode": launch.returncode,
            "stop_reason": launch.stop_reason,
            "signals": launch.signals,
            "turns": state.turns,
            "turn_cap_overrun": state.overrun,
            "tool_calls": state.calls,
            "session_id": (launch.init or {}).get("session_id"),
            "model_id": (launch.init or {}).get("model"),
            "result_subtype": result.get("subtype"),
            "result_is_error": result.get("is_error"),
            "num_turns": result.get("num_turns"),
            "terminal_reason": result.get("terminal_reason"),
            "usage": usage,
            "account_error": state.account_error,
            "compactions": len(state.compactions),
            "hook_failures": len(state.hook_failures),
            "warnings": state.warnings,
            "stderr_tail": self.redact(stderr_tail),
        }
        _append_jsonl(self.commands_path, entry)

    def _check_init(self, launch: Launch) -> None:
        init = launch.init
        if init is None:
            return
        self.inits.append(init_summary(init))
        model = init.get("model")
        if isinstance(model, str) and model:
            self.model_ids.append(model)
        expected = () if launch.purpose == "probe" else CLAUDE_TOOLS
        problems = init_problems(init, expected)
        sid = init.get("session_id")
        if launch.purpose == "probe":
            if sid == self.uuid:
                problems.append("the probe did not fork: it reported the main session id")
        elif sid != self.uuid:
            problems.append(f"the launch continued session {sid!r}, not {self.uuid}")
        if problems:
            raise SafetyStop(f"launch {launch.number} ({launch.purpose}): " + "; ".join(problems))

    def new_records(self) -> list[dict[str, Any]]:
        """Hook records written since the last call, in order."""
        records = [
            r
            for r in snapshot_hook.read_records(self.hooks_path)
            if isinstance(r.get("seq"), int) and r["seq"] >= self.next_seq
        ]
        if records:
            self.next_seq = max(int(r["seq"]) for r in records) + 1
        for record in records:
            blobs = record.get("blobs")
            if isinstance(blobs, Mapping):
                self.blobs.update({str(k): str(v) for k, v in blobs.items()})
            path = record.get("transcript_path")
            if isinstance(path, str) and path and record.get("session_id") == self.uuid:
                self.transcript = Path(path)
            if record.get("kind") == "error":
                self.hook_errors.append(str(record.get("error", "")))
        return records

    def snapshot(self) -> dict[str, str]:
        """The workspace now (the driver's own snapshot), merging new blobs."""
        fs, blobs, _ = snapshot_hook.snapshot_dir(self.ws.root)
        for digest, text in blobs.items():
            self.blobs.setdefault(digest, self.redact(text))
        return fs

    def transcript_path(self) -> Path | None:
        """The main conversation's transcript file, if known and present."""
        if self.transcript is not None and self.transcript.is_file():
            return self.transcript
        base = Path(self.cfg.env.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
        key = re.sub(r"[^A-Za-z0-9]", "-", str(self.ws.root))
        guess = base / "projects" / key / f"{self.uuid}.jsonl"
        return guess if guess.is_file() else None


def probe_failure(launch: Launch) -> str | None:
    """Why a probe launch produced no usable answer, or None if it did."""
    state = launch.state
    if launch.init is None:
        return f"no init event (exit {launch.returncode})"
    if state.api_error is not None:
        return f"API {state.api_error['kind']}"
    if launch.stop_reason == "timeout":
        return "stopped at the probe timeout"
    if state.result is None and launch.stop_reason is None:
        return f"no result event (exit {launch.returncode})"
    if state.result is not None and state.result.get("is_error"):
        return f"error result {state.result.get('subtype')!r}"
    return None


@contextmanager
def session_workspace(
    files: Mapping[str, str], *, parent: Path | None, repo_root: Path | None, prefix: str
) -> Iterator[Workspace]:
    """A disposable git workspace outside the repository, with no instruction files
    in any ancestor directory (Claude Code would load them).

    Raises:
        SafetyStop: if the workspace lies inside the repository or an ancestor
            directory has ``CLAUDE.md`` or ``CLAUDE.local.md``.
    """
    with Workspace(files, prefix=prefix, parent=parent) as ws:
        root = ws.root
        if repo_root is not None:
            repo = repo_root.resolve()
            if root == repo or repo in root.parents:
                raise SafetyStop(f"workspace {root} lies inside the repository")
        for ancestor in root.parents:
            for name in ("CLAUDE.md", "CLAUDE.local.md"):
                if (ancestor / name).exists():
                    raise SafetyStop(f"{ancestor / name} would load into the child")
        yield ws


# --------------------------------------------------------------------------- one session


@dataclass
class SessionOutcome:
    """What :meth:`SessionRunner.run` did.

    Attributes:
        record: The result record (None if the session was not saved).
        saved: Whether ``result.json`` was written.
        stop: Why the run must stop after this session (``usage_limit``,
            ``billing_error``, ``auth_error``, ``error``), or None.
        skipped: True if ``result.json`` already existed.
    """

    record: dict[str, Any] | None
    saved: bool
    stop: str | None = None
    skipped: bool = False


class SessionRunner:
    """Runs one :class:`SessionPlan` as one Claude Code conversation."""

    def __init__(self, plan: SessionPlan, template: Template, cfg: DriverConfig) -> None:
        self.plan = plan
        self.template = template
        self.cfg = cfg
        self.out_dir = cfg.out_root / cfg.run_id / plan.session_id
        self.rendered = {c.id: c.render(plan.phrasing) for c in template.constraints}
        self.rules = rules_text(plan, template)
        self.append = append_system_prompt(plan.placement, self.rules)
        self.tool_calls = 0
        self.events: list[ToolEvent] = []
        self.spans: list[SubtaskSpan] = []
        self.checks: list[HarnessCheck] = []
        self.probes: list[dict[str, Any]] = []
        self.probes_skipped: list[dict[str, Any]] = []
        self.compactions: list[dict[str, Any]] = []
        self.compaction_logs: dict[str, dict[str, Any]] = {}
        self.leaks: list[str] = []
        self.leak_scan = "not run"
        self.warnings: list[str] = []
        self.drift: list[dict[str, Any]] = []
        self.reminders: list[int] = []
        self.launches: list[dict[str, Any]] = []
        self.turns_max = 0
        self.last_statement = 0
        self.ended_reason = "completed"
        self.stop: str | None = None
        self.subtask_end: str | None = None
        self.account_error: dict[str, Any] | None = None
        self.api_error: dict[str, Any] | None = None
        self.probed: set[str] = set()
        self.conv: Conversation | None = None
        self.contaminants: list[str] = list(cfg.contaminants)
        self.prompts: dict[str, dict[str, Any]] = {}
        """System prompts seen in the transcript, by SHA-256 of their text."""
        self.prompt_launches: list[dict[str, Any]] = []
        """For each main launch, the SHA-256 of the latest recorded system prompt."""

    # ------------------------------------------------------------------ entry

    def run(self) -> SessionOutcome:
        """Run the session (or load it if done) and save its outputs.

        Raises:
            BillingGuardRefused: before any launch, if the guard is not clean.
            CapExceeded: if today's agent-run or single-call cap has no room.
            UnsupportedPlan: if the plan asks for something this driver cannot do.
            SafetyStop: if a child was not clean or not instrumented.
        """
        done = self.out_dir / "result.json"
        if done.exists():
            stored_plan = self.out_dir / "plan.json"
            stored = (
                json.loads(stored_plan.read_text(encoding="utf-8"))
                if stored_plan.is_file()
                else None
            )
            if stored != json.loads(json.dumps(self.plan.to_dict())):
                # D-050 item 6: never bank a result saved under one plan as another's.
                raise UnsupportedPlan(
                    f"{self.plan.session_id}: the stored plan differs from the plan being run "
                    f"(use a new run id, or remove {self.out_dir})"
                )
            loaded: dict[str, Any] = json.loads(done.read_text(encoding="utf-8"))
            return SessionOutcome(loaded, saved=True, skipped=True)
        require_clean_guard(self.cfg.guard)
        problems = plan_problems(self.plan, self.template, network_logging=self.cfg.network_logging)
        if problems:
            raise UnsupportedPlan(f"{self.plan.session_id}: " + "; ".join(problems))
        self._require_caps()
        remove_tree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        files = dict(self.template.files)
        if self.plan.placement == "pinned_file":
            files[PINNED_FILE] = pinned_file_text(self.rules)
        start = self.cfg.clock()
        with session_workspace(
            files,
            parent=self.cfg.workspace_parent,
            repo_root=self.cfg.repo_root,
            prefix="halflife-cc-",
        ) as ws:
            conv = Conversation(
                self.cfg,
                workspace=ws,
                out_dir=self.out_dir,
                append=self.append,
                session_id=self.plan.session_id,
                experiment=self.plan.experiment,
            )
            self.conv = conv
            try:
                self._baseline_checks(ws)
                self._loop(ws)
                self._leak_scan()
            finally:
                if conv.main_launches:
                    self._agent_run_row(self.cfg.clock() - start)
            log = ActionLog(
                session_id=self.plan.session_id,
                template_id=self.plan.template_id,
                constraints=self.template.constraints,
                baseline_fs=dict(ws.baseline),
                blobs=dict(sorted(conv.blobs.items())),
                events=list(self.events),
                checks=list(self.checks),
                spans=list(self.spans),
            )
        if self.stop is not None and not self.spans:
            remove_tree(self.out_dir)
            return SessionOutcome(None, saved=False, stop=self.stop)
        record = self._save(log, self.cfg.clock() - start)
        return SessionOutcome(record, saved=True, stop=self.stop)

    def _require_caps(self) -> None:
        cfg = self.cfg
        caps = cfg.caps
        runs = used_today(cfg.ledger_path, "claude_agent_run")
        if runs >= caps.agent_runs_per_day:
            raise CapExceeded(
                f"claude_agent_runs_per_day = {caps.agent_runs_per_day} already used ({runs})"
            )
        probes = 0
        if self.plan.probe_before_dp:
            probes = sum(1 for i, s in enumerate(self.plan.subtasks) if s.is_dp and i > 0)
        singles = used_today(cfg.ledger_path, "claude_single_call")
        if singles + probes > caps.single_calls_per_day:
            raise CapExceeded(
                f"claude_single_calls_per_day = {caps.single_calls_per_day}: {singles} used, "
                f"this session needs {probes} probes"
            )

    # ------------------------------------------------------------------ the loop

    def _baseline_checks(self, ws: Workspace) -> None:
        for spec in self.template.checks:
            code = self.cfg.check(ws, spec.cmd)
            self.checks.append(HarnessCheck(spec.id, BASELINE_SUBTASK, -1, code))

    def _message(self, number: int, subtask: Subtask) -> str:
        if number == 0:
            return first_message(self.plan.placement, self.rules, subtask.prompt)
        text = subtask.prompt
        every = self.plan.reminder_every
        if self.plan.placement == "reminder" and self.tool_calls - self.last_statement >= every:
            text = f"{text}\n\n{reminder_text(self.rules)}"
            self.last_statement = self.tool_calls
            self.reminders.append(self.tool_calls)
        return text

    def _loop(self, ws: Workspace) -> None:
        plan = self.plan
        for number, subtask in enumerate(plan.subtasks):
            if self.tool_calls >= plan.max_tool_calls:
                self.ended_reason = "tool_cap"
                break
            if number > 0:
                self._boundary_probes(subtask)
            elif subtask.is_dp and plan.probe_before_dp:
                self.probes_skipped.append(
                    {"checkpoint": f"before_dp:{subtask.id}", "reason": "no conversation yet"}
                )
            if self.stop is not None:
                break
            start = self.tool_calls
            self.subtask_end = None
            turns = self._subtask(ws, subtask, self._message(number, subtask))
            if self.stop is not None and turns == 0:
                # The model never answered this message, so the subtask was not reached.
                break
            success = None
            if subtask.success is not None:
                success = self.cfg.check(ws, subtask.success.cmd) == subtask.success.expect_exit
            self.spans.append(
                SubtaskSpan(
                    subtask.id,
                    subtask.kind,
                    start,
                    self.tool_calls,
                    constraint=subtask.constraint,
                    success=success,
                    ended=self.subtask_end,
                )
            )
            for spec in self.template.checks:
                code = self.cfg.check(ws, spec.cmd)
                self.checks.append(HarnessCheck(spec.id, subtask.id, self.tool_calls - 1, code))
            if self.stop is not None:
                break
        if self.stop is None:
            self._compaction_probes()

    def _subtask(self, ws: Workspace, subtask: Subtask, message: str) -> int:
        """Launch one subtask message; return the assistant turns it produced.

        Sets :attr:`subtask_end` to how the message ended (DECISIONS.md D-050, item 5):
        ``done`` when the CLI finished the turn itself, the stop reason that cut it
        (``turn_cap``, ``subtask_tool_cap``, ``session_tool_cap``, ``timeout``), or
        ``error`` for an account problem or a failed launch.
        """
        conv = self._conv()
        limits = Limits(
            turn_cap=self.cfg.caps.max_turns,
            subtask_calls=self.plan.max_calls_per_subtask,
            session_calls=self.plan.max_tool_calls - self.tool_calls,
        )
        before = self.events[-1].fs_after if self.events else ws.baseline
        launch = conv.launch(
            purpose="subtask",
            message=message,
            limits=limits,
            first_index=self.tool_calls,
            subtask_id=subtask.id,
        )
        self._absorb(launch, subtask, before)
        state = launch.state
        result = state.result or {}
        if state.account_error is not None:
            self.account_error = state.account_error
            self.stop = self.ended_reason = str(state.account_error["kind"])
            self.subtask_end = "error"
            return state.turns
        error: str | None = None
        if launch.init is None or (state.result is None and launch.stop_reason is None):
            error = f"error: launch {launch.number} exited {launch.returncode}"
        elif state.api_error is not None:
            self.api_error = state.api_error
            error = f"error: API {state.api_error['kind']} in launch {launch.number}"
        elif result.get("is_error") and launch.stop_reason is None:
            error = f"error: launch {launch.number} ended with {result.get('subtype')!r}"
        if error is not None:
            self.stop = "error"
            self.ended_reason = error
            self.warnings.append(error)
        elif launch.stop_reason == "timeout":
            self.warnings.append(f"subtask {subtask.id} stopped at the launch timeout")
        self.subtask_end = "error" if error is not None else (launch.stop_reason or "done")
        return state.turns

    def _absorb(self, launch: Launch, subtask: Subtask, before: Mapping[str, str]) -> None:
        conv = self._conv()
        state = launch.state
        records = conv.new_records()
        if conv.hook_errors:
            raise SafetyStop(f"snapshot hook failed: {conv.hook_errors[-1]}")
        if state.hook_failures:
            raise SafetyStop(f"a hook exited with an error in launch {launch.number}")
        tool_records = [r for r in records if r.get("kind") == "tool"]
        for record in records:
            if record.get("kind") == "instructions":
                loaded = instruction_load(record)
                conv.instructions.append(
                    loaded | {"launch": launch.number, "tool_call_index": state.first_index}
                )
                if not loaded["in_root"]:
                    self._contaminant(foreign_instruction(loaded))
        matched, leftover = align_records(state.tool_uses, tool_records)
        succeeded = [u for u in state.tool_uses if u.has_result and not u.is_error]
        if succeeded and not any(u.tool_use_id in matched for u in succeeded):
            raise SafetyStop(
                f"launch {launch.number} ran tool calls but no snapshot hook fired: the "
                "--settings file was not applied, so the sandbox may have been off"
            )
        missing = [
            u.tool_use_id
            for u in state.tool_uses
            if u.has_result and not u.denied and u.tool_use_id not in matched
        ]
        if missing:
            self.warnings.append(f"launch {launch.number}: no hook record for {missing}")
        if leftover:
            self.warnings.append(f"launch {launch.number}: {len(leftover)} unmatched hook records")
        end = conv.snapshot()
        events, drift = tool_events(
            state.tool_uses,
            matched,
            subtask_id=subtask.id,
            fs_before=before,
            fs_end=end,
            roots=conv.roots,
            redact=conv.redact,
        )
        if drift:
            self.drift.append({"launch": launch.number, "paths": drift})
        self.events.extend(events)
        self.tool_calls = state.next_index
        self.turns_max = max(self.turns_max, state.turns)
        entries = read_transcript(conv.transcript_path())
        self._track_prompts(launch, entries)
        summaries = compact_summaries(entries)
        for seen in state.compactions:
            if seen.summary is None and len(self.compactions) < len(summaries):
                seen.summary = summaries[len(self.compactions)]
                seen.summary_source = "transcript"
            self._compaction(seen, launch, subtask)
        self.launches.append(
            {
                "launch": launch.number,
                "subtask_id": subtask.id,
                "turns": state.turns,
                "tool_calls": state.calls,
                "stop_reason": launch.stop_reason,
                "overrun": state.overrun,
            }
        )

    def _compaction(self, seen: CompactionSeen, launch: Launch, subtask: Subtask) -> None:
        event_id = f"c{len(self.compactions) + 1}"
        retained = {
            cid: (text in seen.summary) if seen.summary is not None else None
            for cid, text in self.rendered.items()
        }
        log = {
            "event_id": event_id,
            "strategy": COMPACTION_STRATEGY,
            "subtype": seen.subtype,
            "trigger": seen.trigger,
            "tool_call_index": seen.tool_call_index,
            "subtask_id": subtask.id,
            "launch": launch.number,
            "pre_tokens": seen.pre_tokens,
            "post_tokens": seen.post_tokens,
            "summary": seen.summary,
            "summary_source": seen.summary_source,
            "constraint_text_retained": retained,
            "autocompact_tokens": AUTOCOMPACT_TOKENS,
            "event": seen.event,
        }
        self.compaction_logs[event_id] = log
        self.compactions.append(
            {
                "run_id": self.cfg.run_id,
                "session_id": self.plan.session_id,
                "event_id": event_id,
                "strategy": COMPACTION_STRATEGY,
                "tool_call_index": seen.tool_call_index,
                "scheduled": False,
                "trigger": seen.trigger,
                "tokens_before": seen.pre_tokens,
                "tokens_after": seen.post_tokens,
                "constraint_text_retained": retained,
                "log_path": f"compactions/{event_id}.json",
            }
        )

    # ------------------------------------------------------------------ probes

    def _boundary_probes(self, subtask: Subtask) -> None:
        self._compaction_probes()
        if subtask.is_dp and self.plan.probe_before_dp and self.stop is None:
            self._probe(f"before_dp:{subtask.id}", {"dp_id": subtask.id})

    def _compaction_probes(self) -> None:
        if not self.plan.probe_after_compaction:
            return
        for event in self.compactions:
            event_id = str(event["event_id"])
            if event_id in self.probed or self.stop is not None:
                continue
            self.probed.add(event_id)
            self._probe(
                f"after_compaction:{event_id}",
                {"compaction_tool_call_index": event["tool_call_index"]},
            )

    def _probe(self, checkpoint: str, extra: Mapping[str, Any]) -> None:
        conv = self._conv()
        cfg = self.cfg
        if conv.main_launches == 0:
            self.probes_skipped.append({"checkpoint": checkpoint, "reason": "no conversation"})
            return
        used = used_today(cfg.ledger_path, "claude_single_call")
        if used >= cfg.caps.single_calls_per_day:
            self.probes_skipped.append(
                {"checkpoint": checkpoint, "reason": "claude_single_calls_per_day reached"}
            )
            self.warnings.append(f"probe {checkpoint} skipped at the single-call cap")
            return
        transcript = conv.transcript_path()
        before = file_sha256(transcript)
        launch = conv.launch(
            purpose="probe",
            message=PROBE_QUESTION,
            limits=Limits(turn_cap=1, subtask_calls=1, session_calls=1),
            first_index=self.tool_calls,
            checkpoint=checkpoint,
            ledger_purpose="probe",
        )
        state = launch.state
        after = file_sha256(transcript)
        if transcript is None:
            leak_check = "unverified: main transcript not found"
        else:
            leak_check = "verified"
            if before != after:
                self.leaks.append(f"probe {checkpoint} changed the main transcript")
        if state.account_error is not None:
            self.account_error = state.account_error
            self.stop = self.ended_reason = str(state.account_error["kind"])
            return
        failure = probe_failure(launch)
        if failure is not None:
            # A failed fork says nothing about recall, so it is not scored as "nothing recalled".
            self.probes_skipped.append({"checkpoint": checkpoint, "reason": f"failed: {failure}"})
            self.warnings.append(f"probe {checkpoint} failed: {failure}")
            return
        answer = conv.redact(state.final_text().strip())
        scored = unfence(answer)
        result = ProbeResult(
            checkpoint=checkpoint,
            tool_call_index=self.tool_calls,
            answer=answer,
            recalled={c.id: recalled_in(scored, c) for c in self.template.constraints},
            context_sha256=before or "",
        )
        self.probes.append(
            result.to_dict()
            | dict(extra)
            | {
                "fork_session_id": (launch.init or {}).get("session_id"),
                "launch": launch.number,
                "leak_check": leak_check,
                "turns": state.turns,
            }
        )

    # ------------------------------------------------------------------ provenance

    def _leak_scan(self) -> None:
        """Search the whole main transcript for probe material once the session is over."""
        transcript = self._conv().transcript_path()
        if transcript is None:
            self.leak_scan = "unverified: main transcript not found"
            return
        answers = [str(p.get("answer", "")) for p in self.probes]
        for what in transcript_leaks(read_transcript(transcript), answers):
            self.leaks.append(f"main transcript contains {what}")
        self.leak_scan = "verified"

    def _contaminant(self, text: str) -> None:
        """Record a known contaminant once; warn if the smoke run did not report it."""
        if text in self.contaminants:
            return
        self.contaminants.append(text)
        self.warnings.append(f"contaminant not reported by the smoke run: {text}")

    def _clean(self, text: str) -> str:
        """Redact secrets and show the home directory as ``~`` (for saved prompt text)."""
        text = self._conv().redact(text)
        for home in {self.cfg.env.get("HOME", ""), str(Path.home())}:
            if home not in ("", "/"):
                text = text.replace(home, "~")
        return text

    def _track_prompts(self, launch: Launch, entries: Sequence[Mapping[str, Any]]) -> None:
        """Record the system prompt the CLI recorded and sent for this launch."""
        latest: str | None = None
        for blocks in prompt_snapshots(entries):
            text = [self._clean(block) for block in blocks]
            digest = sha256_text(json.dumps(text, ensure_ascii=False))
            if digest not in self.prompts:
                self.prompts[digest] = {
                    "sha256": digest,
                    "first_launch": launch.number,
                    "system_prompt": text,
                }
            latest = digest
        self.prompt_launches.append({"launch": launch.number, "prompt_sha256": latest})

    def _system_prompt(self) -> dict[str, Any]:
        """Provenance of the system prompt in effect, and the checks made on it."""
        joined = ["\n".join(p["system_prompt"]) for p in self.prompts.values()]
        rules_in = [self.rules in text for text in joined]
        if not self.prompts:
            self.warnings.append(
                "no prompt_snapshot entry in the transcript: the system prompt in effect is "
                "known only from the placement and the appended text"
            )
        elif self.append is not None and not all(self.append in text for text in joined):
            self.warnings.append("a recorded system prompt lacks the --append-system-prompt text")
        elif self.plan.placement in {"first_user", "reminder"} and any(rules_in):
            self.warnings.append(
                f"the rules appear in the system prompt under {self.plan.placement} placement"
            )
        return {
            "base": f"Claude Code default system prompt (CLI {self.cfg.cli.version})",
            "snapshot": "on: recorded on the first request, reused until compaction",
            "placement": self.plan.placement,
            "append": self.append,
            "append_sha256": sha256_text(self.append) if self.append is not None else None,
            "pinned_file": PINNED_FILE if self.plan.placement == "pinned_file" else None,
            "recorded": SYSTEM_PROMPT_FILE if self.prompts else None,
            "recorded_sha256": sorted(self.prompts),
            "rules_in_recorded": rules_in,
            "launches": self.prompt_launches,
        }

    # ------------------------------------------------------------------ outputs

    def _conv(self) -> Conversation:
        if self.conv is None:
            raise DriverError("session has no conversation yet")
        return self.conv

    def _agent_run_row(self, wall: float) -> None:
        """One ``claude_agent_run`` row for the session's own launches (probes have theirs)."""
        conv = self._conv()
        append_ledger(
            self.cfg.ledger_path,
            {
                "ts": utc_stamp(),
                "kind": "claude_agent_run",
                "backend": LEDGER_BACKEND,
                "model": self.cfg.model,
                "model_id": conv.model_ids[-1] if conv.model_ids else None,
                "run_id": self.cfg.run_id,
                "experiment": self.plan.experiment,
                "session_id": self.plan.session_id,
                "purpose": "session",
                "turns": self.turns_max,
                "launches": conv.main_launches,
                "tool_calls": self.tool_calls,
                "ended_reason": self.ended_reason,
                **usage_fields(conv.usage),
                "wall_s": round(wall, 3),
                "cli_version": self.cfg.cli.version,
            },
        )

    def model_id(self) -> str | None:
        """The exact model id the CLI reported first (None before any launch reported one)."""
        conv = self.conv
        return conv.model_ids[0] if conv is not None and conv.model_ids else None

    def _extras(self) -> dict[str, Any]:
        """Fields added to every session and opportunity row.

        ``model`` in the rows is the tier (docs/ARTIFACTS.md, G8); the exact id
        the CLI reported is ``model_id``.
        """
        return {
            "driver": "claude_code",
            "label": LABEL,
            "tier": self.cfg.model,
            "model_id": self.model_id(),
            "cli_version": self.cfg.cli.version,
            "strategy": COMPACTION_STRATEGY,
            "plan_strategy": self.plan.compaction.strategy,
            "turn_cap": self.cfg.caps.max_turns,
            "contaminated": bool(self.contaminants),
        }

    def _save(self, log: ActionLog, wall: float) -> dict[str, Any]:
        conv = self._conv()
        cfg = self.cfg
        out = self.out_dir
        violations = detect_all(log)
        facts = RunFacts(
            family=FAMILY,
            model=cfg.model,
            compaction_indices=tuple(int(c["tool_call_index"]) for c in self.compactions),
            probes=tuple(self.probes),
            ended_reason=self.ended_reason,
            prereg=cfg.prereg,
        )
        extras = self._extras()
        _write_json(out / "plan.json", self.plan.to_dict())
        (out / "action_log.json").write_text(log.dumps() + "\n", encoding="utf-8")
        compactions = []
        for event in self.compactions:
            path = out / str(event["log_path"])
            _write_json(path, self.compaction_logs[str(event["event_id"])])
            compactions.append(dict(event) | {"log_sha256": file_sha256(path)})
        _write_json(out / "probes.json", self.probes)
        _write_json(out / "violations.json", [v.to_dict() for v in violations])
        # Claude Code compacts inside the subtask launches, so "step" covers compaction too.
        _write_json(
            out / "usage.json",
            {"step": purpose_usage(conv.usage), "probe": purpose_usage(conv.probe_usage)},
        )
        distinct_models = sorted(set(conv.model_ids))
        if len(distinct_models) > 1:
            self.warnings.append(f"the CLI reported several models {distinct_models}")
        pinned_seen = any(
            i.get("in_root") and str(i.get("file_path", "")).endswith(PINNED_FILE)
            for i in conv.instructions
        )
        if self.plan.placement == "pinned_file" and not pinned_seen:
            self.warnings.append("no InstructionsLoaded record for the pinned CLAUDE.md")
        system_prompt = self._system_prompt()
        if self.prompts:
            _write_json(
                out / SYSTEM_PROMPT_FILE,
                {
                    "note": (
                        "System prompt text Claude Code recorded for this conversation "
                        "(prompt_snapshot entries of its transcript), secrets redacted and "
                        "the home directory shown as ~."
                    ),
                    "placement": self.plan.placement,
                    "append": self.append,
                    "snapshots": list(self.prompts.values()),
                    "launches": self.prompt_launches,
                },
            )
        record = {
            "run_id": cfg.run_id,
            "summary": {
                "session_id": self.plan.session_id,
                "ended_reason": self.ended_reason,
                "tool_calls": len(log.events),
                "compactions": len(self.compactions),
                "probes": len(self.probes),
                "probes_skipped": self.probes_skipped,
                "leaks": list(self.leaks),
                "leak_scan": self.leak_scan,
                "wall_s": round(wall, 2),
                "model_calls": conv.started,
                "launches": conv.main_launches,
                "turns_max": self.turns_max,
                "turn_cap": cfg.caps.max_turns,
            },
            "session": session_row(self.plan, log, violations, facts) | extras,
            "opportunities": [
                row | extras for row in opportunity_rows(self.plan, log, violations, facts)
            ],
            "compactions": compactions,
            "settings": {
                "driver": DRIVER_CONTRACT,
                "model": cfg.model,
                "tools": list(CLAUDE_TOOLS),
                "permission_mode": PERMISSION_MODE,
                "sandbox": dict(SANDBOX_SETTINGS),
                "autocompact_tokens": AUTOCOMPACT_TOKENS,
                "turn_cap": cfg.caps.max_turns,
                "cap_share": cfg.caps.share,
                "max_tool_calls": self.plan.max_tool_calls,
                "max_calls_per_subtask": self.plan.max_calls_per_subtask,
                "launch_timeout_s": cfg.launch_timeout_s,
                "probe_timeout_s": cfg.probe_timeout_s,
                "harness_sandbox": cfg.sandbox_method,
            },
            "claude": {
                "cli": cfg.cli.to_dict(),
                "session_uuid": conv.uuid,
                "model_ids": distinct_models,
                "init": conv.inits[0] if conv.inits else None,
                "launches": self.launches,
            },
            "system_prompt": system_prompt,
            "instructions_loaded": conv.instructions,
            "contaminants": list(self.contaminants),
            "reminders": self.reminders,
            "usage": {"session": dict(conv.usage), "probes": dict(conv.probe_usage)},
            "account_error": self.account_error,
            "api_error": self.api_error,
            "fs_drift": self.drift,
            "warnings": self.warnings,
        }
        _write_json(out / "result.json", record)
        return record


# --------------------------------------------------------------------------- smoke


def smoke_checks(
    launch: Launch,
    matched: Mapping[str, Mapping[str, Any]],
    *,
    probe_file: Path,
    netlog_text: str,
    instructions: Sequence[Mapping[str, Any]],
    expected_python: str,
    expected_curl: str,
    subject_env: bool,
) -> dict[str, dict[str, Any]]:
    """The smoke run's checks, each ``{"ok": bool, "detail": str, "required": bool}``.

    Required checks decide whether the smoke passes. ``network_logging``,
    ``instructions_observed``, and ``instructions_clean`` are reported but do
    not: without network logging, plans with ``no_network`` rules are refused;
    an instruction file from outside the workspace (for example a user-level
    ``CLAUDE.md``) is a known contaminant that ``run`` makes the operator
    acknowledge (section 8 of the brief: record it, never edit the user's files).
    """
    state = launch.state
    init = launch.init or {}
    problems = init_problems(init, CLAUDE_TOOLS) if launch.init else ["no init event"]
    bash = [u for u in state.tool_uses if u.name == "Bash" and u.has_result]
    output = "\n".join(u.result_text or "" for u in bash)
    ran = any(SMOKE_MARKER in (u.result_text or "") for u in bash)
    lines = [line.strip() for line in output.splitlines()]
    executed = [u for u in state.tool_uses if u.has_result and not u.denied]
    hooks_ok = bool(executed) and all(u.tool_use_id in matched for u in executed)
    foreign = [i for i in instructions if not i.get("in_root")]
    own = [i for i in instructions if i.get("in_root")]
    checks: dict[str, dict[str, Any]] = {
        "init_clean": {
            "ok": not problems,
            "detail": "; ".join(problems) or "tools, MCP servers, plugins, mode, auth as set",
            "required": True,
        },
        "command_ran": {
            "ok": ran and SMOKE_END in output,
            "detail": f"{len(bash)} Bash call(s); marker {'found' if ran else 'missing'}",
            "required": True,
        },
        "hooks_loaded": {
            "ok": hooks_ok,
            "detail": f"{len(matched)} hook record(s) for {len(executed)} executed call(s)",
            "required": True,
        },
        "write_confined": {
            "ok": ran and not probe_file.exists(),
            "detail": "write outside the workspace was blocked"
            if ran and not probe_file.exists()
            else "the sandbox did not block a write outside the workspace, or it never ran",
            "required": True,
        },
        "subject_path": {
            "ok": expected_python in lines and expected_curl in lines,
            "detail": "python and curl resolve to the subject environment and the shims"
            if expected_python in lines and expected_curl in lines
            else "shell startup files change PATH inside the child",
            "required": True,
        },
        "git_isolated": {
            "ok": "/dev/null" in lines,
            "detail": "GIT_CONFIG_GLOBAL=/dev/null inside the child",
            "required": True,
        },
        "instructions_clean": {
            "ok": not foreign,
            "detail": "no instruction file from outside the workspace loaded"
            if not foreign
            else "loaded from outside the workspace (a known contaminant to record): "
            + ", ".join(f"{i.get('memory_type')}:{i.get('file_path')}" for i in foreign),
            "required": False,
        },
        "instructions_observed": {
            "ok": bool(own),
            "detail": "the InstructionsLoaded hook reported the workspace CLAUDE.md"
            if own
            else "no InstructionsLoaded record, so instruction loads cannot be verified",
            "required": False,
        },
        "subject_env": {
            "ok": subject_env,
            "detail": "commands use the subject environment from make subject-env"
            if subject_env
            else "no subject environment; run make subject-env (DECISIONS.md D-025)",
            "required": True,
        },
        "network_logging": {
            "ok": "example.invalid" in netlog_text,
            "detail": "the curl shim logged the attempt"
            if "example.invalid" in netlog_text
            else "no network log line; no_network rules cannot be measured",
            "required": False,
        },
        "turns_within_cap": {
            "ok": not state.overrun,
            "detail": f"{state.turns} turn(s)",
            "required": True,
        },
    }
    return checks


def run_smoke(cfg: DriverConfig, *, host_home: Path | None = None) -> dict[str, Any]:
    """One tiny session that checks the child is clean and instrumented.

    Writes ``runs/<run_id>/smoke.json`` and returns it. One agent run and one
    single call go into the ledger.

    Raises:
        BillingGuardRefused: if the billing guard is not clean.
        CapExceeded: if today's caps have no room.
    """
    require_clean_guard(cfg.guard)
    runs = used_today(cfg.ledger_path, "claude_agent_run")
    singles = used_today(cfg.ledger_path, "claude_single_call")
    if runs >= cfg.caps.agent_runs_per_day or singles >= cfg.caps.single_calls_per_day:
        raise CapExceeded("no room under today's Claude caps for a smoke run")
    run_dir = cfg.out_root / cfg.run_id
    out_dir = run_dir / "smoke"
    remove_tree(out_dir)
    outside = out_dir / "outside"
    outside.mkdir(parents=True)
    probe_file = outside / "sandbox-probe.txt"
    home = host_home or Path.home()
    files = {"README.md": "# Smoke\n", "hello.py": "print('hello')\n", PINNED_FILE: SMOKE_CLAUDE_MD}
    start = cfg.clock()
    findings: dict[str, Any] = {}
    with session_workspace(
        files, parent=cfg.workspace_parent, repo_root=cfg.repo_root, prefix="halflife-cc-smoke-"
    ) as ws:
        conv = Conversation(
            cfg,
            workspace=ws,
            out_dir=out_dir,
            append=None,
            session_id="smoke",
            experiment="smoke",
        )
        launch: Launch | None = None
        error: str | None = None
        try:
            launch = conv.launch(
                purpose="smoke",
                message=smoke_prompt(probe_file),
                limits=Limits(turn_cap=cfg.caps.max_turns, subtask_calls=4, session_calls=4),
                first_index=0,
                subtask_id="smoke",
            )
        except SafetyStop as exc:
            error = str(exc)
        finally:
            if conv.started:
                _smoke_run_row(cfg, conv, cfg.clock() - start)
        records = conv.new_records()
        if launch is not None:
            tool_records = [r for r in records if r.get("kind") == "tool"]
            matched, _ = align_records(launch.state.tool_uses, tool_records)
            instructions = [instruction_load(r) for r in records if r.get("kind") == "instructions"]
            checks = smoke_checks(
                launch,
                matched,
                probe_file=probe_file,
                netlog_text=conv.netlog.read_text(encoding="utf-8", errors="replace"),
                instructions=instructions,
                expected_python=str(subject_bin(cfg.env) / "python"),
                expected_curl=str(shims_dir() / "curl"),
                subject_env=has_subject_env(cfg.env),
            )
            changed = diff_snapshots(ws.baseline, conv.snapshot()).changed
            findings = {
                "instructions_loaded": instructions,
                "workspace_changes": list(changed),
                "account_error": launch.state.account_error,
                "turns": launch.state.turns,
            }
            if launch.state.account_error is None and checks["command_ran"]["ok"]:
                checks["probe_fork"] = _smoke_probe(conv)
        else:
            checks = {"launch": {"ok": False, "detail": error or "no launch", "required": True}}
        findings["inits"] = list(conv.inits)
    with suppress(OSError):
        probe_file.unlink()
    required = [c for c in checks.values() if c["required"]]
    user_claude_md = (home / ".claude" / "CLAUDE.md").exists()
    contaminants = [
        f"{name}: {c['detail']}"
        for name, c in checks.items()
        if name in {"init_clean", "subject_path", "launch"} and not c["ok"]
    ]
    contaminants += [
        foreign_instruction(i) for i in findings.get("instructions_loaded", []) if not i["in_root"]
    ]
    if user_claude_md and not checks.get("instructions_observed", {}).get("ok", False):
        contaminants.append(
            "possible: ~/.claude/CLAUDE.md exists and instruction loads were not observable"
        )
    report = {
        "driver": DRIVER_CONTRACT,
        "run_id": cfg.run_id,
        "timestamp": utc_stamp(),
        "model": cfg.model,
        "cli": cfg.cli.to_dict(),
        "cli_version": cfg.cli.version,
        "passed": bool(required) and all(c["ok"] for c in required),
        "network_logging": bool(checks.get("network_logging", {}).get("ok", False)),
        "checks": checks,
        "contaminants": contaminants,
        "host": {
            "user_claude_md_exists": user_claude_md,
            "user_settings_exists": (home / ".claude" / "settings.json").exists(),
        },
        "findings": findings,
        "usage": {"session": dict(conv.usage), "probes": dict(conv.probe_usage)},
        "wall_s": round(cfg.clock() - start, 2),
    }
    _write_json(run_dir / "smoke.json", report)
    return report


def has_subject_env(env: Mapping[str, str]) -> bool:
    """Whether commands use a subject environment, not the harness interpreter (D-025)."""
    path = subject_bin(env)
    return bool(env.get("HALFLIFE_SUBJECT_BIN")) or path == SUBJECT_BIN_DEFAULT


def _smoke_run_row(cfg: DriverConfig, conv: Conversation, wall: float) -> None:
    mains = [r for r in _read_jsonl(conv.commands_path) if r.get("purpose") != "probe"]
    append_ledger(
        cfg.ledger_path,
        {
            "ts": utc_stamp(),
            "kind": "claude_agent_run",
            "backend": LEDGER_BACKEND,
            "model": cfg.model,
            "model_id": conv.model_ids[-1] if conv.model_ids else None,
            "run_id": cfg.run_id,
            "experiment": "smoke",
            "session_id": "smoke",
            "purpose": "smoke",
            "turns": max((int(r.get("turns", 0)) for r in mains), default=0),
            "launches": conv.main_launches,
            **usage_fields(conv.usage),
            "wall_s": round(wall, 3),
            "cli_version": cfg.cli.version,
        },
    )


def _smoke_probe(conv: Conversation) -> dict[str, Any]:
    transcript = conv.transcript_path()
    before = file_sha256(transcript)
    try:
        launch = conv.launch(
            purpose="probe",
            message=PROBE_QUESTION,
            limits=Limits(turn_cap=1, subtask_calls=1, session_calls=1),
            first_index=0,
            checkpoint="smoke",
            ledger_purpose="smoke_probe",
        )
    except SafetyStop as exc:
        return {"ok": False, "detail": str(exc), "required": True}
    after = file_sha256(transcript)
    answer = launch.state.final_text().strip()
    failure = probe_failure(launch)
    untouched = transcript is not None and before == after
    ok = bool(answer) and untouched and failure is None
    detail = "fork answered; main transcript unchanged" if ok else "fork failed or left no proof"
    if failure is not None:
        detail = f"fork failed: {failure}"
    elif transcript is None:
        detail = "main transcript not found, so the fork could not be verified"
    return {"ok": ok, "detail": detail, "required": True, "answer": answer[:1000]}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [r for r in (parse_line(line) for line in lines) if r is not None]


# --------------------------------------------------------------------------- command line


def repo_root() -> Path:
    """The repository root (``src/halflife/drivers/`` is three levels below it)."""
    return Path(__file__).resolve().parents[3]


def resolve_claude(env: Mapping[str, str], explicit: str | None = None) -> str | None:
    """Absolute path of the ``claude`` executable on ``PATH`` (or ``explicit``)."""
    if explicit:
        return explicit
    found = shutil.which("claude", path=env.get("PATH"))
    return str(Path(found).resolve()) if found else None


def load_smoke(path: Path, cli: CliInfo) -> tuple[dict[str, Any] | None, str]:
    """A passing smoke report for this CLI version and driver contract, or why not."""
    if not path.is_file():
        return None, f"no smoke report at {path}; run the smoke command first"
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("passed") is not True:
        return None, f"the smoke report at {path} did not pass"
    if report.get("driver") != DRIVER_CONTRACT:
        return None, f"the smoke report is for {report.get('driver')}, not {DRIVER_CONTRACT}"
    if report.get("cli_version") != cli.version:
        return None, f"the smoke ran on CLI {report.get('cli_version')}, now {cli.version}"
    return report, ""


@contextmanager
def driver_lock(out_root: Path) -> Iterator[bool]:
    """Hold an exclusive lock under ``out_root`` for one command; yields False if it is taken.

    Two driver processes would each count the same ledger rows and could pass
    a daily cap together, so only one may run at a time.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / LOCK_NAME).open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _git_sha(root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or None


@dataclass
class Setup:
    """What both subcommands prepare before any child runs."""

    cfg: DriverConfig
    root: Path


def prepare(
    args: argparse.Namespace,
    *,
    root: Path,
    launcher: Launcher,
    env: Mapping[str, str],
    guard: Callable[[], billing_guard.GuardReport],
    claude_bin: str | None,
) -> Setup:
    """Billing guard, model-call switch, CLI preflight, caps, and the config.

    Raises:
        BillingGuardRefused: first, before anything else, if the guard is not clean.
        PreflightError: if the CLI is missing or lacks a required flag.
        CapExceeded: if the caps cannot be read or round to nothing.
    """
    require_clean_guard(guard)
    if env_flag(env, NO_MODEL_CALLS_ENV) and not launcher.offline_safe:
        raise PreflightError(f"{NO_MODEL_CALLS_ENV}=1 forbids launching Claude children")
    path = resolve_claude(env, claude_bin)
    if path is None:
        raise PreflightError("the claude CLI is not on PATH")
    probe_env = {k: v for k, v in env.items() if k not in PARENT_SESSION_MARKERS}
    cli = preflight(launcher, path, child_env(probe_env, netlog=Path(os.devnull)))
    caps = Caps.load(root / "budget.toml", env)
    cfg = DriverConfig(
        run_id=args.run_id,
        model=args.model,
        claude_bin=path,
        cli=cli,
        caps=caps,
        out_root=root / args.out_root,
        ledger_path=root / "artifacts" / "usage" / "ledger.jsonl",
        launcher=launcher,
        guard=guard,
        env=dict(env),
        repo_root=root,
        launch_timeout_s=args.launch_timeout_s,
    )
    return Setup(cfg, root)


def cmd_smoke(setup: Setup) -> int:
    """Run the smoke session and print one line per check."""
    report = run_smoke(setup.cfg)
    for name, check in report["checks"].items():
        mark = "ok" if check["ok"] else ("FAIL" if check["required"] else "warn")
        print(f"{name}: {mark} ({check['detail']})")
    print(f"SMOKE: {'PASS' if report['passed'] else 'FAIL'} -> {setup.cfg.run_id}/smoke.json")
    return EXIT_OK if report["passed"] else EXIT_SAFETY


def cmd_run(setup: Setup, args: argparse.Namespace, *, checker: Checker | None = None) -> int:
    """Run plans in order (resumable); see the module docstring."""
    from halflife import bench
    from halflife.experiment import check_plans
    from halflife.runner import load_plans
    from halflife.sandbox import detect_method

    cfg, root = setup.cfg, setup.root
    smoke_path = root / args.smoke
    smoke, why = load_smoke(smoke_path, cfg.cli)
    if smoke is None:
        print(f"refusing: {why}")
        return EXIT_REFUSED
    contaminants = [str(c) for c in smoke.get("contaminants") or []]
    if contaminants and not args.accept_contaminants:
        for line in contaminants:
            print(f"known contaminant: {line}")
        print(
            "refusing: the smoke run found configuration from outside the workspace. Record it "
            "as a known contaminant in DECISIONS.md (section 8 of the brief; never edit or "
            "delete the user's files), then rerun with --accept-contaminants."
        )
        return EXIT_REFUSED
    cfg.contaminants = contaminants
    method = args.sandbox_method or detect_method()
    if method == "none" and checker is None:
        print("refusing: no sandbox method for harness checks (DECISIONS.md D-019)")
        return EXIT_REFUSED
    cfg.sandbox_method = method
    cfg.checker = checker
    cfg.network_logging = bool(smoke.get("network_logging"))
    cfg.prereg = args.prereg
    plans = load_plans(root / args.plans)
    if args.index is not None:
        plans = plans[args.index : args.index + 1]
    if args.limit is not None:
        plans = plans[: args.limit]
    templates = {t.id: t for t in bench.load_all()}
    problems = check_plans(plans, templates)
    for plan in plans:
        item = templates.get(plan.template_id)
        if item is not None:
            problems += [
                f"{plan.session_id}: {p}"
                for p in plan_problems(plan, item.template, network_logging=cfg.network_logging)
            ]
    if problems:
        for problem in problems:
            print(f"refusing: {problem}")
        return EXIT_REFUSED
    run_dir = cfg.out_root / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pid_file = run_dir / "pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    record: dict[str, Any] = {
        "run_id": cfg.run_id,
        "driver": DRIVER_CONTRACT,
        "model": {"key": f"claude-code/{cfg.model}", "family": FAMILY, "tier": cfg.model},
        "cli": cfg.cli.to_dict(),
        "caps": cfg.caps.to_dict(),
        "plans": args.plans,
        "plans_sha256": file_sha256(root / args.plans),
        "prereg": args.prereg,
        "git_sha": _git_sha(root),
        "harness_sandbox": method,
        "subject_python": snapshot_hook.display_path(str(subject_bin(cfg.env) / "python")),
        "smoke": {"path": args.smoke, "sha256": file_sha256(smoke_path)},
        "contaminants": contaminants,
        "label": LABEL,
        "command": "python -m halflife.drivers.claude_code " + " ".join(sys.argv[1:]),
        "started": utc_stamp(),
    }
    _append_jsonl(run_dir / "invocations.jsonl", record)
    _write_json(run_dir / "run.json", record)
    status = EXIT_ERROR
    try:
        status = run_plans(plans, {k: v.template for k, v in templates.items()}, cfg, run_dir)
    finally:
        record["finished"] = utc_stamp()
        record["status"] = status
        _write_json(run_dir / "run.json", record)
        pid_file.unlink(missing_ok=True)
    return status


def run_plans(
    plans: Sequence[SessionPlan],
    templates: Mapping[str, Template],
    cfg: DriverConfig,
    run_dir: Path,
) -> int:
    """Run plans in order, skipping finished sessions; return the exit status.

    Stops at the first session that hits a cap (``EXIT_LIMIT``), an account
    problem: a usage limit, a billing or authentication error, or overage
    (``EXIT_LIMIT``), a CLI or API error (``EXIT_ERROR``), a plan it cannot run
    (``EXIT_REFUSED``), or a safety stop (``EXIT_SAFETY``).
    """
    for plan in plans:
        try:
            outcome = SessionRunner(plan, templates[plan.template_id], cfg).run()
        except CapExceeded as exc:
            print(f"stopping: {exc}")
            return EXIT_LIMIT
        except SafetyStop as exc:
            print(f"stopping for safety: {exc}")
            return EXIT_SAFETY
        except UnsupportedPlan as exc:
            print(f"refusing: {exc}")
            return EXIT_REFUSED
        if not outcome.skipped and outcome.record is not None:
            summary = outcome.record["summary"]
            line = {
                "session_id": plan.session_id,
                "ended_reason": summary["ended_reason"],
                "tool_calls": summary["tool_calls"],
                "violations": outcome.record["session"]["n_violations"],
                "opportunities": len(outcome.record["opportunities"]),
                "finished": utc_stamp(),
            }
            _append_jsonl(run_dir / "progress.jsonl", line)
            print(json.dumps(line, sort_keys=True), flush=True)
        if outcome.stop is not None:
            _write_json(
                run_dir / "stopped.json",
                {"session_id": plan.session_id, "reason": outcome.stop, "at": utc_stamp()},
            )
            print(f"stopping: {outcome.stop} in {plan.session_id}")
            return EXIT_LIMIT if outcome.stop in ACCOUNT_STOPS else EXIT_ERROR
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """The ``python -m halflife.drivers.claude_code`` argument parser."""
    parser = argparse.ArgumentParser(prog="python -m halflife.drivers.claude_code")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="one tiny session that checks the child is clean")
    smoke.add_argument("--run-id", default="claude-smoke")
    smoke.add_argument("--model", default="haiku")
    run = sub.add_parser("run", help="run E2 plans (resumable)")
    run.add_argument("--plans", required=True, help="plans JSONL, relative to the repo root")
    run.add_argument("--run-id", required=True)
    run.add_argument("--model", required=True, help="model tier, for example haiku or sonnet")
    run.add_argument("--prereg", default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--index", type=int, default=None)
    run.add_argument("--smoke", default="runs/claude-smoke/smoke.json")
    run.add_argument("--sandbox-method", default=None, help="halflife sandbox for checks")
    run.add_argument(
        "--accept-contaminants",
        action="store_true",
        help="run although the smoke reported known contaminants (record them in DECISIONS.md)",
    )
    for p in (smoke, run):
        p.add_argument("--out-root", default="runs")
        p.add_argument("--launch-timeout-s", type=float, default=1800.0)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path | None = None,
    launcher: Launcher | None = None,
    env: Mapping[str, str] | None = None,
    guard: Callable[[], billing_guard.GuardReport] | None = None,
    claude_bin: str | None = None,
    checker: Checker | None = None,
) -> int:
    """Entry point; the keyword arguments exist for tests (a fake launcher and guard)."""
    args = build_parser().parse_args(argv)
    try:
        setup = prepare(
            args,
            root=root or repo_root(),
            launcher=launcher or SubprocessLauncher(),
            env=dict(os.environ) if env is None else env,
            guard=guard or billing_guard.check,
            claude_bin=claude_bin,
        )
    except BillingGuardRefused as exc:
        print(f"refusing: {exc}")
        return EXIT_REFUSED
    except SafetyStop as exc:
        print(f"refusing for safety: {exc}")
        return EXIT_SAFETY
    except (PreflightError, UnsupportedPlan, CapExceeded, OSError, ValueError) as exc:
        print(f"refusing: {exc}")
        return EXIT_REFUSED
    with driver_lock(setup.cfg.out_root) as locked:
        if not locked:
            print(f"refusing: another driver process holds {setup.cfg.out_root / LOCK_NAME}")
            return EXIT_REFUSED
        try:
            if args.command == "smoke":
                return cmd_smoke(setup)
            return cmd_run(setup, args, checker=checker)
        except BillingGuardRefused as exc:
            print(f"refusing: {exc}")
            return EXIT_REFUSED
        except CapExceeded as exc:
            print(f"stopping: {exc}")
            return EXIT_LIMIT
        except SafetyStop as exc:
            print(f"stopping for safety: {exc}")
            return EXIT_SAFETY
        except OSError as exc:
            print(f"error: {exc}")
            return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
