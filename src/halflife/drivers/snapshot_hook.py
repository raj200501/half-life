"""Workspace snapshot hook for the Claude Code driver (E2).

Claude Code runs this command, as configured by
:mod:`halflife.drivers.claude_code` through ``--settings``, after every tool
call (``PostToolUse`` and ``PostToolUseFailure``), whenever it loads an
instruction file such as ``CLAUDE.md`` (``InstructionsLoaded``), and at the
start of every launch (``SessionStart``)::

    python -I -m halflife.drivers.snapshot_hook --out <jsonl> --root <workspace> [--netlog <jsonl>]

It reads the hook input (one JSON object) on stdin and appends one JSON line to
``--out``:

* tool calls: ``{"kind": "tool", "tool_name", "tool_input", "tool_use_id",
  "fs": {path: sha256}, "blobs": {sha256: text}, "skipped": {sha256: reason},
  "net": [{"target", "via"}], ...}``. ``fs`` uses exactly the snapshot rules of
  :meth:`halflife.workspace.Workspace.snapshot` (``.git/``, ``__pycache__/``,
  ``.pytest_cache/``, and ``*.pyc`` excluded), so detectors read the same
  ``fs_after`` as in minimal-agent runs. ``blobs`` holds the text of every file
  version not already in an earlier line of the same file. ``net`` holds the
  network attempts the sandbox shims logged to ``--netlog`` since the last
  record.
* instruction loads: ``{"kind": "instructions", "file_path", "memory_type",
  "load_reason", "in_root"}``. Only the path is recorded, never the content, and
  the home directory is shown as ``~``.
* any other event, such as ``SessionStart``: ``{"kind": "other", ...}`` with
  the event's ``source``, ``trigger``, or ``reason`` when it has one.
* failures of the hook itself: ``{"kind": "error", "error"}``.

Every record carries ``hook_event_name``, ``session_id``, ``transcript_path``,
and ``cwd`` from the hook input, so the driver learns the transcript path at
the start of each launch.

Every line also has ``seq`` (its 0-based line number) and ``ts``. Appends hold an
exclusive lock, so parallel tool calls cannot interleave lines.

The hook never blocks a tool: it always exits 0 and never writes to stdout (in
Claude Code, hook stdout can reach the transcript). It runs with ``-I`` so a
module planted in the agent's workspace cannot shadow the standard library.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from halflife.sandbox import parse_netlog
from halflife.secrets_scan import PATTERNS
from halflife.workspace import Workspace

TOOL_EVENTS: frozenset[str] = frozenset({"PostToolUse", "PostToolUseFailure"})
"""Hook events that carry a tool call and get a workspace snapshot."""

INSTRUCTION_EVENTS: frozenset[str] = frozenset({"InstructionsLoaded"})
"""Hook events that report a loaded instruction file."""

MAX_ERROR_CHARS = 500


class DirectorySnapshot(Workspace):
    """The workspace snapshot rules applied to an existing directory.

    This reuses :meth:`Workspace.snapshot` (exclusions, symlink and special-file
    markers, blob limits) without creating, committing, or deleting anything:
    the directory belongs to the session, so :meth:`close` is a no-op.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        self._closed = True
        self.blobs = {}
        self.skipped = {}
        self.baseline_commit = None


def snapshot_dir(root: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Hash every file under ``root`` as :meth:`Workspace.snapshot` does.

    Returns:
        ``(fs, blobs, skipped)``: path to SHA-256, SHA-256 to text for every
        text version seen, and SHA-256 to the reason a version has no text.

    Raises:
        FileNotFoundError: if ``root`` is not a directory (an empty snapshot
            would read as every file deleted).
    """
    if not Path(root).is_dir():
        raise FileNotFoundError(f"workspace root {root} is not a directory")
    snap = DirectorySnapshot(root)
    fs = snap.snapshot()
    return fs, dict(snap.blobs), dict(snap.skipped)


def redact_text(text: str, values: Iterable[str] = ()) -> str:
    """Replace credential-shaped text (``halflife.secrets_scan`` patterns) and exact
    secret values with a ``[redacted:<rule>]`` marker."""
    for rule, pattern in PATTERNS.items():
        text = pattern.sub(f"[redacted:{rule}]", text)
    for value in values:
        if value:
            text = text.replace(value, "[redacted:env]")
    return text


def redact_json(data: Any, values: Sequence[str] = ()) -> Any:
    """:func:`redact_text` applied to every string inside JSON-like data (keys too)."""
    if isinstance(data, str):
        return redact_text(data, values)
    if isinstance(data, list | tuple):
        return [redact_json(item, values) for item in data]
    if isinstance(data, Mapping):
        return {redact_text(str(k), values): redact_json(v, values) for k, v in data.items()}
    return data


def display_path(path: str) -> str:
    """``path`` with the home directory shown as ``~`` (keeps user names out of logs)."""
    home = str(Path.home())
    if home not in ("", "/") and (path == home or path.startswith(home + os.sep)):
        return "~" + path[len(home) :]
    return path


def inside(path: str, root: Path) -> bool:
    """Whether ``path`` (absolute, or relative to ``root``) lies inside ``root``."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    real_root = root.resolve()
    real = candidate.resolve(strict=False)
    return real == real_root or real_root in real.parents


def utc_stamp() -> str:
    """Current UTC time with milliseconds."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class HookArgs:
    """Command-line arguments of the hook."""

    out: Path
    root: Path
    netlog: Path | None = None


def parse_args(argv: Sequence[str]) -> HookArgs:
    """Parse ``--out``, ``--root``, and the optional ``--netlog``.

    A hand-written parser, because argparse prints to stdout for ``--help``
    and exits on errors; the hook must do neither.

    Raises:
        ValueError: for an unknown or incomplete argument.
    """
    values: dict[str, str] = {}
    tokens = iter(argv)
    for token in tokens:
        if token not in ("--out", "--root", "--netlog"):
            raise ValueError(f"unknown argument {token!r}")
        value = next(tokens, None)
        if not value:
            raise ValueError(f"{token} needs a value")
        values[token] = value
    if "--out" not in values or "--root" not in values:
        raise ValueError("--out and --root are required")
    netlog = values.get("--netlog")
    return HookArgs(Path(values["--out"]), Path(values["--root"]), Path(netlog) if netlog else None)


def parse_payload(text: str) -> dict[str, Any]:
    """The hook input object.

    Raises:
        ValueError: if the text is not a JSON object.
    """
    data = json.loads(text) if text.strip() else None
    if not isinstance(data, dict):
        raise ValueError("hook input is not a JSON object")
    return data


def _ledger_state(text: str) -> tuple[set[str], int, int]:
    """Known blob hashes, the last netlog offset, and the line count of an output file."""
    known: set[str] = set()
    offset = 0
    lines = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        lines += 1
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        for key in ("blobs", "skipped"):
            value = record.get(key)
            if isinstance(value, dict):
                known.update(str(k) for k in value)
        if isinstance(record.get("netlog_offset"), int):
            offset = max(offset, int(record["netlog_offset"]))
    return known, offset, lines


def read_netlog(path: Path, offset: int) -> tuple[list[dict[str, str]], int]:
    """Network attempts logged at or after byte ``offset``, and the new offset.

    Only complete lines are consumed, so a line still being written is read by
    the next record. A file shorter than ``offset`` was replaced and is read
    from the start.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return [], offset
    if len(data) < offset:
        offset = 0
    chunk = data[offset:]
    end = chunk.rfind(b"\n")
    if end < 0:
        return [], offset
    complete = chunk[: end + 1].decode("utf-8", errors="replace")
    attempts = [{"target": a.target, "via": a.via} for a in parse_netlog(complete)]
    return attempts, offset + end + 1


def append_record(
    out: Path,
    record: Mapping[str, Any],
    *,
    blobs: Mapping[str, str] | None = None,
    skipped: Mapping[str, str] | None = None,
    netlog: Path | None = None,
) -> dict[str, Any]:
    """Append one record to ``out`` under an exclusive lock and return it.

    Adds ``seq`` and ``ts``; keeps only blobs and skip reasons whose hashes no
    earlier line has; with ``netlog``, adds the network attempts logged since
    the last record (``net``) and the new ``netlog_offset``. Text is redacted
    with :func:`redact_text` before it is written.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            known, offset, lines = _ledger_state(handle.read())
            full: dict[str, Any] = dict(record)
            full["seq"] = lines
            full["ts"] = utc_stamp()
            if blobs is not None:
                full["blobs"] = {h: t for h, t in sorted(blobs.items()) if h not in known}
            if skipped is not None:
                full["skipped"] = {h: r for h, r in sorted(skipped.items()) if h not in known}
            if netlog is not None:
                full["net"], full["netlog_offset"] = read_netlog(netlog, offset)
            safe: dict[str, Any] = redact_json(full)
            handle.write(json.dumps(safe, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return safe


def read_records(out: Path) -> list[dict[str, Any]]:
    """Every parseable record in an output file, in file order (missing file: none)."""
    try:
        text = out.read_text(encoding="utf-8")
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def write_baseline(
    out: Path,
    fs: Mapping[str, str],
    blobs: Mapping[str, str],
    skipped: Mapping[str, str],
) -> dict[str, Any]:
    """Write the baseline record (``seq`` 0), so later records skip baseline blobs."""
    return append_record(
        out,
        {"kind": "baseline", "fs": dict(sorted(fs.items())), "netlog_offset": 0},
        blobs=blobs,
        skipped=skipped,
    )


def handle(payload: Mapping[str, Any], args: HookArgs) -> dict[str, Any]:
    """Record one hook input and return the line written."""
    event = str(payload.get("hook_event_name", ""))
    common = {
        "hook_event_name": event,
        "session_id": payload.get("session_id"),
        "transcript_path": payload.get("transcript_path"),
        "cwd": payload.get("cwd"),
    }
    if event in TOOL_EVENTS:
        start = time.monotonic()
        fs, blobs, skipped = snapshot_dir(args.root)
        record = {
            "kind": "tool",
            **common,
            "tool_name": payload.get("tool_name"),
            "tool_input": payload.get("tool_input"),
            "tool_use_id": payload.get("tool_use_id"),
            "duration_ms": payload.get("duration_ms"),
            "failed": event == "PostToolUseFailure",
            "fs": fs,
            "snapshot_ms": round((time.monotonic() - start) * 1000, 3),
        }
        return append_record(args.out, record, blobs=blobs, skipped=skipped, netlog=args.netlog)
    if event in INSTRUCTION_EVENTS:
        path = str(payload.get("file_path", ""))
        record = {
            "kind": "instructions",
            **common,
            "file_path": display_path(path),
            "memory_type": payload.get("memory_type"),
            "load_reason": payload.get("load_reason"),
            "in_root": bool(path) and inside(path, args.root),
        }
        return append_record(args.out, record)
    extra = {
        key: payload[key]
        for key in ("source", "trigger", "reason")
        if isinstance(payload.get(key), str)
    }
    return append_record(args.out, {"kind": "other", **common, **extra})


def main(argv: Sequence[str] | None = None, stdin: IO[str] | None = None) -> int:
    """Hook entry point. Always returns 0 and never writes to stdout."""
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
    except Exception:
        return 0
    try:
        stream = sys.stdin if stdin is None else stdin
        handle(parse_payload(stream.read()), args)
    except BaseException as exc:  # the hook must never fail the tool call
        with suppress(BaseException):
            message = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
            append_record(args.out, {"kind": "error", "error": message})
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
