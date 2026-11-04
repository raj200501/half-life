"""halflife-pin: write the spec's rules into CLAUDE.md or AGENTS.md.

Claude Code loads the project-root CLAUDE.md at the start of every session and
re-reads it from disk after compaction ("Project-root CLAUDE.md survives
compaction: after /compact, Claude re-reads it from disk and re-injects it into
the session"); it reads AGENTS.md instead when the repository has no CLAUDE.md
(https://code.claude.com/docs/en/memory, retrieved 2026-11-04). Rules pinned
there survive compaction by construction, unlike rules stated only in the
conversation.

The rules go into a managed block between ``<!-- halflife:begin -->`` and
``<!-- halflife:end -->``. Updating a file replaces only that block (or appends
it when there is none), keeps every other byte, keeps the file's line endings,
and is idempotent: a second run changes nothing. A file with a begin marker but
no end marker, or with two blocks, is refused rather than guessed at.

Usage::

    halflife-pin --spec halflife.yaml --out CLAUDE.md
    halflife-pin --format agents --with-reasons --out AGENTS.md
    halflife-pin --out CLAUDE.md --check     # exit 1 if the block is stale
    halflife-pin                             # print the block

``--compact-instructions`` (CLAUDE.md only) adds a "Compact Instructions"
section asking Claude Code to copy every rule into its compaction summary
(:mod:`halflife.kit.compaction_prompt`).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Template
from typing import NoReturn, TextIO

from halflife.kit.compaction_prompt import compact_instructions_section
from halflife.kit.spec import DEFAULT_SPEC_NAME, SpecError, find_spec, load_spec
from halflife.schema import Constraint, Phrasing

__all__ = [
    "BEGIN",
    "END",
    "FORMATS",
    "PinError",
    "apply_block",
    "main",
    "render_block",
    "render_rules",
    "update_file",
]

BEGIN = "<!-- halflife:begin -->"
"""Marker that opens the managed block."""
END = "<!-- halflife:end -->"
"""Marker that closes the managed block."""
FORMATS = ("claude", "agents")
"""Supported targets: CLAUDE.md (Claude Code) and AGENTS.md (other coding agents)."""
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

EXIT_OK = 0
EXIT_STALE = 1
EXIT_ERROR = 2


class PinError(ValueError):
    """Raised for a target file whose managed block cannot be updated safely."""


def render_rules(constraints: Sequence[Constraint], *, with_reasons: bool = False) -> str:
    """The rules as a numbered Markdown list, one rule per line.

    With ``with_reasons`` each rule carries its reason, in the same words the
    study's ``reason`` phrasing uses (:meth:`halflife.schema.Constraint.render`).
    """
    phrasing: Phrasing = "reason" if with_reasons else "bare"
    return "\n".join(
        f"{number}. {constraint.render(phrasing)}"
        for number, constraint in enumerate(constraints, start=1)
    )


def render_block(
    constraints: Sequence[Constraint],
    *,
    fmt: str = "claude",
    with_reasons: bool = False,
    source: str = DEFAULT_SPEC_NAME,
    compact_instructions: bool = False,
) -> str:
    """The managed block, markers included, ending with a newline.

    Args:
        constraints: Rules to pin, in order.
        fmt: ``claude`` (CLAUDE.md) or ``agents`` (AGENTS.md).
        with_reasons: Append each rule's reason.
        source: Spec file name shown in the block.
        compact_instructions: Add a "Compact Instructions" section (``claude`` only).

    Raises:
        PinError: for an unknown format, no constraints, or compact
            instructions requested for AGENTS.md.
    """
    if fmt not in FORMATS:
        raise PinError(f"unknown format {fmt!r}; use one of {list(FORMATS)}")
    if not constraints:
        raise PinError("the spec has no constraints to pin")
    if compact_instructions and fmt != "claude":
        raise PinError("compact instructions apply to CLAUDE.md only")
    template = Template((TEMPLATES_DIR / f"{fmt}.md").read_text(encoding="utf-8"))
    body = template.substitute(
        source=source, rules=render_rules(constraints, with_reasons=with_reasons)
    )
    if compact_instructions:
        body = f"{body.rstrip()}\n\n{compact_instructions_section()}"
    note = f"<!-- Written by halflife-pin from {source}; edit the spec, not this block. -->"
    return f"{BEGIN}\n{note}\n{body.rstrip()}\n{END}\n"


def _locate(text: str) -> tuple[int, int] | None:
    """Start and end offsets of the managed block, or None when there is none."""
    begin = text.find(BEGIN)
    end = text.find(END)
    if begin < 0 and end < 0:
        return None
    if begin < 0 or 0 <= end < begin:
        raise PinError(f"found {END} without a {BEGIN} before it")
    end = text.find(END, begin)
    if end < 0:
        raise PinError(f"found {BEGIN} without a matching {END}")
    stop = end + len(END)
    if BEGIN in text[stop:] or END in text[stop:]:
        raise PinError("found more than one halflife block; keep one and run again")
    return begin, stop


def apply_block(text: str, block: str) -> str:
    """``text`` with its managed block replaced by ``block`` (or ``block`` appended).

    Everything outside the block is kept exactly. When the text uses CRLF line
    endings, so does the inserted block.

    Raises:
        PinError: for malformed or duplicated markers.
    """
    if "\r\n" in text:
        block = block.replace("\r\n", "\n").replace("\n", "\r\n")
    newline = "\r\n" if "\r\n" in text else "\n"
    span = _locate(text)
    if span is not None:
        begin, stop = span
        inner = block[: -len(newline)] if block.endswith(newline) else block
        return text[:begin] + inner + text[stop:]
    if not text:
        return block
    if text.endswith(newline * 2):
        return text + block
    if text.endswith(newline):
        return text + newline + block
    return text + newline * 2 + block


def update_file(path: str | os.PathLike[str], block: str, *, check: bool = False) -> bool:
    """Put ``block`` into the file at ``path``, creating it if needed.

    Args:
        path: CLAUDE.md, AGENTS.md, or another Markdown file.
        block: Output of :func:`render_block`.
        check: Only report whether the file is stale; never write.

    Returns:
        True when the file changed (or, with ``check``, would change).

    Raises:
        PinError: for malformed markers.
        OSError: if the file cannot be read or written.
    """
    target = Path(path)
    try:
        with target.open(encoding="utf-8", newline="") as handle:
            current = handle.read()
    except FileNotFoundError:
        current = ""
    updated = apply_block(current, block)
    if updated == current:
        return False
    if check:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
        if target.exists():
            os.chmod(temp, target.stat().st_mode & 0o7777)
        os.replace(temp, target)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return True


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="halflife-pin",
        description="Write the rules of a halflife.yaml spec into CLAUDE.md or AGENTS.md.",
    )
    parser.add_argument("--spec", help="spec file (default: $HALFLIFE_SPEC or ./halflife.yaml)")
    parser.add_argument(
        "--format",
        choices=FORMATS,
        help="claude (CLAUDE.md) or agents (AGENTS.md); default: from --out, else claude",
    )
    parser.add_argument("--with-reasons", action="store_true", help="append each rule's reason")
    parser.add_argument(
        "--compact-instructions",
        action="store_true",
        help="add a Compact Instructions section (CLAUDE.md only)",
    )
    parser.add_argument("--out", help="file to update in place (default: print the block)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="with --out: exit 1 if the block is stale, write nothing",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run halflife-pin.

    Returns:
        0 on success, 1 when ``--check`` finds the block stale, 2 on an error.
    """
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        args = _parser().parse_args(argv)
        if args.check and not args.out:
            raise _UsageError("--check needs --out")
        spec_path = find_spec(args.spec, env=env)
        if spec_path is None:
            raise SpecError(f"no spec found (pass --spec or create {DEFAULT_SPEC_NAME})")
        fmt = args.format or (
            "agents" if args.out and Path(args.out).name.upper() == "AGENTS.MD" else "claude"
        )
        block = render_block(
            load_spec(spec_path),
            fmt=fmt,
            with_reasons=args.with_reasons,
            source=spec_path.name,
            compact_instructions=args.compact_instructions,
        )
        if not args.out:
            out.write(block)
            return EXIT_OK
        changed = update_file(args.out, block, check=args.check)
    except (_UsageError, SpecError, PinError, OSError) as exc:
        err.write(f"halflife-pin: {exc}\n")
        return EXIT_ERROR
    if args.check:
        out.write(f"halflife-pin: {args.out} is {'stale' if changed else 'up to date'}\n")
        return EXIT_STALE if changed else EXIT_OK
    out.write(f"halflife-pin: {'updated' if changed else 'no change to'} {args.out}\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
