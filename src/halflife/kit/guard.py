"""halflife-guard: a Claude Code PreToolUse hook that enforces a constraint spec.

Claude Code runs a PreToolUse command hook before each matching tool call. It
passes JSON on stdin with ``tool_name``, ``tool_input``, ``cwd``, and
``hook_event_name`` among the common fields. A hook that prints
``{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision":
"deny", "permissionDecisionReason": "..."}}`` and exits 0 prevents the call,
and Claude sees the reason. Printing nothing and exiting 0 leaves the call to
the normal permission flow. Any exit code other than 0 and 2 is a non-blocking
error: the call goes ahead and the transcript shows a hook error notice with
the first line of stderr, unless the hook entry sets ``"onFailure": "block"``.
For Write and Edit, ``tool_input.file_path`` is always absolute. (Hooks
reference, https://code.claude.com/docs/en/hooks, retrieved 2026-11-04.)

This module follows that contract:

* **Deny**: print the deny JSON on stdout and exit 0. The reason says what the
  call would do and names the rule with its text, so the agent can look for
  the compliant route. It never quotes the command line, so a secret in a
  command never reaches the transcript or a log through the guard.
* **Allow**: print nothing and exit 0.
* **Fail open**: malformed hook input, an unreadable or invalid spec, a usage
  error, or an unexpected error prints one ``halflife-guard: ...`` line on
  stderr and exits 1, so the call goes ahead with a visible hook error notice.
  Set ``"onFailure": "block"`` on the hook entry (Claude Code v2.1.295 or later)
  to fail closed instead. With no ``--spec``, no ``$HALFLIFE_SPEC``, and no
  ``halflife.yaml`` in the working directory or in ``$CLAUDE_PROJECT_DIR``,
  there is nothing to enforce: the guard exits 0 with a note on stderr.

The decision core, :func:`decide`, depends only on the tool call, the
constraints, and the files on disk: it reads files and may run read-only git
queries, and it never writes. :func:`decide_minimal_agent` maps the minimal
agent's tools onto Claude Code's (``write_file`` to Write, ``replace_in_file``
to Edit, ``run`` to Bash), so the E4 guard arm runs exactly the shipped logic.

What it enforces, by detector kind:

* ``protected_paths``: Write, Edit, MultiEdit, and NotebookEdit on a matching
  path, and Bash commands that would create, modify, or delete one. Globs are
  anchored at the project root with the semantics of
  :func:`halflife.detect.glob_match`. Paths the workspace snapshot ignores
  (``.git``, ``__pycache__``, ``.pytest_cache``, ``*.pyc``) never count.
* ``forbidden_commands``: Bash commands matching a pattern (``re.search`` on the
  whole command, as the detector does).
* ``no_network``: Bash commands that would use the network (see
  :func:`analyze_command`).
* ``no_new_dependencies``: Write, Edit, and MultiEdit on a manifest that would
  add a dependency name (parsed as :func:`halflife.detect.manifest_dependencies`
  parses it), any Bash write to a manifest, and Bash commands that install a
  named package.
* ``max_file_lines``, ``file_invariant``, ``stable_api``: Write, Edit, and
  MultiEdit whose resulting content breaks the rule, computed from the file on
  disk with the detector's own line count, pattern, and public-API functions.
  Bash writes to a ``file_invariant`` file or a ``stable_api`` module are
  refused, since their result cannot be computed; Bash writes are not checked
  against ``max_file_lines``.
* ``changelog_procedure``, ``tests_before_done``, ``max_new_files``,
  ``max_test_runs``, and ``invariant_command`` need session state or a test
  run; the guard does not enforce them.

The limits of the static Bash analysis are listed in docs/MITIGATIONS.md.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import glob as globlib
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NoReturn, TextIO

from halflife.detect import (
    api_breaks,
    compile_glob,
    manifest_dependencies,
    match_any,
    public_api,
    resolve_module,
)
from halflife.kit.spec import PROJECT_DIR_ENV, SpecError, find_spec, load_spec
from halflife.schema import Action, Constraint
from halflife.workspace import EXCLUDED_DIRS, EXCLUDED_SUFFIXES

__all__ = [
    "ALLOW",
    "ENFORCED_KINDS",
    "FILE_TOOLS",
    "GUARDED_TOOLS",
    "Decision",
    "analyze_command",
    "could_match_under",
    "coverage",
    "decide",
    "decide_minimal_agent",
    "deny_output",
    "main",
    "minimal_agent_guard",
]

FILE_TOOLS: tuple[str, ...] = ("Write", "Edit", "MultiEdit", "NotebookEdit")
"""Claude Code file tools the guard checks (MultiEdit for versions that still have it)."""
GUARDED_TOOLS: tuple[str, ...] = (*FILE_TOOLS, "Bash")
"""Every tool the guard checks; calls to other tools are always allowed."""
ENFORCED_KINDS: tuple[str, ...] = (
    "protected_paths",
    "forbidden_commands",
    "no_network",
    "no_new_dependencies",
    "max_file_lines",
    "file_invariant",
    "stable_api",
)
"""Detector kinds the guard enforces; the others need session state or a test run."""
CONTENT_KINDS = frozenset({"no_new_dependencies", "max_file_lines", "file_invariant", "stable_api"})
"""Kinds whose file-tool check needs the file's content before and after the call."""

EXIT_OK = 0
EXIT_FAIL_OPEN = 1
"""Claude Code treats exit 1 as a non-blocking error: the call goes ahead."""
MAX_READ_BYTES = 1_000_000
"""Files larger than this are not read for content checks, scripts, or patches."""
MAX_EXPANSIONS = 256
"""A word with more possible values than this counts as unknown."""
MAX_DEPTH = 4
"""Nesting limit for ``bash -c``, ``eval``, scripts, and command substitutions."""
GIT_TIMEOUT_S = 5.0
"""Time limit for each read-only git query."""


@dataclass(frozen=True)
class Decision:
    """The guard's verdict on one tool call.

    Attributes:
        allow: False when the call must be refused.
        constraint_id: The rule that refuses it (None when allowed).
        reason: For a refusal, what the call would do plus the rule's id and
            text; for an allowed call, empty or a note on what was not checked.
    """

    allow: bool
    constraint_id: str | None = None
    reason: str = ""


ALLOW = Decision(True)
"""The plain allow decision."""


def _deny(constraint: Constraint, detail: str) -> Decision:
    return Decision(False, constraint.id, f"{detail}. Rule {constraint.id}: {constraint.text}")


def _unchecked(why: str) -> Decision:
    return Decision(True, None, f"not checked: {why}")


# --------------------------------------------------------------------------- paths


def _excluded(rel: str) -> bool:
    """Paths the workspace snapshot never records, so no detector sees them change."""
    parts = [p for p in rel.split("/") if p]
    return any(p in EXCLUDED_DIRS for p in parts) or rel.endswith(EXCLUDED_SUFFIXES)


def _relative(path: str, cwd: str, root: str) -> str | None:
    """Root-relative POSIX form of ``path`` (absolute, or relative to ``cwd``).

    Backslashes count as separators (Claude Code sends Windows paths with
    them), ``~`` is expanded, and symbolic links are resolved. Returns ``""``
    for the root itself and None for a path outside the root.
    """
    text = os.path.expanduser(path.replace("\\", "/"))
    joined = text if os.path.isabs(text) else os.path.join(cwd, text)
    try:
        rel = os.path.relpath(os.path.realpath(joined), os.path.realpath(root))
    except ValueError:  # pragma: no cover - paths on different Windows drives
        return None
    rel = rel.replace(os.sep, "/")
    if rel == ".":
        return ""
    if rel == ".." or rel.startswith("../"):
        return None
    return rel


def _segments(pattern: str) -> list[str]:
    text = pattern.strip()
    while text.startswith("./"):
        text = text[2:]
    text = text.lstrip("/")
    if text.endswith("/"):
        text += "**"
    return [s for s in text.split("/") if s]


def could_match_under(pattern: str, directory: str) -> bool:
    """Whether a workspace glob matches some path strictly under ``directory``.

    ``directory`` is root-relative (``""`` for the root). The guard uses it for
    targets that stand for a whole tree, such as ``rm -r dir`` or
    ``git checkout -- .``.
    """
    segs = _segments(pattern)
    dirs = [s for s in directory.split("/") if s]

    def walk(i: int, j: int) -> bool:
        if j == len(dirs):
            return i < len(segs)
        if i == len(segs):
            return False
        if segs[i] == "**":
            return i == len(segs) - 1 or walk(i + 1, j) or walk(i, j + 1)
        if compile_glob(segs[i]).fullmatch(dirs[j]) is None:
            return False
        return walk(i + 1, j + 1)

    return walk(0, 0)


def _hits(rel: str, globs: Sequence[str], recursive: bool) -> bool:
    if _excluded(rel):
        return False
    if rel and match_any(rel, globs):
        return True
    return recursive and any(could_match_under(g, rel) for g in globs)


def _read_text(path: str) -> str | None:
    """UTF-8 text of a file, or None if it is missing, too large, or not text."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_READ_BYTES + 1)
    except OSError:
        return None
    if len(data) > MAX_READ_BYTES or b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _line_count(text: str) -> int:
    """Lines as :func:`halflife.detect.detect_max_file_lines` counts them."""
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


# --------------------------------------------------------------------------- shell lexer


@dataclass(frozen=True)
class _Part:
    """One piece of a shell word: ``lit`` text, a ``var`` name, a ``subst`` index, or ``expr``."""

    kind: str
    text: str


@dataclass(frozen=True)
class _Word:
    """A shell word after quote removal, with what the shell may still expand in it."""

    parts: tuple[_Part, ...]
    glob: bool = False
    brace: bool = False
    tilde: bool = False

    @property
    def literal(self) -> bool:
        """True when the word holds no parameter, command, or arithmetic expansion."""
        return all(p.kind == "lit" for p in self.parts)

    @property
    def text(self) -> str:
        """The word as text, with expansions shown as written."""
        shown = {"var": "$", "subst": "$(", "expr": ""}
        return "".join(
            p.text if p.kind == "lit" else shown[p.kind] + (p.text if p.kind != "subst" else "...)")
            for p in self.parts
        )


def _lit(text: str) -> _Word:
    return _Word((_Part("lit", text),))


def _is_assignment(word: _Word) -> bool:
    first = word.parts[0] if word.parts else None
    return first is not None and first.kind == "lit" and _ASSIGNMENT.match(first.text) is not None


@dataclass
class _Redirect:
    op: str
    target: _Word | None
    heredoc: str | None = None
    strip_tabs: bool = False
    delimiter: str = ""


@dataclass
class _Simple:
    """A simple command: its words and redirections."""

    words: list[_Word] = field(default_factory=list)
    redirects: list[_Redirect] = field(default_factory=list)


_OPS = (
    "&&",
    "||",
    ";;&",
    ";;",
    ";&",
    "|&",
    "&>>",
    "&>",
    ">>",
    ">|",
    ">&",
    "<<<",
    "<<-",
    "<<",
    "<&",
    "<>",
    ">",
    "<",
    "|",
    "&",
    ";",
    "(",
    ")",
)
_OPERATOR = re.compile("|".join(re.escape(op) for op in _OPS))
_REDIRECT_OPS = frozenset({"&>>", "&>", ">>", ">|", ">&", "<<<", "<<-", "<<", "<&", "<>", ">", "<"})
_WRITE_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>", "<>", ">&"})
_FD_PREFIX = re.compile(r"\d+(?=[<>])")
_VAR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9@*#?$!-]")
_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\+?=")
_ANSI = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"', "e": "\x1b"}
_WORD_END = frozenset(" \t\r\n;&|<>()")


class _Lexer:
    """Split a shell command into words, operators, and redirections.

    Handles single, double, and ``$'...'`` quotes, backslash escapes, line
    continuations, comments, ``$name`` and ``${name}``, ``$(...)``, backticks,
    ``$((...))``, process substitution, file-descriptor prefixes such as
    ``2>``, and heredoc bodies. Command substitutions are collected in
    :attr:`substitutions` and referenced from words by index.
    """

    def __init__(self, text: str) -> None:
        self.s = text
        self.i = 0
        self.n = len(text)
        self.tokens: list[tuple[str, Any]] = []
        self.substitutions: list[str] = []
        self.pending: list[_Redirect] = []

    def lex(self) -> list[tuple[str, Any]]:
        """Tokens as ``("word", _Word)``, ``("op", str)``, or ``("redir", _Redirect)``."""
        s = self.s
        while self.i < self.n:
            c = s[self.i]
            if c in " \t\r":
                self.i += 1
            elif s.startswith("\\\n", self.i):
                self.i += 2
            elif c == "\n":
                self.tokens.append(("op", "\n"))
                self.i += 1
                self._heredocs()
            elif c == "#":
                end = s.find("\n", self.i)
                self.i = self.n if end < 0 else end
            elif s.startswith(("<(", ">("), self.i):
                start = self.i + 2
                self.i = self._balanced(start, "(", ")")
                self.tokens.append(("word", _Word((self._subst(s[start : self.i - 1]),))))
            else:
                fd = _FD_PREFIX.match(s, self.i)
                op = _OPERATOR.match(s, fd.end() if fd else self.i)
                if op and (fd is None or op.group() in _REDIRECT_OPS):
                    self.i = op.end()
                    if op.group() in _REDIRECT_OPS:
                        self._redirect(op.group())
                    else:
                        self.tokens.append(("op", op.group()))
                else:
                    self.tokens.append(("word", self._word()))
        return self.tokens

    def _subst(self, command: str) -> _Part:
        self.substitutions.append(command)
        return _Part("subst", str(len(self.substitutions) - 1))

    def _redirect(self, op: str) -> None:
        while self.i < self.n and self.s[self.i] in " \t":
            self.i += 1
        target = None
        if self.i < self.n and self.s[self.i] not in _WORD_END:
            target = self._word()
        redirect = _Redirect(op, target)
        if op in ("<<", "<<-") and target is not None:
            redirect.delimiter = target.text
            redirect.strip_tabs = op == "<<-"
            self.pending.append(redirect)
        self.tokens.append(("redir", redirect))

    def _heredocs(self) -> None:
        for redirect in self.pending:
            lines: list[str] = []
            while self.i < self.n:
                end = self.s.find("\n", self.i)
                line = self.s[self.i : self.n if end < 0 else end]
                self.i = self.n if end < 0 else end + 1
                if (line.lstrip("\t") if redirect.strip_tabs else line) == redirect.delimiter:
                    break
                lines.append(line)
            redirect.heredoc = "".join(f"{line}\n" for line in lines)
        self.pending = []

    def _balanced(self, start: int, opening: str, closing: str) -> int:
        """Index just past the bracket that closes one opened right before ``start``."""
        s, depth, j = self.s, 1, start
        while j < self.n:
            c = s[j]
            if c == "\\":
                j += 2
            elif c == "'":
                end = s.find("'", j + 1)
                j = self.n if end < 0 else end + 1
            elif c == '"':
                j += 1
                while j < self.n and s[j] != '"':
                    j += 2 if s[j] == "\\" else 1
                j += 1
            else:
                if c == opening:
                    depth += 1
                elif c == closing:
                    depth -= 1
                    if depth == 0:
                        return j + 1
                j += 1
        return self.n

    def _dollar(self) -> _Part:
        """Read one ``$`` expansion starting at the current position."""
        s, i = self.s, self.i
        if s.startswith("$((", i):
            self.i = self._balanced(i + 3, "(", ")")
            if self.i < self.n and s[self.i] == ")":
                self.i += 1
            return _Part("expr", s[i : self.i])
        if s.startswith("$(", i):
            self.i = self._balanced(i + 2, "(", ")")
            return self._subst(s[i + 2 : self.i - 1])
        if s.startswith("${", i):
            self.i = self._balanced(i + 2, "{", "}")
            inner = s[i + 2 : self.i - 1]
            if _VAR_NAME.fullmatch(inner):
                return _Part("var", inner)
            return _Part("expr", s[i : self.i])
        name = _VAR_NAME.match(s, i + 1)
        if name:
            self.i = name.end()
            return _Part("var", name.group())
        self.i = i + 1
        return _Part("lit", "$")

    def _backtick(self) -> _Part:
        s, j = self.s, self.i + 1
        buf: list[str] = []
        while j < self.n and s[j] != "`":
            if s[j] == "\\" and j + 1 < self.n:
                buf.append(s[j + 1])
                j += 2
            else:
                buf.append(s[j])
                j += 1
        self.i = j + 1
        return self._subst("".join(buf))

    def _word(self) -> _Word:
        s = self.s
        parts: list[_Part] = []
        buf: list[str] = []
        glob = brace = False
        tilde = s.startswith("~", self.i)

        def flush() -> None:
            if buf:
                parts.append(_Part("lit", "".join(buf)))
                buf.clear()

        while self.i < self.n and s[self.i] not in _WORD_END:
            c = s[self.i]
            if c == "\\":
                if not s.startswith("\\\n", self.i) and self.i + 1 < self.n:
                    buf.append(s[self.i + 1])
                self.i += 2
            elif c == "'":
                end = s.find("'", self.i + 1)
                end = self.n if end < 0 else end
                buf.append(s[self.i + 1 : end])
                self.i = end + 1
            elif s.startswith("$'", self.i):
                j = self.i + 2
                while j < self.n and s[j] != "'":
                    if s[j] == "\\" and j + 1 < self.n:
                        buf.append(_ANSI.get(s[j + 1], s[j + 1]))
                        j += 2
                    else:
                        buf.append(s[j])
                        j += 1
                self.i = j + 1
            elif c == '"':
                self.i += 1
                while self.i < self.n and s[self.i] != '"':
                    d = s[self.i]
                    if d == "\\" and self.i + 1 < self.n and s[self.i + 1] in '$`"\\\n':
                        if s[self.i + 1] != "\n":
                            buf.append(s[self.i + 1])
                        self.i += 2
                    elif d in "$`":
                        flush()
                        parts.append(self._dollar() if d == "$" else self._backtick())
                    else:
                        buf.append(d)
                        self.i += 1
                self.i += 1
            elif c in "$`":
                flush()
                parts.append(self._dollar() if c == "$" else self._backtick())
            else:
                glob = glob or c in "*?["
                brace = brace or c == "{"
                buf.append(c)
                self.i += 1
        flush()
        return _Word(tuple(parts) or (_Part("lit", ""),), glob, brace, tilde)


_PUSH = "push"
_POP = "pop"
_Item = tuple[str, list[_Simple]]


def _parse(tokens: Sequence[tuple[str, Any]]) -> list[_Item]:
    """Group tokens into pipelines of simple commands, with subshell push and pop markers."""
    items: list[_Item] = []
    pipeline: list[_Simple] = []
    current = _Simple()

    def end_command() -> None:
        nonlocal current
        if current.words or current.redirects:
            pipeline.append(current)
        current = _Simple()

    def end_pipeline() -> None:
        nonlocal pipeline
        end_command()
        if pipeline:
            items.append(("pipeline", pipeline))
        pipeline = []

    for kind, value in tokens:
        if kind == "word":
            current.words.append(value)
        elif kind == "redir":
            current.redirects.append(value)
        elif value in ("|", "|&"):
            end_command()
        else:
            end_pipeline()
            if value in ("(", ")"):
                items.append((_PUSH if value == "(" else _POP, []))
    end_pipeline()
    return items


def _brace_options(body: str) -> list[str] | None:
    """The alternatives of one ``{...}`` group, or None if bash would not expand it."""
    seq = re.fullmatch(r"(-?\d+)\.\.(-?\d+)|([A-Za-z])\.\.([A-Za-z])", body)
    if seq:
        if seq.group(1) is not None:
            a, b = int(seq.group(1)), int(seq.group(2))
            step = 1 if b >= a else -1
            return [str(v) for v in range(a, b + step, step)][:MAX_EXPANSIONS]
        x, y = ord(seq.group(3)), ord(seq.group(4))
        step = 1 if y >= x else -1
        return [chr(v) for v in range(x, y + step, step)]
    options: list[str] = []
    buf: list[str] = []
    depth = 0
    for char in body:
        if char == "," and depth == 0:
            options.append("".join(buf))
            buf = []
            continue
        depth += {"{": 1, "}": -1}.get(char, 0)
        buf.append(char)
    options.append("".join(buf))
    return options if len(options) > 1 else None


def _brace_expand(text: str) -> list[str]:
    """Bash brace expansion: ``a{b,c}`` and ``x{1..3}``, nested, capped at MAX_EXPANSIONS."""
    depth, start = 0, -1
    for index, char in enumerate(text):
        if char == "{":
            start = index if depth == 0 else start
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth:
                continue
            options = _brace_options(text[start + 1 : index])
            head, tail = text[:start], text[index + 1 :]
            if options is None:
                return [text[: index + 1] + rest for rest in _brace_expand(tail)]
            out: list[str] = []
            for option in options:
                out.extend(head + rest for rest in _brace_expand(option + tail))
                if len(out) >= MAX_EXPANSIONS:
                    return out[:MAX_EXPANSIONS]
            return out
    return [text]


# --------------------------------------------------------------------------- effects


@dataclass(frozen=True)
class _Value:
    """A possible value of a word; ``recursive`` stands for any path under it."""

    text: str
    recursive: bool = False


@dataclass(frozen=True)
class _Write:
    path: str
    recursive: bool
    via: str


@dataclass
class _Effects:
    """What a command may do: write paths, use the network, install packages."""

    writes: list[_Write] = field(default_factory=list)
    network: list[str] = field(default_factory=list)
    installs: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


_KEYWORDS = frozenset(
    {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "esac", "!", "{", "}",
     "[[", "]]", "in", "function", "coproc"}
)  # fmt: skip
_WRAPPERS: dict[str, tuple[str, ...]] = {
    "sudo": ("-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U"),
    "doas": ("-u", "-C"),
    "nice": ("-n", "--adjustment"),
    "nohup": (),
    "time": ("-f", "--format", "-o", "--output"),
    "command": (),
    "builtin": (),
    "exec": ("-a",),
    "stdbuf": ("-i", "-o", "-e", "--input", "--output", "--error"),
    "ionice": ("-c", "-n", "-p", "--class", "--classdata"),
    "chronic": (),
    "unbuffer": (),
}
_NETWORK_COMMANDS = frozenset(
    {"nc", "ncat", "netcat", "telnet", "ssh", "scp", "sftp", "ftp", "aria2c", "http", "https",
     "ping", "dig", "nslookup", "host", "socat", "gh", "lynx", "w3m", "links", "uvx", "npx",
     "pnpx", "bunx", "pipx"}
)  # fmt: skip
_HINT_COMMANDS = frozenset({"find", "ls", "grep", "egrep", "fgrep", "rg", "fd", "fdfind", "tree"})
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})
_EDITORS = frozenset({"ed", "ex", "vi", "vim", "nvim", "nano", "emacs"})
_PYTHON = re.compile(r"^(?:python|pypy)(?:\d+(?:\.\d+)*)?$")
_PIP = re.compile(r"^pip(?:\d+(?:\.\d+)*)?$")
_NET_PREFIXES = (
    "requests.",
    "httpx.",
    "aiohttp.",
    "urllib3.",
    "pycurl.",
    "paramiko.",
    "websocket.",
    "websockets.",
)
"""Python modules whose every call counts as network use (HTTP and SSH clients)."""
_NET_FUNCTIONS = frozenset(
    {"urlopen", "urlretrieve", "socket.socket", "socket.create_connection",
     "socket.getaddrinfo", "socket.gethostbyname", "http.client.HTTPConnection",
     "http.client.HTTPSConnection", "ftplib.FTP", "ftplib.FTP_TLS", "smtplib.SMTP",
     "smtplib.SMTP_SSL", "telnetlib.Telnet", "xmlrpc.client.ServerProxy"}
)  # fmt: skip
"""Python calls that open a network connection (``urlopen`` and ``urlretrieve`` by any path)."""
_FORMATTERS: dict[str, tuple[str, ...]] = {
    # name: the words that make it rewrite files (empty: it always does)
    "black": (),
    "isort": (),
    "autopep8": ("-i", "--in-place"),
    "autoflake": ("-i", "--in-place"),
    "yapf": ("-i", "--in-place"),
    "docformatter": ("-i", "--in-place"),
    "prettier": ("--write", "-w"),
    "eslint": ("--fix",),
    "gofmt": ("-w",),
    "rustfmt": (),
    "clang-format": ("-i",),
}
_PIP_VALUES = (
    "-r", "--requirement", "-c", "--constraint", "-e", "--editable", "-i", "--index-url",
    "--extra-index-url", "-f", "--find-links", "-t", "--target", "--prefix", "--root", "--src",
    "--upgrade-strategy", "--python-version", "--platform", "--implementation", "--abi",
    "--no-binary", "--only-binary", "--progress-bar", "--log", "--proxy", "--retries",
    "--timeout", "--exists-action", "--trusted-host", "--cert", "--client-cert", "--cache-dir",
    "--report", "-C", "--config-settings", "--python",
)  # fmt: skip
_CURL_VALUES = (
    "-o", "--output", "-d", "--data", "-H", "--header", "-X", "--request", "-u", "--user", "-A",
    "--user-agent", "-e", "--referer", "-b", "--cookie", "-c", "--cookie-jar", "-T",
    "--upload-file", "--data-binary", "--data-raw", "-F", "--form", "--connect-timeout", "-m",
    "--max-time", "-x", "--proxy", "-w", "--write-out", "-K", "--config", "--output-dir",
)  # fmt: skip
_GIT_VALUES = (
    "-b", "-B", "--orphan", "--conflict", "--pathspec-from-file", "-s", "--source", "-e",
    "--exclude", "-m", "--message", "-o", "--origin", "--depth", "--branch", "-u",
    "--upload-pack", "-p", "--directory", "--include", "-c", "-C",
)  # fmt: skip
_XARGS_VALUES = (
    "-a", "--arg-file", "-d", "--delimiter", "-E", "-e", "-L", "-l", "-n", "--max-args", "-P",
    "--max-procs", "-s", "--max-chars", "--process-slot-var",
)  # fmt: skip
_RUNNERS = frozenset({"uv", "poetry", "pipenv", "pdm", "hatch"})
"""Project tools whose ``run`` subcommand runs another command in the project environment."""
_RUNNER_VALUES = (
    "--with", "--python", "-p", "--directory", "--project", "--package", "--extra", "--group",
    "--env-file", "-e", "--env",
)  # fmt: skip
_SUBMODULE_NETWORK = ("update", "add", "sync")
_REMOTE_NETWORK = ("update", "show", "prune")
_Handler = Callable[[str, list["_Word"], "_Simple", "list[_Value] | None"], None]


def _options(
    words: Sequence[_Word], with_value: Iterable[str], *, posix: bool = False
) -> tuple[dict[str, list[str]], list[_Word]]:
    """GNU-style option parsing into options (with values) and operands.

    ``--`` ends the options. With ``posix`` the first operand ends them too,
    as for wrappers such as ``sudo`` whose remaining words are a command.
    """
    takes = frozenset(with_value)
    opts: dict[str, list[str]] = {}
    operands: list[_Word] = []
    index = 0
    while index < len(words):
        word, text = words[index], words[index].text
        index += 1
        if text == "--":
            operands.extend(words[index:])
            break
        if text.startswith("--") and len(text) > 2:
            name, eq, value = text.partition("=")
            if not eq and name in takes and index < len(words):
                value = words[index].text
                index += 1
            opts.setdefault(name, []).append(value)
        elif text.startswith("-") and len(text) > 1 and word.literal:
            for position, char in enumerate(text[1:], start=2):
                name = "-" + char
                if name in takes:
                    value = text[position:]
                    if not value and index < len(words):
                        value = words[index].text
                        index += 1
                    opts.setdefault(name, []).append(value)
                    break
                opts.setdefault(name, []).append("")
        else:
            operands.append(word)
            if posix:
                operands.extend(words[index:])
                break
    return opts, operands


def _has(opts: Mapping[str, list[str]], *names: str) -> bool:
    return any(name in opts for name in names)


def _first(opts: Mapping[str, list[str]], *names: str) -> str | None:
    for name in names:
        if opts.get(name):
            return opts[name][-1]
    return None


def _diff_targets(text: str, strip: int) -> list[str]:
    """Paths a unified diff touches, as written and with ``strip`` leading components removed."""
    targets: list[str] = []
    for line in text.splitlines():
        if not line.startswith(("+++ ", "--- ")):
            continue
        path = line[4:].split("\t", 1)[0].strip()
        if path in ("/dev/null", ""):
            continue
        parts = path.split("/")
        for count in sorted({0, strip}):
            candidate = "/".join(parts[count:])
            if candidate and candidate not in targets:
                targets.append(candidate)
    return targets


class _Shell:
    """Static analysis of one shell command line: what it may write, fetch, or install."""

    def __init__(
        self,
        cwd: str,
        *,
        use_git: bool,
        effects: _Effects | None = None,
        variables: dict[str, list[_Value]] | None = None,
        depth: int = 0,
    ) -> None:
        self.cwd = cwd
        self.use_git = use_git
        self.effects = effects if effects is not None else _Effects()
        self.vars: dict[str, list[_Value]] = (
            variables
            if variables is not None
            else {"PWD": [_Value(cwd)], "HOME": [_Value(os.path.expanduser("~"))]}
        )
        self.depth = depth
        self.dirs: list[str] = []
        self.substitutions: list[str] = []
        self.subst_hints: dict[int, list[_Value] | None] = {}
        self.handlers: dict[str, _Handler] = {
            "sed": self._sed,
            "gsed": self._sed,
            "perl": self._perl,
            "awk": self._awk,
            "gawk": self._awk,
            "tee": self._tee,
            "sponge": self._tee,
            "cp": self._copy,
            "mv": self._copy,
            "install": self._copy,
            "ln": self._copy,
            "rsync": self._rsync,
            "rm": self._rm,
            "unlink": self._rm,
            "shred": self._rm,
            "truncate": self._truncate,
            "touch": self._touch,
            "dd": self._dd,
            "tar": self._tar,
            "unzip": self._unzip,
            "patch": self._patch,
            "find": self._find,
            "xargs": self._xargs,
            "git": self._git,
            "sort": self._sort,
            "uniq": self._uniq,
            "ruff": self._ruff,
            "eval": self._eval,
            "source": self._source,
            ".": self._source,
            "cd": self._cd,
            "pushd": self._cd,
            "popd": self._popd,
            "export": self._export,
            "declare": self._export,
            "typeset": self._export,
            "local": self._export,
            "readonly": self._export,
            "read": self._read,
            "curl": self._curl,
            "wget": self._wget,
            "uv": self._uv,
            "poetry": self._poetry,
            "npm": self._node_pm,
            "pnpm": self._node_pm,
            "yarn": self._node_pm,
            "bun": self._node_pm,
            "conda": self._conda,
            "mamba": self._conda,
            "micromamba": self._conda,
            "apt": self._system_pm,
            "apt-get": self._system_pm,
            "brew": self._system_pm,
            "gem": self._system_pm,
            "cargo": self._cargo,
            "go": self._go,
            "docker": self._docker,
            "podman": self._docker,
        }

    # ------------------------------------------------------------------ driver

    def run(self, command: str) -> None:
        """Analyze ``command`` and add what it may do to :attr:`effects`."""
        lexer = _Lexer(command)
        tokens = lexer.lex()
        self.substitutions = lexer.substitutions
        stack: list[str] = []
        for kind, pipeline in _parse(tokens):
            if kind == _PUSH:
                stack.append(self.cwd)
            elif kind == _POP:
                self.cwd = stack.pop() if stack else self.cwd
            else:
                upstream: list[_Value] | None = None
                for position, simple in enumerate(pipeline):
                    upstream = self._simple(simple, upstream if position else None)
        for index in range(len(self.substitutions)):
            self._substitution(index)

    def _nested(self, command: str, what: str) -> None:
        if self.depth >= MAX_DEPTH:
            self.effects.notes.append(f"{what} nested too deeply to analyze")
            return
        _Shell(
            self.cwd, use_git=self.use_git, effects=self.effects, variables=self.vars,
            depth=self.depth + 1,
        ).run(command)  # fmt: skip

    def _substitution(self, index: int) -> list[_Value] | None:
        """Analyze command substitution ``index`` once; return the paths its output may name."""
        if index not in self.subst_hints:
            command = self.substitutions[index]
            self.subst_hints[index] = None
            self._nested(command, "command substitution")
            self.subst_hints[index] = self._hints_of(command)
        return self.subst_hints[index]

    def _hints_of(self, command: str) -> list[_Value] | None:
        """Paths the output of ``command`` may name (from find, ls, grep, git ls-files, echo)."""
        lexer = _Lexer(command)
        items = _parse(lexer.lex())
        probe = _Shell(
            self.cwd, use_git=False, effects=_Effects(), variables=dict(self.vars), depth=MAX_DEPTH
        )
        probe.substitutions = lexer.substitutions
        hints: list[_Value] = []
        found = False
        for kind, pipeline in items:
            for simple in pipeline if kind == "pipeline" else []:
                resolved = probe._command_words(simple.words)
                result = probe._output_hints(*resolved) if resolved else None
                if result is not None:
                    found = True
                    hints.extend(result)
        return hints if found else None

    # ------------------------------------------------------------------ words

    def _abs(self, text: str) -> str:
        expanded = os.path.expanduser(text) if text.startswith("~") else text
        return os.path.normpath(
            expanded if os.path.isabs(expanded) else os.path.join(self.cwd, expanded)
        )

    def _values(self, word: _Word) -> list[_Value] | None:
        """Every value ``word`` may take, or None when it depends on run-time data."""
        combos: list[tuple[str, bool]] = [("", False)]
        for part in word.parts:
            options: list[_Value] | None
            if part.kind == "lit":
                options = [_Value(part.text)]
            elif part.kind == "var":
                options = self.vars.get(part.text)
            elif part.kind == "subst":
                options = self._substitution(int(part.text))
            else:
                options = None
            if not options:
                return None
            combos = [(t + o.text, r or o.recursive) for t, r in combos for o in options]
            if len(combos) > MAX_EXPANSIONS:
                return None
        values: list[_Value] = []
        for text, recursive in combos:
            for item in _brace_expand(text) if word.brace else [text]:
                item = os.path.expanduser(item) if word.tilde else item
                matches = self._glob(item) if word.glob else [item]
                values.extend(_Value(m, recursive) for m in matches)
        return values

    def _glob(self, pattern: str) -> list[str]:
        """Pathname expansion against the files on disk now; no match keeps the pattern."""
        absolute = pattern if os.path.isabs(pattern) else os.path.join(self.cwd, pattern)
        try:
            matches = sorted(globlib.glob(absolute))
        except (OSError, ValueError, re.error):  # pragma: no cover - glob rarely raises
            matches = []
        return matches[:MAX_EXPANSIONS] or [pattern]

    def _texts(self, word: _Word) -> list[str]:
        return [v.text for v in self._values(word) or []]

    def _write(self, word: _Word, via: str, *, recursive: bool = False) -> None:
        values = self._values(word)
        if values is None:
            self.effects.notes.append(f"{via}: target {word.text!r} depends on run-time values")
            return
        for value in values:
            if value.text:
                self._write_text(value.text, via, recursive=recursive or value.recursive)

    def _write_text(self, text: str, via: str, *, recursive: bool = False) -> None:
        self.effects.writes.append(_Write(self._abs(text), recursive, via))

    def _command_words(self, words: Sequence[_Word]) -> tuple[str, list[_Word]] | None:
        """Strip assignments, keywords, and wrappers; return the command name and arguments."""
        rest = list(words)
        while rest and (_is_assignment(rest[0]) or (rest[0].literal and rest[0].text in _KEYWORDS)):
            rest.pop(0)
        if not rest:
            return None
        values = self._values(rest[0])
        if not values or len(values) != 1:
            self.effects.notes.append(f"command name {rest[0].text!r} depends on run-time values")
            return None
        name, args = values[0].text.rsplit("/", 1)[-1], rest[1:]
        if name in _WRAPPERS:
            opts, operands = _options(args, _WRAPPERS[name], posix=True)
            if name == "command" and _has(opts, "-v", "-V"):
                return None
            return self._command_words(operands)
        if name == "env":
            index = 0
            while index < len(args):
                text = args[index].text
                if text in ("-u", "--unset", "-C", "--chdir", "-S", "--split-string"):
                    index += 2
                elif text.startswith("-") or _is_assignment(args[index]):
                    index += 1
                else:
                    break
            return self._command_words(args[index:])
        if name == "timeout":
            _opts, operands = _options(args, ("-s", "--signal", "-k", "--kill-after"), posix=True)
            return self._command_words(operands[1:])
        if name in _RUNNERS and args and args[0].text == "run":
            opts, operands = _options(args[1:], _RUNNER_VALUES, posix=True)
            extra = opts.get("--with", [])
            if extra and not _has(opts, "--offline"):
                self.effects.network.append(f"{name} run --with")
            if extra:
                self.effects.installs.append((f"{name} run --with", tuple(extra)))
            return self._command_words(operands)
        return name, args

    # ------------------------------------------------------------------ commands

    def _simple(self, command: _Simple, upstream: list[_Value] | None) -> list[_Value] | None:
        """Analyze one simple command; return the paths its output may name."""
        for redirect in command.redirects:
            target = redirect.target
            if target is None:
                continue
            if redirect.op in _WRITE_REDIRECTS:
                if redirect.op == ">&" and (target.text.isdigit() or target.text == "-"):
                    continue
                self._write(target, f"redirect {redirect.op}")
            else:
                self._values(target)
        words = command.words
        for word in words:
            for part in word.parts:
                if part.kind == "subst":
                    self._substitution(int(part.text))
        if words and all(_is_assignment(w) for w in words):
            for word in words:
                self._assign(word)
            return None
        rest = list(words)
        while rest and rest[0].literal and rest[0].text in _KEYWORDS:
            rest.pop(0)
        if rest and rest[0].literal and rest[0].text in ("for", "select"):
            self._for(rest[1:])
            return None
        resolved = self._command_words(rest)
        if resolved is None:
            return None
        name, args = resolved
        if name in _NETWORK_COMMANDS:
            self.effects.network.append(name)
        if _PYTHON.match(name):
            self._python(name, args, self._stdin_text(command))
        elif _PIP.match(name):
            self._pip(args, "pip")
        elif name in _SHELLS:
            self._shell(name, args, self._stdin_text(command))
        elif name in _EDITORS:
            for word in _options(args, ("-c", "-S", "-u", "-i", "-T", "--cmd"))[1]:
                if not word.text.startswith("+"):
                    self._write(word, name)
        elif name in _FORMATTERS:
            self._formatter(name, args)
        elif name in self.handlers:
            self.handlers[name](name, args, command, upstream)
        return self._output_hints(name, args)

    def _stdin_text(self, command: _Simple) -> str | None:
        """Text a command reads on stdin from a heredoc, a here-string, or a ``<`` file."""
        for redirect in command.redirects:
            if redirect.op in ("<<", "<<-") and redirect.heredoc is not None:
                return redirect.heredoc
            if redirect.op == "<<<" and redirect.target is not None:
                return redirect.target.text
            if redirect.op == "<" and redirect.target is not None:
                texts = self._texts(redirect.target)
                if len(texts) == 1:
                    return _read_text(self._abs(texts[0]))
        return None

    def _output_hints(self, name: str, args: list[_Word]) -> list[_Value] | None:
        """Paths a command's output may name, for ``xargs``, ``read`` loops, and ``$(...)``."""
        if name in ("echo", "printf"):
            words = [w for w in args if not w.text.startswith("-")]
            if name == "printf":
                words = words[1:]
            values = [v for w in words for v in self._values(w) or []]
            return values or None
        if name == "git":
            if not args or args[0].text not in ("ls-files", "diff", "grep", "status"):
                return None
            operands = _options(args[1:], ("-e", "-f", "-x", "-X"))[1]
            return self._hint_values(operands[1:] if args[0].text == "grep" else operands)
        if name not in _HINT_COMMANDS:
            return None
        if name == "find":
            roots = []
            for word in args:
                if word.text.startswith(("-", "(", "!", ")", ",")):
                    break
                roots.append(word)
            return self._hint_values(roots)
        opts, operands = _options(
            args, ("-e", "-f", "-m", "-A", "-B", "-C", "--include", "--exclude", "-g", "-t")
        )
        if name in ("grep", "egrep", "fgrep", "rg") and not _has(opts, "-e", "-f", "--files"):
            operands = operands[1:]
        return self._hint_values(operands)

    def _hint_values(self, operands: Sequence[_Word]) -> list[_Value]:
        out = [_Value(v.text, True) for w in operands for v in self._values(w) or []]
        return out or [_Value(self.cwd, True)]

    # ------------------------------------------------------------------ shell state

    def _assign(self, word: _Word) -> None:
        """``NAME=value``: remember the value's possible values, or forget the name."""
        first = word.parts[0]
        match = _ASSIGNMENT.match(first.text)
        if match is None:  # pragma: no cover - callers check _is_assignment
            return
        rest = (_Part("lit", first.text[match.end() :]), *word.parts[1:])
        tilde = rest[0].text.startswith("~")
        values = self._values(_Word(rest, word.glob, word.brace, tilde))
        if values is None:
            self.vars.pop(match.group(1), None)
        else:
            self.vars[match.group(1)] = values

    def _export(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        for word in args:
            if _is_assignment(word):
                self._assign(word)

    def _for(self, words: list[_Word]) -> None:
        if not words or not words[0].literal:
            return
        name = words[0].text
        values: list[_Value] = []
        if len(words) < 2 or words[1].text != "in":
            self.vars.pop(name, None)
            return
        for word in words[2:]:
            found = self._values(word)
            if found is None:
                self.vars.pop(name, None)
                return
            values.extend(found)
        self.vars[name] = values

    def _read(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        for word in _options(args, ("-d", "-n", "-N", "-p", "-t", "-u", "-a"))[1]:
            if upstream:
                self.vars[word.text] = [_Value(v.text, True) for v in upstream]
            else:
                self.vars.pop(word.text, None)

    def _cd(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        operands = [w for w in args if w.text == "-" or not w.text.startswith("-")]
        values = self._values(operands[0]) if operands else [_Value(os.path.expanduser("~"))]
        if not values or len(values) != 1 or (operands and operands[0].text == "-"):
            self.effects.notes.append(f"{name}: run-time directory; paths use the last known one")
            return
        if name == "pushd":
            self.dirs.append(self.cwd)
        self.cwd = self._abs(values[0].text)
        self.vars["PWD"] = [_Value(self.cwd)]

    def _popd(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        if self.dirs:
            self.cwd = self.dirs.pop()
            self.vars["PWD"] = [_Value(self.cwd)]

    # ------------------------------------------------------------------ writers

    def _sed(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        opts, operands = _options(args, ("-e", "--expression", "-f", "--file", "-l"))
        if _has(opts, "-i", "--in-place"):
            script_given = _has(opts, "-e", "--expression", "-f", "--file")
            for word in operands if script_given else operands[1:]:
                self._write(word, "sed -i")

    def _perl(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        opts, operands = _options(args, ("-e", "-E", "-I", "-M", "-m", "-x", "-F", "-C"))
        if _has(opts, "-i"):
            for word in operands if _has(opts, "-e", "-E") else operands[1:]:
                self._write(word, "perl -i")

    def _awk(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        opts, operands = _options(
            args, ("-f", "-v", "-F", "-i", "-l", "-E", "--include", "--file", "--assign")
        )
        if not _has(opts, "-f", "--file", "-E") and operands:
            for target in re.findall(r'>{1,2}\s*"([^"]+)"', operands[0].text):
                self._write_text(target, f"{name} print >")
            operands = operands[1:]
        if {"inplace", "inplace.awk"} & set(opts.get("-i", []) + opts.get("--include", [])):
            for word in operands:
                if not _is_assignment(word):
                    self._write(word, f"{name} -i inplace")

    def _tee(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        for word in _options(args, ("--output-error",))[1]:
            self._write(word, name)

    def _copy(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        values = ["-t", "--target-directory", "-S", "--suffix"]
        if name == "install":
            values += ["-m", "--mode", "-o", "--owner", "-g", "--group"]
        opts, operands = _options(args, values)
        if name == "install" and _has(opts, "-d", "--directory"):
            return
        target = _first(opts, "-t", "--target-directory")
        sources = operands
        if target is not None:
            self._write_text(target, name, recursive=True)
        elif name == "ln" and len(operands) == 1:
            for text in self._texts(operands[0]):
                self._write_text(os.path.basename(text.rstrip("/")), name)
            sources = []
        elif len(operands) >= 2:
            *sources, dest = operands
            recursive = (
                len(sources) > 1
                or dest.text.endswith("/")
                or _has(opts, "-r", "-R", "-a", "--recursive", "--archive")
                or any(os.path.isdir(self._abs(t)) for t in self._texts(dest))
            )
            self._write(dest, name, recursive=recursive)
        else:
            sources = []
        if name == "mv":
            for word in sources:
                self._write(word, "mv (source)", recursive=True)

    def _rsync(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        opts, operands = _options(
            args,
            ("-e", "--rsh", "-f", "--filter", "-T", "--temp-dir", "--exclude", "--include",
             "--exclude-from", "--include-from", "--files-from", "--port", "--timeout"),
        )  # fmt: skip
        remote = re.compile(r"^(?:rsync://|[^/]*:)")
        if any(remote.match(w.text) for w in operands):
            self.effects.network.append("rsync to or from a remote")
        if len(operands) >= 2:
            *sources, dest = operands
            if not remote.match(dest.text):
                self._write(dest, "rsync", recursive=True)
            if _has(opts, "--remove-source-files"):
                for word in sources:
                    self._write(word, "rsync --remove-source-files", recursive=True)

    def _rm(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        opts, operands = _options(args, ("-n", "--iterations", "-s", "--size"))
        recursive = _has(opts, "-r", "-R", "--recursive")
        for word in operands:
            self._write(word, name, recursive=recursive)

    def _truncate(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        for word in _options(args, ("-s", "--size", "-r", "--reference"))[1]:
            self._write(word, "truncate")

    def _touch(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        for word in _options(args, ("-d", "--date", "-r", "--reference", "-t", "--time"))[1]:
            self._write(word, "touch")

    def _dd(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        for word in args:
            if word.text.startswith("of="):
                self._write_text(word.text[3:], "dd of=")

    def _tar(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        first = args[0].text if args else ""
        extract = (not first.startswith("-") and "x" in first) or any(
            w.text in ("--extract", "--get") or re.match(r"^-[A-Za-z]*x", w.text) for w in args
        )
        if not extract:
            return
        opts, _operands = _options(args, ("-C", "--directory", "-f", "--file"))
        directory = _first(opts, "-C", "--directory")
        self._write_text(directory or self.cwd, "tar -x", recursive=True)

    def _unzip(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        opts, _operands = _options(args, ("-d", "-x", "-P"))
        if not _has(opts, "-l", "-t", "-v", "-Z", "-p", "-z"):
            self._write_text(_first(opts, "-d") or self.cwd, "unzip", recursive=True)

    def _patch(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        opts, operands = _options(
            args,
            ("-p", "--strip", "-i", "--input", "-d", "--directory", "-o", "--output", "-r", "-B",
             "-F", "-D", "-V", "-z"),
        )  # fmt: skip
        if _has(opts, "--dry-run"):
            return
        source = _first(opts, "-i", "--input")
        stdin = None if source else self._stdin_text(command)
        saved = self.cwd
        directory = _first(opts, "-d", "--directory")
        self.cwd = self._abs(directory) if directory else self.cwd
        try:
            output = _first(opts, "-o", "--output")
            if output or operands:
                if output:
                    self._write_text(output, "patch -o")
                else:
                    self._write(operands[0], "patch")
                return
            text = _read_text(self._abs(source)) if source else stdin
            if text is None:
                self.effects.notes.append("patch: could not read the patch to find its targets")
                return
            strip = _first(opts, "-p", "--strip") or "1"
            for target in _diff_targets(text, int(strip) if strip.isdigit() else 1):
                self._write_text(target, "patch")
        finally:
            self.cwd = saved

    def _find(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        index = 0
        while index < len(args) and not args[index].text.startswith(("-", "(", "!", ")", ",")):
            index += 1
        roots, expression = self._hint_values(args[:index]), args[index:]
        names = [
            expression[i + 1].text
            for i in range(len(expression) - 1)
            if expression[i].text in ("-name", "-iname")
        ]
        ignored = bool(names) and all(n.endswith(".pyc") or n in EXCLUDED_DIRS for n in names)
        position = 0
        while position < len(expression):
            word = expression[position].text
            position += 1
            if word in ("-fprint", "-fprint0", "-fprintf", "-fls") and position < len(expression):
                self._write(expression[position], f"find {word}")
            elif word == "-delete" and not ignored:
                for root in roots:
                    self._write_text(root.text, "find -delete", recursive=True)
            elif word in ("-exec", "-execdir", "-ok", "-okdir"):
                inner: list[_Word] = []
                while position < len(expression) and expression[position].text not in (";", "+"):
                    inner.append(expression[position])
                    position += 1
                position += 1
                if inner and not ignored:
                    marker = "\0find"
                    self.vars[marker] = roots
                    found = _Word((_Part("var", marker),))
                    self._simple(_Simple([found if w.text == "{}" else w for w in inner]), None)
                    self.vars.pop(marker, None)

    def _xargs(
        self, name: str, args: list[_Word], command: _Simple, upstream: list[_Value] | None
    ) -> None:
        takes = _XARGS_VALUES
        index, replace = 0, None
        while index < len(args) and args[index].text.startswith("-"):
            text = args[index].text
            index += 1
            if text == "--":
                break
            if text in takes:
                index += 1
            elif text == "-I" and index < len(args):
                replace = args[index].text
                index += 1
            elif text.startswith("-I"):
                replace = text[2:]
            elif text == "-i" or text.startswith("--replace"):
                replace = text.partition("=")[2] or "{}"
        marker = "\0xargs"
        self.vars[marker] = list(upstream or [])
        if not upstream:
            self.effects.notes.append("xargs: its input names no known paths")
        piped = _Word((_Part("var", marker),))
        inner = list(args[index:]) or [_lit("echo")]
        inner = [piped if w.text == replace else w for w in inner] if replace else [*inner, piped]
        self._simple(_Simple(inner), None)
        self.vars.pop(marker, None)

    def _sort(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        opts, _operands = _options(args, ("-o", "--output", "-k", "--key", "-t", "-S", "-T"))
        output = _first(opts, "-o", "--output")
        if output:
            self._write_text(output, "sort -o")

    def _uniq(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        operands = _options(args, ("-f", "-s", "-w"))[1]
        if len(operands) >= 2:
            self._write(operands[1], "uniq")

    def _formatter(self, name: str, args: list[_Word]) -> None:
        triggers = _FORMATTERS[name]
        texts = {w.text for w in args}
        if (triggers and not texts & set(triggers)) or texts & {"--check", "--diff", "-l"}:
            return
        for word in [w for w in args if not w.text.startswith("-")] or [_lit(".")]:
            self._write(word, name, recursive=True)

    def _ruff(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        sub, rest = (args[0].text, args[1:]) if args else ("", [])
        texts = {w.text for w in rest}
        if (sub == "format" and not texts & {"--check", "--diff"}) or (
            sub == "check" and texts & {"--fix", "--unsafe-fixes"}
        ):
            operands = _options(rest, ("--config", "--select", "--ignore", "--exclude"))[1]
            for word in operands or [_lit(".")]:
                self._write(word, f"ruff {sub}", recursive=True)

    # ------------------------------------------------------------------ interpreters

    def _shell(self, name: str, args: list[_Word], stdin: str | None) -> None:
        index = 0
        while index < len(args):
            text = args[index].text
            if text in ("-o", "+o", "-O", "+O"):
                index += 2
            elif text.startswith("-") and not text.startswith("--") and "c" in text[1:]:
                if index + 1 < len(args):
                    script = args[index + 1]
                    if script.literal:
                        self._nested(script.text, f"{name} -c")
                    else:
                        self.effects.notes.append(f"{name} -c with a run-time command string")
                return
            elif text.startswith(("-", "+")) and text != "-":
                index += 1
            else:
                break
        if index < len(args) and args[index].text != "-":
            for path in self._texts(args[index]):
                script_text = _read_text(self._abs(path))
                if script_text is not None:
                    self._nested(script_text, f"{name} script")
        elif stdin is not None:
            self._nested(stdin, f"{name} reading stdin")

    def _eval(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        if all(w.literal for w in args):
            self._nested(" ".join(w.text for w in args), "eval")
        else:
            self.effects.notes.append("eval of run-time text")

    def _source(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        for path in self._texts(args[0]) if args else []:
            script = _read_text(self._abs(path))
            if script is not None:
                self._nested(script, f"{name} script")

    def _python(self, name: str, args: list[_Word], stdin: str | None) -> None:
        index = 0
        while index < len(args):
            text = args[index].text
            if text == "-c" or re.fullmatch(r"-[A-Za-z]+c", text):
                if index + 1 < len(args):
                    code = args[index + 1]
                    if code.literal:
                        _PythonScan(self).scan(code.text)
                    else:
                        self.effects.notes.append(f"{name} -c with run-time code")
                return
            if text == "-m":
                if index + 1 < len(args) and args[index + 1].text == "pip":
                    self._pip(args[index + 2 :], f"{name} -m pip")
                return
            if text in ("-W", "-X", "--check-hash-based-pycs"):
                index += 2
            elif text.startswith("-") and text != "-":
                index += 1
            else:
                break
        if index < len(args) and args[index].text != "-":
            for path in self._texts(args[index]):
                script = _read_text(self._abs(path))
                if script is not None:
                    _PythonScan(self).scan(script)
        elif stdin is not None:
            _PythonScan(self).scan(stdin)

    # ------------------------------------------------------------------ network and packages

    @staticmethod
    def _urls(args: Sequence[_Word]) -> list[str]:
        return [w.text for w in args if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", w.text)]

    def _fetch(self, name: str, args: list[_Word]) -> None:
        urls = self._urls(args)
        if not urls or not all(u.lower().startswith("file://") for u in urls):
            self.effects.network.append(name)

    def _curl(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        self._fetch(name, args)
        opts, _operands = _options(args, _CURL_VALUES)
        for target in opts.get("-o", []) + opts.get("--output", []):
            self._write_text(target, "curl -o")
        if _has(opts, "-O", "--remote-name"):
            for url in self._urls(args):
                base = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
                if base and "://" not in base:
                    self._write_text(base, "curl -O")

    def _wget(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        self._fetch(name, args)
        opts, _operands = _options(
            args, ("-O", "--output-document", "-P", "--directory-prefix", "-o", "--output-file")
        )
        for target in opts.get("-O", []) + opts.get("--output-document", []):
            if target != "-":
                self._write_text(target, "wget -O")
        prefix = _first(opts, "-P", "--directory-prefix")
        if prefix:
            self._write_text(prefix, "wget -P", recursive=True)

    def _packages(self, operands: Sequence[_Word]) -> tuple[str, ...]:
        """Operands that name packages to fetch (not local paths or requirement files)."""
        names = []
        for word in operands:
            text = word.text
            local = text.startswith((".", "/", "~")) or text.endswith((".txt", ".toml"))
            if text and not local and not os.path.exists(self._abs(text)):
                names.append(text)
        return tuple(names)

    def _install(self, via: str, operands: Sequence[_Word]) -> None:
        packages = self._packages(operands)
        if packages:
            self.effects.installs.append((via, packages))

    def _pip(self, args: Sequence[_Word], via: str) -> None:
        opts, operands = _options(args, _PIP_VALUES)
        sub = operands[0].text if operands else ""
        if sub not in ("install", "download", "wheel"):
            return
        if not _has(opts, "--no-index"):
            self.effects.network.append(f"{via} {sub}")
        if sub == "install":
            editable = opts.get("-e", []) + opts.get("--editable", [])
            remote = [_lit(v) for v in editable if "://" in v or v.startswith("git+")]
            self._install(f"{via} install", [*operands[1:], *remote])

    def _uv(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        opts, operands = _options(
            args, ("--python", "-p", "--directory", "--project", "--index", "--group", "--extra")
        )
        sub = operands[0].text if operands else ""
        if sub == "pip":
            self._pip(operands[1:], "uv pip")
            return
        tool = sub == "tool" and len(operands) > 1 and operands[1].text in ("install", "run")
        if (sub in ("add", "sync", "lock") or tool) and not _has(opts, "--offline"):
            self.effects.network.append(f"uv {sub}")
        if sub == "add":
            self._install("uv add", operands[1:])

    def _poetry(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        operands = _options(args, ("--group", "-G", "--source", "-E", "--extras", "-C"))[1]
        sub = operands[0].text if operands else ""
        if sub in ("add", "install", "update", "lock", "self"):
            self.effects.network.append(f"poetry {sub}")
        if sub == "add":
            self._install("poetry add", operands[1:])

    def _node_pm(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        operands = _options(args, ("--prefix", "-C", "--dir", "--registry", "--cwd", "-w"))[1]
        sub = operands[0].text if operands else ("install" if name == "yarn" else "")
        network = ("install", "i", "add", "ci", "update", "upgrade", "up", "exec", "dlx", "x",
                   "create", "publish", "view", "info")  # fmt: skip
        if sub in network:
            self.effects.network.append(f"{name} {sub}")
        if sub in ("install", "i", "add"):
            self._install(f"{name} {sub}", operands[1:])

    def _conda(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        operands = _options(args, ("-n", "--name", "-p", "--prefix", "-c", "--channel"))[1]
        sub = operands[0].text if operands else ""
        if sub in ("install", "create", "update", "upgrade"):
            self.effects.network.append(f"{name} {sub}")
        if sub == "install":
            self._install(f"{name} install", operands[1:])

    def _system_pm(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        operands = _options(args, ("-o", "-t", "--target-release"))[1]
        sub = operands[0].text if operands else ""
        if sub in ("install", "update", "upgrade", "download", "fetch", "reinstall", "tap"):
            self.effects.network.append(f"{name} {sub}")

    def _cargo(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        operands = _options(args, ("--manifest-path", "-p", "--package", "--features", "-F"))[1]
        sub = operands[0].text if operands else ""
        if sub in ("install", "add", "fetch", "update", "search", "publish"):
            self.effects.network.append(f"cargo {sub}")
        if sub == "add":
            self._install("cargo add", operands[1:])

    def _go(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        sub = args[0].text if args else ""
        download = sub == "mod" and len(args) > 1 and args[1].text == "download"
        if sub in ("get", "install") or download:
            self.effects.network.append(f"go {sub}")
        if sub == "get":
            self._install("go get", [w for w in args[1:] if not w.text.startswith("-")])

    def _docker(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        if args and args[0].text in ("pull", "push", "login", "search"):
            self.effects.network.append(f"{name} {args[0].text}")

    # ------------------------------------------------------------------ git

    def _git(self, name: str, args: list[_Word], command: _Simple, upstream: Any) -> None:
        index, directory = 0, self.cwd
        while index < len(args) and args[index].text.startswith("-"):
            text = args[index].text
            if text == "-C" and index + 1 < len(args):
                directory = os.path.normpath(
                    os.path.join(directory, os.path.expanduser(args[index + 1].text))
                )
            index += 2 if text in ("-C", "-c", "--git-dir", "--work-tree", "--namespace") else 1
        if index < len(args):
            saved, self.cwd = self.cwd, directory
            try:
                self._git_sub(args[index].text, args[index + 1 :])
            finally:
                self.cwd = saved

    def _local_remote(self, word: _Word | None) -> bool:
        if word is None:
            return False
        text = word.text
        return text.startswith(("/", "./", "../", "~", "file://")) or os.path.isdir(self._abs(text))

    @staticmethod
    def _after_dashdash(words: Sequence[_Word]) -> list[_Word] | None:
        for index, word in enumerate(words):
            if word.text == "--":
                return list(words[index + 1 :])
        return None

    def _git_sub(self, sub: str, rest: list[_Word]) -> None:
        opts, operands = _options(rest, _GIT_VALUES)
        if sub in ("clone", "fetch", "pull", "push", "ls-remote"):
            if not self._local_remote(operands[0] if operands else None):
                self.effects.network.append(f"git {sub}")
            if sub == "clone" and operands:
                if len(operands) > 1:
                    self._write(operands[1], "git clone", recursive=True)
                else:
                    base = operands[0].text.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
                    self._write_text(base or ".", "git clone", recursive=True)
        elif sub in ("submodule", "remote"):
            actions = _SUBMODULE_NETWORK if sub == "submodule" else _REMOTE_NETWORK
            if operands and operands[0].text in actions:
                self.effects.network.append(f"git {sub} {operands[0].text}")
        elif sub == "checkout":
            if _has(opts, "-b", "-B", "--orphan"):
                return
            paths = self._after_dashdash(rest)
            single = paths is None and len(operands) == 1 and self.use_git
            if single and self._git_changes("checkout", ["diff", "--name-only", operands[0].text]):
                return
            for word in operands if paths is None else paths:
                self._write(word, "git checkout", recursive=True)
        elif sub == "switch":
            if operands and not _has(opts, "-c", "-C", "--create", "--orphan") and self.use_git:
                self._git_changes("switch", ["diff", "--name-only", operands[0].text])
        elif sub == "restore":
            if not _has(opts, "-S", "--staged") or _has(opts, "-W", "--worktree"):
                for word in operands:
                    self._write(word, "git restore", recursive=True)
        elif sub == "rm":
            if not _has(opts, "--cached", "-n", "--dry-run"):
                for word in operands:
                    self._write(word, "git rm", recursive=True)
        elif sub == "mv":
            if len(operands) >= 2:
                for word in operands:
                    self._write(word, "git mv", recursive=True)
        elif sub == "clean":
            self._git_clean(opts, operands)
        elif sub == "reset":
            if _has(opts, "--hard", "--merge", "--keep") and self.use_git:
                target = operands[0].text if operands else "HEAD"
                self._git_changes("reset", ["diff", "--name-only", target])
        elif sub == "stash":
            self._git_stash(rest, operands)
        elif sub in ("apply", "am"):
            self._git_apply(sub, opts, operands)

    def _git_clean(self, opts: Mapping[str, list[str]], operands: list[_Word]) -> None:
        if _has(opts, "-n", "--dry-run") or not _has(opts, "-f", "--force"):
            return
        if self.use_git:
            flags = [f for f in ("-d", "-x", "-X") if f in opts]
            listed = self._git_query(["clean", "-n", *flags, "--", *[w.text for w in operands]])
            if listed is not None:
                for line in listed.splitlines():
                    if line.startswith("Would remove "):
                        path = line[len("Would remove ") :]
                        self._write_text(path, "git clean", recursive=path.endswith("/"))
                return
        for word in operands or [_lit(".")]:
            self._write(word, "git clean", recursive=True)

    def _git_stash(self, rest: list[_Word], operands: list[_Word]) -> None:
        action = operands[0].text if operands else "push"
        if action in ("push", "save"):
            paths = self._after_dashdash(rest)
            if paths:
                for word in paths:
                    self._write(word, "git stash", recursive=True)
            elif self.use_git:
                self._git_changes("stash", ["diff", "--name-only", "HEAD"])
        elif action in ("pop", "apply") and self.use_git:
            ref = operands[1].text if len(operands) > 1 else "stash@{0}"
            self._git_changes(f"stash {action}", ["stash", "show", "--name-only", ref])

    def _git_apply(self, sub: str, opts: Mapping[str, list[str]], operands: list[_Word]) -> None:
        if sub == "apply" and (
            _has(opts, "--check", "--stat", "--numstat", "--summary")
            or (_has(opts, "--cached") and not _has(opts, "--index"))
        ):
            return
        strip_text = _first(opts, "-p") or "1"
        strip = int(strip_text) if strip_text.isdigit() else 1
        for word in operands:
            for path in self._texts(word):
                text = _read_text(self._abs(path))
                if text is None:
                    self.effects.notes.append(f"git {sub}: could not read {path}")
                    continue
                for target in _diff_targets(text, strip):
                    self._write_text(target, f"git {sub}")

    def _git_query(self, query: list[str]) -> str | None:
        """Output of a read-only git command in the current directory, or None on failure."""
        try:
            result = subprocess.run(
                ["git", "--no-optional-locks", "-C", self.cwd, *query],
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout if result.returncode == 0 else None

    def _git_changes(self, what: str, query: list[str]) -> bool:
        """Record the files git says a whole-tree command would change; False if git cannot say."""
        listed = self._git_query(query)
        top = self._git_query(["rev-parse", "--show-toplevel"])
        if listed is None or top is None:
            self.effects.notes.append(f"git {what}: git could not say which files would change")
            return False
        for line in listed.splitlines():
            if line.strip():
                path = os.path.join(top.strip(), line.strip())
                self.effects.writes.append(_Write(os.path.normpath(path), False, f"git {what}"))
        return True


class _PythonScan(ast.NodeVisitor):
    """Find file writes, shell commands, and network use in Python source (straight-line)."""

    OPENERS = frozenset({"open", "io.open", "codecs.open", "builtins.open"})
    PATH_TYPES = frozenset({"Path", "PurePath", "PosixPath", "WindowsPath"})
    JOINS = frozenset({"os.path.join", "posixpath.join", "ntpath.join"})
    PASSTHROUGH = frozenset(
        {"os.path.abspath", "os.path.expanduser", "os.path.normpath", "os.path.realpath", "str"}
    )
    PATH_METHODS = frozenset(
        {"write_text", "write_bytes", "unlink", "rmdir", "touch", "symlink_to", "hardlink_to"}
    )
    MOVES = frozenset({"rename", "replace"})
    OS_ONE = frozenset({"os.remove", "os.unlink", "os.rmdir", "os.removedirs", "os.truncate"})
    OS_TWO = frozenset({"os.rename", "os.replace", "os.renames"})
    COPIES = frozenset({"os.symlink", "os.link", "shutil.copy", "shutil.copy2", "shutil.copyfile"})
    SHELL_CALLS = frozenset(
        {"os.system", "os.popen", "subprocess.run", "subprocess.call", "subprocess.check_call",
         "subprocess.check_output", "subprocess.Popen", "subprocess.getoutput",
         "subprocess.getstatusoutput"}
    )  # fmt: skip
    WRITE_FLAGS = frozenset({"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"})

    def __init__(self, shell: _Shell) -> None:
        self.shell = shell
        self.env: dict[str, tuple[str, bool]] = {}
        self.aliases: dict[str, str] = {}

    def scan(self, code: str) -> None:
        """Visit ``code``.

        Code this interpreter cannot parse (for example newer syntax that the
        subject's Python accepts) gets a conservative fallback: ``open()`` calls
        with a write mode, and imports of common network modules.
        """
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            pattern = r"""open\(\s*(['"])(.+?)\1\s*,\s*(['"])[^'"]*[wax+][^'"]*\3"""
            for match in re.finditer(pattern, code):
                self.shell._write_text(match.group(2), "python open()")
            if re.search(r"\b(?:import|from)\s+(?:urllib\.request|requests|httpx|socket)\b", code):
                self._network()
            return
        self.visit(tree)

    def _network(self) -> None:
        if "python network call" not in self.shell.effects.network:
            self.shell.effects.network.append("python network call")

    def _dotted(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = self._dotted(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def _path(self, node: ast.expr | None) -> tuple[str, bool] | None:
        """Static value of a path expression, and whether it is a ``pathlib`` path."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value, False
        if isinstance(node, ast.Name):
            return self.env.get(node.id)
        if isinstance(node, ast.JoinedStr):
            if all(isinstance(v, ast.Constant) and isinstance(v.value, str) for v in node.values):
                texts = [str(v.value) for v in node.values if isinstance(v, ast.Constant)]
                return "".join(texts), False
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left, right = self._path(node.left), self._path(node.right)
            if left and right and left[1]:
                return os.path.join(left[0], right[0]), True
            return None
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            base = self._path(node.value)
            return (os.path.dirname(base[0].rstrip("/")) or ".", True) if base and base[1] else None
        if not isinstance(node, ast.Call):
            return None
        name = self._dotted(node.func) or ""
        if name.rsplit(".", 1)[-1] in self.PATH_TYPES or name in self.JOINS:
            parts = [self._path(a) for a in node.args]
            if any(p is None for p in parts) or (name in self.JOINS and not parts):
                return None
            joined = os.path.join(*[p[0] for p in parts if p]) if parts else "."
            return joined, name not in self.JOINS
        if name in self.PASSTHROUGH and node.args:
            inner = self._path(node.args[0])
            return (inner[0], False) if inner else None
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("resolve", "absolute"):
            return self._path(node.func.value)
        return None

    @staticmethod
    def _arg(node: ast.Call, position: int, keyword: str) -> ast.expr | None:
        if len(node.args) > position:
            return node.args[position]
        return next((k.value for k in node.keywords if k.arg == keyword), None)

    def _sink(self, node: ast.expr | None, via: str, *, recursive: bool = False) -> None:
        value = self._path(node)
        if value is not None:
            self.shell._write_text(value[0], via, recursive=recursive)
        elif node is not None:
            self.shell.effects.notes.append(f"{via}: target computed at run time")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self.aliases[alias.asname] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            full = f"{module}.{alias.name}" if module else alias.name
            self.aliases[alias.asname or alias.name] = full

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        value = self._path(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if value is None:
                    self.env.pop(target.id, None)
                else:
                    self.env[target.id] = value

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        name = self._dotted(node.func) or ""
        if name in self.OPENERS:
            self._open(self._arg(node, 0, "file"), self._arg(node, 1, "mode"))
        elif name in self.OS_ONE:
            self._sink(self._arg(node, 0, "path"), f"python {name}()")
        elif name in self.OS_TWO or name == "shutil.move":
            self._sink(self._arg(node, 0, "src"), f"python {name}()", recursive=True)
            self._sink(self._arg(node, 1, "dst"), f"python {name}()", recursive=True)
        elif name in self.COPIES or name == "shutil.copytree":
            self._sink(self._arg(node, 1, "dst"), f"python {name}()", recursive=True)
        elif name == "shutil.rmtree":
            self._sink(self._arg(node, 0, "path"), "python shutil.rmtree()", recursive=True)
        elif name == "os.open":
            flags = self._arg(node, 1, "flags")
            walked = ast.walk(flags) if flags is not None else iter(())
            used = {n.attr for n in walked if isinstance(n, ast.Attribute)}
            if used & self.WRITE_FLAGS:
                self._sink(self._arg(node, 0, "path"), "python os.open()")
        elif name in self.SHELL_CALLS:
            self._shell_call(node, name)
        elif (
            name.startswith(_NET_PREFIXES)
            or name in _NET_FUNCTIONS
            or (name.rsplit(".", 1)[-1] in ("urlopen", "urlretrieve"))
        ):
            self._network()
        elif isinstance(node.func, ast.Attribute):
            self._path_method(node, node.func)

    def _open(self, target: ast.expr | None, mode: ast.expr | None) -> None:
        """``open(target, mode)``: a write when the mode can write (or is not a constant)."""
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
            if any(flag in mode.value for flag in "wax+"):
                self._sink(target, "python open()")
        elif mode is not None:
            self._sink(target, "python open()")

    def _path_method(self, node: ast.Call, func: ast.Attribute) -> None:
        """Writes through ``pathlib`` methods (``str.replace`` and the like are not paths)."""
        method = func.attr
        receiver = self._path(func.value)
        if method == "open" and receiver is not None and receiver[1]:
            self._open(func.value, self._arg(node, 0, "mode"))
        elif method in self.PATH_METHODS or method in self.MOVES:
            if receiver is not None and receiver[1]:
                self.shell._write_text(receiver[0], f"python Path.{method}()")
                if method in self.MOVES:
                    self._sink(self._arg(node, 0, "target"), f"python Path.{method}()")
            elif receiver is None and method in self.PATH_METHODS:
                self.shell.effects.notes.append(f"python .{method}(): target computed at run time")

    def _shell_call(self, node: ast.Call, name: str) -> None:
        first = self._arg(node, 0, "args")
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            self.shell._nested(first.value, f"python {name}()")
            return
        if isinstance(first, ast.List | ast.Tuple):
            items = [e.value for e in first.elts if isinstance(e, ast.Constant)]
            if len(items) == len(first.elts) and all(isinstance(i, str) for i in items):
                self.shell._nested(shlex.join(str(i) for i in items), f"python {name}()")
                return
        self.shell.effects.notes.append(f"python {name}(): command computed at run time")


def analyze_command(command: str, cwd: str, *, use_git: bool = True) -> _Effects:
    """Statically analyze a shell command line.

    Args:
        command: The command line, as the Bash tool would run it.
        cwd: Directory it runs in.
        use_git: Allow read-only git queries for whole-tree git commands.

    Returns:
        The paths it may write (absolute, with how), network uses, named
        package installs, and notes on what could not be analyzed.
    """
    shell = _Shell(cwd, use_git=use_git)
    shell.run(command)
    return shell.effects


# --------------------------------------------------------------------------- decisions


def _strings(params: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = params.get(name)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(v) for v in value)
    return ()


def _norm(path: str) -> str:
    text = path.strip()
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def _module_path(entry: str, root: str) -> str:
    """Root-relative path of a ``stable_api`` module entry, resolved as the detector does."""
    stem = entry.replace(".", "/")
    candidates = [f"{stem}.py", f"{stem}/__init__.py", f"src/{stem}.py", f"src/{stem}/__init__.py"]
    present = {c: "" for c in candidates if os.path.isfile(os.path.join(root, c))}
    return _norm(resolve_module(entry, present))


@dataclass(frozen=True)
class _FileChange:
    rel: str
    before: str | None
    after: str | None


def _apply_edit(text: str, old: object, new: object, replace_all: object) -> str | None:
    if not isinstance(old, str) or not isinstance(new, str) or not old:
        return None
    count = text.count(old)
    if count == 0 or (count > 1 and replace_all is not True):
        return None
    return text.replace(old, new) if replace_all is True else text.replace(old, new, 1)


def _would_be(tool_name: str, tool_input: Mapping[str, Any], before: str | None) -> str | None:
    """The file's content after the call, or None when it cannot be computed."""
    if tool_name == "Write":
        content = tool_input.get("content")
        return content if isinstance(content, str) else None
    if before is None:
        return None
    if tool_name == "Edit":
        return _apply_edit(
            before,
            tool_input.get("old_string"),
            tool_input.get("new_string"),
            tool_input.get("replace_all"),
        )
    edits = tool_input.get("edits") if tool_name == "MultiEdit" else None
    if not isinstance(edits, list) or not edits:
        return None
    text: str | None = before
    for edit in edits:
        if text is None or not isinstance(edit, Mapping):
            return None
        text = _apply_edit(
            text, edit.get("old_string"), edit.get("new_string"), edit.get("replace_all")
        )
    return text


def _check_file(
    constraint: Constraint, change: _FileChange, root: str, verb: str
) -> Decision | None:
    params, kind = constraint.detector.params, constraint.detector.kind
    rel, before, after = change.rel, change.before, change.after
    if kind == "protected_paths":
        if _hits(rel, _strings(params, "globs"), recursive=False):
            return _deny(constraint, f"this call would {verb} {rel}, a protected path")
        return None
    if after is None:
        return None
    if kind == "no_new_dependencies" and match_any(rel, _strings(params, "manifests")):
        added = sorted(manifest_dependencies(rel, after) - manifest_dependencies(rel, before or ""))
        if added:
            return _deny(constraint, f"this call would add {', '.join(added)} to {rel}")
    elif kind == "max_file_lines" and match_any(rel, _strings(params, "globs")):
        limit = params.get("limit")
        lines, old = _line_count(after), _line_count(before or "")
        if isinstance(limit, int) and lines > limit and lines > old:
            detail = f"this call would leave {rel} at {lines} lines (limit {limit})"
            return _deny(constraint, detail)
    elif kind == "file_invariant":
        path, regex = _strings(params, "path"), _strings(params, "regex")
        if path and regex and rel == _norm(path[0]) and before is not None:
            pattern = re.compile(regex[0])
            if pattern.search(before) and not pattern.search(after):
                return _deny(constraint, f"this call would make {rel} stop matching {regex[0]!r}")
    elif kind == "stable_api":
        modules = {_module_path(m, root) for m in _strings(params, "modules")}
        baseline = public_api(before) if rel in modules and before is not None else None
        if baseline is not None:
            breaks = api_breaks(baseline, public_api(after))
            if breaks:
                shown = "; ".join(breaks[:3])
                detail = f"this call would change the public API of {rel} ({shown})"
                return _deny(constraint, detail)
    return None


def _check_shell(
    constraint: Constraint, command: str, effects: _Effects, targets: list[tuple[str, bool, str]],
    root: str,
) -> Decision | None:  # fmt: skip
    params, kind = constraint.detector.params, constraint.detector.kind
    if kind == "forbidden_commands":
        for pattern in _strings(params, "patterns"):
            if re.search(pattern, command):
                return _deny(constraint, f"the command matches the forbidden pattern {pattern!r}")
        return None
    if kind == "no_network":
        if effects.network:
            return _deny(constraint, f"the command would use the network ({effects.network[0]})")
        return None
    if kind == "no_new_dependencies" and effects.installs:
        via, packages = effects.installs[0]
        return _deny(constraint, f"the command would install {', '.join(packages)} ({via})")
    if kind == "protected_paths":
        globs = _strings(params, "globs")
        for rel, recursive, via in targets:
            if _hits(rel, globs, recursive):
                shown = rel or "the project root"
                what = f"files under {shown}" if recursive and not match_any(rel, globs) else shown
                return _deny(constraint, f"the command would modify {what} ({via})")
        return None
    watched: tuple[str, ...] = ()
    if kind == "no_new_dependencies":
        watched = _strings(params, "manifests")
    elif kind == "file_invariant":
        watched = tuple(_norm(p) for p in _strings(params, "path"))
    elif kind == "stable_api":
        watched = tuple(_module_path(m, root) for m in _strings(params, "modules"))
    for rel, recursive, via in targets:
        if watched and _hits(rel, watched, recursive):
            return _deny(
                constraint,
                f"the command would rewrite {rel or 'files in the project'} ({via}); use the "
                "Edit or Write tool for this file so the guard can check the change",
            )
    return None


def decide(
    tool_name: str,
    tool_input: Mapping[str, Any],
    constraints: Sequence[Constraint],
    cwd: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
    use_git: bool = True,
) -> Decision:
    """Decide whether one Claude Code tool call may run under ``constraints``.

    Args:
        tool_name: Claude Code tool name. Write, Edit, MultiEdit, NotebookEdit,
            and Bash are checked; every other tool is allowed.
        tool_input: The tool's input object from the hook JSON.
        constraints: Rules in force, checked in order; the first refusal wins.
        cwd: Directory that relative paths resolve against (the hook's ``cwd``,
            where Bash commands run).
        root: Directory the spec's globs are anchored at (default: ``cwd``).
        use_git: Allow read-only git queries for whole-tree git commands.

    Returns:
        The :class:`Decision`. Malformed input is allowed with a "not checked" note.
    """
    here = os.fspath(cwd)
    base = os.path.realpath(os.fspath(root) if root is not None else here)
    if not isinstance(tool_input, Mapping):
        return _unchecked("tool_input is not an object")
    enforced = [c for c in constraints if c.detector.kind in ENFORCED_KINDS]
    if tool_name in FILE_TOOLS:
        key = "notebook_path" if tool_name == "NotebookEdit" else "file_path"
        path = tool_input.get(key, tool_input.get("file_path"))
        if not isinstance(path, str) or not path:
            return _unchecked(f"{tool_name} input has no {key}")
        rel = _relative(path, here, base)
        if rel is None:
            return ALLOW
        before = after = None
        if any(c.detector.kind in CONTENT_KINDS for c in enforced):
            before = _read_text(os.path.join(base, rel))
            after = _would_be(tool_name, tool_input, before)
        change = _FileChange(rel, before, after)
        verb = "write" if tool_name == "Write" else "edit"
        for constraint in enforced:
            decision = _check_file(constraint, change, base, verb)
            if decision is not None:
                return decision
        return ALLOW
    if tool_name == "Bash":
        command = tool_input.get("command")
        if not isinstance(command, str):
            return _unchecked("Bash input has no command string")
        effects = analyze_command(command, here, use_git=use_git)
        targets = []
        for write in effects.writes:
            rel = _relative(write.path, here, base)
            if rel is not None:
                targets.append((rel, write.recursive, write.via))
        for constraint in enforced:
            decision = _check_shell(constraint, command, effects, targets, base)
            if decision is not None:
                return decision
        return ALLOW
    return ALLOW


def decide_minimal_agent(
    action: Action, constraints: Sequence[Constraint], workspace_root: str | os.PathLike[str]
) -> Decision:
    """:func:`decide` for one minimal-agent action, through the matching Claude Code tool.

    ``write_file`` becomes Write, ``replace_in_file`` becomes Edit (one exact
    occurrence, as the minimal agent's tool requires), and ``run`` becomes Bash
    in the workspace root; ``list_files``, ``read_file``, and ``done`` are
    always allowed.
    """
    root = os.fspath(workspace_root)
    args = action.args
    if action.tool == "run":
        return decide("Bash", {"command": args.get("cmd")}, constraints, root)
    if action.tool not in ("write_file", "replace_in_file"):
        return ALLOW
    path = args.get("path")
    if not isinstance(path, str) or not path:
        return _unchecked(f"{action.tool} without a path")
    file_path = os.path.join(root, path)
    if action.tool == "write_file":
        tool_input = {"file_path": file_path, "content": args.get("content")}
        return decide("Write", tool_input, constraints, root)
    tool_input = {
        "file_path": file_path,
        "old_string": args.get("old"),
        "new_string": args.get("new"),
        "replace_all": False,
    }
    return decide("Edit", tool_input, constraints, root)


def minimal_agent_guard(
    constraints: Sequence[Constraint], workspace_root: str | os.PathLike[str]
) -> Callable[[Action], str | None]:
    """A :data:`halflife.agent.Guard` for the E4 guard arm: the refusal reason, or None."""

    def guard(action: Action) -> str | None:
        decision = decide_minimal_agent(action, constraints, workspace_root)
        return None if decision.allow else decision.reason

    return guard


_COVERAGE = {
    "protected_paths": "refuses file-tool calls on matching paths and Bash commands changing one",
    "forbidden_commands": "refuses Bash commands that match a pattern",
    "no_network": "refuses Bash commands that would use the network",
    "no_new_dependencies": (
        "refuses manifest edits that add a dependency, Bash writes to manifests, and installs"
    ),
    "max_file_lines": "refuses Write, Edit, and MultiEdit that would grow a file over the limit",
    "file_invariant": "refuses file-tool calls that break the pattern, and Bash writes to the file",
    "stable_api": "refuses file-tool calls that change the public API, and Bash writes to it",
}


def coverage(constraints: Sequence[Constraint]) -> list[tuple[str, str, str]]:
    """For each constraint: its id, its detector kind, and what the guard does about it."""
    fallback = "not enforced by the guard (needs session state or a test run)"
    return [(c.id, c.detector.kind, _COVERAGE.get(c.detector.kind, fallback)) for c in constraints]


# --------------------------------------------------------------------------- command line


def deny_output(reason: str) -> dict[str, Any]:
    """The PreToolUse hook output that refuses a tool call and shows Claude ``reason``."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"halflife-guard: {reason}",
        }
    }


class _UsageError(Exception):
    """A command-line usage error, reported as fail open instead of argparse's exit 2."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="halflife-guard",
        description=(
            "Claude Code PreToolUse hook: reads the hook JSON on stdin and refuses tool "
            "calls that would break a rule in the halflife.yaml constraint spec."
        ),
    )
    parser.add_argument(
        "--spec",
        help="spec file (default: $HALFLIFE_SPEC, ./halflife.yaml, or "
        "$CLAUDE_PROJECT_DIR/halflife.yaml)",
    )
    parser.add_argument(
        "--root",
        help="directory the spec's globs are anchored at (default: $CLAUDE_PROJECT_DIR, "
        "else the hook's cwd)",
    )
    parser.add_argument("--log", help="append one JSON line per decision to this file")
    parser.add_argument("--no-git", action="store_true", help="never run read-only git queries")
    parser.add_argument(
        "--explain", action="store_true", help="print what the guard enforces for each rule"
    )
    return parser


def _note(stream: TextIO, message: str) -> None:
    stream.write(f"halflife-guard: {message}\n")


def _log(path: str, record: Mapping[str, Any], err: TextIO) -> None:
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        _note(err, f"could not write the log {path}: {exc}")


def _hook_call(raw: str) -> tuple[dict[str, Any], str, dict[str, Any]] | str | None:
    """The hook JSON, tool name, and tool input; None for another hook event; or why the
    input cannot be checked."""
    try:
        data = json.loads(raw)
    except ValueError:
        return "hook input is not JSON"
    if not isinstance(data, dict):
        return "hook input is not a JSON object"
    if data.get("hook_event_name", "PreToolUse") != "PreToolUse":
        return None
    tool_name, tool_input = data.get("tool_name"), data.get("tool_input")
    if not isinstance(tool_name, str) or not isinstance(tool_input, dict):
        return "hook input lacks tool_name or tool_input"
    return data, tool_name, tool_input


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run the hook: read the hook JSON on stdin and print a deny decision when needed.

    Returns:
        0 after a decision, for other hook events, and when there is no spec to
        enforce; 1 when the call could not be checked (fail open: Claude Code
        lets the call go ahead and shows a hook error notice).
    """
    inp = sys.stdin if stdin is None else stdin
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    environ = os.environ if env is None else env
    unchecked = "the tool call was not checked (fail open)"
    try:
        args = _parser().parse_args(argv)
    except _UsageError as exc:
        _note(err, f"usage error: {exc}; {unchecked}")
        return EXIT_FAIL_OPEN
    try:
        spec_path = find_spec(args.spec, env=environ)
        if args.explain:
            if spec_path is None:
                _note(err, "no spec found")
                return EXIT_FAIL_OPEN
            for cid, kind, what in coverage(load_spec(spec_path)):
                out.write(f"{cid} ({kind}): {what}\n")
            return EXIT_OK
        call = _hook_call(inp.read())
        if call is None:
            return EXIT_OK
        if isinstance(call, str):
            _note(err, f"{call}; {unchecked}")
            return EXIT_FAIL_OPEN
        data, tool_name, tool_input = call
        if tool_name not in GUARDED_TOOLS:
            return EXIT_OK
        if spec_path is None:
            _note(err, "no halflife.yaml found; nothing to enforce")
            return EXIT_OK
        constraints = load_spec(spec_path)
        hook_cwd = data.get("cwd")
        cwd = hook_cwd if isinstance(hook_cwd, str) and os.path.isdir(hook_cwd) else os.getcwd()
        root = args.root or environ.get(PROJECT_DIR_ENV) or cwd
        use_git = not args.no_git
        decision = decide(tool_name, tool_input, constraints, cwd, root=root, use_git=use_git)
    except SpecError as exc:
        _note(err, f"{exc}; {unchecked}")
        return EXIT_FAIL_OPEN
    except Exception as exc:  # fail open on anything unexpected, never a traceback
        _note(err, f"internal error ({type(exc).__name__}: {exc}); {unchecked}")
        return EXIT_FAIL_OPEN
    if args.log:
        record = {
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "session_id": data.get("session_id"),
            "tool_use_id": data.get("tool_use_id"),
            "tool_name": tool_name,
            "allow": decision.allow,
            "constraint_id": decision.constraint_id,
            "reason": decision.reason,
        }
        _log(args.log, record, err)
    if not decision.allow:
        out.write(json.dumps(deny_output(decision.reason), sort_keys=True) + "\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
