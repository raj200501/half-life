"""Integrity tooling for ship gates G10, G11, and G13.

Four command-line checks live here, each run as ``python -m halflife.integrity.<name>``:

* :mod:`halflife.integrity.prereg` (``make prereg-verify``, G11): preregistrations are
  frozen once committed and predate every confirmatory artifact that cites them.
* :mod:`halflife.integrity.claimcheck` (``make claimcheck``, G10): every result number
  in the results documents is a claim marker backed by a banked artifact.
* :mod:`halflife.integrity.freshness` (``make freshness``, G10): every live claim is
  re-derived from lower-level artifacts and matches the artifact and the documents.
* :mod:`halflife.integrity.citecheck` (``make citecheck``, G13): every cited URL is
  banked as verified in ``artifacts/citecheck.json``.

They use the standard library only, so they run in a fresh clone, and find the
repository root with ``git rev-parse --show-toplevel`` so they work from any directory
inside the repository. Conventions are documented in docs/CLAIMS.md. This module holds
the helpers they share: read-only git queries, repository-relative path checks, strict
artifact loading, and :class:`CheckLog`, the one-line-per-check report every tool prints.

Every check follows one rule from raise-v1 (see NOTICE): anything that cannot be
verified is a failure, never a skip.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

__all__ = [
    "CheckLog",
    "GitError",
    "IntegrityError",
    "blob_id",
    "commits_touching",
    "first_added_commits",
    "git",
    "head_commit",
    "head_files",
    "in_head",
    "is_proper_ancestor",
    "is_shallow",
    "load_artifact",
    "load_json",
    "load_jsonl",
    "parse_json",
    "repo_root",
    "safe_relpath",
    "short",
    "snapshot",
    "worktree_files",
    "worktree_matches_head",
]


class GitError(RuntimeError):
    """Raised when a git command fails or the path is not inside a git work tree."""


class IntegrityError(ValueError):
    """Raised for one problem that the calling check reports as a single FAIL line."""


# Settings that would change what read-only git output looks like are pinned, so a
# user's configuration (no root-commit diffs, signatures, colour) cannot hide a commit.
_GIT_CONFIG = (
    "-c",
    "core.quotePath=false",
    "-c",
    "log.showRoot=true",
    "-c",
    "log.showSignature=false",
    "-c",
    "color.ui=false",
)


def _run_git(root: Path, args: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", *_GIT_CONFIG, "-C", str(root), *args],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise GitError(f"cannot run git: {exc}") from exc


def git(root: Path, *args: str) -> str:
    """Run a read-only git command in ``root`` and return its stdout.

    Raises:
        GitError: when git exits non-zero or cannot be started.
    """
    proc = _run_git(root, args)
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def repo_root(start: Path | None = None) -> Path:
    """Top level of the git work tree that contains ``start`` (default: the cwd)."""
    return Path(git(start or Path.cwd(), "rev-parse", "--show-toplevel").strip())


def head_commit(root: Path) -> str | None:
    """SHA of HEAD, or None when the repository has no commits yet."""
    proc = _run_git(root, ("rev-parse", "--verify", "--quiet", "HEAD^{commit}"))
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else None


def is_shallow(root: Path) -> bool:
    """True when the repository is a shallow clone (history is incomplete)."""
    return git(root, "rev-parse", "--is-shallow-repository").strip() == "true"


def short(sha: str | None) -> str:
    """First 12 characters of a commit SHA, or ``uncommitted`` for None."""
    return sha[:12] if sha else "uncommitted"


def first_added_commits(root: Path, *pathspecs: str) -> dict[str, str]:
    """Map every path ever added under ``pathspecs`` to the commit that first added it.

    One ``git log`` call covers the whole history. Rename detection is off, so a renamed
    file counts as added at its new path. Merge commits are diffed against every parent
    (``-m``), so a file that first appears in a merge is still found. Commits are walked
    parents first (``--topo-order --reverse``), so "first" means topologically earliest.
    """
    if head_commit(root) is None:
        return {}
    out = git(
        root,
        "log",
        "-m",
        "--no-renames",
        "--topo-order",
        "--reverse",
        "--diff-filter=A",
        "--name-only",
        "--format=%x01%H",
        "--",
        *pathspecs,
    )
    first: dict[str, str] = {}
    commit = ""
    for line in out.splitlines():
        if line.startswith("\x01"):
            commit = line[1:].strip()
        elif line.strip() and commit:
            first.setdefault(line, commit)
    return first


def commits_touching(root: Path, rel: str) -> list[str]:
    """Every commit (full history, newest first) whose tree changes ``rel``."""
    if head_commit(root) is None:
        return []
    out = git(root, "log", "--full-history", "--no-renames", "--format=%H", "--", rel)
    return out.split()


def blob_id(root: Path, rev: str, rel: str) -> str | None:
    """Object id of ``rel`` in commit ``rev``, or None if the path is absent there."""
    proc = _run_git(root, ("rev-parse", "--verify", "--quiet", f"{rev}:{rel}"))
    oid = proc.stdout.strip()
    return oid if proc.returncode == 0 and oid else None


def in_head(root: Path, rel: str) -> bool:
    """True when ``rel`` is a file committed at HEAD."""
    return blob_id(root, "HEAD", rel) is not None


def worktree_matches_head(root: Path, rel: str) -> bool:
    """True when the working-tree file ``rel`` has the same content as at HEAD."""
    head = blob_id(root, "HEAD", rel)
    if head is None or not (root / rel).is_file():
        return False
    return git(root, "hash-object", "--", rel).strip() == head


def is_proper_ancestor(root: Path, earlier: str, later: str) -> bool:
    """True when commit ``earlier`` is an ancestor of ``later`` and not the same commit."""
    if earlier == later:
        return False
    return _run_git(root, ("merge-base", "--is-ancestor", earlier, later)).returncode == 0


def head_files(root: Path, *pathspecs: str) -> list[str]:
    """Sorted paths of the files committed at HEAD under ``pathspecs``."""
    if head_commit(root) is None:
        return []
    out = git(root, "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", *pathspecs)
    return sorted(p for p in out.split("\0") if p)


def worktree_files(root: Path, *pathspecs: str) -> list[str]:
    """Sorted paths that exist on disk and are tracked or untracked but not ignored."""
    out = git(
        root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *pathspecs
    )
    return sorted({p for p in out.split("\0") if p and (root / p).is_file()})


def snapshot(root: Path, rels: Iterable[str]) -> dict[str, tuple[int, int] | None]:
    """Size and modification time of each file (None when absent), to detect writes."""
    state: dict[str, tuple[int, int] | None] = {}
    for rel in rels:
        try:
            info = (root / rel).stat()
        except OSError:
            state[rel] = None
        else:
            state[rel] = (info.st_size, info.st_mtime_ns)
    return state


def safe_relpath(rel: object, prefix: str | None = None) -> str:
    """Validate a repository-relative POSIX path and return it.

    The path must be a non-empty string in normal form (no ``.`` or ``..`` segments, no
    doubled or backward slashes, not absolute) and, when ``prefix`` is given, lie under it.

    Raises:
        IntegrityError: when the path is malformed or outside ``prefix``.
    """
    if not isinstance(rel, str) or not rel:
        raise IntegrityError(f"path {rel!r} is not a non-empty string")
    if "\\" in rel or rel.startswith("/") or "//" in rel:
        raise IntegrityError(f"path {rel!r} is not a relative POSIX path")
    parts = PurePosixPath(rel).parts
    if any(part in {".", ".."} for part in parts) or str(PurePosixPath(rel)) != rel:
        raise IntegrityError(f"path {rel!r} is not in normal form")
    if prefix is not None and not (rel == prefix or rel.startswith(prefix + "/")):
        raise IntegrityError(f"path {rel!r} is not under {prefix}/")
    return rel


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    dupes = sorted({key for key in keys if keys.count(key) > 1})
    if dupes:
        raise IntegrityError(f"duplicate key(s) {', '.join(map(repr, dupes))}")
    return dict(pairs)


def parse_json(text: str, what: str = "JSON") -> Any:
    """Parse JSON text, rejecting duplicate object keys (a reader would see two values).

    Raises:
        IntegrityError: naming ``what`` when the text is not valid JSON or repeats a key.
    """
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except IntegrityError as exc:
        raise IntegrityError(f"{what} has a {exc}") from exc
    except ValueError as exc:
        raise IntegrityError(f"{what} is not valid JSON: {exc}") from exc


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IntegrityError(f"{path.name} cannot be read as UTF-8 text: {exc}") from exc


def load_json(path: Path) -> Any:
    """Parse a UTF-8 JSON file strictly (see :func:`parse_json`).

    Raises:
        IntegrityError: when the file cannot be read, does not parse, or repeats a key.
    """
    return parse_json(_read(path), path.name)


def load_jsonl(path: Path) -> list[Any]:
    """Parse a JSON-lines file strictly; blank lines are skipped.

    Raises:
        IntegrityError: naming the first line that cannot be parsed.
    """
    rows: list[Any] = []
    for number, line in enumerate(_read(path).splitlines(), start=1):
        if line.strip():
            rows.append(parse_json(line, f"{path.name} line {number}"))
    return rows


def load_artifact(path: Path) -> Any:
    """Load a ``.json`` artifact as its value or a ``.jsonl`` artifact as a list of rows.

    Raises:
        IntegrityError: for another file type or a file that does not parse.
    """
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    if path.suffix == ".json":
        return load_json(path)
    raise IntegrityError(f"{path.name}: only .json and .jsonl artifacts are supported")


@dataclass
class CheckLog:
    """One printed line per check, plus a running count of problems.

    Lines start with ``ok``, ``FAIL``, or ``note``; only ``FAIL`` lines count as problems.
    """

    lines: list[str] = field(default_factory=list)
    problems: int = 0

    def ok(self, check: str, detail: str) -> None:
        """Record a passing check."""
        self.lines.append(f"ok   {check}: {detail}")

    def fail(self, check: str, detail: str) -> None:
        """Record a failing check (one problem)."""
        self.problems += 1
        self.lines.append(f"FAIL {check}: {detail}")

    def note(self, text: str) -> None:
        """Record an informational line that is neither a pass nor a problem."""
        self.lines.append(f"note {text}")

    def verdict(self, tool: str, passed: str = "") -> str:
        """Final line: ``<tool>: PASS[ <passed>]`` or ``<tool>: FAIL <n> problem(s)``."""
        if self.problems:
            return f"{tool}: FAIL {self.problems} problem(s)"
        return f"{tool}: PASS{' ' + passed if passed else ''}"

    def emit(self, stream: TextIO | None = None) -> None:
        """Print every recorded line to ``stream`` (default: stdout)."""
        out = stream if stream is not None else sys.stdout
        for line in self.lines:
            print(line, file=out)
