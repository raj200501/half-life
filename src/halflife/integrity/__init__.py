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

All of them use the standard library only, so they run in a fresh clone, and find the
repository root with ``git rev-parse --show-toplevel`` so they work from any directory
inside the repository. This module holds the small helpers they share.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

__all__ = [
    "CheckLog",
    "GitError",
    "first_added_commit",
    "git",
    "head_commit",
    "is_proper_ancestor",
    "is_shallow",
    "is_tracked",
    "repo_root",
]


class GitError(RuntimeError):
    """Raised when a git command fails or the path is not inside a git work tree."""


def git(root: Path, *args: str) -> str:
    """Run a read-only git command in ``root`` and return its stdout.

    Raises:
        GitError: when git exits non-zero or cannot be started.
    """
    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise GitError(f"cannot run git: {exc}") from exc
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def repo_root(start: Path | None = None) -> Path:
    """Top level of the git work tree that contains ``start`` (default: the cwd)."""
    return Path(git(start or Path.cwd(), "rev-parse", "--show-toplevel").strip())


def head_commit(root: Path) -> str | None:
    """SHA of HEAD, or None when the repository has no commits yet."""
    try:
        return git(root, "rev-parse", "--verify", "--quiet", "HEAD^{commit}").strip() or None
    except GitError:
        return None


def is_shallow(root: Path) -> bool:
    """True when the repository is a shallow clone (history is incomplete)."""
    return git(root, "rev-parse", "--is-shallow-repository").strip() == "true"


def first_added_commit(root: Path, rel: str) -> str | None:
    """SHA of the oldest commit that added ``rel``, or None if it was never committed.

    Rename detection is off, so a renamed file counts as added at its new path.
    """
    if head_commit(root) is None:
        return None
    out = git(root, "log", "--no-renames", "--diff-filter=A", "--format=%H", "--reverse", "--", rel)
    lines = out.split()
    return lines[0] if lines else None


def is_proper_ancestor(root: Path, earlier: str, later: str) -> bool:
    """True when commit ``earlier`` is an ancestor of ``later`` and not the same commit."""
    if earlier == later:
        return False
    proc = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), "merge-base", "--is-ancestor", earlier, later],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def is_tracked(root: Path, rel: str) -> bool:
    """True when ``rel`` is in the git index (staged or committed)."""
    return bool(git(root, "ls-files", "--", rel).strip())


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

    def emit(self, stream: TextIO | None = None) -> None:
        """Print every recorded line to ``stream`` (default: stdout)."""
        out = stream if stream is not None else sys.stdout
        for line in self.lines:
            print(line, file=out)
