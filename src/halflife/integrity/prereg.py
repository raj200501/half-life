# Adapted from raise-v1 tools/prereg.py (https://github.com/raj200501/raise-v1-public),
# Copyright 2026 Raj Kashikar, licensed under the Apache License, Version 2.0
# (LICENSES/Apache-2.0.txt; see NOTICE). Modified for half-life: the hash chain and
# beacon anchors are replaced by checks over this repository's git history, the
# preregistrations are Markdown files, and confirmatory artifacts are verified to
# postdate the preregistration they cite. Carried over from the original: the "verify"
# report with one problem per line, frozen content that may never change, ordering that
# cannot be rearranged after the fact, and the rule that a check which cannot be
# completed is a failure.
"""Preregistration integrity check for gate G11 (``make prereg-verify``).

Preregistrations are ``prereg/NNNN-slug.md`` files. ``prereg/.gitkeep`` and README
files are ignored; any other file under ``prereg/`` is a failure, so a misnamed
preregistration cannot escape the freeze check. Using the git history of the
repository it runs in, the check verifies:

a. **Frozen.** Every preregistration's content at HEAD equals its content in the commit
   that first added it, no later commit changed or deleted it (a modify-and-revert is
   caught too), no committed preregistration was deleted or renamed, and the working
   tree does not edit one. Uncommitted preregistrations are drafts and are only noted.
   Each frozen file's SHA-256 is printed.
b. **Amendments.** A preregistration whose slug contains ``amend`` or that has an
   ``Amends:`` line is an amendment. Its ``Amends:`` line cites the file it amends by
   repository path (``prereg/NNNN-slug.md``); that file must exist, be committed, be
   numbered before it, and be first committed no later than it; the amendment carries
   a ``YYYY-MM-DD`` date.
c. **Confirmatory artifacts.** Every ``artifacts/**/provenance.json`` that is
   confirmatory (``"confirmatory": true``, or no ``confirmatory`` key and a non-null
   ``prereg``) governs the files in its directory tree, except subtrees with their own
   ``provenance.json``. Every other JSON artifact with ``"confirmatory": true`` and every
   JSON-lines artifact with a ``"confirmatory": true`` row (for example
   ``sessions.jsonl``) is confirmatory too. Each must name, in ``prereg``, a path (or
   list of paths) to a committed preregistration whose first commit is a proper ancestor
   of the first commit of every file it governs. A ``confirmatory`` value that is not a
   boolean, or an artifact that does not parse, is a failure.
d. **Numbering.** Preregistrations are numbered contiguously from 0001 without
   duplicates (drafts included), and committed ones were first committed in number order.

A shallow clone cannot show first commits, so it fails whenever there is anything to
verify, and so does a committed file whose first commit cannot be found. Output is one
line per check, then ``PREREG-VERIFY: PASS`` or ``PREREG-VERIFY: FAIL <n> problem(s)``;
the exit status is 0 only on a pass (2 when the directory is not inside a git work tree).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from halflife.integrity import (
    CheckLog,
    GitError,
    IntegrityError,
    blob_id,
    commits_touching,
    first_added_commits,
    git,
    head_commit,
    head_files,
    is_proper_ancestor,
    is_shallow,
    load_artifact,
    repo_root,
    short,
    worktree_files,
)

__all__ = [
    "Ancestry",
    "Confirmatory",
    "Prereg",
    "amended_paths",
    "blob_sha256",
    "check_amendment",
    "check_confirmatory",
    "check_frozen",
    "check_numbering",
    "classify_prereg_dir",
    "collect_confirmatory",
    "has_iso_date",
    "is_amendment",
    "main",
    "verify",
]

TOOL = "PREREG-VERIFY"
PREREG_DIR = "prereg"
ARTIFACTS_DIR = "artifacts"
PROVENANCE = "provenance.json"
NAME_RE = re.compile(r"^(?P<num>\d{4})-(?P<slug>[A-Za-z0-9][A-Za-z0-9_-]*)\.md$")
IGNORED_RE = re.compile(r"^(?:\.gitkeep|readme(?:\.[A-Za-z0-9]+)?)$", re.IGNORECASE)
PREREG_PATH_RE = re.compile(r"prereg/\d{4}-[A-Za-z0-9][A-Za-z0-9_-]*\.md")
AMENDS_RE = re.compile(
    r"^[ \t>*_-]*(?:\*\*|__)?amends(?:\*\*|__)?[ \t]*:(?P<rest>.*)$",
    re.IGNORECASE | re.MULTILINE,
)
ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


@dataclass(frozen=True)
class Prereg:
    """One preregistration file.

    Attributes:
        path: Repository-relative path, ``prereg/NNNN-slug.md``.
        number: The ``NNNN`` prefix as an integer.
        slug: The part between the number and ``.md``.
        first_commit: Commit that first added it, or None for an uncommitted draft.
        text: Content as frozen (first commit) or, for a draft, as in the working tree.
    """

    path: str
    number: int
    slug: str
    first_commit: str | None
    text: str

    @property
    def committed(self) -> bool:
        """True when the file is committed (frozen)."""
        return self.first_commit is not None


@dataclass(frozen=True)
class Confirmatory:
    """A confirmatory artifact and the files whose history must follow its preregistration.

    Attributes:
        path: The artifact that declares itself confirmatory.
        preregs: Preregistration paths it cites.
        files: Files whose first commit must come after each cited preregistration.
    """

    path: str
    preregs: tuple[str, ...]
    files: tuple[str, ...]


class Ancestry:
    """Memoized commit-ancestry queries (many files share a few first commits)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[tuple[str, str], bool] = {}

    def proper(self, earlier: str, later: str) -> bool:
        """True when ``earlier`` is an ancestor of ``later`` and not the same commit."""
        key = (earlier, later)
        if key not in self._cache:
            self._cache[key] = is_proper_ancestor(self.root, earlier, later)
        return self._cache[key]

    def same_or_before(self, earlier: str, later: str) -> bool:
        """True when ``earlier`` is ``later`` or one of its ancestors."""
        return earlier == later or self.proper(earlier, later)


def blob_sha256(root: Path, rev: str, rel: str) -> str:
    """SHA-256 of the exact bytes of ``rel`` in commit ``rev``.

    Raises:
        GitError: when the blob cannot be read.
    """
    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), "cat-file", "blob", f"{rev}:{rel}"],
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise GitError(f"cannot run git: {exc}") from exc
    if proc.returncode != 0:
        raise GitError(f"git cat-file blob {rev}:{rel} failed")
    return hashlib.sha256(proc.stdout).hexdigest()


def classify_prereg_dir(paths: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split paths under ``prereg/`` into preregistration files and unrecognized files.

    ``.gitkeep`` and README files directly in ``prereg/`` are dropped. Anything else that
    is not a ``prereg/NNNN-slug.md`` file (including files in subdirectories) is
    unrecognized.
    """
    preregs: list[str] = []
    unknown: list[str] = []
    for rel in sorted(set(paths)):
        inner = PurePosixPath(rel).relative_to(PREREG_DIR)
        if len(inner.parts) == 1 and IGNORED_RE.match(inner.name):
            continue
        if len(inner.parts) == 1 and NAME_RE.match(inner.name):
            preregs.append(rel)
        else:
            unknown.append(rel)
    return preregs, unknown


def amended_paths(text: str) -> list[str]:
    """Preregistration paths named on ``Amends:`` lines, in order of appearance."""
    found: list[str] = []
    for match in AMENDS_RE.finditer(text):
        for path in PREREG_PATH_RE.findall(match.group("rest")):
            if path not in found:
                found.append(path)
    return found


def is_amendment(prereg: Prereg) -> bool:
    """True when the slug contains ``amend`` or the text has an ``Amends:`` line."""
    return "amend" in prereg.slug.lower() or AMENDS_RE.search(prereg.text) is not None


def has_iso_date(text: str) -> bool:
    """True when the text contains a valid calendar date written ``YYYY-MM-DD``."""
    for match in ISO_DATE_RE.finditer(text):
        try:
            dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            continue
        return True
    return False


def _cited_preregs(value: Any) -> tuple[str, ...] | None:
    """The ``prereg`` field as a tuple of strings, or None when it has another shape."""
    if isinstance(value, str) and value:
        return (value,)
    if isinstance(value, list) and value and all(isinstance(v, str) and v for v in value):
        return tuple(dict.fromkeys(value))
    return None


def _governed(directory: str, files: Sequence[str], provenance_dirs: set[str]) -> tuple[str, ...]:
    """Files under ``directory`` whose nearest ``provenance.json`` is the one in it."""
    owned: list[str] = []
    for rel in files:
        parent = PurePosixPath(rel).parent
        if not (str(parent) == directory or str(parent).startswith(directory + "/")):
            continue
        nearest = next(
            (str(d) for d in (parent, *parent.parents) if str(d) in provenance_dirs), directory
        )
        if nearest == directory:
            owned.append(rel)
    return tuple(owned)


def _claims_confirmatory(name: str, data: dict[str, Any]) -> bool:
    """True when a JSON object declares itself confirmatory (see check c)."""
    if data.get("confirmatory") is True:
        return True
    return name == PROVENANCE and "confirmatory" not in data and data.get("prereg") is not None


def collect_confirmatory(root: Path, files: Sequence[str], log: CheckLog) -> list[Confirmatory]:
    """Find every confirmatory artifact among ``files``; shape problems go to ``log``."""
    found: list[Confirmatory] = []
    provenance_dirs = {
        str(PurePosixPath(p).parent) for p in files if PurePosixPath(p).name == PROVENANCE
    }
    for rel in files:
        name = PurePosixPath(rel).name
        if not name.endswith((".json", ".jsonl")):
            continue
        check = f"confirmatory {rel}"
        try:
            data: Any = load_artifact(root / rel)
        except IntegrityError as exc:
            log.fail(check, f"cannot be parsed, so it cannot be verified: {exc}")
            continue
        if name.endswith(".jsonl"):
            _collect_rows(rel, data, found, log)
            continue
        if not isinstance(data, dict):
            if name == PROVENANCE:
                log.fail(check, "provenance is not a JSON object")
            continue
        flag = data.get("confirmatory")
        if "confirmatory" in data and not isinstance(flag, bool):
            log.fail(check, f"'confirmatory' must be true or false, not {flag!r}")
            continue
        if not _claims_confirmatory(name, data):
            continue
        cited = _cited_preregs(data.get("prereg"))
        if cited is None:
            log.fail(check, "is confirmatory but 'prereg' does not name a preregistration path")
            continue
        directory = str(PurePosixPath(rel).parent)
        targets = _governed(directory, files, provenance_dirs) if name == PROVENANCE else (rel,)
        found.append(Confirmatory(rel, cited, targets))
    return found


def _collect_rows(rel: str, rows: list[Any], found: list[Confirmatory], log: CheckLog) -> None:
    check = f"confirmatory {rel}"
    cited: list[str] = []
    bad_flag = missing = 0
    for row in rows:
        if not isinstance(row, dict) or "confirmatory" not in row:
            continue
        flag = row["confirmatory"]
        if not isinstance(flag, bool):
            bad_flag += 1
        elif flag:
            names = _cited_preregs(row.get("prereg"))
            if names is None:
                missing += 1
            else:
                cited.extend(n for n in names if n not in cited)
    if bad_flag:
        log.fail(check, f"{bad_flag} row(s) have a 'confirmatory' value that is not a boolean")
    if missing:
        log.fail(check, f"{missing} confirmatory row(s) do not name a preregistration in 'prereg'")
    if cited:
        found.append(Confirmatory(rel, tuple(cited), (rel,)))


def check_frozen(root: Path, rel: str, first: str, log: CheckLog) -> None:
    """Check (a) for one committed preregistration first added in ``first``."""
    check = f"frozen {rel}"
    frozen = blob_id(root, first, rel)
    if blob_id(root, "HEAD", rel) != frozen:
        log.fail(check, f"content at HEAD differs from its first commit {short(first)}")
        return
    for commit in commits_touching(root, rel):
        if blob_id(root, commit, rel) != frozen:
            log.fail(check, f"changed after it was frozen in {short(first)}: see {short(commit)}")
            return
    if not (root / rel).is_file():
        log.fail(check, "deleted in the working tree after it was frozen")
        return
    if git(root, "hash-object", "--", rel).strip() != frozen:
        log.fail(check, "edited in the working tree after it was frozen")
        return
    digest = blob_sha256(root, first, rel)
    log.ok(check, f"unchanged since its first commit {short(first)}, sha256 {digest}")


def check_amendment(
    prereg: Prereg, known: dict[str, Prereg], ancestry: Ancestry, log: CheckLog
) -> None:
    """Check (b) for one amendment against the known preregistrations."""
    check = f"amendment {prereg.path}"
    targets = amended_paths(prereg.text)
    problems: list[str] = []
    if not targets:
        problems.append("does not cite the file it amends (add 'Amends: prereg/NNNN-slug.md')")
    for target in targets:
        original = known.get(target)
        if target == prereg.path:
            problems.append("cites itself as the file it amends")
        elif original is None:
            problems.append(f"amends {target}, which does not exist")
        elif original.number >= prereg.number:
            problems.append(f"is not numbered after {target}")
        elif original.first_commit is None:
            problems.append(f"amends {target}, which is not committed")
        elif prereg.first_commit is not None and not ancestry.same_or_before(
            original.first_commit, prereg.first_commit
        ):
            problems.append(f"was first committed before {target}")
    if not has_iso_date(prereg.text):
        problems.append("is not dated (write the date as YYYY-MM-DD)")
    for problem in problems:
        log.fail(check, problem)
    if not problems:
        log.ok(check, f"amends {', '.join(targets)}, is numbered after it, and is dated")


def check_numbering(preregs: Sequence[Prereg], ancestry: Ancestry, log: CheckLog) -> None:
    """Check (d): contiguous from 0001, no duplicates, first commits in number order."""
    if not preregs:
        log.ok("numbering", "no preregistrations in prereg/")
        return
    by_number: dict[int, list[str]] = {}
    for prereg in preregs:
        by_number.setdefault(prereg.number, []).append(prereg.path)
    problems = [
        f"number {number:04d} is used by {', '.join(sorted(paths))}"
        for number, paths in sorted(by_number.items())
        if len(paths) > 1
    ]
    gaps = [f"{n:04d}" for n in range(1, max(by_number) + 1) if n not in by_number]
    if gaps:
        problems.append(f"not contiguous from 0001: missing {', '.join(gaps)}")
    for problem in problems:
        log.fail("numbering", problem)
    if not problems:
        last = max(by_number)
        log.ok("numbering", f"{len(preregs)} preregistration(s), contiguous 0001 to {last:04d}")
    committed = sorted((p for p in preregs if p.first_commit is not None), key=lambda p: p.number)
    out_of_order = [
        f"{later.path} was first committed in {short(later.first_commit)}, which does not "
        f"follow {earlier.path} ({short(earlier.first_commit)})"
        for earlier, later in itertools.pairwise(committed)
        if earlier.first_commit is not None
        and later.first_commit is not None
        and not ancestry.same_or_before(earlier.first_commit, later.first_commit)
    ]
    for problem in out_of_order:
        log.fail("order", problem)
    if len(committed) > 1 and not out_of_order:
        log.ok("order", "first commits follow the numbering")


def check_confirmatory(
    artifacts: Sequence[Confirmatory],
    known: dict[str, Prereg],
    first_commits: dict[str, str],
    ancestry: Ancestry,
    log: CheckLog,
    committed: frozenset[str] = frozenset(),
) -> None:
    """Check (c) for every confirmatory artifact.

    ``committed`` lists the artifact files committed at HEAD: each of them needs a known
    first commit, or the check cannot be completed and fails.
    """
    if not artifacts:
        log.ok("confirmatory", "no confirmatory artifacts under artifacts/")
        return
    for artifact in artifacts:
        check = f"confirmatory {artifact.path}"
        problems = [
            f"{rel} is committed but its first commit cannot be found"
            for rel in artifact.files
            if rel in committed and rel not in first_commits
        ]
        for cited in artifact.preregs:
            prereg = known.get(cited)
            if not PREREG_PATH_RE.fullmatch(cited):
                problems.append(f"cites {cited!r}, which is not a prereg/NNNN-slug.md path")
            elif prereg is None:
                problems.append(f"cites {cited}, which does not exist")
            elif prereg.first_commit is None:
                problems.append(f"cites {cited}, which is not committed")
            else:
                problems.extend(
                    f"{rel} was first committed in {short(first_commits[rel])}, which does not "
                    f"come after {cited} ({short(prereg.first_commit)})"
                    for rel in artifact.files
                    if rel in first_commits
                    and not ancestry.proper(prereg.first_commit, first_commits[rel])
                )
        for problem in problems:
            log.fail(check, problem)
        if problems:
            continue
        pending = [rel for rel in artifact.files if rel not in first_commits]
        done = len(artifact.files) - len(pending)
        log.ok(check, f"cites {', '.join(artifact.preregs)}; {done} file(s) committed after it")
        if pending:
            log.note(f"{artifact.path}: {len(pending)} file(s) not committed yet: {pending[0]}")


def _load_preregs(
    root: Path,
    paths: Sequence[str],
    committed: set[str],
    first_commits: dict[str, str],
    log: CheckLog,
) -> dict[str, Prereg]:
    known: dict[str, Prereg] = {}
    for rel in paths:
        match = NAME_RE.match(PurePosixPath(rel).name)
        assert match is not None  # classify_prereg_dir only returns matching names
        number, slug = int(match.group("num")), match.group("slug")
        if rel in committed and rel not in first_commits:
            log.fail(f"frozen {rel}", "committed but its first commit cannot be found")
            continue
        first = first_commits.get(rel) if rel in committed else None
        try:
            if first is not None:
                text = git(root, "show", f"{first}:{rel}")
            else:
                text = (root / rel).read_text(encoding="utf-8")
        except (GitError, OSError, UnicodeDecodeError) as exc:
            log.fail(f"read {rel}", f"cannot be read, so it cannot be verified: {exc}")
            continue
        known[rel] = Prereg(rel, number, slug, first, text)
    return known


def verify(root: Path) -> CheckLog:
    """Run checks (a) to (d) on the repository at ``root``; the last line is the verdict."""
    log = CheckLog()
    committed_paths = head_files(root, PREREG_DIR)
    disk_paths = worktree_files(root, PREREG_DIR)
    prereg_paths, unknown = classify_prereg_dir([*committed_paths, *disk_paths])
    for rel in unknown:
        log.fail("layout", f"{rel} is not a preregistration name (prereg/NNNN-slug.md)")
    if not unknown:
        log.ok("layout", f"prereg/ holds {len(prereg_paths)} preregistration file(s), nothing else")

    confirmatory = collect_confirmatory(root, worktree_files(root, ARTIFACTS_DIR), log)
    if head_commit(root) is not None and is_shallow(root) and (committed_paths or confirmatory):
        log.fail("history", "shallow clone: first commits cannot be trusted (fetch full history)")
        log.lines.append(log.verdict(TOOL))
        return log

    first_commits = first_added_commits(root, PREREG_DIR, ARTIFACTS_DIR)
    committed = set(committed_paths)
    ever_added, _ = classify_prereg_dir(p for p in first_commits if p.startswith(PREREG_DIR + "/"))
    for rel in ever_added:
        if rel not in committed:
            first = short(first_commits[rel])
            log.fail(f"frozen {rel}", f"deleted or renamed after it was frozen in {first}")

    known = _load_preregs(root, prereg_paths, committed, first_commits, log)
    for rel, prereg in sorted(known.items()):
        if prereg.first_commit is None:
            log.note(f"draft {rel}: not committed, so not frozen yet")
        else:
            check_frozen(root, rel, prereg.first_commit, log)
    ancestry = Ancestry(root)
    for _, prereg in sorted(known.items()):
        if is_amendment(prereg):
            check_amendment(prereg, known, ancestry, log)
    check_numbering(list(known.values()), ancestry, log)
    committed_artifacts = frozenset(head_files(root, ARTIFACTS_DIR))
    check_confirmatory(confirmatory, known, first_commits, ancestry, log, committed_artifacts)
    if not prereg_paths and not confirmatory and not log.problems:
        log.note("nothing to verify: no preregistrations and no confirmatory artifacts")
    log.lines.append(log.verdict(TOOL))
    return log


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; prints the report and returns the exit status."""
    parser = argparse.ArgumentParser(description="Verify the preregistration chain (gate G11).")
    parser.add_argument("--root", type=Path, help="any directory inside the repository")
    args = parser.parse_args(argv)
    try:
        root = repo_root(args.root)
    except GitError as exc:
        print(f"{TOOL}: FAIL not inside a git work tree ({exc})")
        return 2
    try:
        log = verify(root)
    except GitError as exc:
        print(f"{TOOL}: FAIL a git query failed, so the check cannot be completed ({exc})")
        return 1
    log.emit()
    return 1 if log.problems else 0


if __name__ == "__main__":
    sys.exit(main())
