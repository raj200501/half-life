"""Disposable git workspaces with per-call file hashing and a blob store.

A :class:`Workspace` materializes a template's ``files`` mapping into a fresh
temporary directory outside the repository, makes it a git repository with one
deterministic baseline commit, and hashes every file on demand
(:meth:`Workspace.snapshot`). Detectors decide whether a file changed only from
these hashes, never from parsing commands (docs/DESIGN.md).

Snapshot rules:

* Every regular file is hashed with SHA-256 over its bytes.
* Directories named ``.git``, ``__pycache__``, or ``.pytest_cache`` (at any
  depth) and files ending in ``.pyc`` are excluded. Everything else is
  included, so other tool caches (for example ``.mypy_cache``) do show up.
* Symbolic links are never followed. Their hash is the SHA-256 of
  ``b"symlink\\0" + target``, so retargeting a link counts as a change.
* Other special files (FIFOs, sockets, devices) hash a fixed marker per kind.

Blob store: every regular-file version seen by :meth:`Workspace.snapshot` is
kept as text, keyed by its SHA-256, unless it is larger than
:data:`MAX_BLOB_BYTES`, is not valid UTF-8, or contains a NUL byte. Skipped
versions are recorded in :attr:`Workspace.skipped` with the reason.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType

from halflife.schema import Template

MAX_BLOB_BYTES = 256 * 1024
"""Largest file version (in bytes) kept in the blob store."""

EXCLUDED_DIRS: frozenset[str] = frozenset({".git", "__pycache__", ".pytest_cache"})
"""Directory names skipped at any depth by snapshots and listings."""

EXCLUDED_SUFFIXES: tuple[str, ...] = (".pyc",)
"""File name suffixes skipped by snapshots and listings."""

BASELINE_DATE = "2026-01-01T00:00:00+00:00"
"""Author and committer date of every baseline commit (keeps commit ids stable)."""

GIT_IDENTITY: Mapping[str, str] = {
    "GIT_AUTHOR_NAME": "half-life",
    "GIT_AUTHOR_EMAIL": "half-life@example.invalid",
    "GIT_COMMITTER_NAME": "half-life",
    "GIT_COMMITTER_EMAIL": "half-life@example.invalid",
}
"""Fixed git identity for workspace commits."""

SkipReason = str
"""Why a file version is missing from the blob store: ``too_large``, ``binary``,
``symlink``, ``special``, or ``unreadable``."""


class WorkspaceError(RuntimeError):
    """Raised when a workspace cannot be created or used."""


class WorkspacePathError(ValueError):
    """Raised for a path that is absolute, malformed, or escapes the workspace."""


def hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def is_excluded(rel: str) -> bool:
    """Whether a workspace-relative POSIX path is left out of snapshots and listings."""
    parts = PurePosixPath(rel).parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return True
    return rel.endswith(EXCLUDED_SUFFIXES)


def git_env(home: Path | str) -> dict[str, str]:
    """A minimal, deterministic environment for the harness's own git commands.

    It ignores system and global git configuration, fixes the identity and the
    dates, and keeps only ``PATH`` from the host environment.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_DATE": BASELINE_DATE,
        "GIT_COMMITTER_DATE": BASELINE_DATE,
    }
    env.update(GIT_IDENTITY)
    return env


@dataclass(frozen=True)
class SnapshotDiff:
    """Paths that differ between two snapshots, each tuple sorted."""

    created: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]

    @property
    def changed(self) -> tuple[str, ...]:
        """Every created, modified, or deleted path, sorted."""
        return tuple(sorted((*self.created, *self.modified, *self.deleted)))

    def __bool__(self) -> bool:
        """True when anything changed."""
        return bool(self.created or self.modified or self.deleted)


def diff_snapshots(before: Mapping[str, str], after: Mapping[str, str]) -> SnapshotDiff:
    """Compare two ``{path: sha256}`` snapshots."""
    created = sorted(p for p in after if p not in before)
    deleted = sorted(p for p in before if p not in after)
    modified = sorted(p for p in after if p in before and before[p] != after[p])
    return SnapshotDiff(tuple(created), tuple(modified), tuple(deleted))


def remove_tree(path: Path) -> None:
    """Delete a directory tree, restoring write permission where needed."""
    if not path.exists() and not path.is_symlink():
        return
    try:
        shutil.rmtree(path)
        return
    except OSError:
        pass
    for dirpath, dirnames, _ in os.walk(path):
        for name in dirnames:
            full = os.path.join(dirpath, name)
            if not os.path.islink(full):
                try:
                    os.chmod(full, stat.S_IRWXU)
                except OSError:
                    continue
    with suppress(OSError):
        os.chmod(path, stat.S_IRWXU)
    shutil.rmtree(path, ignore_errors=True)


class Workspace:
    """A temporary git workspace seeded from a template's files.

    Use it as a context manager (or call :meth:`close`) so the directory is
    removed afterwards.

    Args:
        files: Seed repository, workspace-relative POSIX path to text content.
        git: Initialize a git repository with one baseline commit.
        prefix: Prefix of the temporary directory name.
        parent: Directory to create the workspace in (default: the system
            temporary directory). It must lie outside this repository.
        uncommitted: Files written after the baseline commit, so they show up
            as local, uncommitted edits (for example a user's work in progress
            that ``git checkout --`` or ``git stash`` would discard). They are
            part of the baseline snapshot.
    """

    def __init__(
        self,
        files: Mapping[str, str],
        *,
        git: bool = True,
        prefix: str = "halflife-ws-",
        parent: Path | None = None,
        uncommitted: Mapping[str, str] | None = None,
    ) -> None:
        self._root = Path(tempfile.mkdtemp(prefix=prefix, dir=parent)).resolve()
        self._closed = False
        self.blobs: dict[str, str] = {}
        """Text of every file version seen, by SHA-256."""
        self.skipped: dict[str, SkipReason] = {}
        """File versions seen but not stored, by SHA-256, with the reason."""
        self.baseline_commit: str | None = None
        """Commit id of the baseline commit (None without git)."""
        try:
            self._seed(files)
            if git:
                self.baseline_commit = self._git_baseline()
            self._seed(uncommitted or {})
            self.baseline: dict[str, str] = self.snapshot()
            """Snapshot taken right after seeding (before tool call 0)."""
        except BaseException:
            self.close()
            raise

    def _seed(self, files: Mapping[str, str]) -> None:
        for rel, content in sorted(files.items()):
            target = self.resolve(rel)
            if target == self._root:
                raise WorkspacePathError(f"file path {rel!r} names the workspace root")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="") as handle:
                handle.write(content)

    @classmethod
    def from_template(
        cls,
        template: Template,
        *,
        git: bool = True,
        prefix: str = "halflife-ws-",
        parent: Path | None = None,
    ) -> Workspace:
        """Create a workspace from a :class:`~halflife.schema.Template`'s files."""
        return cls(template.files, git=git, prefix=prefix, parent=parent)

    # ------------------------------------------------------------------ lifecycle

    @property
    def root(self) -> Path:
        """Absolute, symlink-free path of the workspace directory."""
        return self._root

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has removed the directory."""
        return self._closed

    def close(self) -> None:
        """Remove the workspace directory (idempotent)."""
        if not self._closed:
            self._closed = True
            remove_tree(self._root)

    def __enter__(self) -> Workspace:
        """Enter the context; returns the workspace."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Remove the workspace directory."""
        self.close()

    # ------------------------------------------------------------------ git

    def _git(self, *args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self._root,
                env=git_env(self._root),
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorkspaceError(f"git {args[0]} failed: {exc}") from exc
        if result.returncode != 0:
            raise WorkspaceError(f"git {args[0]} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def _git_baseline(self) -> str:
        self._git("-c", "init.defaultBranch=main", "init", "-q")
        self._git("add", "-A", "--force", ".")
        self._git(
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--no-verify",
            "--allow-empty",
            "-m",
            "baseline",
        )
        return self._git("rev-parse", "HEAD")

    # ------------------------------------------------------------------ paths

    def resolve(self, rel: str) -> Path:
        """Absolute path for a workspace-relative POSIX path.

        ``"."`` names the workspace root. Raises :class:`WorkspacePathError` for
        an empty or absolute path, a NUL byte, ``..`` that climbs above the
        root, or a symbolic link that leads outside the workspace.
        """
        if not isinstance(rel, str) or not rel:
            raise WorkspacePathError("path must be a non-empty string")
        if "\0" in rel:
            raise WorkspacePathError("path contains a NUL byte")
        pure = PurePosixPath(rel)
        if pure.is_absolute():
            raise WorkspacePathError(f"path {rel!r} is absolute; use a workspace-relative path")
        parts: list[str] = []
        for part in pure.parts:
            if part == "..":
                if not parts:
                    raise WorkspacePathError(f"path {rel!r} escapes the workspace")
                parts.pop()
            elif part != ".":
                parts.append(part)
        candidate = self._root.joinpath(*parts)
        real = candidate.resolve(strict=False)
        if real != self._root and self._root not in real.parents:
            raise WorkspacePathError(f"path {rel!r} leads outside the workspace")
        return candidate

    def relpath(self, path: Path) -> str:
        """Workspace-relative POSIX form of an absolute path inside the workspace."""
        return path.relative_to(self._root).as_posix()

    # ------------------------------------------------------------------ files

    def read_text(self, rel: str) -> str:
        """Text content of a workspace file (UTF-8, newlines untouched)."""
        data = self.resolve(rel).read_bytes()
        if b"\0" in data:
            raise UnicodeDecodeError("utf-8", data, 0, 1, "file contains NUL bytes")
        return data.decode("utf-8")

    def write_text(self, rel: str, content: str) -> int:
        """Write a workspace file, creating parent directories; returns characters written."""
        target = self.resolve(rel)
        if target == self._root:
            raise IsADirectoryError(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            return handle.write(content)

    def list_files(self, rel: str = ".") -> list[str]:
        """Sorted workspace-relative paths of files under ``rel`` (snapshot exclusions apply)."""
        base = self.resolve(rel)
        if base.is_symlink() or base.is_file():
            name = self.relpath(base)
            return [] if is_excluded(name) else [name]
        if not base.is_dir():
            raise FileNotFoundError(rel)
        return sorted(name for name, _ in self._walk(base))

    def _walk(self, base: Path) -> Iterator[tuple[str, Path]]:
        for dirpath, dirnames, filenames in os.walk(base):
            current = Path(dirpath)
            keep: list[str] = []
            for name in dirnames:
                full = current / name
                if name in EXCLUDED_DIRS:
                    continue
                if full.is_symlink():
                    filenames.append(name)
                else:
                    keep.append(name)
            dirnames[:] = sorted(keep)
            for name in filenames:
                full = current / name
                rel = self.relpath(full)
                if not is_excluded(rel):
                    yield rel, full

    def snapshot(self) -> dict[str, str]:
        """Hash every file: ``{relative_posix_path: sha256}`` with keys sorted.

        Also records every regular-file version seen in :attr:`blobs` (or in
        :attr:`skipped`). See the module docstring for the exclusion rules.
        """
        hashes: dict[str, str] = {}
        for rel, full in self._walk(self._root):
            hashes[rel] = self._hash_entry(full)
        return dict(sorted(hashes.items()))

    def _hash_entry(self, full: Path) -> str:
        try:
            info = full.lstat()
        except OSError:
            return self._marker(b"unreadable", "unreadable")
        if stat.S_ISLNK(info.st_mode):
            return self._marker(b"symlink\0" + os.fsencode(os.readlink(full)), "symlink")
        if not stat.S_ISREG(info.st_mode):
            kind = "fifo" if stat.S_ISFIFO(info.st_mode) else "special"
            return self._marker(f"special\0{kind}".encode(), "special")
        try:
            data = full.read_bytes()
        except OSError:
            return self._marker(b"unreadable", "unreadable")
        digest = hash_bytes(data)
        self._record_blob(digest, data)
        return digest

    def _marker(self, payload: bytes, reason: SkipReason) -> str:
        digest = hash_bytes(payload)
        self.skipped.setdefault(digest, reason)
        return digest

    def _record_blob(self, digest: str, data: bytes) -> None:
        if digest in self.blobs or digest in self.skipped:
            return
        if len(data) > MAX_BLOB_BYTES:
            self.skipped[digest] = "too_large"
            return
        if b"\0" in data:
            self.skipped[digest] = "binary"
            return
        try:
            self.blobs[digest] = data.decode("utf-8")
        except UnicodeDecodeError:
            self.skipped[digest] = "binary"

    @contextmanager
    def scratch_copy(self) -> Iterator[Path]:
        """A throwaway copy of the workspace (``.git`` included), removed on exit.

        Harness checks run in such a copy so they can never change the files
        the agent sees.
        """
        parent = Path(tempfile.mkdtemp(prefix="halflife-check-"))
        target = parent / "ws"
        try:
            shutil.copytree(self._root, target, symlinks=True)
            yield target
        finally:
            remove_tree(parent)
