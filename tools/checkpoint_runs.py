#!/usr/bin/env python3
"""Checkpoint finished sessions of long runs into the repository, and restore them.

Runs write to ``runs/`` (git-ignored), which lives only in the machine that runs them.
A multi-day study would lose every finished session with that machine, so each finished
session (a directory with ``result.json``) is packed once into
``<out>/<run_id>/<session_id>.tar.gz``. Archives are reproducible (sorted members, no
owners, zero times), are never rewritten, and are raw outputs only: nothing is analyzed,
so a checkpoint is not an interim analysis (prereg/0001-e1.md, stopping rules).
``--restore`` unpacks every archive into ``<target>/<run_id>/<session_id>/`` for a
session the target lacks, so the resumable runner skips it.

Usage:
    python tools/checkpoint_runs.py --out artifacts/e1/raw runs/e1-<model> ...
    python tools/checkpoint_runs.py --restore --out artifacts/e1/raw runs
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path


def _archive_bytes(session: Path) -> bytes:
    """A reproducible gzipped tar of ``session``'s files, paths relative to it."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(p for p in session.rglob("*") if p.is_file()):
            info = tarfile.TarInfo(path.relative_to(session).as_posix())
            data = path.read_bytes()
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    return gzip.compress(raw.getvalue(), compresslevel=9, mtime=0)


def checkpoint(run_dirs: Sequence[Path], out: Path) -> list[Path]:
    """Archive every finished session not archived yet; return the archives written."""
    written: list[Path] = []
    for run in run_dirs:
        for session in sorted(p for p in run.iterdir() if (p / "result.json").is_file()):
            target = out / run.name / f"{session.name}.tar.gz"
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_archive_bytes(session))
            written.append(target)
    return written


def restore(out: Path, target_root: Path) -> list[Path]:
    """Unpack every archive under ``out`` whose session ``target_root`` lacks.

    Raises:
        ValueError: when an archive holds a path outside its session directory.
    """
    restored: list[Path] = []
    for archive in sorted(out.glob("*/*.tar.gz")):
        session = target_root / archive.parent.name / archive.name.removesuffix(".tar.gz")
        if session.exists():
            continue
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            base = session.resolve()
            for member in members:
                dest = (session / member.name).resolve()
                if base != dest and base not in dest.parents:
                    raise ValueError(f"{archive}: member {member.name!r} lies outside the session")
                if not member.isfile():
                    raise ValueError(f"{archive}: member {member.name!r} is not a regular file")
            session.mkdir(parents=True)
            for member in members:
                source = tar.extractfile(member)
                assert source is not None  # regular files, checked above
                dest = session / member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(source.read())
        restored.append(session)
    return restored


def main(argv: Sequence[str] | None = None) -> int:
    """Checkpoint run directories, or restore archives with ``--restore``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="archive root")
    parser.add_argument("--restore", action="store_true", help="unpack archives instead")
    parser.add_argument("paths", nargs="+", type=Path, help="run directories, or the target")
    args = parser.parse_args(argv)
    if args.restore:
        if len(args.paths) != 1:
            parser.error("--restore takes one target directory")
        for session in restore(args.out, args.paths[0]):
            print(f"restored {session}")
        return 0
    for archive in checkpoint(args.paths, args.out):
        print(f"archived {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
