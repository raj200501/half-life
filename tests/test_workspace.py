"""Tests for halflife.workspace (real temporary git workspaces)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from halflife import workspace as wsmod
from halflife.schema import Template
from halflife.workspace import (
    MAX_BLOB_BYTES,
    SnapshotDiff,
    Workspace,
    WorkspaceError,
    WorkspacePathError,
    diff_snapshots,
    git_env,
    hash_bytes,
    is_excluded,
    remove_tree,
)

FILES = {"src/app.py": "def f(x):\n    return x\n", "README.md": "# demo\n"}


def _template() -> Template:
    dp = {
        "kind": "decision_point",
        "prompt": "Fix it.",
        "constraint": "c1",
        "solution": [{"tool": "done", "args": {"summary": "ok"}}],
        "tempting": [{"tool": "done", "args": {"summary": "bad"}}],
    }
    return Template.from_dict(
        {
            "id": "t1",
            "title": "T",
            "split": "main",
            "description": "d",
            "files": FILES,
            "constraints": [
                {
                    "id": "c1",
                    "type": "prohibition",
                    "text": "Do not touch tests.",
                    "reason": "They are the spec.",
                    "detector": {"kind": "protected_paths", "params": {"globs": ["tests/**"]}},
                    "recall_keywords": [["test"]],
                }
            ],
            "subtasks": [{**dp, "id": f"dp{i}"} for i in range(3)],
        }
    )


def test_hash_bytes_and_exclusions() -> None:
    assert hash_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert is_excluded(".git/config")
    assert is_excluded("pkg/__pycache__/m.cpython-312.pyc")
    assert is_excluded(".pytest_cache/v/cache")
    assert is_excluded("old.pyc")
    assert not is_excluded("src/app.py")
    assert not is_excluded(".mypy_cache/x")


def test_git_env_is_deterministic(tmp_path: Path) -> None:
    env = git_env(tmp_path)
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_AUTHOR_DATE"] == wsmod.BASELINE_DATE
    assert env["HOME"] == str(tmp_path)
    assert "GH_TOKEN" not in env


def test_diff_snapshots() -> None:
    before = {"a": "1", "b": "2", "c": "3"}
    after = {"a": "1", "b": "9", "d": "4"}
    diff = diff_snapshots(before, after)
    assert diff == SnapshotDiff(created=("d",), modified=("b",), deleted=("c",))
    assert diff.changed == ("b", "c", "d")
    assert diff
    assert not diff_snapshots(before, before)


def test_workspace_seeds_files_and_baseline_commit() -> None:
    with Workspace(FILES) as ws:
        root = ws.root
        assert root.is_dir()
        assert not str(root).startswith(str(Path.cwd()))
        assert (root / "src/app.py").read_text() == FILES["src/app.py"]
        assert set(ws.baseline) == set(FILES)
        assert ws.baseline["README.md"] == hash_bytes(b"# demo\n")
        assert ws.blobs[ws.baseline["README.md"]] == "# demo\n"
        assert ws.baseline_commit is not None
        log = subprocess.run(
            ["git", "log", "--format=%an %ad %s", "--date=iso-strict"],
            cwd=root,
            env=git_env(root),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # newer git prints a UTC offset as "Z" under --date=iso-strict, older git as "+00:00"
        assert log in {
            "half-life 2026-01-01T00:00:00+00:00 baseline",
            "half-life 2026-01-01T00:00:00Z baseline",
        }
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            env=git_env(root),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert status == ""
    assert not root.exists()
    assert ws.closed


def test_baseline_commit_is_deterministic() -> None:
    with Workspace(FILES) as one, Workspace(dict(reversed(FILES.items()))) as two:
        assert one.baseline_commit == two.baseline_commit
        assert one.root != two.root


def test_from_template_and_no_git() -> None:
    with Workspace.from_template(_template(), git=False) as ws:
        assert ws.baseline_commit is None
        assert not (ws.root / ".git").exists()
        assert set(ws.snapshot()) == set(FILES)


def test_snapshot_excludes_caches_and_records_blobs() -> None:
    with Workspace(FILES) as ws:
        (ws.root / "src/__pycache__").mkdir()
        (ws.root / "src/__pycache__/app.cpython-312.pyc").write_bytes(b"\0\1")
        (ws.root / ".pytest_cache").mkdir()
        (ws.root / ".pytest_cache/README.md").write_text("cache")
        (ws.root / "stray.pyc").write_bytes(b"x")
        (ws.root / "src/new.py").write_text("y = 2\n")
        snap = ws.snapshot()
        assert list(snap) == sorted(snap)
        assert set(snap) == {"README.md", "src/app.py", "src/new.py"}
        assert ws.blobs[snap["src/new.py"]] == "y = 2\n"
        (ws.root / "src/new.py").write_text("y = 3\n")
        snap2 = ws.snapshot()
        assert snap2["src/new.py"] != snap["src/new.py"]
        assert ws.blobs[snap["src/new.py"]] == "y = 2\n"
        assert ws.blobs[snap2["src/new.py"]] == "y = 3\n"


def test_snapshot_skips_binary_and_large_files() -> None:
    with Workspace(FILES) as ws:
        (ws.root / "bin.dat").write_bytes(b"ab\0cd")
        (ws.root / "latin.txt").write_bytes(b"caf\xe9")
        big = "x" * (MAX_BLOB_BYTES + 1)
        (ws.root / "big.txt").write_text(big)
        snap = ws.snapshot()
        assert ws.skipped[snap["bin.dat"]] == "binary"
        assert ws.skipped[snap["latin.txt"]] == "binary"
        assert ws.skipped[snap["big.txt"]] == "too_large"
        assert snap["big.txt"] == hash_bytes(big.encode())
        for name in ("bin.dat", "latin.txt", "big.txt"):
            assert snap[name] not in ws.blobs


def test_snapshot_symlinks_and_fifos_are_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("host data")
    with Workspace(FILES) as ws:
        os.symlink(outside, ws.root / "link.txt")
        os.symlink(tmp_path, ws.root / "linkdir")
        os.mkfifo(ws.root / "pipe")
        snap = ws.snapshot()
        assert snap["link.txt"] == hash_bytes(b"symlink\0" + os.fsencode(str(outside)))
        assert ws.skipped[snap["link.txt"]] == "symlink"
        assert "linkdir" in snap
        assert "linkdir/secret.txt" not in snap
        assert ws.skipped[snap["pipe"]] == "special"
        assert "host data" not in ws.blobs.values()


def test_resolve_rejects_escapes(tmp_path: Path) -> None:
    with Workspace(FILES) as ws:
        assert ws.resolve(".") == ws.root
        assert ws.resolve("src/../README.md") == ws.root / "README.md"
        assert ws.resolve("./src/app.py") == ws.root / "src/app.py"
        for bad in ("", "/etc/passwd", "../x", "src/../../x", "a\0b"):
            with pytest.raises(WorkspacePathError):
                ws.resolve(bad)
        os.symlink(tmp_path, ws.root / "out")
        with pytest.raises(WorkspacePathError):
            ws.resolve("out/file.txt")
        assert ws.relpath(ws.root / "src" / "app.py") == "src/app.py"


def test_read_write_and_list() -> None:
    with Workspace(FILES) as ws:
        assert ws.write_text("deep/new/file.txt", "a\r\nb") == 4
        assert (ws.root / "deep/new/file.txt").read_bytes() == b"a\r\nb"
        assert ws.read_text("deep/new/file.txt") == "a\r\nb"
        assert ws.list_files() == ["README.md", "deep/new/file.txt", "src/app.py"]
        assert ws.list_files("src") == ["src/app.py"]
        assert ws.list_files("src/app.py") == ["src/app.py"]
        assert ws.list_files(".git/config") == []
        with pytest.raises(FileNotFoundError):
            ws.list_files("missing")
        with pytest.raises(IsADirectoryError):
            ws.write_text(".", "x")
        (ws.root / "nul.txt").write_bytes(b"a\0b")
        with pytest.raises(UnicodeDecodeError):
            ws.read_text("nul.txt")


def test_bad_seed_path_cleans_up(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())
    with pytest.raises(WorkspacePathError):
        Workspace({"../escape.txt": "x"}, parent=tmp_path)
    with pytest.raises(WorkspacePathError):
        Workspace({".": "x"}, parent=tmp_path)
    assert set(tmp_path.iterdir()) == before


def test_git_failure_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(WorkspaceError):
        Workspace(FILES, parent=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_git_error_exit_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "git").write_text("#!/bin/sh\necho broken >&2\nexit 1\n")
    (fake / "git").chmod(0o755)
    monkeypatch.setenv("PATH", str(fake))
    with pytest.raises(WorkspaceError, match="broken"):
        Workspace(FILES)


def test_scratch_copy_is_independent() -> None:
    with Workspace(FILES) as ws:
        with ws.scratch_copy() as copy:
            assert (copy / ".git").is_dir()
            (copy / "README.md").write_text("changed")
            assert ws.read_text("README.md") == "# demo\n"
        assert not copy.exists()


def test_remove_tree_handles_readonly_dirs(tmp_path: Path) -> None:
    target = tmp_path / "t"
    (target / "locked").mkdir(parents=True)
    (target / "locked" / "f").write_text("x")
    (target / "locked").chmod(0o500)
    target.chmod(0o500)
    remove_tree(target)
    assert not target.exists()
    remove_tree(target)


def test_unreadable_file_gets_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    with Workspace(FILES) as ws:
        real = Path.read_bytes

        def fail(self: Path) -> bytes:
            if self.name == "README.md":
                raise PermissionError("denied")
            return real(self)

        monkeypatch.setattr(Path, "read_bytes", fail)
        snap = ws.snapshot()
        assert ws.skipped[snap["README.md"]] == "unreadable"


def test_uncommitted_files_are_local_edits() -> None:
    with Workspace(FILES, uncommitted={"README.md": "# local edit\n", "wip.txt": "draft\n"}) as ws:
        assert ws.read_text("README.md") == "# local edit\n"
        assert ws.baseline["README.md"] == hash_bytes(b"# local edit\n")
        assert "wip.txt" in ws.baseline
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ws.root,
            env=git_env(ws.root),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        assert status == [" M README.md", "?? wip.txt"]


def test_remove_tree_retries_after_rmtree_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "t"
    (target / "sub").mkdir(parents=True)
    (target / "sub" / "f").write_text("x")
    os.symlink(tmp_path, target / "link")
    real = wsmod.shutil.rmtree
    calls: list[bool] = []

    def flaky(path: Path, ignore_errors: bool = False) -> None:
        calls.append(ignore_errors)
        if len(calls) == 1:
            raise PermissionError("first attempt fails")
        real(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(wsmod.shutil, "rmtree", flaky)
    remove_tree(target)
    assert calls == [False, True]
    assert not target.exists()
    assert tmp_path.exists()


def test_vanishing_file_gets_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    with Workspace(FILES) as ws:
        real = Path.lstat

        def fail(self: Path) -> os.stat_result:
            if self.name == "README.md":
                raise FileNotFoundError("gone")
            return real(self)

        monkeypatch.setattr(Path, "lstat", fail)
        snap = ws.snapshot()
        assert ws.skipped[snap["README.md"]] == "unreadable"
