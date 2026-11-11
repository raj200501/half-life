"""Tests for the secrets scanner.

Fake credentials are assembled at runtime so this file never contains a
string that the scanner itself would flag.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from halflife import secrets_scan

FAKE_GITHUB = "ghp" + "_" + "A1b2C3d4" * 5
FAKE_ANTHROPIC = "sk-" + "ant-" + "x" * 30
FAKE_PRIVATE = "-----BEGIN " + "RSA PRIVATE KEY-----"


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)


def _track(root: Path, name: str, content: bytes) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    subprocess.run(["git", "-C", str(root), "add", name], check=True)


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        (f"token = {FAKE_GITHUB}", "github_token"),
        (f"key: {FAKE_ANTHROPIC}", "anthropic_key"),
        (FAKE_PRIVATE, "private_key_block"),
        ("AKIA" + "ABCDEFGHIJKLMNOP", "aws_access_key"),
    ],
)
def test_patterns_detect_known_formats(text: str, rule: str) -> None:
    findings = secrets_scan.scan_text("f.txt", f"line one\n{text}\n", {})
    assert [f.rule for f in findings] == [rule]
    assert findings[0].line == 2


def test_variable_names_alone_are_not_findings() -> None:
    text = "export GEMINI_API_KEY=...\nOPENROUTER_API_KEY is optional\n"
    assert secrets_scan.scan_text("README.md", text, {}) == []


def test_env_secret_values_filter_by_name_and_length() -> None:
    env = {
        "MY_API_KEY": "z" * 20,
        "SHORT_TOKEN": "abc",
        "HOME": "/root/" + "a" * 30,
    }
    assert secrets_scan.env_secret_values(env) == {"MY_API_KEY": "z" * 20}


def test_live_env_value_is_found_and_not_printed() -> None:
    value = "live-" + "q" * 24
    findings = secrets_scan.scan_text(
        "log.txt", f"ok\ncalled with {value}\n", {"SERVICE_TOKEN": value}
    )
    assert len(findings) == 1
    rendered = findings[0].render()
    assert rendered == "log.txt:2: env:SERVICE_TOKEN"
    assert value not in rendered


def test_scan_tracked_files_skips_binary_and_untracked(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _track(tmp_path, "clean.py", b"print('hello')\n")
    _track(tmp_path, "blob.bin", b"\0\0" + FAKE_GITHUB.encode())
    _track(tmp_path, "sub/leak.md", f"x\n{FAKE_GITHUB}\n".encode())
    (tmp_path / "untracked.txt").write_text(FAKE_GITHUB, encoding="utf-8")

    files = secrets_scan.tracked_files(tmp_path)
    findings = secrets_scan.scan_files(tmp_path, files, env={})

    assert [f.render() for f in findings] == ["sub/leak.md:2: github_token"]


def test_unreadable_file_is_skipped(tmp_path: Path) -> None:
    assert secrets_scan.scan_files(tmp_path, [tmp_path / "gone.txt"], env={}) == []


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _init_repo(tmp_path)
    _track(tmp_path, "a.txt", b"nothing here\n")
    assert secrets_scan.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "SECRETS-SCAN: CLEAN 1 files"

    _track(tmp_path, "b.txt", FAKE_PRIVATE.encode())
    assert secrets_scan.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "b.txt:1: private_key_block" in out
    assert out.strip().endswith("SECRETS-SCAN: FAIL 1 finding(s) in 2 files")
