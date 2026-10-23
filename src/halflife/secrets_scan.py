"""Scan tracked files for secrets (rule 6 of the build brief).

Two kinds of findings:

* pattern: text shaped like a known credential format (API keys, tokens, private keys);
* environment: the exact value of a secret-looking environment variable that is
  set in the scanning process, so a leaked live token is caught even when its
  format is unknown.

Findings name the file, line, and rule only. Matched text is never printed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PATTERNS: dict[str, re.Pattern[str]] = {
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    "openrouter_key": re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"),
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"),
    "github_pat": re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    "private_key_block": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
}

_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
MIN_ENV_SECRET_LENGTH = 16


@dataclass(frozen=True)
class Finding:
    """One suspected secret.

    Attributes:
        path: File path relative to the scanned root.
        line: 1-based line number.
        rule: Pattern name, or ``env:<VARIABLE>`` for a live environment value.
    """

    path: str
    line: int
    rule: str

    def render(self) -> str:
        """Format the finding without the matched text."""
        return f"{self.path}:{self.line}: {self.rule}"


def env_secret_values(env: Mapping[str, str]) -> dict[str, str]:
    """Map secret-looking variable names to values long enough to scan for."""
    return {
        name: value
        for name, value in env.items()
        if _SECRET_NAME.search(name) and len(value) >= MIN_ENV_SECRET_LENGTH
    }


def scan_text(path: str, text: str, env_secrets: Mapping[str, str]) -> list[Finding]:
    """Scan one file's text and return its findings in line order."""
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in PATTERNS.items():
            if pattern.search(line):
                findings.append(Finding(path, number, rule))
        for name, value in env_secrets.items():
            if value in line:
                findings.append(Finding(path, number, f"env:{name}"))
    return findings


def tracked_files(root: Path) -> list[Path]:
    """List files tracked by git under ``root``."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    names = [name for name in result.stdout.decode("utf-8").split("\0") if name]
    return [root / name for name in names]


def scan_files(
    root: Path, files: Iterable[Path], env: Mapping[str, str] | None = None
) -> list[Finding]:
    """Scan ``files`` (binary files are skipped) and return all findings."""
    env_secrets = env_secret_values(os.environ if env is None else env)
    findings: list[Finding] = []
    for path in files:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        text = data.decode("utf-8", errors="replace")
        findings.extend(scan_text(str(path.relative_to(root)), text, env_secrets))
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``make secrets-scan``; exits non-zero on any finding."""
    parser = argparse.ArgumentParser(description="Scan git-tracked files for secrets.")
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    files = tracked_files(root)
    findings = scan_files(root, files)
    for finding in findings:
        print(finding.render())
    if findings:
        print(f"SECRETS-SCAN: FAIL {len(findings)} finding(s) in {len(files)} files")
        return 1
    print(f"SECRETS-SCAN: CLEAN {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
