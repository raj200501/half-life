"""Billing guard for launching Claude children (section 8 of the build brief).

Claude may only be used through the Max plan login. Any variable or setting that
could route a child to per-token API billing makes the Claude backend unusable.
This module reports *names* of offending variables and settings files, never values.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

FORBIDDEN_ENV_EXACT: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
)
FORBIDDEN_ENV_PREFIX = "CLAUDE_CODE_USE_"


@dataclass(frozen=True)
class GuardReport:
    """Result of a billing-guard check.

    Attributes:
        env_vars: Names of forbidden environment variables that are set.
        api_key_helper_files: Settings files that define ``apiKeyHelper``.
    """

    env_vars: tuple[str, ...] = field(default_factory=tuple)
    api_key_helper_files: tuple[str, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        """True when nothing could route a Claude child to API billing."""
        return not self.env_vars and not self.api_key_helper_files

    def problems(self) -> list[str]:
        """Human-readable problem lines, naming variables and files only."""
        lines = [f"environment variable {name} is set" for name in self.env_vars]
        lines += [f"apiKeyHelper is configured in {path}" for path in self.api_key_helper_files]
        return lines


def forbidden_env_vars(env: Mapping[str, str]) -> tuple[str, ...]:
    """Return the sorted names of forbidden variables present in ``env``.

    A variable counts as present even when its value is empty, because the
    Claude CLI may still treat it as configuration.
    """
    names = {name for name in env if name in FORBIDDEN_ENV_EXACT}
    names |= {name for name in env if name.startswith(FORBIDDEN_ENV_PREFIX)}
    return tuple(sorted(names))


def default_settings_paths(home: Path, project: Path) -> list[Path]:
    """Settings files the Claude CLI may read, in user, project, and managed scope."""
    return [
        home / ".claude" / "settings.json",
        home / ".claude" / "settings.local.json",
        project / ".claude" / "settings.json",
        project / ".claude" / "settings.local.json",
        Path("/etc/claude-code/managed-settings.json"),
        Path("/Library/Application Support/ClaudeCode/managed-settings.json"),
    ]


def files_with_api_key_helper(paths: Iterable[Path]) -> tuple[str, ...]:
    """Return the paths whose JSON object has a top-level ``apiKeyHelper`` key.

    Only the presence of the key is inspected. Unreadable or non-JSON files are
    skipped, since the CLI would not load them either.
    """
    found: list[str] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "apiKeyHelper" in data:
            found.append(str(path))
    return tuple(found)


def check(
    env: Mapping[str, str] | None = None,
    settings_paths: Iterable[Path] | None = None,
) -> GuardReport:
    """Run the billing guard against an environment and settings files.

    Args:
        env: Environment to inspect; defaults to ``os.environ``.
        settings_paths: Settings files to inspect; defaults to
            :func:`default_settings_paths` for the current user and directory.
    """
    env = os.environ if env is None else env
    if settings_paths is None:
        settings_paths = default_settings_paths(Path.home(), Path.cwd())
    return GuardReport(
        env_vars=forbidden_env_vars(env),
        api_key_helper_files=files_with_api_key_helper(settings_paths),
    )
