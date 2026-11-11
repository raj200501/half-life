#!/usr/bin/env python3
"""Quick executable check of a bench template's scripted routes (authoring aid).

For each route it materializes the template files in a temporary git repo,
applies the actions (``write_file``, ``replace_in_file``, ``run``; reads and
``done`` are no-ops), runs the success checks, and reports anything that does
not execute: an ``old`` string that does not occur exactly once, a command that
fails unexpectedly, a success check that fails on the compliant route, or a
harness check that fails at baseline.

It does not run the detectors; ``python -m halflife.bench verify`` does the
full verification. Commands run with the subject Python (only pytest added) at
``~/.cache/halflife/subject-venv`` first on PATH inside ``unshare -rn`` (no
network); writes are not confined, so use it only on templates written for
this project.

Usage: uv run --frozen python tools/route_check.py src/halflife/bench/templates/<id>.yaml [...]
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from halflife.schema import Action, Subtask, Template, load_template

SUBJECT_BIN = Path.home() / ".cache" / "halflife" / "subject-venv" / "bin"
USAGE = (
    "usage: uv run --frozen python tools/route_check.py "
    "src/halflife/bench/templates/<id>.yaml [...]"
)


def _env() -> dict[str, str]:
    env = {
        "PATH": f"{SUBJECT_BIN}:/usr/bin:/bin",
        "HOME": tempfile.gettempdir(),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
    }
    return env


def _sh(cmd: str, cwd: Path) -> tuple[int, str]:
    # Tempting routes may try the network, so every command runs in a fresh
    # network namespace with proxies pointed at a closed port.
    proc = subprocess.run(
        ["unshare", "-rn", "/bin/sh", "-c", cmd],
        cwd=cwd,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return proc.returncode, (proc.stdout + proc.stderr)[-800:]


def _materialize(template: Template, root: Path) -> None:
    for rel, text in template.files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for cmd in (
        "git init -q",
        "git add -A",
        "git -c user.name=halflife -c user.email=halflife@example.invalid commit -q -m baseline",
    ):
        code, out = _sh(cmd, root)
        if code:
            raise RuntimeError(f"git setup failed: {out}")


def _apply(action: Action, root: Path) -> str | None:
    """Apply one action; return a problem description or None."""
    args = action.args
    if action.tool in {"list_files", "read_file", "done"}:
        if action.tool == "read_file" and not (root / str(args.get("path", ""))).is_file():
            return f"read_file of missing {args.get('path')}"
        return None
    if action.tool == "write_file":
        path = root / str(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(args["content"]), encoding="utf-8")
        return None
    if action.tool == "replace_in_file":
        path = root / str(args["path"])
        if not path.is_file():
            return f"replace_in_file on missing {args['path']}"
        text = path.read_text(encoding="utf-8")
        count = text.count(str(args["old"]))
        if count != 1:
            snippet = str(args["old"])[:60]
            return f"replace_in_file old text occurs {count} times in {args['path']}: {snippet!r}"
        path.write_text(text.replace(str(args["old"]), str(args["new"])), encoding="utf-8")
        return None
    if action.tool == "run":
        _sh(str(args["cmd"]), root)
        return None
    return f"unknown tool {action.tool}"


def _route(
    template: Template, steps: list[tuple[Subtask, tuple[Action, ...]]], label: str, check: bool
) -> list[str]:
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="halflife-route-") as tmp:
        root = Path(tmp)
        _materialize(template, root)
        if check:
            for spec in template.checks:
                code, out = _sh(spec.cmd, root)
                if code:
                    problems.append(
                        f"{label}: harness check {spec.id} fails at baseline: {out.strip()[-200:]}"
                    )
        for subtask, actions in steps:
            for action in actions:
                problem = _apply(action, root)
                if problem:
                    problems.append(f"{label}/{subtask.id}: {problem}")
            if check and subtask.success is not None:
                code, out = _sh(subtask.success.cmd, root)
                if code != subtask.success.expect_exit:
                    problems.append(
                        f"{label}/{subtask.id}: success check exit {code}: {out.strip()[-200:]}"
                    )
            if check:
                for spec in template.checks:
                    code, out = _sh(spec.cmd, root)
                    if code:
                        tail = out.strip()[-200:]
                        problems.append(
                            f"{label}/{subtask.id}: harness check {spec.id} fails: {tail}"
                        )
    return problems


def check_template(template: Template) -> list[str]:
    """All problems found on the compliant routes and the tempting routes."""
    core = list(template.subtasks)
    fillers = list(template.fillers)
    compliant_early = [(core[0], core[0].solution)] + [(f, f.solution) for f in fillers]
    compliant_early += [(s, s.solution) for s in core[1:]]
    compliant_late = [(s, s.solution) for s in core] + [(f, f.solution) for f in reversed(fillers)]
    problems = _route(template, compliant_early, "compliant(fillers-first)", True)
    problems += _route(template, compliant_late, "compliant(fillers-last)", True)
    for index, subtask in enumerate(core):
        if subtask.is_dp:
            prefix = [(s, s.solution) for s in core[:index]]
            problems += _route(
                template, [*prefix, (subtask, subtask.tempting)], f"tempting/{subtask.id}", False
            )
    return problems


def main(argv: list[str]) -> int:
    """Check each template file given on the command line (usage and exit 2 without one)."""
    names = argv[1:]
    if not names or any(name in ("-h", "--help") for name in names):
        print(USAGE)
        return 2
    if not SUBJECT_BIN.is_dir():
        print(f"missing subject Python at {SUBJECT_BIN}")
        return 2
    failed = 0
    for name in names:
        template = load_template(Path(name))
        problems = check_template(template)
        status = "OK" if not problems else f"{len(problems)} problem(s)"
        print(f"{template.id}: {status}")
        for problem in problems:
            print(f"  - {problem}")
        failed += bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.exit(main(sys.argv))
