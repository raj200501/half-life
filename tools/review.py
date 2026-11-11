#!/usr/bin/env python3
"""``make review``: validate and print the latest independent review.

Section 10.1 step 8 of the brief: the reviewer runs through headless
``claude -p --model opus`` when that backend is usable. Otherwise the
``reviewer`` subagent (.claude/agents/reviewer.md) runs and writes
``artifacts/review/latest.json`` itself; the main session never edits that file.

The headless path is refused while the billing guard trips (D-002), so this
script only validates and prints the subagent's file. Its last
line is the review's final line (``REVIEW: SHIP`` or ``REVIEW: BLOCK``), or
``REVIEW: STALE`` / ``REVIEW: INVALID`` / ``REVIEW: MISSING``.

Schema of latest.json (see docs/ARTIFACTS.md):
    reviewed_commit: 40-hex SHA the reviewer read.
    reviewer: "subagent" or "headless".
    timestamp: ISO-8601 UTC.
    verdict: "SHIP" or "BLOCK".
    blocking: list of {"id", "title", "file", "evidence"}; empty iff SHIP.
    non_blocking: list of short strings (also appended to docs/REVIEW_LOG.md).
    final_line: "REVIEW: SHIP" or "REVIEW: BLOCK".
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
LATEST = REPO / "artifacts" / "review" / "latest.json"
EXEMPT_PREFIXES = ("artifacts/review/",)
# Equal to REVIEW_EXEMPT_FILES in tools/ship_check.py (D-051); a test checks it.
EXEMPT_FILES = frozenset({"STATUS.md", "docs/LOG.md", "docs/REVIEW_LOG.md"})


def git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run git in the repository without taking optional locks."""
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def schema_problems(review: dict[str, Any]) -> list[str]:
    """Ways the review file breaks its schema or contradicts itself."""
    problems: list[str] = []
    if not re.fullmatch(r"[0-9a-f]{40}", str(review.get("reviewed_commit", ""))):
        problems.append("reviewed_commit is not a full SHA")
    if review.get("reviewer") not in {"subagent", "headless"}:
        problems.append("reviewer must be 'subagent' or 'headless'")
    verdict = review.get("verdict")
    blocking = review.get("blocking")
    if verdict not in {"SHIP", "BLOCK"}:
        problems.append("verdict must be SHIP or BLOCK")
    if not isinstance(blocking, list):
        problems.append("blocking must be a list")
    elif (verdict == "SHIP") != (len(blocking) == 0):
        problems.append("verdict SHIP requires an empty blocking list, and BLOCK a non-empty one")
    else:
        for issue in blocking:
            if not all(issue.get(k) for k in ("id", "title", "file", "evidence")):
                problems.append("each blocking issue needs id, title, file, evidence")
                break
    if review.get("final_line") != f"REVIEW: {verdict}":
        problems.append("final_line must be 'REVIEW: <verdict>'")
    return problems


def staleness(reviewed: str) -> str | None:
    """Why the review no longer covers HEAD, or None if it does."""
    head = git("rev-parse", "HEAD").stdout.strip()
    if reviewed == head:
        return None
    if git("merge-base", "--is-ancestor", reviewed, head).returncode != 0:
        return f"reviewed commit {reviewed[:12]} is not an ancestor of HEAD {head[:12]}"
    changed = git("diff", "--name-only", "--no-renames", reviewed, head).stdout.split()
    outside = [p for p in changed if not p.startswith(EXEMPT_PREFIXES) and p not in EXEMPT_FILES]
    if outside:
        return f"{len(outside)} file(s) changed since review, e.g. {outside[0]}"
    return None


def main() -> int:
    """Validate and print artifacts/review/latest.json."""
    if not LATEST.is_file():
        print("No review banked. Run the reviewer subagent (.claude/agents/reviewer.md).")
        print("REVIEW: MISSING")
        return 1
    try:
        review = json.loads(LATEST.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"artifacts/review/latest.json is not valid JSON: {exc}")
        print("REVIEW: INVALID")
        return 1
    problems = schema_problems(review) if isinstance(review, dict) else ["not a JSON object"]
    if problems:
        for problem in problems:
            print(f"invalid: {problem}")
        print("REVIEW: INVALID")
        return 1
    source = f"{review['reviewer']}, {review.get('timestamp')}"
    print(f"reviewed_commit: {review['reviewed_commit']} ({source})")
    for issue in review["blocking"]:
        print(f"BLOCKING {issue['id']}: {issue['title']} [{issue['file']}]")
    print(f"non-blocking issues: {len(review.get('non_blocking', []))} (see docs/REVIEW_LOG.md)")
    stale = staleness(review["reviewed_commit"])
    if stale:
        print(f"stale: {stale}")
        print("REVIEW: STALE")
        return 1
    print(review["final_line"])
    return 0 if review["verdict"] == "SHIP" else 1


if __name__ == "__main__":
    sys.exit(main())
