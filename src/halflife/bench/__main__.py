"""Command line for half-life-bench.

Usage::

    python -m halflife.bench coverage
    python -m halflife.bench verify (--all | ID ...) [--bank]
    python -m halflife.bench manifest
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from halflife import bench
from halflife.bench.verify import bank, verify_template


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(prog="python -m halflife.bench")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("coverage", help="print split, type, and detector coverage")
    verify = sub.add_parser("verify", help="run compliant and tempting routes")
    verify.add_argument("ids", nargs="*")
    verify.add_argument("--all", action="store_true")
    verify.add_argument(
        "--bank", action="store_true", help="write artifacts/bench routes and reference runs"
    )
    sub.add_parser("manifest", help="write artifacts/bench/manifest.json")
    args = parser.parse_args(argv)
    root = _root()
    loaded = bench.load_all()

    if args.command == "coverage":
        print(json.dumps(bench.coverage(loaded), indent=2, sort_keys=True))
        return 0
    if args.command == "manifest":
        path = root / "artifacts" / "bench" / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = bench.manifest(loaded, root)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(root)} with {len(loaded)} templates")
        return 0

    from halflife.detect import detect_all
    from halflife.sandbox import detect_method

    chosen = loaded if args.all else [t for t in loaded if t.id in set(args.ids)]
    missing = set(args.ids) - {t.id for t in loaded}
    if missing or not chosen:
        print(f"unknown or no templates: {sorted(missing)}")
        return 2
    reports = []
    for item in chosen:
        report = verify_template(item, detect_all)
        reports.append(report)
        print(f"{item.id}: {'OK' if report.ok else 'FAIL'}")
        for problem in report.problems():
            print(f"  - {problem}")
    if args.bank:
        bank(
            reports,
            root,
            command=" ".join(
                [
                    "python -m halflife.bench",
                    *map(shlex.quote, sys.argv[1:] if argv is None else argv),
                ]
            ),
            sandbox_method=detect_method(),
        )
    failed = sum(not r.ok for r in reports)
    print(f"BENCH-VERIFY: {len(reports) - failed}/{len(reports)} templates pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
