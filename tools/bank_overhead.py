#!/usr/bin/env python3
"""Bank artifacts/e4/overhead.json: the measured overhead of halflife-guard and halflife-pin.

Gate G9 and rule 15 need each shipped tool's cost. The guard is timed on the
PreToolUse hook fixtures in tests/fixtures/claude_hooks/ (written from the Claude
Code docs, not captured) against the rules of examples/halflife.yaml, in a
temporary project holding the files those fixtures name, both in process and as
a fresh hook subprocess per call, as Claude Code runs it. The pin block is
measured for every held-out template, the E4 sample; the record's ``halflife-pin``
entry is the template with the median block size, and every template's numbers
are kept beside it. Load averages before and after are recorded, because the
machine may be shared with a model run.

Usage: uv run --frozen python tools/bank_overhead.py [--repeats 30] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from halflife import bench
from halflife.kit import overhead
from halflife.kit.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "claude_hooks"
SPEC = ROOT / "examples" / "halflife.yaml"
PROJECT = {
    "tests/test_core.py": "def test():\n    assert True\n",
    "mypkg/__init__.py": "",
    "mypkg/core.py": "def add(a, b):\n    return a + b\n",
    "requirements.txt": "pyyaml==6.0.1\n",
    "README.md": "# mypkg\n",
}


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _load() -> list[float]:
    return [round(x, 2) for x in os.getloadavg()]


def measure(repeats: int) -> dict[str, Any]:
    """Every measurement and its provenance, in the docs/ARTIFACTS.md G9 layout."""
    load_before = _load()
    constraints = load_spec(SPEC)
    fixtures = sorted(FIXTURES.glob("pretooluse_*.json"))
    with tempfile.TemporaryDirectory(prefix="halflife-overhead-") as tmp:
        for rel, text in PROJECT.items():
            path = Path(tmp) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        guard = overhead.measure_guard_latency(fixtures, constraints, repeats, workspace=tmp)
    pins: list[dict[str, Any]] = []
    for loaded in bench.load_all():
        if loaded.template.split != "heldout":
            continue
        rules = loaded.template.constraints
        pins.append(
            {
                "template_id": loaded.id,
                "tokens": overhead.measure_pin_tokens(rules),
                "latency": overhead.measure_pin_latency(rules, repeats),
            }
        )
    pins.sort(key=lambda p: (p["tokens"]["tokens"], p["template_id"]))
    median = pins[len(pins) // 2]
    record = overhead.overhead_record(guard, median["tokens"], median["latency"])
    record["halflife-pin"]["template_id"] = median["template_id"]
    return record | {
        "guard_detail": guard,
        "pin_by_template": pins,
        "provenance": {
            "command": "python tools/bank_overhead.py " + " ".join(sys.argv[1:]),
            "git_sha": _git_sha(),
            "created": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "spec": SPEC.relative_to(ROOT).as_posix(),
            "fixtures": [f.relative_to(ROOT).as_posix() for f in fixtures],
            "repeats": repeats,
            "cpu_count": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": _load(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Measure and write the artifact."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "e4" / "overhead.json")
    args = parser.parse_args(argv)
    record = measure(args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    guard, pin = record["halflife-guard"], record["halflife-pin"]
    print(
        f"wrote {args.out}: guard median {guard['latency_ms']:.1f} ms per call, "
        f"pin {pin['tokens']} tokens ({pin['template_id']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
