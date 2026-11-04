#!/usr/bin/env python3
"""Run plans round-robin across model families, one model-heavy process at a time.

For each plan index in order, run that one plan with every model before moving
to the next index, so slow periods on a shared machine hit every family alike
(DECISIONS.md D-026). Each call is
``python -m halflife.experiment run --plans P --index I --run-id <prefix>-<model> --model M``,
which is resumable, so rerunning this script skips finished sessions.

Usage: uv run --frozen python tools/round_robin.py --plans PATH --prefix pilot \
    --models key1 key2 ... [--indices 0 1 2] [--prereg prereg/NNNN-slug.md]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run every (index, model) pair in round-robin order; return the worst exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--prereg", default=None, help="preregistration that governs the runs")
    args = parser.parse_args(argv)
    total = sum(1 for line in args.plans.read_text(encoding="utf-8").splitlines() if line.strip())
    indices = args.indices if args.indices is not None else list(range(total))
    worst = 0
    for index in indices:
        for model in args.models:
            cmd = [
                sys.executable,
                "-m",
                "halflife.experiment",
                "run",
                "--plans",
                str(args.plans),
                "--index",
                str(index),
                "--run-id",
                f"{args.prefix}-{model}",
                "--model",
                model,
                "--n-ctx",
                str(args.n_ctx),
                *(["--prereg", args.prereg] if args.prereg else []),
            ]
            print(json.dumps({"index": index, "model": model, "cmd": cmd}), flush=True)
            code = subprocess.run(cmd, check=False).returncode
            print(json.dumps({"index": index, "model": model, "exit": code}), flush=True)
            if code == 3:
                print("stopping: a cap or the no-model-calls flag stopped the run", flush=True)
                return code
            worst = max(worst, code)
    return worst


if __name__ == "__main__":
    sys.exit(main())
