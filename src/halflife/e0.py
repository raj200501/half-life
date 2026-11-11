"""E0 instrument validity: detector precision and recall on seeded logs (gate G5).

Each seeded case (:mod:`halflife.e0_cases`, written blind to the detector code)
is executed for real by the scripted runner, so its action log has genuine
``fs_after`` hashes, sandbox network logs, and harness checks. The detector for
the case's constraint then predicts whether the log contains a violation, and
the prediction is scored against the case's ground-truth label.

The banked artifact (``artifacts/e0/detector_validation.json``) holds every
case's label and prediction plus precision and recall with Wilson 95%
intervals, which ``tools/ship_check.py`` recomputes from the cases.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from halflife.schema import ActionLog, Constraint, Violation
from halflife.scripted import StepSpec, run_scripted_session

Z95 = 1.959963984540054
Detector = Callable[[ActionLog, Constraint], list[Violation]]


def wilson(successes: int, total: int) -> tuple[float, float]:
    """Wilson score 95% interval (same formula as the ship check)."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + Z95**2 / total
    centre = (p + Z95**2 / (2 * total)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / total + Z95**2 / (4 * total**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_case(case: Any, detect: Detector) -> dict[str, Any]:
    """Execute one seeded case and score the detector on it."""
    steps = [
        (StepSpec(id=step_id, kind=kind, constraint=constraint), tuple(actions))
        for step_id, kind, constraint, actions in case.steps
    ]
    log = run_scripted_session(
        case.files,
        (case.constraint,),
        case.checks,
        steps,
        session_id=f"e0-{case.id}",
        template_id="e0-seeded",
    )
    violations = detect(log, case.constraint)
    return {
        "case_id": case.id,
        "family": case.family,
        "kind": case.kind,
        "constraint_id": case.constraint.id,
        "constraint_type": case.constraint.type,
        "detector": case.constraint.detector.kind,
        "label": bool(case.label),
        "predicted": bool(violations),
        "violations": [v.to_dict() for v in violations],
    }


def score(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Precision and recall with Wilson intervals, plus confusion counts."""
    tp = sum(1 for r in rows if r["label"] and r["predicted"])
    fp = sum(1 for r in rows if not r["label"] and r["predicted"])
    fn = sum(1 for r in rows if r["label"] and not r["predicted"])
    tn = sum(1 for r in rows if not r["label"] and not r["predicted"])
    out: dict[str, Any] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    for name, k, n in (("precision", tp, tp + fp), ("recall", tp, tp + fn)):
        lo, hi = wilson(k, n)
        out[name] = {"value": k / n if n else 0.0, "lo": lo, "hi": hi, "k": k, "n": n}
    return out


def by_detector(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Confusion counts per detector kind (descriptive)."""
    out: dict[str, Counter[str]] = {}
    for r in rows:
        key = ("t" if r["label"] == r["predicted"] else "f") + ("p" if r["predicted"] else "n")
        out.setdefault(r["detector"], Counter())[key] += 1
    return {k: {c: v[c] for c in ("tp", "fp", "fn", "tn")} for k, v in sorted(out.items())}


def _git_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def build_artifact(
    rows: Sequence[dict[str, Any]], *, root: Path, command: str, method: str
) -> dict[str, Any]:
    """The banked detector-validation artifact."""
    scored = score(rows)
    return {
        "synthetic": True,
        "description": (
            "Detector validation on seeded action logs (E0). "
            "Cases are synthetic and executed for real."
        ),
        "cases": [
            {
                k: r[k]
                for k in (
                    "case_id",
                    "family",
                    "kind",
                    "constraint_id",
                    "constraint_type",
                    "detector",
                    "label",
                    "predicted",
                )
            }
            for r in rows
        ],
        "precision": {k: scored["precision"][k] for k in ("value", "lo", "hi")},
        "recall": {k: scored["recall"][k] for k in ("value", "lo", "hi")},
        "confusion": {k: scored[k] for k in ("tp", "fp", "fn", "tn")},
        "by_detector": by_detector(rows),
        "interval_method": "wilson-95",
        "provenance": {
            "git_sha": _git_sha(root),
            "command": command,
            "sandbox_method": method,
            "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run every seeded case, print the scores, and optionally bank the artifact."""
    from halflife.detect import detect
    from halflife.e0_cases import build_cases
    from halflife.sandbox import detect_method

    parser = argparse.ArgumentParser(description="E0 detector validation on seeded logs.")
    parser.add_argument(
        "--bank", action="store_true", help="write artifacts/e0/detector_validation.json"
    )
    parser.add_argument("--only", nargs="*", default=None, help="case ids to run")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    cases = [c for c in build_cases() if args.only is None or c.id in set(args.only)]
    rows = []
    for case in cases:
        row = run_case(case, detect)
        rows.append(row)
        if row["label"] != row["predicted"]:
            label, predicted = row["label"], row["predicted"]
            print(f"MISMATCH {row['case_id']} kind={row['kind']} {label=} {predicted=}")
    scored = score(rows)
    p, r = scored["precision"], scored["recall"]
    print(
        f"cases={len(rows)} tp={scored['tp']} fp={scored['fp']} fn={scored['fn']} tn={scored['tn']}"
    )
    print(f"precision={p['value']:.4f} [{p['lo']:.4f}, {p['hi']:.4f}]")
    print(f"recall={r['value']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]")
    if args.bank:
        artifact = build_artifact(
            rows, root=root, command="python -m halflife.e0 --bank", method=detect_method()
        )
        path = root / "artifacts" / "e0" / "detector_validation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
