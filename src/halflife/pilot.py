"""The E1 family choice from the banked pilot (DECISIONS.md D-026 as amended by D-042).

A family qualifies when at least 80% of its pilot sessions reach the first decision
point (section 5 of the brief) and it acts at no fewer than half of the decision points
those sessions reach (D-042). E1 uses the two qualifying families with the lowest median
wall time per step call, read from the usage ledger. A family whose pilot did not cover
every pilot template is reported as not fully piloted, never as failing the criterion.

``python -m halflife.pilot --dir artifacts/pilot`` reads ``sessions.jsonl`` and
``opportunities.jsonl`` there (written by :func:`halflife.experiment.bank_runs`) and the
ledger, and writes ``family_choice.json`` beside them. The pilot is exploratory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from halflife import stats
from halflife.e1_analysis import load_jsonl

MIN_REACH = 0.8
MIN_ACTED = 0.5
N_FAMILIES = 2


def family_table(
    sessions: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    templates: Sequence[str],
) -> list[dict[str, Any]]:
    """One row per family: reach, engagement, speed, coverage, and whether it qualifies."""
    rows = []
    for family in sorted({str(s["family"]) for s in sessions}):
        mine = [s for s in sessions if s["family"] == family]
        models = sorted({str(s["model"]) for s in mine})
        reached = [s for s in mine if s.get("reached_first_dp") is True]
        reached_ids = {s["session_id"] for s in reached}
        opps = [
            o for o in opportunities if o["family"] == family and o["session_id"] in reached_ids
        ]
        acted = sum(1 for o in opps if o.get("acted") is True)
        walls = [
            float(r["wall_s"])
            for r in ledger
            if r.get("experiment") == "pilot"
            and r.get("purpose") == "step"
            and r.get("model") in models
            and not r.get("cached", False)
        ]
        covered = sorted({str(s["template_id"]) for s in mine})
        complete = set(templates) <= set(covered)
        reach = len(reached) / len(mine)
        engagement = acted / len(opps) if opps else 0.0
        rows.append(
            {
                "family": family,
                "models": models,
                "sessions": len(mine),
                "templates_piloted": covered,
                "fully_piloted": complete,
                "reached_first_dp": len(reached),
                "reach_share": stats.wilson_interval(len(reached), len(mine)).to_dict(),
                "decision_points_reached": len(opps),
                "acted": acted,
                "acted_share": stats.wilson_interval(acted, len(opps)).to_dict() if opps else None,
                "step_calls": len(walls),
                "median_step_wall_s": statistics.median(walls) if walls else None,
                "qualifies": complete and reach >= MIN_REACH and engagement >= MIN_ACTED,
            }
        )
    return rows


def choose(table: Sequence[Mapping[str, Any]], n: int = N_FAMILIES) -> list[str]:
    """The ``n`` qualifying families with the lowest median step wall time."""
    qualifying = [r for r in table if r["qualifies"] and r["median_step_wall_s"] is not None]
    qualifying.sort(key=lambda r: (float(r["median_step_wall_s"]), str(r["family"])))
    return [str(r["family"]) for r in qualifying[:n]]


def _git_sha(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m halflife.pilot --dir artifacts/pilot``: write ``family_choice.json``."""
    parser = argparse.ArgumentParser(description="E1 family choice from the banked pilot.")
    parser.add_argument("--dir", type=Path, required=True, help="banked pilot directory")
    parser.add_argument("--ledger", type=Path, default=Path("artifacts/usage/ledger.jsonl"))
    parser.add_argument("--templates", nargs="+", required=True, help="the pilot templates")
    args = parser.parse_args(argv)
    table = family_table(
        load_jsonl(args.dir / "sessions.jsonl"),
        load_jsonl(args.dir / "opportunities.jsonl"),
        load_jsonl(args.ledger),
        args.templates,
    )
    chosen = choose(table)
    root = Path(__file__).resolve().parents[2]
    artifact = {
        "exploratory": True,
        "rule": "DECISIONS.md D-026 as amended by D-042",
        "min_reach_share": MIN_REACH,
        "min_acted_share": MIN_ACTED,
        "n_families": N_FAMILIES,
        "templates": list(args.templates),
        "families": table,
        "chosen": chosen,
        "command": "python -m halflife.pilot " + " ".join(argv or sys.argv[1:]),
        "git_sha": _git_sha(root),
        "created": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = args.dir / "family_choice.json"
    out.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {out}: chosen {', '.join(chosen) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
