"""E1 power analysis by simulation, banked before the E1 preregistration.

Reads banked pilot rows (``sessions.jsonl`` and ``opportunities.jsonl`` from
:func:`halflife.experiment.bank_runs`) of the families E1 uses (``family_choice.json``,
when banked) under the E1 analysis-set exclusions, for the observed per-opportunity
violation rate, decision points per session, session length, and compactions per
session (the unpadded pilot shape, :data:`SHAPE_NOTE`), then
runs :func:`halflife.stats.power_by_simulation` over a grid of sessions per family,
effect sizes (log odds ratios), and base rates. A pilot of a few sessions estimates the
base rate poorly, so the grid also holds fixed planning rates. Every simulated number is
synthetic and labeled so; the artifact records the pilot inputs it used.

Sessions per family: the hazard design (as in H2b) simulates all ``n`` sessions of a
family; the two-arm contrast (as in H2a, which compares the two compaction arms a
family's sessions are randomized to) puts ``n // 2`` sessions in each arm, so both
designs describe one family of ``n`` sessions.
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

SESSIONS_GRID = (40, 48, 56, 64)
EFFECTS_GRID = (0.4, 0.7, 1.0)
PLANNING_RATES = (0.02, 0.05, 0.10, 0.20)
DEFAULT_N_SIM = 400
DEFAULT_SEED = 20261105


SHAPE_NOTE = (
    "Session shape from the exploratory pilot, whose plans are not padded to 20 subtasks "
    "and force 1 to 3 compactions in [8, 120); E1 pads plans (D-043) and forces 1 or 2 "
    "compactions in [8, 70) (D-047), so its sessions are longer and see more compactions "
    "before their decision points (unpadded pilot shape), which raises the power for the "
    "compaction effect. The simulation also places compactions at uniformly random times, "
    "while E1's token-budget compactions recur at nearly regular intervals, which ties the "
    "compaction count to position and lowers that power, so the simulated compaction power "
    "can be too high or too low for E1."
)


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("family", "")), str(row["session_id"])


def pilot_inputs(
    sessions: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Simulation inputs measured in the pilot, with the observed violation rate.

    Only the ``families`` E1 uses count (the pilot's family choice, D-042; every family
    when None), and the E1 analysis-set exclusions apply: sessions that reached their
    first decision point without a probe leak, and their decision points with at least
    one tool call in the span. A session is ``(family, session_id)``, since every family
    runs the same plans.
    """
    present = sorted({str(s.get("family", "")) for s in sessions})
    used = sorted(set(families)) if families is not None else present
    mine = [s for s in sessions if str(s.get("family", "")) in used]
    reached = [s for s in mine if s.get("reached_first_dp") is True and not s.get("probe_leaks")]
    keys = {_key(s) for s in reached}
    kept = [o for o in opportunities if _key(o) in keys and int(o.get("span_calls", 1)) > 0]
    k = sum(1 for o in kept if o["violated"])
    n = len(kept)
    if not reached or n == 0:
        raise stats.StatsError("the pilot has no session that reached a decision point")
    per_session = {key: 0 for key in keys}
    for row in kept:
        per_session[_key(row)] += 1
    return {
        "families": used,
        "families_excluded": [f for f in present if f not in used],
        "shape_note": SHAPE_NOTE,
        "sessions": len(mine),
        "sessions_reaching_first_dp": len(reached),
        "violations": k,
        "opportunities": n,
        "violation_rate": stats.wilson_interval(k, n).to_dict(),
        "opportunities_per_session": max(1, round(statistics.median(per_session.values()))),
        "max_position": max(2, round(statistics.median(int(s["n_tool_calls"]) for s in reached))),
        "compactions_per_session": max(
            1, round(statistics.median(int(s["n_compactions"]) for s in reached))
        ),
    }


def chosen_families(pilot_dir: Path) -> list[str] | None:
    """The E1 families banked in ``family_choice.json`` (D-042), or None before it exists."""
    path = pilot_dir / "family_choice.json"
    if not path.is_file():
        return None
    chosen = json.loads(path.read_text(encoding="utf-8")).get("chosen")
    return [str(f) for f in chosen] if isinstance(chosen, list) else None


def len_dps(sessions: Sequence[Mapping[str, Any]]) -> list[int]:
    """Decision points reached per session."""
    return [int(s["dps_reached"]) for s in sessions]


def grid(
    inputs: Mapping[str, Any],
    *,
    sessions_grid: Sequence[int] = SESSIONS_GRID,
    effects_grid: Sequence[float] = EFFECTS_GRID,
    rates: Sequence[float] = PLANNING_RATES,
    n_sim: int = DEFAULT_N_SIM,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Simulated power for every grid point, from the pilot's session shape."""
    observed = float(inputs["violation_rate"]["value"])
    base_rates = sorted({*rates, *([observed] if 0.0 < observed < 1.0 else [])})
    out = []
    for index, (n_sessions, effect, rate) in enumerate(
        (n, e, r) for n in sessions_grid for e in effects_grid for r in base_rates
    ):
        result = stats.power_by_simulation(
            effect,
            n_sessions,
            int(inputs["opportunities_per_session"]),
            rate,
            n_sim,
            seed + index,
            max_position=int(inputs["max_position"]),
            compactions_per_session=min(
                int(inputs["compactions_per_session"]), int(inputs["max_position"])
            ),
            arm_sessions=max(1, n_sessions // 2),
        )
        out.append({"pilot_rate": rate == observed} | result.to_dict())
    return out


def _git_sha(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m halflife.power --pilot artifacts/pilot --out artifacts/power/e1.json``."""
    parser = argparse.ArgumentParser(description="E1 power analysis from banked pilot rows.")
    parser.add_argument("--pilot", type=Path, required=True, help="banked pilot directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-sim", type=int, default=DEFAULT_N_SIM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    families = chosen_families(args.pilot)
    inputs = pilot_inputs(
        load_jsonl(args.pilot / "sessions.jsonl"),
        load_jsonl(args.pilot / "opportunities.jsonl"),
        families,
    )
    rows = grid(inputs, n_sim=args.n_sim, seed=args.seed)
    root = Path(__file__).resolve().parents[2]
    artifact = {
        "synthetic": True,
        "description": "Simulated power for E1 sizing; inputs measured in the exploratory pilot.",
        "pilot_dir": args.pilot.as_posix(),
        "families": families,
        "pilot": inputs,
        "grid": rows,
        "command": "python -m halflife.power " + " ".join(argv or sys.argv[1:]),
        "git_sha": _git_sha(root),
        "created": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out} ({len(rows)} grid points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
