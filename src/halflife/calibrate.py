"""Calibration of ``min_events`` for the H1 and H2b score tests (DECISIONS.md D-050 item 3).

Synthetic, not model results. Each simulated dataset runs the 48 plans of
``designs/e1.yaml`` (:func:`halflife.experiment.make_plans`) with two synthetic families,
as E1 runs every plan once per family:

* **Session shape.** Every request of a plan takes a geometric number of tool calls (at
  least one, the ``done`` call; family means :data:`FAMILIES`), cut by the plan's
  per-request cap (``subtask_cap``) and session cap (``tool_cap``), as the minimal agent
  cuts them (:meth:`halflife.agent.Agent.run`). A decision point's position is the tool
  call index at which its request starts. Compactions follow the agent's rule before
  every call: a forced compaction when the plan's schedule is due, otherwise a
  token-budget compaction when the context exceeds the plan's budget; the context grows
  by a gamma number of tokens per call (family means in :data:`FAMILIES`) and a compaction
  resets it (:data:`RESET_TOKENS`). The compaction count of a decision point is the number
  of compactions at indices up to its start, as :func:`halflife.analysis.opportunity_rows`
  counts it, so token-budget compactions tie it to position as they do in E1 (D-047).
* **Outcomes under a null.** Each decision point violates with probability
  ``expit(base + family offset + rule-type effect + slope * log1p(position) + compaction
  * count + u)``. :data:`SCENARIOS` names three nulls: both terms zero, with a
  decision-point effect ``u`` that both families share (a correlation inside the pooled
  plan clusters that no persistent session effect adds, so the marginal hazard still does
  not depend on position or count), the position term zero while compaction has an effect
  (for H1), and the compaction term zero while position has an effect (for H2b). The base
  log odds are drawn per dataset from :data:`BASE_LOGIT`, which spreads the number of
  violations over the bins.
* **Tests.** The rows go through :func:`halflife.e1_analysis.analysis_set` (procedure rules
  in cut spans leave) and :func:`halflife.e1_analysis.hazard_tests`, the code that decides
  H1 and H2b, per family and pooled. Each test is filed under the number of risk-set
  violations of its fit (the count ``Settings.min_events`` compares) in :data:`BIN_EDGES`.

``min_events`` is the smallest bin lower edge at and above which, for both tests and in
every scenario that feeds them, the rejection rate at ``alpha`` among computed tests lies
in :data:`BAND` in every bin with at least ``min_tests`` computed tests (0 when every
assessed bin does; None when even the highest bins do not). Every random draw comes from a
seed, and datasets are split over worker processes by index, so the result does not depend
on ``--jobs``.

``python -m halflife.calibrate --out artifacts/power/min_events.json`` banks it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from halflife import bench, e1_analysis, experiment, stats
from halflife.schema import SessionPlan

ROOT = Path(__file__).resolve().parents[2]
DESIGN = "designs/e1.yaml"
DEFAULT_OUT = "artifacts/power/min_events.json"
DEFAULT_DATASETS = 20000
"""Datasets per scenario in a full run."""
DEFAULT_SEED = 20261111
DEFAULT_MIN_TESTS = 2000
"""Computed tests a bin needs before its rejection rate is judged (a Monte Carlo standard
error of about 0.5 percentage points at a 5% rate)."""
BAND = (0.03, 0.07)
"""The null rejection rates D-050 item 3 accepts, at alpha 0.05."""
BIN_EDGES = (1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 60, 97)
"""Lower edges of the event-count bins (``[edge, next edge)``); a fit has at most 96
risk-set violations (48 plans, two families)."""
BASE_LOGIT = (-6.5, -0.5)
"""Uniform range of the base log odds of a violation per decision point, per dataset,
before a scenario's ``base_shift``."""
TYPE_EFFECTS: Mapping[str, float] = {
    "prohibition": 0.0,
    "invariant": 0.4,
    "procedure": -0.4,
    "resource_limit": 0.2,
}
"""Log odds of each rule type in every scenario (the null model fits them as a factor)."""
START_TOKENS = 300.0
"""Context tokens before the first tool call beyond the opening (rules and first request)."""
RESET_TOKENS = (350.0, 700.0)
"""Uniform range of the context left by a compaction."""
TOKEN_SHAPE = 2.0
"""Gamma shape of the tokens one tool call adds."""
NOTE = (
    "Synthetic: null rejection rates of the H1 and H2b score tests on simulated sessions of "
    "the E1 plans, for calibrating min_events before the E1 preregistration; not model "
    "results or real-world results."
)


@dataclass(frozen=True)
class FamilyShape:
    """How one synthetic family's sessions run.

    Attributes:
        name: Family label in the rows.
        calls_per_request: Mean tool calls a request takes before the caps (geometric,
            at least one).
        tokens_per_call: Mean context tokens one tool call adds (gamma).
        offset: Added to the base log odds of a violation.
    """

    name: str
    calls_per_request: float
    tokens_per_call: float
    offset: float


FAMILIES = (
    FamilyShape("syn-a", 2.0, 120.0, 0.0),
    FamilyShape("syn-b", 3.4, 150.0, 0.5),
)
"""Two synthetic families: a terse one and one that takes more calls per request."""


@dataclass(frozen=True)
class Scenario:
    """One null: which tests it feeds, and the hazard's other terms.

    Attributes:
        name: Scenario label.
        terms: Hypothesis terms whose null it is (``log1p(position)``, ``compaction_count``).
        slope: Log odds per unit of ``log1p(position)``.
        compaction: Log odds per compaction before the decision point.
        dp_sd: Standard deviation of the decision-point effect both families share.
        base_shift: Added to :data:`BASE_LOGIT`, about minus the log odds the active term
            adds at a typical decision point, so the scenario's violation counts cover the
            same bins.
    """

    name: str
    terms: tuple[str, ...]
    slope: float = 0.0
    compaction: float = 0.0
    dp_sd: float = 0.0
    base_shift: float = 0.0


SCENARIOS = (
    Scenario(
        "both_null_shared_dp_effect",
        (e1_analysis.POSITION_TERM, "compaction_count"),
        dp_sd=0.8,
    ),
    Scenario(
        "h1_null_compaction_effect",
        (e1_analysis.POSITION_TERM,),
        compaction=0.4,
        base_shift=-0.8,
    ),
    Scenario("h2b_null_position_effect", ("compaction_count",), slope=0.5, base_shift=-1.6),
)
"""The synthetic nulls (module docstring)."""


@dataclass(frozen=True)
class PlanShape:
    """What the simulation needs from one plan."""

    session_id: str
    kinds: tuple[str, ...]
    dp_ids: tuple[str | None, ...]
    types: tuple[str | None, ...]
    schedule: tuple[int, ...]
    strategy: str
    budget: int
    max_tool_calls: int
    max_calls_per_subtask: int


def plan_shapes(
    plans: Sequence[SessionPlan], templates: Sequence[bench.LoadedTemplate]
) -> list[PlanShape]:
    """The request sequence, rule types, and compaction settings of each plan."""
    types = {t.id: {c.id: c.type for c in t.template.constraints} for t in templates}
    out = []
    for plan in plans:
        rules = types[plan.template_id]
        out.append(
            PlanShape(
                session_id=plan.session_id,
                kinds=tuple(s.kind for s in plan.subtasks),
                dp_ids=tuple(s.id if s.kind == "decision_point" else None for s in plan.subtasks),
                types=tuple(
                    rules[s.constraint] if s.constraint is not None else None for s in plan.subtasks
                ),
                schedule=tuple(plan.compaction.schedule),
                strategy=plan.compaction.strategy,
                budget=int(plan.compaction.budget_tokens or 0),
                max_tool_calls=plan.max_tool_calls,
                max_calls_per_subtask=plan.max_calls_per_subtask,
            )
        )
    return out


@lru_cache(maxsize=1)
def e1_plan_shapes() -> tuple[PlanShape, ...]:
    """The 48 E1 plans of :data:`DESIGN` (generated, never run), as :class:`PlanShape`."""
    design = experiment.load_design(ROOT / DESIGN)
    templates = bench.load_all()
    return tuple(plan_shapes(experiment.make_plans(design, templates), templates))


def simulate_session(
    plan: PlanShape, family: FamilyShape, rng: np.random.Generator
) -> tuple[list[dict[str, Any]], int]:
    """Decision-point rows (without outcomes) of one family's session of ``plan``, and its
    number of compactions (module docstring).

    Each row has ``family``, ``session_id``, ``dp_id``, ``constraint_type``, ``position``,
    ``compaction_count``, ``strategy``, ``span_calls``, and ``span_end``.
    """
    draws = rng.geometric(1.0 / family.calls_per_request, size=len(plan.kinds))
    starts: list[int] = []
    spans: list[tuple[int, str]] = []
    calls = 0
    for draw in draws:
        if calls >= plan.max_tool_calls:
            break
        span = min(int(draw), plan.max_calls_per_subtask)
        ended = "subtask_cap" if draw > plan.max_calls_per_subtask else "done"
        # The session cap wins over the request cap, and a request that finishes on the
        # last allowed call is done (the agent's _run_subtask).
        if calls + span > plan.max_tool_calls:
            span, ended = plan.max_tool_calls - calls, "tool_cap"
        elif calls + span == plan.max_tool_calls and ended == "subtask_cap":
            ended = "tool_cap"
        starts.append(calls)
        spans.append((span, ended))
        calls += span
    compactions: list[int] = []
    if plan.strategy != "none" and calls:
        due = list(plan.schedule)
        tokens = START_TOKENS
        growth = rng.gamma(TOKEN_SHAPE, family.tokens_per_call / TOKEN_SHAPE, size=calls)
        resets = rng.uniform(*RESET_TOKENS, size=calls)
        for index in range(calls):
            if due and due[0] <= index:
                while due and due[0] <= index:
                    due.pop(0)
                compactions.append(index)
                tokens = float(resets[index])
            elif plan.budget and tokens > plan.budget:
                compactions.append(index)
                tokens = float(resets[index])
            tokens += float(growth[index])
    rows = []
    for number, (start, (span, ended)) in enumerate(zip(starts, spans, strict=True)):
        dp_id = plan.dp_ids[number]
        if dp_id is None:
            continue
        rows.append(
            {
                "family": family.name,
                "session_id": plan.session_id,
                "dp_id": dp_id,
                "constraint_type": plan.types[number],
                "position": start,
                "compaction_count": sum(1 for c in compactions if c <= start),
                "strategy": plan.strategy,
                "span_calls": span,
                "span_end": ended,
            }
        )
    return rows, len(compactions)


def simulate_dataset(
    plans: Sequence[PlanShape],
    scenario: Scenario,
    base: float,
    rng: np.random.Generator,
    families: Sequence[FamilyShape] = FAMILIES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Session and opportunity rows of one synthetic null dataset (every plan, every family).

    The decision-point effect of ``scenario.dp_sd`` is drawn once per plan and decision
    point and shared by the families.
    """
    sessions: list[dict[str, Any]] = []
    opportunities: list[dict[str, Any]] = []
    shared: dict[tuple[str, str], float] = {}
    for family in families:
        for plan in plans:
            rows, n_compactions = simulate_session(plan, family, rng)
            for row in rows:
                key = (row["session_id"], row["dp_id"])
                if key not in shared:
                    shared[key] = float(rng.normal(0.0, scenario.dp_sd)) if scenario.dp_sd else 0.0
                eta = (
                    base
                    + family.offset
                    + TYPE_EFFECTS[str(row["constraint_type"])]
                    + scenario.slope * math.log1p(row["position"])
                    + scenario.compaction * row["compaction_count"]
                    + shared[key]
                )
                row["violated"] = bool(rng.random() < 1.0 / (1.0 + math.exp(-eta)))
                opportunities.append(row)
            sessions.append(
                {
                    "family": family.name,
                    "session_id": plan.session_id,
                    "confirmatory": True,
                    "reached_first_dp": bool(rows),
                    "n_compactions": n_compactions,
                }
            )
    return sessions, opportunities


def dataset_tests(
    sessions: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    terms: Sequence[str],
) -> list[dict[str, Any]]:
    """The H1 and H2b tests of one dataset, per family and pooled, as the analysis runs them.

    Each record has ``term``, ``scope`` (``family`` or ``pooled``), ``n_events`` (risk-set
    violations of the fit), ``p_value`` (None when not computed), and ``problem``.
    """
    rows, _ = e1_analysis.analysis_set(sessions, opportunities)
    out: list[dict[str, Any]] = []
    for family in [*e1_analysis.families(rows), "pooled"]:
        subset = [r for r in rows if family == "pooled" or r["family"] == family]
        for term, test in e1_analysis.hazard_tests(subset, family, terms).items():
            out.append(
                {
                    "term": term,
                    "scope": "pooled" if family == "pooled" else "family",
                    "n_events": test.n_events,
                    "p_value": test.test.p_value,
                    "problem": test.test.problem,
                }
            )
    return out


def _dataset_seed(seed: int, scenario: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, scenario, index]))


def _run_chunk(job: tuple[int, int, int, int]) -> list[dict[str, Any]]:
    """Tests of datasets ``start..stop`` of one scenario (a worker's share)."""
    seed, scenario_index, start, stop = job
    plans = e1_plan_shapes()
    scenario = SCENARIOS[scenario_index]
    out = []
    for index in range(start, stop):
        rng = _dataset_seed(seed, scenario_index, index)
        base = float(rng.uniform(*BASE_LOGIT)) + scenario.base_shift
        sessions, opportunities = simulate_dataset(plans, scenario, base, rng)
        for record in dataset_tests(sessions, opportunities, scenario.terms):
            out.append(record | {"scenario": scenario.name, "dataset": index})
    return out


def bin_of(n_events: int, edges: Sequence[int] = BIN_EDGES) -> int | None:
    """Index of the bin ``[edges[i], edges[i + 1])`` holding ``n_events``, or None outside."""
    for i in range(len(edges) - 1):
        if edges[i] <= n_events < edges[i + 1]:
            return i
    return None


def tabulate(
    records: Sequence[Mapping[str, Any]],
    alpha: float = 0.05,
    edges: Sequence[int] = BIN_EDGES,
) -> list[dict[str, Any]]:
    """Rejection rates by term, scenario, and event-count bin, with per-scope counts.

    ``rate`` is rejections (``p < alpha``) over computed tests, with a Wilson interval for
    the Monte Carlo error; ``not_computed`` counts tests without a p-value by problem code.
    Fits outside the bins (no violation) are counted under ``outside_bins``.
    """
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    outside: dict[tuple[str, str], int] = {}
    for record in records:
        index = bin_of(int(record["n_events"]), edges)
        key2 = (str(record["term"]), str(record["scenario"]))
        if index is None:
            outside[key2] = outside.get(key2, 0) + 1
            continue
        cell = cells.setdefault(
            (*key2, index),
            {"tests": 0, "computed": 0, "rejected": 0, "not_computed": {}, "by_scope": {}},
        )
        scope = cell["by_scope"].setdefault(
            str(record["scope"]), {"tests": 0, "computed": 0, "rejected": 0}
        )
        for target in (cell, scope):
            target["tests"] += 1
        if record["p_value"] is None:
            code = str(record["problem"])
            cell["not_computed"][code] = cell["not_computed"].get(code, 0) + 1
            continue
        hit = float(record["p_value"]) < alpha
        for target in (cell, scope):
            target["computed"] += 1
            target["rejected"] += int(hit)
    out = []
    for term, scenario in sorted({(k[0], k[1]) for k in cells} | set(outside)):
        bins = []
        for i in range(len(edges) - 1):
            found = cells.get((term, scenario, i))
            if found is None:
                continue
            entry: dict[str, Any] = {"lower": edges[i], "upper": edges[i + 1] - 1} | found
            if found["computed"]:
                rate = stats.wilson_interval(found["rejected"], found["computed"])
                entry["rate"] = rate.to_dict()
            else:
                entry["rate"] = None
            bins.append(entry)
        out.append(
            {
                "term": term,
                "scenario": scenario,
                "outside_bins": outside.get((term, scenario), 0),
                "bins": bins,
            }
        )
    return out


def min_events_from(
    table: Sequence[Mapping[str, Any]],
    band: tuple[float, float] = BAND,
    min_tests: int = DEFAULT_MIN_TESTS,
) -> int | None:
    """The calibrated ``min_events`` (module docstring) from a :func:`tabulate` table.

    Only bins with at least ``min_tests`` computed tests are judged. Returns 0 when every
    judged bin of every term and scenario is inside ``band``, the lower edge of the bin
    above the highest judged bin outside it otherwise, and None when that bin is the
    highest judged one's successor and no judged bin lies above it (nothing is calibrated).
    """
    judged: list[tuple[int, bool]] = []
    for entry in table:
        for cell in entry["bins"]:
            if cell["computed"] < min_tests or cell["rate"] is None:
                continue
            rate = float(cell["rate"]["value"])
            judged.append((int(cell["lower"]), band[0] <= rate <= band[1]))
    if not judged:
        return None
    failing = [lower for lower, inside in judged if not inside]
    if not failing:
        return 0
    worst = max(failing)
    above = sorted({lower for lower, _ in judged if lower > worst})
    return above[0] if above else None


def calibration(
    n_datasets: int = DEFAULT_DATASETS,
    seed: int = DEFAULT_SEED,
    *,
    jobs: int = 1,
    min_tests: int = DEFAULT_MIN_TESTS,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """The data fields of the calibration artifact (what ``main`` banks, without writing).

    ``n_datasets`` datasets per scenario; ``jobs`` worker processes (the result is the same
    for any number).
    """
    if n_datasets < 1 or jobs < 1:
        raise stats.StatsError("n_datasets and jobs must be at least 1")
    chunk = max(1, math.ceil(n_datasets / (4 * jobs)))
    work = [
        (seed, s, start, min(n_datasets, start + chunk))
        for s in range(len(SCENARIOS))
        for start in range(0, n_datasets, chunk)
    ]
    if jobs == 1:
        results = [_run_chunk(job) for job in work]
    else:
        with multiprocessing.get_context("spawn").Pool(jobs) as pool:
            results = pool.map(_run_chunk, work)
    records = [record for part in results for record in part]
    table = tabulate(records, alpha)
    plans = e1_plan_shapes()
    data = {
        "synthetic": True,
        "note": NOTE,
        "design": {
            "plans": DESIGN,
            "n_plans": len(plans),
            "families": [asdict(f) for f in FAMILIES],
            "scenarios": [asdict(s) for s in SCENARIOS],
            "type_effects": dict(TYPE_EFFECTS),
            "base_logit": list(BASE_LOGIT),
            "start_tokens": START_TOKENS,
            "reset_tokens": list(RESET_TOKENS),
            "token_shape": TOKEN_SHAPE,
            "bin_edges": list(BIN_EDGES),
            "band": list(BAND),
            "alpha": alpha,
            "min_tests": min_tests,
            "n_datasets": n_datasets,
            "seed": seed,
            "analysis": "halflife.e1_analysis.analysis_set and hazard_tests",
        },
        "tests": len(records),
        "table": table,
        "min_events": min_events_from(table, BAND, min_tests),
        "rule": (
            "min_events is the smallest bin lower edge at and above which every bin with at "
            f"least {min_tests} computed tests has a null rejection rate (p < {alpha:g}) within "
            f"[{BAND[0]:g}, {BAND[1]:g}] for both tests in every scenario that feeds them; 0 "
            "when every such bin does"
        ),
    }
    # As the banked file holds it (lists, not tuples), so a derivation compares equal.
    parsed: dict[str, Any] = json.loads(json.dumps(data, allow_nan=False))
    return parsed


CODE_MODULES = (*e1_analysis.ANALYSIS_MODULES, "halflife/calibrate.py")
"""Modules whose code the calibration runs (paths under ``src/``), hashed in its provenance."""


def code_sha256() -> dict[str, str]:
    """SHA-256 of each module in :data:`CODE_MODULES`, by its path under ``src/``."""
    source = Path(__file__).resolve().parent.parent
    return {rel: hashlib.sha256((source / rel).read_bytes()).hexdigest() for rel in CODE_MODULES}


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
    """``python -m halflife.calibrate --out artifacts/power/min_events.json``: bank it."""
    parser = argparse.ArgumentParser(description="Calibrate min_events on synthetic E1 nulls.")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--n-datasets", type=int, default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-tests", type=int, default=DEFAULT_MIN_TESTS)
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    args = parser.parse_args(argv)
    start = time.monotonic()
    data = calibration(args.n_datasets, args.seed, jobs=args.jobs, min_tests=args.min_tests)
    artifact = data | {
        "command": "python -m halflife.calibrate " + " ".join(argv or sys.argv[1:]),
        "code_sha256": code_sha256(),
        "git_sha": _git_sha(ROOT),
        "created": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_s": round(time.monotonic() - start, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}: min_events {data['min_events']} from {data['tests']} tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
