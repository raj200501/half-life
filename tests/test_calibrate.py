"""Tests for the min_events calibration (synthetic sessions; not model results)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from halflife import calibrate, e1_analysis, stats
from halflife.calibrate import FamilyShape, PlanShape, Scenario


def _plan(
    schedule: tuple[int, ...] = (),
    *,
    budget: int = 10**9,
    max_tool_calls: int = 150,
    per_request: int = 12,
) -> PlanShape:
    """A hand-made plan: work, a prohibition decision point, work, a procedure one."""
    return PlanShape(
        session_id="p0",
        kinds=("work", "decision_point", "work", "decision_point"),
        dp_ids=(None, "dp1", None, "dp2"),
        types=(None, "prohibition", None, "procedure"),
        schedule=schedule,
        strategy="full_summary",
        budget=budget,
        max_tool_calls=max_tool_calls,
        max_calls_per_subtask=per_request,
    )


ONE_CALL = FamilyShape("one", 1.0, 1.0, 0.0)
"""Every request takes exactly one tool call (a geometric draw with p = 1)."""


def test_positions_and_forced_compactions_follow_the_agent_rules() -> None:
    rng = np.random.default_rng(1)
    rows, compactions = calibrate.simulate_session(_plan((2,)), ONE_CALL, rng)
    # One call per request: the decision points start at tool calls 1 and 3; the forced
    # compaction at index 2 runs before call 2, so only the second decision point saw it.
    assert [(r["dp_id"], r["position"], r["compaction_count"]) for r in rows] == [
        ("dp1", 1, 0),
        ("dp2", 3, 1),
    ]
    assert compactions == 1
    assert [r["constraint_type"] for r in rows] == ["prohibition", "procedure"]
    assert {r["span_end"] for r in rows} == {"done"} and {r["span_calls"] for r in rows} == {1}
    assert {r["family"] for r in rows} == {"one"} and {r["strategy"] for r in rows} == {
        "full_summary"
    }
    # A compaction at the decision point's own start index runs before its first call.
    rows, _ = calibrate.simulate_session(_plan((3,)), ONE_CALL, rng)
    assert [r["compaction_count"] for r in rows] == [0, 1]


def test_token_budget_compactions_run_before_every_call_once_the_budget_is_passed() -> None:
    # Every call adds far more tokens than the budget, so from index 1 on each call starts
    # with a compaction: the count at a position is the position itself.
    flood = FamilyShape("flood", 1.0, 1e9, 0.0)
    rows, compactions = calibrate.simulate_session(
        _plan(budget=3000), flood, np.random.default_rng(2)
    )
    assert [(r["position"], r["compaction_count"]) for r in rows] == [(1, 1), (3, 3)]
    assert compactions == 3
    none = dataclasses.replace(_plan(budget=3000), strategy="none")
    assert calibrate.simulate_session(none, flood, np.random.default_rng(2))[1] == 0


def test_request_and_session_caps_cut_spans_as_the_agent_does() -> None:
    # A request needs far more than 12 calls, so every one is cut at 12 (subtask_cap); the
    # session cap of 30 cuts the third request after 6 calls (tool_cap) and the fourth
    # never starts.
    slow = FamilyShape("slow", 1e6, 1.0, 0.0)
    rows, _ = calibrate.simulate_session(_plan(max_tool_calls=30), slow, np.random.default_rng(3))
    assert [(r["dp_id"], r["position"], r["span_calls"], r["span_end"]) for r in rows] == [
        ("dp1", 12, 12, "subtask_cap")
    ]
    capped = _plan(max_tool_calls=36)
    rows, _ = calibrate.simulate_session(capped, slow, np.random.default_rng(3))
    # The fourth request starts at 36, the session cap: it is not reached.
    assert [r["dp_id"] for r in rows] == ["dp1"]
    edge = _plan(max_tool_calls=48)
    rows, _ = calibrate.simulate_session(edge, slow, np.random.default_rng(3))
    # Cut by its own cap on the last allowed call: the session cap wins.
    assert rows[-1]["span_end"] == "tool_cap" and rows[-1]["span_calls"] == 12


def test_the_e1_plans_are_the_design_files_48_plans() -> None:
    plans = calibrate.e1_plan_shapes()
    assert len(plans) == 48 and len({p.session_id for p in plans}) == 48
    assert {p.strategy for p in plans} == {"full_summary", "sham"}
    assert {len(p.kinds) for p in plans} == {20}
    for plan in plans:
        dps = [t for k, t in zip(plan.kinds, plan.types, strict=True) if k == "decision_point"]
        assert 3 <= len(dps) <= 5 and set(dps) <= set(calibrate.TYPE_EFFECTS)
        assert plan.max_tool_calls == 150 and plan.budget == 3000
        assert 1 <= len(plan.schedule) <= 2 and all(8 <= c < 70 for c in plan.schedule)


def test_datasets_run_every_plan_for_both_families_with_the_stated_hazard() -> None:
    plans = calibrate.e1_plan_shapes()[:6]
    scenario = Scenario("test", ("compaction_count",))
    sessions, rows = calibrate.simulate_dataset(plans, scenario, 50.0, np.random.default_rng(4))
    assert len(sessions) == 12 and {s["family"] for s in sessions} == {"syn-a", "syn-b"}
    assert all(r["violated"] for r in rows)  # log odds 50: every decision point violates
    _, quiet = calibrate.simulate_dataset(plans, scenario, -50.0, np.random.default_rng(4))
    assert not any(r["violated"] for r in quiet)
    by_family = {f: {r["session_id"] for r in rows if r["family"] == f} for f in ("syn-a", "syn-b")}
    assert by_family["syn-a"] == by_family["syn-b"]  # the families run the same plans
    # A shared decision-point effect is drawn once per plan and decision point.
    shared = Scenario("shared", ("compaction_count",), dp_sd=50.0)
    _, both = calibrate.simulate_dataset(plans, shared, 0.0, np.random.default_rng(5))
    outcome = {}
    for row in both:
        outcome.setdefault((row["session_id"], row["dp_id"]), set()).add(row["violated"])
    assert sum(len(v) == 1 for v in outcome.values()) >= 0.9 * len(outcome)


def test_dataset_tests_are_the_analysis_tests_on_the_analysis_set() -> None:
    plans = calibrate.e1_plan_shapes()
    scenario = calibrate.SCENARIOS[0]
    sessions, opportunities = calibrate.simulate_dataset(
        plans, scenario, -2.0, np.random.default_rng(6)
    )
    records = calibrate.dataset_tests(sessions, opportunities, scenario.terms)
    assert [(r["scope"], r["term"]) for r in records] == [
        (scope, term)
        for scope in ("family", "family", "pooled")
        for term in (e1_analysis.POSITION_TERM, "compaction_count")
    ]
    rows, _ = e1_analysis.analysis_set(sessions, opportunities)
    pooled = e1_analysis.hazard_tests(rows, "pooled")
    for record in records[4:]:
        expected = pooled[record["term"]]
        assert record["p_value"] == expected.test.p_value
        assert record["n_events"] == expected.n_events
    only = calibrate.dataset_tests(sessions, opportunities, ("compaction_count",))
    assert {r["term"] for r in only} == {"compaction_count"} and len(only) == 3


def _records(rates: dict[int, tuple[int, int]], scenario: str = "s") -> list[dict[str, Any]]:
    """Synthetic test records: for each event count, (computed tests, rejections)."""
    out = []
    for events, (computed, rejected) in rates.items():
        for i in range(computed):
            out.append(
                {
                    "term": "t",
                    "scenario": scenario,
                    "scope": "family" if i % 2 else "pooled",
                    "n_events": events,
                    "p_value": 0.01 if i < rejected else 0.5,
                    "problem": None,
                }
            )
    return out


def test_tabulate_counts_rejections_by_bin_and_scope() -> None:
    records = _records({1: (10, 1), 3: (20, 4), 50: (40, 2)})
    records.append(dict(records[0], p_value=None, problem="null_separation"))
    records.append(dict(records[0], n_events=0, p_value=None, problem="no_events"))
    table = calibrate.tabulate(records, edges=(1, 2, 10, 97))
    (entry,) = table
    assert entry["outside_bins"] == 1
    first, second, third = entry["bins"]
    assert (first["lower"], first["upper"], first["tests"], first["computed"]) == (1, 1, 11, 10)
    assert first["not_computed"] == {"null_separation": 1}
    assert first["rate"]["k"] == 1 and first["rate"]["n"] == 10
    assert (second["lower"], second["upper"], second["rejected"]) == (2, 9, 4)
    assert second["rate"]["value"] == pytest.approx(0.2)
    assert third["by_scope"]["pooled"]["computed"] + third["by_scope"]["family"]["computed"] == 40
    assert calibrate.bin_of(0) is None and calibrate.bin_of(96) == len(calibrate.BIN_EDGES) - 2
    assert calibrate.bin_of(97) is None


def _table(rates: dict[int, float], computed: int = 2000) -> list[dict[str, Any]]:
    return [
        {
            "term": "t",
            "scenario": "s",
            "bins": [
                {"lower": lower, "computed": computed, "rate": {"value": rate}}
                for lower, rate in sorted(rates.items())
            ],
        }
    ]


def test_min_events_is_the_lowest_edge_above_every_bin_outside_the_band() -> None:
    every = {1: 0.05, 2: 0.04, 5: 0.06, 10: 0.05}
    assert calibrate.min_events_from(_table(every)) == 0
    low = {1: 0.11, 2: 0.02, 5: 0.06, 10: 0.05}
    assert calibrate.min_events_from(_table(low)) == 5
    gap = {1: 0.05, 2: 0.08, 5: 0.06, 10: 0.05}
    assert calibrate.min_events_from(_table(gap)) == 5  # every bin at and above 5 is inside
    high = {1: 0.05, 2: 0.04, 5: 0.06, 10: 0.075}
    assert calibrate.min_events_from(_table(high)) is None  # nothing above the failing bin
    edge = {1: 0.03, 2: 0.07}
    assert calibrate.min_events_from(_table(edge)) == 0  # the band is closed
    # Bins with too few computed tests are not judged.
    assert calibrate.min_events_from(_table(low, computed=10)) is None
    sparse = _table(low) + _table({1: 0.2}, computed=5)
    assert calibrate.min_events_from(sparse) == 5
    # Every term and scenario counts.
    other = _table({1: 0.05, 2: 0.05, 5: 0.05, 10: 0.05}) + _table({1: 0.05, 2: 0.09, 5: 0.05})
    assert calibrate.min_events_from(other) == 5


def test_calibration_is_seeded_and_does_not_depend_on_the_number_of_workers() -> None:
    one = calibrate.calibration(2, 11, jobs=1, min_tests=1)
    assert one["synthetic"] is True and one["note"].startswith("Synthetic")
    assert one["design"]["n_plans"] == 48 and one["design"]["n_datasets"] == 2
    assert one["design"]["band"] == [0.03, 0.07] and one["design"]["alpha"] == 0.05
    terms = {(e["term"], e["scenario"]) for e in one["table"]}
    assert terms == {
        (e1_analysis.POSITION_TERM, "both_null_shared_dp_effect"),
        ("compaction_count", "both_null_shared_dp_effect"),
        (e1_analysis.POSITION_TERM, "h1_null_compaction_effect"),
        ("compaction_count", "h2b_null_position_effect"),
    }
    # Two datasets of the first scenario (six tests each) and of the others (three each).
    assert one["tests"] == 2 * 6 + 2 * 3 + 2 * 3
    assert calibrate.calibration(2, 11, jobs=2, min_tests=1) == one
    assert calibrate.calibration(2, 12, jobs=1, min_tests=1) != one
    json.dumps(one, allow_nan=False)
    with pytest.raises(stats.StatsError, match="at least 1"):
        calibrate.calibration(0, 1)


def test_main_banks_a_labeled_artifact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "power" / "min_events.json"
    argv = ["--out", str(out), "--n-datasets", "1", "--min-tests", "1", "--seed", "3"]
    assert calibrate.main(argv) == 0
    banked = json.loads(out.read_text(encoding="utf-8"))
    assert banked["synthetic"] is True and banked["command"].startswith(
        "python -m halflife.calibrate"
    )
    assert {"git_sha", "created", "runtime_s", "min_events", "rule", "table"} <= set(banked)
    assert banked["code_sha256"] == calibrate.code_sha256()
    assert set(banked["code_sha256"]) == {*e1_analysis.ANALYSIS_MODULES, "halflife/calibrate.py"}
    assert banked == banked | calibrate.calibration(1, 3, min_tests=1)
    assert "min_events" in capsys.readouterr().out


def test_settings_min_events_is_the_banked_calibration() -> None:
    """D-050 item 3: Settings.min_events cites artifacts/power/min_events.json."""
    banked = json.loads((calibrate.ROOT / calibrate.DEFAULT_OUT).read_text(encoding="utf-8"))
    assert banked["synthetic"] is True
    assert banked["min_events"] == e1_analysis.Settings.min_events
    design = banked["design"]
    assert design["n_datasets"] == calibrate.DEFAULT_DATASETS and design["n_plans"] == 48
    assert design["seed"] == calibrate.DEFAULT_SEED
    assert design["min_tests"] == calibrate.DEFAULT_MIN_TESTS
    assert design["bin_edges"] == list(calibrate.BIN_EDGES)
    # The rule applied to the banked table gives the banked value.
    assert calibrate.min_events_from(banked["table"]) == banked["min_events"]
