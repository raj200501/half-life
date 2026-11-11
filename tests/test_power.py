"""Tests for the E1 power analysis on synthetic pilot rows (labeled synthetic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from halflife import power, stats


def _pilot(violations: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sessions = [
        {"session_id": f"s{i}", "reached_first_dp": True, "dps_reached": 4}
        | {"n_tool_calls": 40 + i, "n_compactions": 2}
        for i in range(3)
    ] + [{"session_id": "s9", "reached_first_dp": False, "dps_reached": 0}]
    opportunities = [{"session_id": f"s{i % 3}", "violated": i < violations} for i in range(12)]
    return sessions, opportunities


def test_pilot_inputs_measure_the_session_shape() -> None:
    inputs = power.pilot_inputs(*_pilot(3))
    assert inputs["sessions"] == 4 and inputs["sessions_reaching_first_dp"] == 3
    assert inputs["opportunities_per_session"] == 4 and inputs["max_position"] == 41
    assert inputs["compactions_per_session"] == 2
    assert inputs["violation_rate"]["value"] == pytest.approx(0.25)
    with pytest.raises(stats.StatsError):
        power.pilot_inputs([], [])


def test_pilot_inputs_use_the_e1_families_and_the_analysis_set() -> None:
    # Synthetic pilot: family "fast" qualified for E1, "idle" did not. Its zero-call and
    # leaked sessions' decision points leave the measured rate as they leave E1's.
    sessions = [
        {"session_id": f"s{i}", "family": family, "reached_first_dp": True, "dps_reached": 2}
        | {"n_tool_calls": calls, "n_compactions": 2}
        for family, calls in (("fast", 60), ("idle", 20))
        for i in range(3)
    ]
    sessions[2]["probe_leaks"] = [{"call": 3}]
    opportunities = [
        {"session_id": s["session_id"], "family": s["family"], "violated": j == 0}
        | {"span_calls": 0 if (s["session_id"], j) == ("s1", 1) else 2}
        for s in sessions
        for j in range(2)
    ]
    inputs = power.pilot_inputs(sessions, opportunities, families=["fast"])
    assert inputs["families"] == ["fast"] and inputs["families_excluded"] == ["idle"]
    # fast: s0 (2 kept), s1 (1 kept: one zero-call decision point), s2 leaked.
    assert inputs["sessions"] == 3 and inputs["sessions_reaching_first_dp"] == 2
    assert inputs["opportunities"] == 3 and inputs["violations"] == 2
    assert inputs["max_position"] == 60
    every = power.pilot_inputs(sessions, opportunities)
    assert every["families"] == ["fast", "idle"] and every["families_excluded"] == []
    assert "unpadded" in inputs["shape_note"]


def test_grid_adds_the_pilot_rate_only_when_usable() -> None:
    inputs = power.pilot_inputs(*_pilot(3))
    rows = power.grid(inputs, sessions_grid=(4,), effects_grid=(1.0,), rates=(0.1,), n_sim=5)
    assert [r["inputs"]["base_rate"] for r in rows] == [0.1, 0.25]
    assert [r["pilot_rate"] for r in rows] == [False, True]
    assert all(r["synthetic"] is True for r in rows)
    zero = power.pilot_inputs(*_pilot(0))
    rows = power.grid(zero, sessions_grid=(4,), effects_grid=(1.0,), rates=(0.1,), n_sim=5)
    assert len(rows) == 1 and rows[0]["inputs"]["max_position"] == 41


def test_grid_splits_each_family_between_the_two_contrast_arms() -> None:
    # H2a compares the two compaction arms within a family, so a family of n sessions puts
    # n // 2 of them in each arm; the hazard design keeps all n.
    inputs = power.pilot_inputs(*_pilot(3))
    rows = power.grid(inputs, sessions_grid=(40, 41), effects_grid=(1.0,), rates=(0.1,), n_sim=3)
    assert [r["inputs"]["n_sessions"] for r in rows] == [40, 40, 41, 41]
    for row in rows:
        half = row["inputs"]["n_sessions"] // 2
        assert row["inputs"]["arm_sessions"] == half
        assert row["contrast"]["design"].startswith(f"{half} sessions per arm")


def test_main_banks_the_grid(tmp_path: Path) -> None:
    sessions, opportunities = _pilot(2)
    for name, rows in (("sessions", sessions), ("opportunities", opportunities)):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    out = tmp_path / "power" / "e1.json"
    assert power.main(["--pilot", str(tmp_path), "--out", str(out), "--n-sim", "3"]) == 0
    banked = json.loads(out.read_text(encoding="utf-8"))
    assert banked["synthetic"] is True and banked["pilot"]["violations"] == 2
    assert banked["families"] is None  # no family choice banked yet: every family
    # Once the pilot's family choice is banked, power uses the chosen families only.
    (tmp_path / "family_choice.json").write_text(json.dumps({"chosen": ["x"]}), encoding="utf-8")
    with pytest.raises(stats.StatsError, match="no session that reached"):
        power.main(["--pilot", str(tmp_path), "--out", str(out), "--n-sim", "3"])
    grid_size = len(power.SESSIONS_GRID) * len(power.EFFECTS_GRID) * (len(power.PLANNING_RATES) + 1)
    assert len(banked["grid"]) == grid_size and banked["command"].startswith("python -m")


def test_pilot_inputs_drop_procedure_rules_in_cut_spans_as_e1_does() -> None:
    # D-050 item 5 (synthetic pilot rows): a procedure decision point whose span was cut,
    # or whose row predates span_end, leaves the measured rate as it leaves E1's.
    sessions, opportunities = _pilot(6)
    for i, row in enumerate(opportunities):
        row["constraint_type"] = "procedure" if i % 2 else "prohibition"
        if i in (1, 3):
            row["span_end"] = "done"
        elif i == 5:
            row["span_end"] = "subtask_cap"
    inputs = power.pilot_inputs(sessions, opportunities)
    # Odd rows are procedures: 1 and 3 finished, 5 was cut, 7, 9, and 11 have no span_end.
    assert inputs["opportunities_procedure_cut"] == 4
    assert inputs["opportunities"] == 8
    assert inputs["violations"] == sum(
        1 for i, o in enumerate(opportunities) if o["violated"] and i not in (5, 7, 9, 11)
    )
