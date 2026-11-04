"""Tests for the E3 and E4 analyses on synthetic rows (labeled synthetic; not model results)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from halflife import arms_analysis, stats
from halflife.e1_analysis import Settings

FAST = Settings(n_boot=60, seed=7)
TYPES = ("prohibition", "invariant", "procedure", "resource_limit")


def _synthetic(
    arm_key: str, arms: list[dict[str, str]], families: tuple[str, ...], per_arm: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthetic sessions and opportunities; arm index raises the violation chance."""
    rng = random.Random(seed)
    sessions: list[dict[str, Any]] = []
    opportunities: list[dict[str, Any]] = []
    for family in families:
        for a, arm in enumerate(arms):
            for i in range(per_arm):
                sid = f"syn-{arm_key}-{a}-{i}"
                sessions.append(
                    {
                        "session_id": sid,
                        "family": family,
                        "model": f"{family}-model",
                        "template_id": f"tpl-{i % 3}",
                        "confirmatory": True,
                        "reached_first_dp": True,
                        "mitigation": arm.get("mitigation", "none"),
                        "subtasks_checked": 4,
                        "subtasks_succeeded": 3 if a % 2 else 4,
                        "usage": {
                            "step": {
                                "calls": 10,
                                "prompt_tokens": 1000 + 100 * a,
                                "completion_tokens": 50,
                                "elapsed_s": 20.0 + a,
                            },
                            "probe": {
                                "calls": 3,
                                "prompt_tokens": 999,
                                "completion_tokens": 9,
                                "elapsed_s": 9.0,
                            },
                        },
                    }
                )
                for j in range(4):
                    row = {
                        "session_id": sid,
                        "family": family,
                        "constraint_type": TYPES[j],
                        "position": 5 + 10 * j + i % 5,
                        "compaction_count": rng.randint(0, 2),
                        "violated": rng.random() < 0.1 + 0.08 * a,
                        "recalled_before": rng.random() < 0.6,
                        "span_calls": 2,
                        "placement": "first_user",
                        "phrasing": "bare",
                        "strategy": "full_summary",
                        "mitigation": "none",
                    } | arm
                    opportunities.append(row)
    return sessions, opportunities


E3_ARMS = [
    {"placement": p, "phrasing": ph}
    for p in ("first_user", "system", "pinned_file", "reminder")
    for ph in ("bare", "reason")
]
E4_ARMS = [{"mitigation": m} for m in ("none", *arms_analysis.MITIGATIONS)]


def test_e3_outputs_have_cells_contrasts_and_hazard() -> None:
    sessions, opportunities = _synthetic("e3", E3_ARMS, ("fa", "fb"), 4, 1)
    out = arms_analysis.analyze_e3(sessions, opportunities, FAST)
    assert set(out) == {"rates", "contrasts", "hazard", "summary"}
    assert out["summary"]["arm"] == "E3" and out["summary"]["n_sessions"] == len(sessions)
    pooled = next(f for f in out["rates"]["by_family"] if f["family"] == "pooled")
    assert len(pooled["cells"]) == 8 and {c["placement"] for c in pooled["placement"]} == {
        "first_user",
        "system",
        "pinned_file",
        "reminder",
    }
    assert sum(c["opportunities"] for c in pooled["cells"]) == len(opportunities)
    contrasts = next(f for f in out["contrasts"]["by_family"] if f["family"] == "pooled")
    assert [c["placement"] for c in contrasts["placement"]] == list(arms_analysis.PLACEMENTS)
    raw = sorted(c["p_value"] for c in contrasts["placement"])
    assert all(c["p_holm"] >= c["p_value"] for c in contrasts["placement"])
    assert min(c["p_holm"] for c in contrasts["placement"]) == pytest.approx(min(1.0, 3 * raw[0]))
    assert [c["phrasing"] for c in contrasts["phrasing"]] == ["reason"]
    terms = {(e["family"], e["term"]) for e in out["hazard"]["estimates"]}
    assert ("pooled", "log1p(position)") in terms
    assert any(t.startswith("family") for f, t in terms if f == "pooled")
    assert any(t.startswith("placement") for f, t in terms if f == "fa")


def test_e4_costs_sit_beside_every_effect() -> None:
    sessions, opportunities = _synthetic("e4", E4_ARMS, ("fa",), 5, 2)
    out = arms_analysis.analyze_e4(sessions, opportunities, FAST)
    summary = out["summary"]
    assert summary["arm"] == "E4" and summary["ran"] is True
    assert summary["n_sessions"] == len(sessions)
    assert summary["held_out_templates"] == ["tpl-0", "tpl-1", "tpl-2"]
    names = [m["name"] for m in summary["mitigations"]]
    assert names == list(arms_analysis.MITIGATIONS)
    for index, mitigation in enumerate(summary["mitigations"], start=1):
        cost = mitigation["cost"]
        assert set(cost) == set(arms_analysis.COST_FIELDS)
        # probes are left out: step tokens only, 10 calls' worth per session
        assert cost["tokens"] == pytest.approx(1000 + 100 * index + 50)
        assert cost["latency_s"] == pytest.approx(20.0 + index)
        assert cost["task_success"] == pytest.approx(0.75 if index % 2 else 1.0)
        assert mitigation["p_holm"] is not None and mitigation["odds_ratio"] is not None
    assert summary["control"]["cost"]["tokens"] == pytest.approx(1050)
    pooled = next(f for f in out["costs"]["by_family"] if f["family"] == "pooled")
    guard = next(a for a in pooled["arms"] if a["mitigation"] == "guard")
    assert guard["tokens_vs_none"]["difference"] == pytest.approx(100)
    assert "tokens_vs_none" not in next(a for a in pooled["arms"] if a["mitigation"] == "none")


def test_session_costs_handle_missing_usage_and_checks() -> None:
    assert arms_analysis.session_costs({"usage": None}) == {
        "tokens": None,
        "latency_s": None,
        "task_success": None,
    }
    costs = arms_analysis.session_costs(
        {
            "usage": {"compaction": {"prompt_tokens": 5, "completion_tokens": 1, "elapsed_s": 2}},
            "subtasks_checked": 2,
            "subtasks_succeeded": 1,
        }
    )
    assert costs == {"tokens": 6.0, "latency_s": 2.0, "task_success": 0.5}


def test_e4_costs_count_missing_usage_and_skip_absent_reference() -> None:
    sessions, _ = _synthetic("e4", E4_ARMS[1:3], ("fa",), 3, 3)
    sessions[0]["usage"] = None
    costs, means = arms_analysis.e4_costs(sessions, FAST)
    pooled = next(f for f in costs["by_family"] if f["family"] == "pooled")
    guard = next(a for a in pooled["arms"] if a["mitigation"] == "guard")
    assert guard["tokens_missing"] == 1 and "tokens_vs_none" not in guard
    assert set(means) == {"guard", "pin"}


def test_contrasts_need_the_reference_arm() -> None:
    _, opportunities = _synthetic("e4", E4_ARMS[1:], ("fa",), 2, 4)
    assert (
        arms_analysis.contrasts_against(opportunities, "mitigation", "none", ["guard"], FAST) == []
    )


def test_empty_analysis_set_is_an_error() -> None:
    with pytest.raises(stats.StatsError):
        arms_analysis.analyze_e3([], [], FAST)
    with pytest.raises(stats.StatsError):
        arms_analysis.analyze_e4([], [], FAST)


@pytest.mark.parametrize("experiment", ["e3", "e4"])
def test_main_writes_every_output(tmp_path: Path, experiment: str) -> None:
    arms = E3_ARMS if experiment == "e3" else E4_ARMS
    sessions, opportunities = _synthetic(experiment, arms, ("fa",), 3, 5)
    for name, rows in (("sessions", sessions), ("opportunities", opportunities)):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    code = arms_analysis.main([experiment, "--dir", str(tmp_path), "--n-boot", "40"])
    assert code == 0
    stems = {"rates", "contrasts", "summary", "hazard" if experiment == "e3" else "costs"}
    assert {p.stem for p in tmp_path.glob("*.json")} == stems
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["arm"] == experiment.upper() and summary["settings"]["n_boot"] == 40


def test_e2_summary_is_ecological_and_lists_tiers(tmp_path: Path) -> None:
    sessions, opportunities = _synthetic("e2", [{}], ("claude",), 4, 6)
    tiers = ("haiku", "sonnet")
    tiered_sessions = [dict(s, model=t, confirmatory=False) for t in tiers for s in sessions]
    tiered_opps = [dict(o, model=t) for t in tiers for o in opportunities]
    tiered_sessions[0]["reached_first_dp"] = False
    out = arms_analysis.analyze_e2(tiered_sessions, tiered_opps, FAST)
    summary = out["summary"]
    assert summary == {
        "arm": "E2",
        "ran": True,
        "n_sessions": 8,
        "label": "ecological",
        "models": ["haiku", "sonnet"],
        "confirmatory": False,
        "settings": FAST.__dict__,
    }
    by_tier = {r["model"]: r for r in out["rates"]["model"]}
    assert by_tier["haiku"]["opportunities"] == 12 and by_tier["sonnet"]["opportunities"] == 16
    assert out["rates"]["exclusions"]["sessions_not_reaching_first_dp"] == 1
    with pytest.raises(stats.StatsError):
        arms_analysis.analyze_e2(tiered_sessions, [], FAST)
    for name, rows in (("sessions", tiered_sessions), ("opportunities", tiered_opps)):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    assert arms_analysis.main(["e2", "--dir", str(tmp_path), "--n-boot", "20"]) == 0
    assert {p.stem for p in tmp_path.glob("*.json")} == {"rates", "summary"}
