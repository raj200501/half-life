"""Tests for the E3 and E4 analyses on synthetic rows (labeled synthetic; not model results)."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from halflife import arms_analysis, stats
from halflife.e1_analysis import Settings
from halflife.e1_analysis import holm as e1_holm

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
                        "subtasks_planned_checked": 4,
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
                        "span_end": "done",
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


def _manual_risk_set(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each (family, session) in position order up to its first violation, by hand."""
    out: list[dict[str, Any]] = []
    for key in sorted({(r["family"], r["session_id"]) for r in rows}):
        mine = sorted(
            (r for r in rows if (r["family"], r["session_id"]) == key), key=lambda r: r["position"]
        )
        for row in mine:
            out.append(dict(row))
            if row["violated"]:
                break
    return out


def test_e3_pooled_counts_and_risk_sets_are_per_family_session() -> None:
    # Both families run the same plans (session ids repeat across families, as in E1).
    sessions, opportunities = _synthetic("e3", E3_ARMS, ("fa", "fb"), 4, 1)
    out = arms_analysis.analyze_e3(sessions, opportunities, FAST)
    pooled = next(f for f in out["rates"]["by_family"] if f["family"] == "pooled")
    assert all(c["sessions"] == 8 for c in pooled["cells"])  # 4 plans x 2 families
    assert all(p["sessions"] == 16 for p in pooled["placement"])
    fa = next(f for f in out["rates"]["by_family"] if f["family"] == "fa")
    assert all(c["sessions"] == 4 for c in fa["cells"])
    factors = (*arms_analysis.E3_FACTORS, "family")
    reference = stats.fit_hazard(
        _manual_risk_set(opportunities),
        arms_analysis.HAZARD_COVARIATES,
        factors,
        arms_analysis.TRANSFORM,
        use_risk_set=False,
    )
    estimates = {e["term"]: e for e in out["hazard"]["estimates"] if e["family"] == "pooled"}
    assert set(estimates) == set(reference.coefficients)
    for term, value in reference.coefficients.items():
        assert estimates[term]["estimate"] == pytest.approx(value, rel=1e-9, abs=1e-12), term
        assert estimates[term]["n_clusters"] == 32  # plans, each holding both families
    contrasts = next(f for f in out["contrasts"]["by_family"] if f["family"] == "pooled")
    system = next(c for c in contrasts["placement"] if c["placement"] == "system")
    assert system["arm"]["n_sessions"] == 16 and system["reference_arm"]["n_sessions"] == 16
    assert system["difference"]["n_clusters"] == 16 and system["wald"]["n_clusters"] == 16
    assert out["rates"]["exclusions_by_family"][0]["family"] == "fa"


def test_e3_recall_counts_reminders_the_probe_did_not_see() -> None:
    _, opportunities = _synthetic("e3", E3_ARMS, ("fa",), 3, 3)
    for i, row in enumerate(opportunities):
        row["reminder_on_request"] = row["placement"] == "reminder" and i % 2 == 0
    found = arms_analysis.rates(opportunities, ("placement",), 0.95)
    by_arm = {e["placement"]: e for e in found}
    reminder = [r for r in opportunities if r["placement"] == "reminder"]
    probed = [r for r in reminder if r["recalled_before"] is not None]
    assert by_arm["reminder"]["recall_before"]["reminders_on_request"] == sum(
        1 for r in probed if r["reminder_on_request"]
    )
    assert by_arm["system"]["recall_before"]["reminders_on_request"] == 0
    assert (
        "reminders_on_request"
        not in (
            arms_analysis.rates(_synthetic("e3", E3_ARMS, ("fa",), 3, 3)[1], ("placement",), 0.95)[
                0
            ]["recall_before"]
        )
    )


def test_e3_hazard_writes_separated_terms_without_a_number() -> None:
    # Synthetic: the reminder placement never violates, which separates its hazard term;
    # it gets no number, and the fit diagnostics say why.
    sessions, opportunities = _synthetic("e3", E3_ARMS, ("fa",), 6, 3)
    for row in opportunities:
        if row["placement"] == "reminder":
            row["violated"] = False
    out = arms_analysis.analyze_e3(sessions, opportunities, FAST)
    hazard = out["hazard"]
    fits = {f["family"]: f for f in hazard["fits"]}
    assert (
        fits["fa"]["separation"] is True
        and "placement[reminder]" in (fits["fa"]["separated_terms"])
    )
    terms = {e["term"]: e for e in hazard["estimates"] if e["family"] == "pooled"}
    reminder = terms["placement[reminder]"]
    assert (reminder["estimate"], reminder["lo"], reminder["hi"]) == (None, None, None)
    assert reminder["not_estimable"] == "separation"
    # No E3 hazard term carries a hypothesis, so none has a decision.
    assert reminder["hypothesis"] is None and reminder["verdict"] is None
    assert reminder["not_tested"] is None and reminder["decided_by"] is None
    assert terms["placement[system]"]["not_estimable"] is None
    json.dumps(out, allow_nan=False)


def test_e4_contrasts_carry_score_p_values_and_name_missing_ones() -> None:
    sessions, opportunities = _synthetic("e4", E4_ARMS, ("fa",), 30, 2)
    for row in opportunities:
        if row["mitigation"] == "guard":
            row["violated"] = False  # separated from none: no Wald test exists
        if row["mitigation"] in {"none", "pin"} and row["session_id"].endswith("-1"):
            row["violated"] = True
    out = arms_analysis.analyze_e4(sessions, opportunities, FAST)
    pooled = next(f for f in out["contrasts"]["by_family"] if f["family"] == "pooled")
    entries = {c["mitigation"]: c for c in pooled["mitigation"]}
    assert all(c["holm_family_size"] == 5 for c in entries.values())
    selfcheck = entries["selfcheck"]
    assert selfcheck["p_value"] is not None and selfcheck["p_value"] < 0.05
    assert selfcheck["p_value"] == selfcheck["score"]["p_value"]
    assert selfcheck["p_holm"] >= selfcheck["p_value"]
    # A guard arm without violations separates the model (no Wald test) but is tested by
    # the score test, and with this large a difference it differs from none.
    guard = entries["guard"]
    assert guard["wald"]["problem"] == "separation" and guard["p_value_missing"] is None
    assert guard["p_value"] < 0.01 and guard["p_holm"] < 0.05
    assert guard["verdict"] == "supported"
    summary = {m["name"]: m for m in out["summary"]["mitigations"]}
    assert summary["guard"]["p_holm"] == guard["p_holm"]
    assert summary["guard"]["verdict"] == "supported"
    assert summary["selfcheck"]["p_holm"] == selfcheck["p_holm"]
    assert out["contrasts"]["holm_note"] == arms_analysis.HOLM_NOTE
    json.dumps(out, allow_nan=False)


def test_e4_contrast_with_no_violation_in_either_arm_is_not_tested() -> None:
    _, opportunities = _synthetic("e4", E4_ARMS, ("fa",), 8, 5)
    for row in opportunities:
        if row["mitigation"] in {"none", "pin"}:
            row["violated"] = False
    entries = {
        e["mitigation"]: e
        for e in arms_analysis.contrasts_against(
            opportunities, "mitigation", "none", arms_analysis.MITIGATIONS, FAST
        )
    }
    pin = entries["pin"]
    assert pin["p_value"] is None and pin["p_value_missing"] == "no_events"
    assert pin["p_holm"] is None and pin["verdict"] == "not tested"
    assert pin["not_tested"] == "no_events"
    assert all(e["holm_family_size"] == 5 for e in entries.values())


def test_holm_family_is_every_preregistered_arm_even_when_one_has_no_rows() -> None:
    # D-040 fixes the E4 family as the five mitigations. With the reinject arm absent, the
    # other four are still adjusted over five tests, never over the four that ran, and the
    # absent arm has a not-tested record with its reason (D-059 item 3), not no entry.
    _, opportunities = _synthetic("e4", E4_ARMS, ("fa",), 8, 5)
    present = [o for o in opportunities if o["mitigation"] != "reinject"]
    entries = arms_analysis.contrasts_against(
        present, "mitigation", "none", arms_analysis.MITIGATIONS, FAST
    )
    assert [e["mitigation"] for e in entries] == list(arms_analysis.MITIGATIONS)
    assert all(e["holm_family_size"] == 5 for e in entries)
    absent = next(e for e in entries if e["mitigation"] == "reinject")
    assert absent["verdict"] == "not tested" and absent["not_tested"] == "arm_absent"
    assert absent["p_value"] is None and absent["p_holm"] is None
    assert absent["missing_arms"] == ["reinject"] and absent["reference_arm"]["n_opportunities"]
    raw = {e["mitigation"]: e["p_value"] for e in entries if e is not absent}
    assert None not in raw.values()
    expected = e1_holm({**raw, "reinject": None})
    for entry in entries:
        if entry is not absent:
            assert entry["p_holm"] == pytest.approx(expected[entry["mitigation"]])
    smallest = min(raw.values())
    tested = [e["p_holm"] for e in entries if e is not absent]
    assert min(tested) == pytest.approx(min(1.0, 5 * smallest))


def test_e4_pooled_costs_resample_whole_plans() -> None:
    sessions, _ = _synthetic("e4", E4_ARMS[:3], ("fa", "fb"), 5, 3)
    costs, _ = arms_analysis.e4_costs(sessions, FAST)
    for entry in costs["by_family"]:
        guard = next(a for a in entry["arms"] if a["mitigation"] == "guard")
        expected = 10 if entry["family"] == "pooled" else 5
        assert guard["sessions"] == expected
        assert guard["tokens"]["n_clusters"] == 5  # plans: both families' sessions together
        assert guard["tokens_vs_none"]["n_clusters"] == 10
        assert guard["tokens_vs_none"]["strata"] == "mitigation"


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
            "subtasks_planned_checked": 2,
            "subtasks_succeeded": 1,
        }
    )
    assert costs == {"tokens": 6.0, "latency_s": 2.0, "task_success": 0.5}


def test_task_success_counts_unreached_checked_subtasks_as_not_passed() -> None:
    # A session stopped at the tool-call cap after 13 of 20 subtasks passed the 4 checks it
    # reached, but its plan had 8: the 4 it never reached count as not passed.
    capped = {"usage": None, "subtasks_checked": 4, "subtasks_succeeded": 4}
    # Rows without the planned count cannot say what was missed: unknown, not 1.0.
    assert arms_analysis.session_costs(capped)["task_success"] is None
    planned = capped | {"subtasks_planned_checked": 8}
    assert arms_analysis.session_costs(planned)["task_success"] == 0.5
    # A session that reached no checked subtask scores 0, not missing.
    none_reached = {"subtasks_checked": 0, "subtasks_succeeded": 0, "subtasks_planned_checked": 7}
    assert arms_analysis.session_costs(none_reached)["task_success"] == 0.0
    no_checks = {"subtasks_checked": 0, "subtasks_succeeded": 0, "subtasks_planned_checked": 0}
    assert arms_analysis.session_costs(no_checks)["task_success"] is None


def test_e4_costs_count_missing_usage_and_skip_absent_reference() -> None:
    sessions, _ = _synthetic("e4", E4_ARMS[1:3], ("fa",), 3, 3)
    sessions[0]["usage"] = None
    costs, means = arms_analysis.e4_costs(sessions, FAST)
    pooled = next(f for f in costs["by_family"] if f["family"] == "pooled")
    guard = next(a for a in pooled["arms"] if a["mitigation"] == "guard")
    assert guard["tokens_missing"] == 1 and "tokens_vs_none" not in guard
    assert set(means) == {"guard", "pin"}


def test_contrasts_without_the_reference_arm_are_not_tested_with_their_reason() -> None:
    # D-059 item 3: without the reference arm nothing can be contrasted, and every
    # preregistered arm says so instead of being left out.
    _, opportunities = _synthetic("e4", E4_ARMS[1:], ("fa",), 2, 4)
    entries = arms_analysis.contrasts_against(
        opportunities, "mitigation", "none", ["guard", "pin"], FAST
    )
    assert [e["mitigation"] for e in entries] == ["guard", "pin"]
    for entry in entries:
        assert entry["verdict"] == "not tested" and entry["not_tested"] == "arm_absent"
        assert entry["missing_arms"] == ["none"] and entry["reference_arm"]["arm"] == "none"
        assert entry["arm"]["n_opportunities"] > 0 and entry["p_holm"] is None
        assert entry["holm_family_size"] == 2 and entry["decided_by"] == "score_test"


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
    assert by_tier["haiku"]["sessions"] == 3 and by_tier["sonnet"]["sessions"] == 4
    assert out["rates"]["exclusions"]["sessions_not_reaching_first_dp"] == 1
    assert out["rates"]["exclusions"]["opportunities_zero_calls"] == 0
    with pytest.raises(stats.StatsError):
        arms_analysis.analyze_e2(tiered_sessions, [], FAST)
    for name, rows in (("sessions", tiered_sessions), ("opportunities", tiered_opps)):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    assert arms_analysis.main(["e2", "--dir", str(tmp_path), "--n-boot", "20"]) == 0
    assert {p.stem for p in tmp_path.glob("*.json")} == {"rates", "summary"}


def test_code_pins_cover_every_module_the_analysis_loads() -> None:
    """D-050 item 6: the E3 and E4 preregistrations pin every module the analysis runs."""
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, halflife.arms_analysis; "
            "print(' '.join(sorted(m for m in sys.modules if m.startswith('halflife'))))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    hashes = arms_analysis.analysis_code_sha256()
    expected = {
        ("halflife/__init__.py" if m == "halflife" else m.replace(".", "/") + ".py") for m in loaded
    }
    assert set(hashes) == expected
    source = Path(arms_analysis.__file__).resolve().parent.parent
    for rel, digest in hashes.items():
        assert digest == hashlib.sha256((source / rel).read_bytes()).hexdigest()


def test_cli_prints_the_code_pins(capsys: pytest.CaptureFixture[str]) -> None:
    assert arms_analysis.main(["e3", "--code-sha256"]) == 0
    assert json.loads(capsys.readouterr().out) == arms_analysis.analysis_code_sha256()
    with pytest.raises(SystemExit):
        arms_analysis.main(["e3"])  # --dir is required unless printing the pins


# --------------------------------------------------------------------------- D-050


def test_pooled_contrasts_are_stratified_by_family_and_per_family_ones_exploratory() -> None:
    # D-050 item 2: the pooled E3 and E4 contrasts are the confirmatory tests and fit their
    # null model within each family; per-family contrasts are exploratory.
    sessions, opportunities = _synthetic("e4", E4_ARMS, ("fa", "fb"), 12, 8)
    for row in opportunities:
        if row["family"] == "fb":
            row["violated"] = row["violated"] or row["mitigation"] == "none"
    out = arms_analysis.analyze_e4(sessions, opportunities, FAST)
    for entry in out["contrasts"]["by_family"]:
        family = entry["family"]
        subset = [r for r in opportunities if family == "pooled" or r["family"] == family]
        for contrast in entry["mitigation"]:
            arm = contrast["mitigation"]
            pair = [r for r in subset if r["mitigation"] in {"none", arm}]
            expected = stats.rate_contrast(
                pair,
                "mitigation",
                "none",
                arm,
                2,
                seed=1,
                strata="family" if family == "pooled" else None,
            )
            assert contrast["p_value"] == pytest.approx(expected.score.p_value, rel=1e-9)
            assert contrast["score_strata"] == ("family" if family == "pooled" else None)
            assert contrast["exploratory"] is (family != "pooled")
            assert contrast["decided_by"] == "score_test" and contrast["alpha"] == 0.05
    summary = {m["name"]: m for m in out["summary"]["mitigations"]}
    pooled = next(f for f in out["contrasts"]["by_family"] if f["family"] == "pooled")
    for contrast in pooled["mitigation"]:
        mine = summary[contrast["mitigation"]]
        assert mine["decided_by"] == "score_test" and mine["alpha"] == 0.05
        assert mine["p_value"] == contrast["p_value"] and mine["p_holm"] == contrast["p_holm"]
        assert mine["verdict"] == contrast["verdict"]
    e3_sessions, e3_opportunities = _synthetic("e3", E3_ARMS, ("fa", "fb"), 4, 1)
    e3 = arms_analysis.analyze_e3(e3_sessions, e3_opportunities, FAST)
    for entry in e3["contrasts"]["by_family"]:
        for contrast in [*entry["placement"], *entry["phrasing"]]:
            assert contrast["exploratory"] is (entry["family"] != "pooled")
            assert contrast["score_strata"] == ("family" if entry["family"] == "pooled" else None)


def test_an_unstable_contrast_interval_decides_nothing() -> None:
    _, opportunities = _synthetic("e4", E4_ARMS, ("fa",), 8, 5)
    strict = Settings(n_boot=20, seed=7, max_failed_share=-1.0)
    plain = Settings(n_boot=20, seed=7)
    every = arms_analysis.MITIGATIONS
    before = arms_analysis.contrasts_against(opportunities, "mitigation", "none", every, plain)
    after = arms_analysis.contrasts_against(opportunities, "mitigation", "none", every, strict)
    for old, new in zip(before, after, strict=True):
        assert new["unstable"] is True and old["unstable"] is False
        assert new["p_holm"] == old["p_holm"] is not None
        assert new["verdict"] == old["verdict"] != "not tested"


def test_e2_drops_procedure_rules_in_cut_spans_and_counts_them() -> None:
    sessions, opportunities = _synthetic("e2", [{}], ("claude",), 4, 6)
    tiered_sessions = [dict(s, model="haiku", confirmatory=False) for s in sessions]
    tiered = [dict(o, model="haiku") for o in opportunities]
    ends = ["done", "turn_cap", "subtask_tool_cap", None]
    for i, row in enumerate(tiered):
        row["span_end"] = ends[i % len(ends)]
    out = arms_analysis.analyze_e2(tiered_sessions, tiered, FAST)
    cut = [o for o in tiered if o["constraint_type"] == "procedure" and o["span_end"] != "done"]
    assert cut and out["rates"]["exclusions"]["opportunities_procedure_cut"] == len(cut)
    analyzed = out["rates"]["exclusions"]["opportunities_analyzed"]
    assert analyzed == len(tiered) - len(cut)
    assert out["rates"]["model"][0]["opportunities"] == analyzed


def test_an_e3_hazard_model_that_cannot_be_fit_gives_not_estimable_records() -> None:
    # D-059 item 3 (audit N6). Synthetic: family fb never sees a compaction before a
    # decision point, so its compaction count is constant and its model cannot be fit.
    sessions, opportunities = _synthetic("e3", E3_ARMS, ("fa", "fb"), 4, 1)
    for row in opportunities:
        if row["family"] == "fb":
            row["compaction_count"] = 0
    out = arms_analysis.analyze_e3(sessions, opportunities, FAST)
    fits = {f["family"]: f for f in out["hazard"]["fits"]}
    assert fits["fb"]["fit_failed"] == "single_value" and "fit_failed" not in fits["fa"]
    fb = [e for e in out["hazard"]["estimates"] if e["family"] == "fb"]
    assert fb and all(e["not_estimable"] == "single_value" for e in fb)
    assert {e["term"] for e in fb} >= {"intercept", "compaction_count", "placement[system]"}
    json.dumps(out, allow_nan=False)
