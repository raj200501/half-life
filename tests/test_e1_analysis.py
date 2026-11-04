"""Tests for the preregistered E1 analysis on synthetic rows (labeled synthetic)."""

from __future__ import annotations

import dataclasses
import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from halflife import e1_analysis, stats

TYPES = ("prohibition", "invariant", "procedure", "resource_limit")


def synthetic(
    n_sessions: int = 60,
    seed: int = 7,
    *,
    shared_plans: bool = False,
    type_effects: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthetic sessions and opportunities from a known hazard model (not model data).

    With ``shared_plans`` both families run the same plans, so session ids repeat across
    families as they do in E1; ``type_effects`` add log odds per rule type.
    """
    rng = random.Random(seed)
    effects = type_effects or {}
    sessions, opps = [], []
    for family in ("fam-a", "fam-b"):
        for i in range(n_sessions):
            sid = f"s{i}" if shared_plans else f"{family}-s{i}"
            strategy = "full_summary" if i % 2 else "sham"
            sessions.append(
                {
                    "session_id": sid,
                    "family": family,
                    "confirmatory": True,
                    "reached_first_dp": i % 13 != 0,
                }
            )
            position = 0
            for j in range(4):
                position += rng.randint(5, 30)
                count = min(3, position // 40)
                ctype = TYPES[(i + j) % 4]
                eta = (
                    -3.0
                    + 0.5 * math.log1p(position)
                    + (0.6 * count if strategy == "full_summary" else 0.0)
                    + effects.get(ctype, 0.0)
                )
                opps.append(
                    {
                        "session_id": sid,
                        "family": family,
                        "position": position,
                        "compaction_count": count,
                        "constraint_type": ctype,
                        "strategy": strategy,
                        "violated": rng.random() < 1 / (1 + math.exp(-eta)),
                        "acted": rng.random() < 0.8,
                        "recalled_before": rng.random() < 0.6,
                        "span_calls": 0 if (i == 1 and j == 3) else 3,
                    }
                )
    return sessions, opps


SETTINGS = e1_analysis.Settings(n_boot=60, seed=3)


def test_analysis_set_exclusions() -> None:
    sessions, opps = synthetic()
    rows, exclusions = e1_analysis.analysis_set(sessions, opps)
    assert exclusions["sessions_not_reaching_first_dp"] == 10
    assert exclusions["opportunities_zero_calls"] == 2
    assert all(r["span_calls"] > 0 for r in rows)
    assert exclusions["opportunities_analyzed"] == len(rows)


def test_full_analysis_outputs_are_complete_and_json_safe(tmp_path: Path) -> None:
    sessions, opps = synthetic()
    out = e1_analysis.analyze(sessions, opps, SETTINGS)
    assert set(out) == {"hazard", "half_life", "km", "contrasts", "descriptive"}
    terms = {(e["family"], e["term"]) for e in out["hazard"]["estimates"]}
    assert ("pooled", "log1p(position)") in terms and ("fam-a", "compaction_count") in terms
    assert any(t == "family[fam-b]" for _, t in terms)
    for est in out["hazard"]["estimates"]:
        assert est["lo"] <= est["estimate"] <= est["hi"]
    assert [e["family"] for e in out["half_life"]["estimates"]] == ["fam-a", "fam-b"]
    assert out["km"]["descriptive"] is True and len(out["km"]["curves"]) == 2
    assert {e["family"] for e in out["contrasts"]["h2a"]} == {"fam-a", "fam-b", "pooled"}
    pairs = out["contrasts"]["h3_pairs"]
    assert len(pairs) == 6 and all(
        0 <= p["p_holm"] <= 1 and p["p_holm"] >= p["p_value"] for p in pairs
    )
    assert out["descriptive"]["by_family"][0]["acted"] > 0
    for data in out.values():
        json.dumps(data, allow_nan=False)
    sdir = tmp_path / "e1"
    sdir.mkdir()
    (sdir / "sessions.jsonl").write_text("\n".join(json.dumps(s) for s in sessions) + "\n")
    (sdir / "opportunities.jsonl").write_text("\n".join(json.dumps(o) for o in opps) + "\n")
    assert e1_analysis.main(["--dir", str(sdir), "--n-boot", "40", "--seed", "1"]) == 0
    assert (sdir / "half_life.json").is_file() and (sdir / "contrasts.json").is_file()


def test_analysis_code_hashes_cover_every_module_the_analysis_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import hashlib
    import subprocess
    import sys

    loaded = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys, halflife.e1_analysis; "
            "print(' '.join(sorted(m for m in sys.modules if m.startswith('halflife'))))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    hashes = e1_analysis.analysis_code_sha256()
    expected = {
        ("halflife/__init__.py" if m == "halflife" else m.replace(".", "/") + ".py") for m in loaded
    }
    assert set(hashes) == expected
    source = Path(e1_analysis.__file__).resolve().parent.parent
    for rel, digest in hashes.items():
        assert digest == hashlib.sha256((source / rel).read_bytes()).hexdigest()
    assert e1_analysis.main(["--code-sha256"]) == 0
    assert json.loads(capsys.readouterr().out) == hashes


def test_holm_and_empty_input() -> None:
    adjusted = e1_analysis.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    with pytest.raises(stats.StatsError):
        e1_analysis.analyze([], [], SETTINGS)


def test_holm_keeps_a_missing_p_value_in_the_family() -> None:
    # Three tests, one not computed: it counts as p = 1, so the others are adjusted over
    # m = 3 (0.01 * 3 and max(0.03, 0.03 * 2)), and its own adjusted value stays missing.
    adjusted = e1_analysis.holm({"a": 0.01, "b": None, "c": 0.03})
    assert list(adjusted) == ["a", "b", "c"]
    assert adjusted["b"] is None
    assert adjusted["a"] == pytest.approx(0.03) and adjusted["c"] == pytest.approx(0.06)
    assert e1_analysis.holm({"a": None}) == {"a": None}
    assert e1_analysis.holm({}) == {}


def reached_pairs(sessions: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (s["family"], s["session_id"])
        for s in sessions
        if s["confirmatory"] is True and s["reached_first_dp"] is True and not s.get("probe_leaks")
    }


def test_analysis_set_and_exclusions_are_per_family_session() -> None:
    # Both families run the same plans. Family fam-b's session of plan s2 did not reach its
    # first decision point and its session of s4 is not confirmatory, while fam-a's sessions
    # of both plans count; fam-b's rows of s2 and s4 must not ride on fam-a's sessions.
    sessions, opps = synthetic(shared_plans=True)
    for row in sessions:
        if row["family"] == "fam-b" and row["session_id"] == "s2":
            row["reached_first_dp"] = False
        if row["family"] == "fam-b" and row["session_id"] == "s4":
            row["confirmatory"] = False
    rows, exclusions = e1_analysis.analysis_set(sessions, opps)
    keep = reached_pairs(sessions)
    expected = [o for o in opps if (o["family"], o["session_id"]) in keep and o["span_calls"] > 0]
    assert [(r["family"], r["session_id"], r["position"]) for r in rows] == [
        (o["family"], o["session_id"], o["position"]) for o in expected
    ]
    assert not any(r["family"] == "fam-b" and r["session_id"] in {"s2", "s4"} for r in rows)
    # i % 13 == 0 misses the first decision point: 5 sessions per family, plus fam-b's s2.
    assert exclusions["sessions_not_reaching_first_dp"] == 11
    assert exclusions["sessions_confirmatory"] == 119 and exclusions["sessions_total"] == 120
    assert exclusions["sessions_analyzed"] == 108 and exclusions["sessions_probe_leak"] == 0
    assert exclusions["opportunities_analyzed"] == len(rows)
    by_family = {e["family"]: e for e in e1_analysis.exclusions_by_family(sessions, opps)}
    assert by_family["fam-a"] == {
        "family": "fam-a",
        "sessions_total": 60,
        "sessions_confirmatory": 60,
        "sessions_probe_leak": 0,
        "sessions_not_reaching_first_dp": 5,
        "sessions_analyzed": 55,
        "opportunities_zero_calls": 1,
        "opportunities_analyzed": 219,
    }
    assert by_family["fam-b"]["sessions_not_reaching_first_dp"] == 6
    assert by_family["fam-b"]["sessions_confirmatory"] == 59
    assert by_family["fam-b"]["sessions_analyzed"] == 53
    assert sum(e["opportunities_analyzed"] for e in by_family.values()) == len(rows)
    out = e1_analysis.analyze(sessions, opps, e1_analysis.Settings(n_boot=10, seed=1))
    for data in out.values():
        assert data["exclusions"] == exclusions
        assert [e["family"] for e in data["exclusions_by_family"]] == ["fam-a", "fam-b"]
    duplicated = [*sessions, dict(sessions[0])]
    with pytest.raises(stats.StatsError, match="duplicate session"):
        e1_analysis.analysis_set(duplicated, opps)


def test_probe_leak_sessions_are_excluded_and_counted() -> None:
    sessions, opps = synthetic(shared_plans=True)
    leaked = next(s for s in sessions if s["family"] == "fam-a" and s["session_id"] == "s3")
    leaked["probe_leaks"] = [{"call": 12, "kind": "probe_question"}]
    rows, exclusions = e1_analysis.analysis_set(sessions, opps)
    assert exclusions["sessions_probe_leak"] == 1
    assert not any(r["family"] == "fam-a" and r["session_id"] == "s3" for r in rows)
    assert any(r["family"] == "fam-b" and r["session_id"] == "s3" for r in rows)
    assert exclusions["sessions_not_reaching_first_dp"] == 10


def manual_risk_set(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def test_pooled_hazard_model_keeps_each_family_sessions_risk_set() -> None:
    sessions, opps = synthetic(shared_plans=True)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    manual = manual_risk_set(rows)
    reference = stats.fit_hazard(
        manual,
        e1_analysis.HAZARD_COVARIATES,
        (*e1_analysis.HAZARD_FACTORS, "family"),
        e1_analysis.TRANSFORM,
        use_risk_set=False,
    )
    out = e1_analysis.hazard(rows, e1_analysis.Settings(n_boot=10, seed=1))
    pooled = {e["term"]: e for e in out["estimates"] if e["family"] == "pooled"}
    assert set(pooled) == set(reference.coefficients)
    for term, value in reference.coefficients.items():
        assert pooled[term]["estimate"] == pytest.approx(value, rel=1e-9, abs=1e-12), term
        assert pooled[term]["cluster"] == "session_id" and pooled[term]["n_clusters"] == 55
    per_family = [e for e in out["estimates"] if e["family"] == "fam-a"]
    assert all(e["n_clusters"] == 55 for e in per_family)
    assert out["model"]["session_key"] == ["family", "session_id"]
    assert out["model"]["cluster"] == "session_id"


def test_a_family_without_violations_has_no_hazard_estimate_and_no_verdict() -> None:
    # Synthetic: family fam-b never violates. Its fit is held only by the ridge, so every
    # term is written as not estimable (no number) and H1 and H2b are not tested; the
    # pooled fit says its slopes come from fam-a alone.
    sessions, opps = synthetic(shared_plans=True)
    for row in opps:
        if row["family"] == "fam-b":
            row["violated"] = False
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    out = e1_analysis.hazard(rows, e1_analysis.Settings(n_boot=40, seed=1))
    fits = {f["family"]: f for f in out["fits"]}
    assert fits["fam-b"]["n_events"] == 0 and fits["fam-b"]["separation"] is True
    assert "all_zero_outcomes" in fits["fam-b"]["warnings"]
    assert fits["fam-a"]["n_events"] > 0 and fits["fam-a"]["n_rows"] > fits["fam-a"]["n_events"]
    assert fits["pooled"]["events_by_family"] == {
        "fam-a": fits["fam-a"]["n_events"],
        "fam-b": 0,
    }
    assert fits["pooled"]["families_without_violations"] == ["fam-b"]
    fam_b = {e["term"]: e for e in out["estimates"] if e["family"] == "fam-b"}
    for entry in fam_b.values():
        assert (entry["estimate"], entry["lo"], entry["hi"]) == (None, None, None)
        assert entry["not_estimable"] == "no_violations"
        assert entry["not_tested"] == "not_estimable"
    assert fam_b["log1p(position)"]["hypothesis"] == "H1"
    assert fam_b["compaction_count"]["hypothesis"] == "H2b"
    assert {fam_b[t]["verdict"] for t in ("log1p(position)", "compaction_count")} == {"not tested"}
    pooled = {e["term"]: e for e in out["estimates"] if e["family"] == "pooled"}
    assert pooled["family[fam-b]"]["not_estimable"] == "separation"
    assert pooled["log1p(position)"]["not_estimable"] is None
    assert pooled["log1p(position)"]["verdict"] in {"supported", "not supported"}
    fam_a = {e["term"]: e for e in out["estimates"] if e["family"] == "fam-a"}
    assert fam_a["intercept"]["hypothesis"] is None and fam_a["intercept"]["verdict"] is None
    json.dumps(out, allow_nan=False)


def test_more_than_a_tenth_of_failed_draws_leaves_a_hypothesis_not_tested() -> None:
    # Section 5: n_failed / n_boot above 0.1 is unstable and its hypothesis not tested;
    # exactly 0.1 is not above it.
    sessions, opps = synthetic(shared_plans=True)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    fam = [dict(r) for r in rows if r["family"] == "fam-a"]
    found = stats.bootstrap_coefficients(
        fam,
        20,
        1,
        covariates=e1_analysis.HAZARD_COVARIATES,
        categorical=e1_analysis.HAZARD_FACTORS,
        position_transform=e1_analysis.TRANSFORM,
    )
    intervals = dict(found.intervals)
    h1, h2b = e1_analysis.POSITION_TERM, "compaction_count"
    intervals[h1] = dataclasses.replace(intervals[h1], n_boot=2000, n_failed=300)
    intervals[h2b] = dataclasses.replace(intervals[h2b], n_boot=2000, n_failed=200)
    records = e1_analysis.term_records(
        stats.HazardCoefficients(intervals, found.fit), e1_analysis.Settings()
    )
    assert records[h1]["unstable"] is True and records[h1]["failed_share"] == 0.15
    assert records[h1]["not_tested"] == "unstable" and records[h1]["verdict"] == "not tested"
    assert records[h2b]["unstable"] is False and records[h2b]["not_tested"] is None
    assert records[h2b]["verdict"] in {"supported", "not supported"}
    # Too few violations in the fit leaves both slopes untested, whatever the interval.
    few = e1_analysis.term_records(found, e1_analysis.Settings(min_events=10**6))
    assert {few[t]["not_tested"] for t in (h1, h2b)} == {"few_violations"}
    half = e1_analysis.half_lives(rows, e1_analysis.Settings(n_boot=20, seed=1))
    for entry in half["estimates"]:
        assert entry["unstable"] is False and entry["first_position"] == min(
            r["position"] for r in rows if r["family"] == entry["family"]
        )


def test_recall_by_compaction_counts_what_the_probe_saw() -> None:
    sessions, opps = synthetic()
    for row in opps:
        # Synthetic: every decision point starts with a compaction that ran after its probe.
        row["compactions_before_probe"] = max(0, int(row["compaction_count"]) - 1)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    out = e1_analysis.descriptive(rows, e1_analysis.Settings(n_boot=10, seed=1))
    for entry in out["by_family"]:
        cells = entry["recall_by_compaction"]
        assert {c["count_field"] for c in cells} == {"compactions_before_probe"}
        mine = [r for r in rows if r["family"] == entry["family"]]
        expected = {int(r["compactions_before_probe"]) for r in mine}
        assert {c["compaction_count"] for c in cells} == expected


def test_pooled_bootstrap_draws_hold_both_families_of_each_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions, opps = synthetic(n_sessions=20, shared_plans=True)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    samples: list[list[tuple[str, str]]] = []
    fit = stats._hazard_statistic_fit

    def spy(sample: list[dict[str, Any]], *args: Any) -> stats.HazardFit:
        samples.append([(r["family"], r["session_id"]) for r in sample])
        return fit(sample, *args)

    monkeypatch.setattr(stats, "_hazard_statistic_fit", spy)
    e1_analysis.hazard(rows, e1_analysis.Settings(n_boot=6, seed=2))
    # Each of the three bootstraps (two families, then pooled) fits the full data once and
    # then one resample per draw; resampled rows carry a "#slot" label on session_id.
    assert len(samples) == 3 * 7
    draws = [s for s in samples if all("#" in sid for _, sid in s)]
    pooled = [s for s in draws if len({family for family, _ in s}) == 2]
    assert len(draws) == 18 and len(pooled) == 6
    for sample in pooled:
        families: dict[str, set[str]] = {}
        for family, sid in sample:
            families.setdefault(sid, set()).add(family)
        assert all(found == {"fam-a", "fam-b"} for found in families.values())


def test_h3_pairs_report_plan_clustered_score_p_values() -> None:
    effects = {"prohibition": -1.5, "invariant": 1.5}
    sessions, opps = synthetic(n_sessions=80, shared_plans=True, type_effects=effects)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    out = e1_analysis.contrasts(rows, e1_analysis.Settings(n_boot=20, seed=1))
    pairs = {p["pair"]: p for p in out["h3_pairs"]}
    strong = pairs["invariant vs prohibition"]
    assert strong["p_value"] is not None and strong["p_value"] < 0.05
    assert strong["p_value_missing"] is None and strong["holm_family_size"] == 6
    assert strong["verdict"] == "supported" and strong["not_tested"] is None
    for pair in pairs.values():
        assert pair["p_value"] == pair["score"]["p_value"]
        assert pair["p_value_test"] == e1_analysis.P_VALUE_TEST
        assert pair["score"]["n_clusters"] == 73 and pair["wald"]["n_clusters"] == 73  # plans
        assert pair["p_holm"] >= pair["p_value"]
        expected = "supported" if pair["p_holm"] < 0.05 else "not supported"
        assert pair["verdict"] == expected
    assert out["holm_note"] == e1_analysis.HOLM_NOTE


def test_h3_pairs_with_a_type_without_violations_are_still_tested() -> None:
    # Synthetic: resource limits and procedures never break. A pair with one such type
    # separates the logistic model, so its Wald test is withheld, but the score test at
    # the null fit is valid and decides; the pair of the two quiet types has no event and
    # cannot be tested, and it stays in the Holm family as p = 1.
    sessions, opps = synthetic(n_sessions=80, shared_plans=True)
    for row in opps:
        if row["constraint_type"] in {"resource_limit", "procedure"}:
            row["violated"] = False
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    out = e1_analysis.contrasts(rows, e1_analysis.Settings(n_boot=20, seed=1))
    pairs = {p["pair"]: p for p in out["h3_pairs"]}
    quiet = pairs["procedure vs resource_limit"]
    assert quiet["p_value"] is None and quiet["p_holm"] is None
    assert quiet["p_value_missing"] == "no_events" and quiet["verdict"] == "not tested"
    assert quiet["not_tested"] == "no_events"
    separated = pairs["prohibition vs resource_limit"]
    assert separated["wald"]["problem"] == "separation" and separated["separation"] is True
    assert separated["p_value"] is not None and separated["p_value"] < 1e-4
    assert separated["verdict"] == "supported"
    computed = sorted(p["p_value"] for p in pairs.values() if p["p_value"] is not None)
    assert len(computed) == 5
    expected, running = [], 0.0
    for rank, p in enumerate(computed):
        running = max(running, min(1.0, (6 - rank) * p))
        expected.append(running)
    got = sorted(p["p_holm"] for p in pairs.values() if p["p_holm"] is not None)
    assert got == pytest.approx(expected)
    json.dumps(out, allow_nan=False)


def test_h3_holm_family_is_every_pair_of_rule_types_even_when_a_type_has_no_rows() -> None:
    # Synthetic: no resource-limit decision point is in the rows. Its three pairs have no
    # entry, but they stay in the preregistered family of all six pairs as p = 1, so the
    # three tested pairs are adjusted over six, not three.
    sessions, opps = synthetic(n_sessions=80, shared_plans=True)
    opps = [o for o in opps if o["constraint_type"] != "resource_limit"]
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    out = e1_analysis.contrasts(rows, e1_analysis.Settings(n_boot=20, seed=1))
    pairs = {p["pair"]: p for p in out["h3_pairs"]}
    assert sorted(pairs) == [
        "invariant vs procedure",
        "invariant vs prohibition",
        "procedure vs prohibition",
    ]
    assert all(p["holm_family_size"] == 6 for p in pairs.values())
    computed = sorted((p["p_value"], name) for name, p in pairs.items())
    running = 0.0
    for rank, (p, name) in enumerate(computed):
        running = max(running, min(1.0, (6 - rank) * p))
        assert pairs[name]["p_holm"] == pytest.approx(running)


def test_unstable_contrasts_are_not_tested_and_count_as_p_one_in_holm() -> None:
    sessions, opps = synthetic(n_sessions=40, shared_plans=True)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    settings = e1_analysis.Settings(n_boot=20, seed=1)
    out = e1_analysis.contrasts(rows, settings)
    # A strict threshold makes any failed draw unstable; with none failing, nothing moves.
    strict = e1_analysis.contrasts(rows, dataclasses.replace(settings, max_failed_share=-1.0))
    for pair in strict["h3_pairs"]:
        assert pair["unstable"] is True and pair["verdict"] == "not tested"
        assert pair["not_tested"] == "unstable" and pair["p_holm"] is None
        assert pair["p_value"] is not None  # the test exists; it is just not used
    for entry in strict["h2a"]:
        assert entry["verdict"] == "not tested" and entry["not_tested"] == "unstable"
    for entry in out["h2a"]:
        assert entry["verdict"] in {"supported", "not supported"}
        assert entry["hypothesis"] == "H2a" and entry["unstable"] is False
