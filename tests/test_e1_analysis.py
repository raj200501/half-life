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
    slope: float = 0.5,
    count_noise: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthetic sessions and opportunities from a known hazard model (not model data).

    With ``shared_plans`` both families run the same plans, so session ids repeat across
    families as they do in E1; ``type_effects`` add log odds per rule type, and ``slope``
    multiplies ``log1p(position)``. The compaction count steps up every 40 tool calls; with
    ``count_noise`` its steps fall at session-specific times (a separate random stream, so
    the other draws do not move), as randomized forced compactions make them in E1.
    """
    rng = random.Random(seed)
    jitter = random.Random(seed + 1)
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
            offset = jitter.randint(0, 39) if count_noise else 0
            for j in range(4):
                position += rng.randint(5, 30)
                count = min(3, (position + offset) // 40)
                ctype = TYPES[(i + j) % 4]
                eta = (
                    -3.0
                    + slope * math.log1p(position)
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
                        "span_end": "done",
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
    for est in [*out["hazard"]["estimates"], *out["hazard"]["h1_estimates"]]:
        # Every term has an interval or the reason it has none (a narrow position band
        # without violations in a family separates its term).
        if est["not_estimable"] is None:
            assert est["lo"] <= est["estimate"] <= est["hi"]
        else:
            assert est["not_estimable"] in stats.NOT_ESTIMABLE and est["estimate"] is None
    for family in ("fam-a", "fam-b", "pooled"):
        for term in e1_analysis.MODELS:
            assert hypothesis_entry(out["hazard"], family, term)["not_estimable"] is None
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
        "opportunities_procedure_cut": 0,
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
    manual = e1_analysis.with_model_fields(manual_risk_set(rows))
    reference = stats.fit_hazard(
        manual,
        e1_analysis.HAZARD_COVARIATES,
        (*e1_analysis.H2B_FACTORS, "family"),
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
    for key in ("estimates", "h1_estimates"):
        for entry in (e for e in out[key] if e["family"] == "fam-b"):
            assert (entry["estimate"], entry["lo"], entry["hi"]) == (None, None, None)
            assert entry["not_estimable"] == "no_violations"
    main = {e["term"]: e for e in out["estimates"] if e["family"] == "fam-b"}
    # D-059 item 1: H1 lives in its own model; the main model's position term is a
    # descriptive term of H2b's model only.
    assert main["log1p(position)"]["hypothesis"] is None
    assert main["compaction_count"]["hypothesis"] == "H2b"
    assert hypothesis_entry(out, "fam-b", "log1p(position)")["hypothesis"] == "H1"
    for term in ("log1p(position)", "compaction_count"):
        # Without a violation the score test has nothing to test, and says so.
        entry = hypothesis_entry(out, "fam-b", term)
        assert entry["verdict"] == "not tested" and entry["not_tested"] == "no_events"
        assert entry["p_value"] is None and entry["p_value_missing"] == "no_events"
    pooled_main = {e["term"]: e for e in out["estimates"] if e["family"] == "pooled"}
    assert pooled_main["family[fam-b]"]["not_estimable"] == "separation"
    assert pooled_main["log1p(position)"]["not_estimable"] is None
    for term in ("log1p(position)", "compaction_count"):
        # Pooled tests are stratified by family (D-050 item 2): fam-b adds nothing, so the
        # pooled score test equals fam-a's, instead of resting on a separated family term.
        pooled = hypothesis_entry(out, "pooled", term)
        assert pooled["verdict"] in {"supported", "not supported"}
        fam_a = hypothesis_entry(out, "fam-a", term)
        assert pooled["p_value"] == pytest.approx(fam_a["p_value"], rel=1e-9)
        assert pooled["score"]["strata_skipped"] == ["fam-b"]
        assert pooled["score_model"]["strata"] == "family"
        assert pooled["score"]["p_value"] == pooled["p_value"]
    intercept = next(
        e for e in out["estimates"] if (e["family"], e["term"]) == ("fam-a", "intercept")
    )
    assert intercept["hypothesis"] is None and intercept["verdict"] is None
    assert intercept["decided_by"] is None and intercept["not_tested"] is None
    json.dumps(out, allow_nan=False)


def hypothesis_entry(out: Mapping[str, Any], family: str, term: str) -> dict[str, Any]:
    """The record that carries a hypothesis: H1 in H1's model, H2b in the main model."""
    key = "h1_estimates" if term == e1_analysis.POSITION_TERM else "estimates"
    found: dict[str, Any] = next(e for e in out[key] if e["family"] == family and e["term"] == term)
    return found


def _fam_a_coefficients(rows: list[dict[str, Any]]) -> stats.HazardCoefficients:
    fam = [dict(r) for r in rows if r["family"] == "fam-a"]
    return stats.bootstrap_coefficients(
        fam,
        20,
        1,
        covariates=e1_analysis.HAZARD_COVARIATES,
        categorical=e1_analysis.HAZARD_FACTORS,
        position_transform=e1_analysis.TRANSFORM,
    )


def test_a_failed_share_above_a_tenth_labels_an_interval_unstable_only() -> None:
    # Section 5 and D-050 item 1: n_failed / n_boot above 0.1 is unstable (exactly 0.1 is
    # not above it); the label is descriptive and the score test still decides.
    sessions, opps = synthetic(shared_plans=True)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    found = _fam_a_coefficients(rows)
    tests = e1_analysis.hazard_tests([r for r in rows if r["family"] == "fam-a"], "fam-a")
    intervals = dict(found.intervals)
    h1, h2b = e1_analysis.POSITION_TERM, "compaction_count"
    intervals[h1] = dataclasses.replace(intervals[h1], n_boot=2000, n_failed=300)
    intervals[h2b] = dataclasses.replace(intervals[h2b], n_boot=2000, n_failed=200)
    records = e1_analysis.term_records(
        stats.HazardCoefficients(intervals, found.fit), e1_analysis.Settings(), tests=tests
    )
    assert records[h1]["unstable"] is True and records[h1]["failed_share"] == 0.15
    assert records[h2b]["unstable"] is False
    for term in (h1, h2b):
        record = records[term]
        assert record["not_tested"] is None and record["decided_by"] == "score_test"
        p_value = tests[term].test.p_value
        assert p_value is not None and record["p_value"] == pytest.approx(p_value)
        expected = "supported" if p_value < 0.05 else "not supported"
        assert record["verdict"] == expected and record["alpha"] == 0.05
    # Too few violations in the fit leaves both slopes untested; the p-value stays on record.
    few = e1_analysis.term_records(found, e1_analysis.Settings(min_events=10**6), tests=tests)
    for term in (h1, h2b):
        assert few[term]["not_tested"] == "few_violations" and few[term]["verdict"] == "not tested"
        assert few[term]["p_value"] == records[term]["p_value"]
    with pytest.raises(stats.StatsError, match="no score test"):
        e1_analysis.term_records(found, e1_analysis.Settings())
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
    # Each of the six bootstraps (H2b's and H1's model, for two families and pooled) fits
    # the full data once and then one resample per draw; resampled rows carry a "#slot"
    # label on session_id.
    assert len(samples) == 6 * 7
    draws = [s for s in samples if all("#" in sid for _, sid in s)]
    pooled = [s for s in draws if len({family for family, _ in s}) == 2]
    assert len(draws) == 36 and len(pooled) == 12
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
    # Synthetic: no resource-limit decision point is in the rows. Its three pairs stay in
    # the preregistered family of all six pairs as p = 1, so the three tested pairs are
    # adjusted over six, not three, and each untested pair has a record saying why (D-059
    # item 3), not no entry.
    sessions, opps = synthetic(n_sessions=80, shared_plans=True)
    opps = [o for o in opps if o["constraint_type"] != "resource_limit"]
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    out = e1_analysis.contrasts(rows, e1_analysis.Settings(n_boot=20, seed=1))
    pairs = {p["pair"]: p for p in out["h3_pairs"]}
    tested = ["invariant vs procedure", "invariant vs prohibition", "procedure vs prohibition"]
    absent = [
        "invariant vs resource_limit",
        "procedure vs resource_limit",
        "prohibition vs resource_limit",
    ]
    assert sorted(pairs) == sorted(tested + absent)
    assert all(p["holm_family_size"] == 6 for p in pairs.values())
    for name in absent:
        pair = pairs[name]
        assert pair["verdict"] == "not tested" and pair["not_tested"] == "arm_absent"
        assert pair["p_value"] is None and pair["p_holm"] is None
        assert pair["missing_arms"] == ["resource_limit"] and pair["arm"]["n_opportunities"] == 0
        assert pair["difference"] is None and pair["decided_by"] == "score_test"
    computed = sorted((pairs[name]["p_value"], name) for name in tested)
    running = 0.0
    for rank, (p, name) in enumerate(computed):
        running = max(running, min(1.0, (6 - rank) * p))
        assert pairs[name]["p_holm"] == pytest.approx(running)
    assert e1_analysis.ARM_ABSENT in e1_analysis.NOT_TESTED_REASONS


def test_an_unstable_interval_is_labeled_but_decides_nothing() -> None:
    # D-050 item 1: more than max_failed_share failed draws labels an interval unstable,
    # and the score test still decides (before D-050 it left the hypothesis not tested).
    sessions, opps = synthetic(n_sessions=40, shared_plans=True)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    settings = e1_analysis.Settings(n_boot=20, seed=1)
    out = e1_analysis.contrasts(rows, settings)
    # A strict threshold makes every interval unstable; no decision moves.
    strict = e1_analysis.contrasts(rows, dataclasses.replace(settings, max_failed_share=-1.0))
    for before, after in zip(out["h3_pairs"], strict["h3_pairs"], strict=True):
        assert after["unstable"] is True and before["unstable"] is False
        assert after["p_holm"] == before["p_holm"] and after["p_holm"] is not None
        assert after["verdict"] == before["verdict"] != "not tested"
        assert after["not_tested"] is None and after["decided_by"] == "score_test"
    for before, after in zip(out["h2a"], strict["h2a"], strict=True):
        assert after["unstable"] is True and after["verdict"] == before["verdict"]
        assert after["hypothesis"] == "H2a" and after["verdict"] in {"supported", "not supported"}
    hazard = e1_analysis.hazard(rows, dataclasses.replace(settings, max_failed_share=-1.0))
    for entry in hazard["estimates"]:
        if entry["hypothesis"] is not None and entry["not_estimable"] is None:
            assert entry["unstable"] is True and entry["not_tested"] is None
            assert entry["verdict"] in {"supported", "not supported"}


# --------------------------------------------------------------------------- D-050


def test_h1_and_h2b_are_decided_by_the_score_test_per_family_and_pooled() -> None:
    # Synthetic: the hazard rises with log1p(position) (slope 1.5 per unit), and compactions
    # fall at session-specific times.
    sessions, opps = synthetic(n_sessions=80, shared_plans=True, slope=1.5, count_noise=True)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    settings = e1_analysis.Settings(n_boot=10, seed=1)
    out = e1_analysis.hazard(rows, settings)
    assert out["decided_by"] == "score_test" and out["alpha"] == settings.alpha
    for family in ("fam-a", "fam-b", "pooled"):
        subset = [r for r in rows if family == "pooled" or r["family"] == family]
        cells = e1_analysis.with_model_fields(subset)
        for term, model in e1_analysis.MODELS.items():
            entry = hypothesis_entry(out, family, term)
            covariates, _ = e1_analysis.model_covariates(model, stats.risk_set(cells))
            expected = stats.hazard_score_test(
                cells,
                model.covariate,
                covariates,
                model.factors,
                e1_analysis.TRANSFORM,
                strata="family" if family == "pooled" else None,
            )
            assert entry["p_value"] == pytest.approx(expected.test.p_value, rel=1e-9)
            assert entry["decided_by"] == "score_test" and entry["alpha"] == 0.05
            assert entry["n_events"] == expected.n_events >= settings.min_events
            assert entry["p_value_test"] == e1_analysis.HAZARD_TEST
            p_value = entry["p_value"]
            assert entry["verdict"] == ("supported" if p_value < 0.05 else "not supported")
            direction = "positive" if (expected.test.z or 0.0) > 0 else "negative"
            assert entry["direction"] == direction
            # The descriptive interval sits beside the decision with its flags.
            assert {"estimate", "lo", "hi", "unstable", "not_estimable"} <= set(entry)
    pooled_h1 = hypothesis_entry(out, "pooled", "log1p(position)")
    assert pooled_h1["verdict"] == "supported" and pooled_h1["direction"] == "positive"
    # Both models are recorded for the report (D-059 item 1, D-060 items 1 and 2): H1 adjusts
    # for compaction with the cell and a count slope, H2b for position with log1p(position)
    # and the position band.
    spec = out["h1_model"]
    assert spec["covariates"] == ["position", "compaction_count"] and spec["hypothesis"] == "H1"
    assert spec["factors"] == ["constraint_type", "compaction_cell"]
    assert spec["compaction_cell"]["levels"] == ["0", "1", "2+"] and "note" in spec
    main = out["model"]
    assert main["hypothesis"] == "H2b" and main["covariates"] == ["position", "compaction_count"]
    assert main["factors"] == ["constraint_type", "strategy", "position_band"]
    assert main["position_band"]["cuts"] == [10, 20, 30, 40, 50, 60, 70]
    assert main["position_band"]["levels"] == [
        "[0,10)",
        "[10,20)",
        "[20,30)",
        "[30,40)",
        "[40,50)",
        "[50,60)",
        "[60,70)",
        "[70,150]",
    ]
    h1_terms = {e["term"] for e in out["h1_estimates"] if e["family"] == "fam-a"}
    assert "strategy[sham]" not in h1_terms
    assert any(t.startswith("compaction_cell[") for t in h1_terms)
    main_terms = {e["term"] for e in out["estimates"] if e["family"] == "fam-a"}
    assert any(t.startswith("position_band[") for t in main_terms)
    assert {f["family"] for f in out["h1_fits"]} == {"fam-a", "fam-b", "pooled"}
    # D-060 item 3: the confirmatory hazard tests are per family; pooled is exploratory.
    for family in ("fam-a", "fam-b", "pooled"):
        for term in e1_analysis.MODELS:
            assert hypothesis_entry(out, family, term)["exploratory"] is (family == "pooled")
    assert "exploratory" in out["verdict_note"] and "per family" in out["verdict_note"]
    assert out["verdict_note"] == e1_analysis.VERDICT_NOTE and "score test" in out["verdict_note"]


def _separated_count_rows(seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthetic: violations only at decision points after a compaction, whose count is
    0 or 1 independently of position, so every violation sits at the largest count."""
    rng = random.Random(seed)
    sessions, opps = [], []
    for i in range(60):
        sessions.append(
            {"session_id": f"s{i}", "family": "fam-a", "confirmatory": True}
            | {"reached_first_dp": True}
        )
        for j in range(4):
            count = int(rng.random() < 0.5)
            opps.append(
                {
                    "session_id": f"s{i}",
                    "family": "fam-a",
                    "position": 5 + 12 * j + rng.randint(0, 9),
                    "compaction_count": count,
                    "constraint_type": TYPES[(i + j) % 4],
                    "strategy": "full_summary" if i % 2 else "sham",
                    "violated": count == 1 and rng.random() < 0.4,
                    "span_calls": 2,
                }
            )
    return sessions, opps


def test_a_term_without_a_finite_estimate_is_still_decided_by_its_score_test() -> None:
    # The compaction term is separated in the full model (no finite estimate, so its
    # interval has no number), but its score test at the null fit is valid and decides.
    sessions, opps = _separated_count_rows(5)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    out = e1_analysis.hazard(rows, e1_analysis.Settings(n_boot=10, seed=1))
    entry = next(
        e for e in out["estimates"] if (e["family"], e["term"]) == ("fam-a", "compaction_count")
    )
    assert entry["not_estimable"] == "separation" and entry["estimate"] is None
    assert entry["p_value"] is not None and entry["p_value"] < 1e-6
    assert entry["verdict"] == "supported" and entry["not_tested"] is None
    assert entry["direction"] == "positive"


def test_h2a_and_h3_are_decided_by_family_stratified_score_tests() -> None:
    sessions, opps = synthetic(n_sessions=80, shared_plans=True)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    settings = e1_analysis.Settings(n_boot=10, seed=1)
    out = e1_analysis.contrasts(rows, settings)
    after = [r for r in rows if r["compaction_count"] >= 1]
    for entry in out["h2a"]:
        family = entry["family"]
        subset = [r for r in after if family == "pooled" or r["family"] == family]
        expected = stats.rate_contrast(
            subset,
            "strategy",
            "sham",
            "full_summary",
            2,
            seed=1,
            strata="family" if family == "pooled" else None,
        )
        assert entry["p_value"] == pytest.approx(expected.score.p_value, rel=1e-9)
        assert entry["score_strata"] == ("family" if family == "pooled" else None)
        assert entry["decided_by"] == "score_test" and entry["alpha"] == 0.05
        # No adjustment across families: each is its own hypothesis.
        assert "p_holm" not in entry
        assert entry["verdict"] == ("supported" if entry["p_value"] < 0.05 else "not supported")
    pooled = next(e for e in out["h2a"] if e["family"] == "pooled")
    assert pooled["verdict"] == "supported"  # the generator's full-summary effect is 0.6
    for pair in out["h3_pairs"]:
        a, b = pair["pair"].split(" vs ")
        subset = [r for r in rows if r["constraint_type"] in {a, b}]
        expected = stats.rate_contrast(subset, "constraint_type", a, b, 2, seed=1, strata="family")
        assert pair["p_value"] == pytest.approx(expected.score.p_value, rel=1e-9)
        assert pair["score_strata"] == "family" and pair["decided_by"] == "score_test"
        assert pair["alpha"] == 0.05


def test_the_half_life_is_marginal_over_compaction() -> None:
    # D-050 item 4: per family, a hazard model with log1p(position) only, searched from the
    # family's first decision point; nothing is held fixed. Synthetic: the compaction count
    # rises with position, as token-budget compactions make it.
    sessions, opps = synthetic(n_sessions=60, shared_plans=True)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    settings = e1_analysis.Settings(n_boot=30, seed=2)
    out = e1_analysis.half_lives(rows, settings)
    assert out["fixed"] == {} and out["covariates"] == ["position"]
    assert out["position_transform"] == "log1p" and out["max_position"] == 150
    for entry in out["estimates"]:
        subset = [dict(r) for r in rows if r["family"] == entry["family"]]
        expected = stats.bootstrap_half_life(
            subset,
            150,
            {},
            30,
            2,
            covariates=("position",),
            categorical=(),
            position_transform="log1p",
        ).to_dict()
        assert entry["model"]["covariates"] == ["position"]
        assert entry["model"]["categorical"] == []
        for key in ("half_life", "lo", "hi", "not_reached_within", "first_position"):
            assert entry.get(key) == expected.get(key), key
        model = stats.fit_hazard(subset, ("position",), (), "log1p")
        point = stats.half_life(model, 150, first_position=entry["first_position"])
        if isinstance(point, stats.NotReached):
            assert entry["half_life"] is None
        else:
            assert entry["half_life"] == pytest.approx(point, rel=1e-9)


def test_procedure_rules_in_cut_spans_leave_the_analysis_and_are_counted() -> None:
    # D-050 item 5 (synthetic rows): a procedure decision point whose span a cap cut, or
    # whose rows predate span_end (null), cannot show whether the procedure was followed.
    sessions, opps = synthetic(shared_plans=True)
    ends = ["done", "subtask_cap", None, "tool_cap", "turn_cap", "timeout"]
    for i, row in enumerate(opps):
        row["span_end"] = ends[i % len(ends)]
    for row in opps[:4]:
        row.pop("span_end")  # rows banked before the field existed
    rows, exclusions = e1_analysis.analysis_set(sessions, opps)
    reached = reached_pairs(sessions)
    candidates = [o for o in opps if (o["family"], o["session_id"]) in reached]
    called = [o for o in candidates if o["span_calls"] > 0]
    cut = [o for o in called if o["constraint_type"] == "procedure" and o.get("span_end") != "done"]
    assert cut and exclusions["opportunities_procedure_cut"] == len(cut)
    assert exclusions["opportunities_zero_calls"] == len(candidates) - len(called)
    assert exclusions["opportunities_analyzed"] == len(called) - len(cut) == len(rows)
    for row in rows:
        assert row["constraint_type"] != "procedure" or row["span_end"] == "done"
    # Other rule types in cut spans stay.
    assert any(r["constraint_type"] != "procedure" and r.get("span_end") != "done" for r in rows)
    by_family = e1_analysis.exclusions_by_family(sessions, opps)
    assert sum(e["opportunities_procedure_cut"] for e in by_family) == len(cut)
    assert e1_analysis.procedure_cut({"constraint_type": "procedure"}) is True
    assert e1_analysis.procedure_cut({"constraint_type": "procedure", "span_end": "done"}) is False
    assert e1_analysis.procedure_cut({"constraint_type": "invariant", "span_end": None}) is False


# --------------------------------------------------------------------------- D-059


def _step_rows(seed: int, step: float, arms: tuple[str, ...]) -> list[dict[str, Any]]:
    """Synthetic: no position effect; the hazard steps up by ``step`` after the first
    compaction in ``arms``, and token-budget compactions every 18 tool calls tie the
    compaction count to position (not model data)."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for family, base in (("fam-a", -3.2), ("fam-b", -2.8)):
        for plan in range(60):
            strategy = "full_summary" if plan % 2 else "sham"
            for j in range(8):
                position = 4 + 14 * j + rng.randint(0, 9)
                count = position // 18
                eta = base + (step if count >= 1 and strategy in arms else 0.0)
                rows.append(
                    {
                        "family": family,
                        "session_id": f"p{plan:03d}",
                        "position": position,
                        "compaction_count": count,
                        "constraint_type": TYPES[(plan + j) % 4],
                        "strategy": strategy,
                        "violated": rng.random() < 1 / (1 + math.exp(-eta)),
                    }
                )
    return rows


@pytest.mark.parametrize("arms", [("full_summary", "sham"), ("full_summary",)])
def test_h1_holds_its_level_under_a_step_at_the_first_compaction(arms: tuple[str, ...]) -> None:
    # D-059 item 1 (audit B2). Synthetic nulls for H1: position has no effect, but the hazard
    # steps up after the first compaction. A straight-line count in H1's null model cannot
    # follow the step, which leaks into the position score; the compaction cell (level 0, 1,
    # 2 or more, crossed with the arm) follows it. Binomial(40, 0.06) exceeds 8 with
    # probability below 0.001.
    old = new = 0
    for seed in range(40):
        rows = _step_rows(3000 + seed, 2.0, arms)
        straight = stats.hazard_score_test(
            rows,
            "position",
            e1_analysis.HAZARD_COVARIATES,
            e1_analysis.HAZARD_FACTORS,
            e1_analysis.TRANSFORM,
            strata="family",
        )
        cell = e1_analysis.hazard_tests(rows, "pooled", [e1_analysis.POSITION_TERM])
        old += (straight.test.p_value or 1.0) < 0.05
        new += (cell[e1_analysis.POSITION_TERM].test.p_value or 1.0) < 0.05
    assert old >= (30 if len(arms) == 2 else 12), old
    assert new <= 8, new


def test_compaction_cells_cross_the_level_with_the_arm() -> None:
    rows = [
        {"strategy": "sham", "compaction_count": c, "violated": False} for c in (0, 1, 2, 5, 2.0)
    ]
    cells = [r["compaction_cell"] for r in e1_analysis.with_compaction_cell(rows)]
    assert cells == ["sham:0", "sham:1", "sham:2+", "sham:2+", "sham:2+"]
    for bad in (-1, 1.5, True, "1"):
        with pytest.raises(stats.StatsError, match="whole number"):
            e1_analysis.compaction_level(bad)


def _family_b_edited(edit: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sessions, opps = synthetic(n_sessions=40, shared_plans=True)
    for row in opps:
        if row["family"] == "fam-b":
            edit(row)
    return sessions, opps


def test_a_model_that_cannot_be_fit_gives_not_estimable_records_not_a_crash() -> None:
    # D-059 item 3 (audit N6). Synthetic: family fam-b never sees a compaction before a
    # decision point, so the compaction count is constant in its risk set and H2b's model
    # cannot be fit. Its terms are written not estimable with the reason, the fit record
    # names the failure, and the score tests still decide: H2b cannot be tested (its term is
    # constant), while H1's model, where compaction is a factor, is fit and decided.
    sessions, opps = _family_b_edited(lambda row: row.update(compaction_count=0))
    out = e1_analysis.analyze(sessions, opps, e1_analysis.Settings(n_boot=10, seed=1))
    hazard = out["hazard"]
    fits = {f["family"]: f for f in hazard["fits"]}
    assert fits["fam-b"]["fit_failed"] == "single_value"
    assert "compaction_count" in fits["fam-b"]["detail"]
    assert fits["fam-b"]["n_rows"] > fits["fam-b"]["n_events"] > 0
    main = {e["term"]: e for e in hazard["estimates"] if e["family"] == "fam-b"}
    assert {"intercept", "log1p(position)", "compaction_count"} <= set(main)
    for entry in main.values():
        assert entry["not_estimable"] == "single_value" and entry["estimate"] is None
    h2b = hypothesis_entry(hazard, "fam-b", "compaction_count")
    assert h2b["verdict"] == "not tested" and h2b["not_tested"] == "rank_deficient"
    h1 = hypothesis_entry(hazard, "fam-b", "log1p(position)")
    assert h1["not_estimable"] is None and h1["verdict"] in {"supported", "not supported"}
    assert "fit_failed" not in {f["family"]: f for f in hazard["h1_fits"]}["fam-b"]
    # Pooled, the count varies over families, so H2b's model is fit there.
    assert "fit_failed" not in fits["pooled"]
    json.dumps(out, allow_nan=False)


def test_a_half_life_model_that_cannot_be_fit_is_not_estimable_with_its_reason() -> None:
    # Synthetic: every decision point of fam-b starts at tool call 9, so position takes one
    # value in its risk set and no half-life model can be fit there.
    sessions, opps = _family_b_edited(lambda row: row.update(position=9))
    out = e1_analysis.analyze(sessions, opps, e1_analysis.Settings(n_boot=10, seed=1))
    half = {e["family"]: e for e in out["half_life"]["estimates"]}
    assert half["fam-b"]["not_estimable"] == "single_value"
    assert half["fam-b"]["half_life"] is None and half["fam-b"]["lo"] is None
    assert half["fam-b"]["model"]["fit"]["fit_failed"] == "single_value"
    assert half["fam-b"]["first_position"] == 9
    assert half["fam-a"]["not_estimable"] is None
    h1 = hypothesis_entry(out["hazard"], "fam-b", "log1p(position)")
    assert h1["not_estimable"] == "single_value" and h1["not_tested"] == "rank_deficient"
    json.dumps(out, allow_nan=False)


def test_h2a_with_an_arm_absent_after_compaction_is_not_tested_with_its_reason() -> None:
    # Synthetic: in fam-b no sham session reaches a decision point after a compaction.
    def edit(row: dict[str, Any]) -> None:
        if row["strategy"] == "sham":
            row["compaction_count"] = 0

    sessions, opps = _family_b_edited(edit)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    out = e1_analysis.contrasts(rows, e1_analysis.Settings(n_boot=10, seed=1))
    h2a = {e["family"]: e for e in out["h2a"]}
    assert set(h2a) == {"fam-a", "fam-b", "pooled"}
    absent = h2a["fam-b"]
    assert absent["verdict"] == "not tested" and absent["not_tested"] == "arm_absent"
    assert absent["p_value"] is None and absent["p_value_missing"] == "arm_absent"
    assert absent["missing_arms"] == ["sham"] and absent["reference_arm"]["arm"] == "sham"
    assert absent["arm"]["n_opportunities"] > 0 and absent["reference_arm"]["n_opportunities"] == 0
    assert absent["hypothesis"] == "H2a" and absent["decided_by"] == "score_test"
    assert absent["difference"] is None and absent["direction"] is None
    assert h2a["pooled"]["verdict"] in {"supported", "not supported"}


# --------------------------------------------------------------------------- D-060


def _crossed_rows(seed: int, kind: str, effect: float) -> list[dict[str, Any]]:
    """Synthetic: compactions every 15 tool calls from a session-specific offset, so the
    count runs to 7 and follows position; the hazard grows by ``effect`` per compaction
    (``growth``, no position effect) or steps up by ``effect`` at 35 tool calls (``step``,
    no compaction effect). Not model data."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for family, base in (("fam-a", -3.4), ("fam-b", -3.0)):
        for plan in range(60):
            strategy = "full_summary" if plan % 2 else "sham"
            offset = rng.randint(0, 14)
            for j in range(8):
                position = 4 + 14 * j + rng.randint(0, 9)
                count = (position + offset) // 15
                if kind == "growth":
                    eta = base + effect * count
                else:
                    eta = base + (effect if position >= 35 else 0.0)
                rows.append(
                    {
                        "family": family,
                        "session_id": f"p{plan:03d}",
                        "position": position,
                        "compaction_count": count,
                        "constraint_type": TYPES[(plan + j) % 4],
                        "strategy": strategy,
                        "violated": rng.random() < 1 / (1 + math.exp(-eta)),
                    }
                )
    return rows


def test_h1_holds_its_level_when_the_compaction_effect_grows_past_two() -> None:
    # D-060 item 1. Synthetic nulls for H1: the hazard grows with every compaction, past the
    # cell's "2 or more" level, and position has no effect. The cell alone lumps counts 2 to
    # 7 together, and the growth among them leaks into the position score; the count slope
    # beside the cell takes it. Binomial(40, 0.06) exceeds 8 with probability below 0.001.
    cell_only = new = 0
    for seed in range(40):
        rows = e1_analysis.with_model_fields(_crossed_rows(4000 + seed, "growth", 0.6))
        before = stats.hazard_score_test(
            rows, "position", ("position",), e1_analysis.H1_FACTORS, "log1p", strata="family"
        )
        after = e1_analysis.hazard_tests(rows, "pooled", [e1_analysis.POSITION_TERM])
        cell_only += (before.test.p_value or 1.0) < 0.05
        new += (after[e1_analysis.POSITION_TERM].test.p_value or 1.0) < 0.05
    assert cell_only >= 30, cell_only
    assert new <= 8, new


def test_h2b_holds_its_level_under_a_step_of_the_hazard_by_position() -> None:
    # D-060 item 2. Synthetic nulls for H2b: the hazard steps up at 35 tool calls and
    # compaction has no effect. A single log1p(position) slope cannot follow the step, which
    # leaks into the compaction score because the count follows position; the position band
    # beside the slope takes most of it.
    straight = new = 0
    for seed in range(80):
        rows = e1_analysis.with_model_fields(_crossed_rows(5000 + seed, "step", 2.0))
        before = stats.hazard_score_test(
            rows,
            "compaction_count",
            e1_analysis.HAZARD_COVARIATES,
            e1_analysis.HAZARD_FACTORS,
            "log1p",
            strata="family",
        )
        after = e1_analysis.hazard_tests(rows, "pooled", ["compaction_count"])
        straight += (before.test.p_value or 1.0) < 0.05
        new += (after["compaction_count"].test.p_value or 1.0) < 0.05
    assert straight >= 9, straight
    assert new <= 9, new


def test_position_bands_are_ten_tool_calls_wide_up_to_70() -> None:
    # D-061 item 1: bands of ten tool calls below 70, and one band from 70 to the cap.
    positions = (0, 9.5, 10, 19, 20, 29, 30, 39, 40, 49, 50, 59, 60, 69, 70, 149)
    bands = [e1_analysis.position_band(p) for p in positions]
    assert bands == [band for band in e1_analysis.POSITION_BANDS for _ in range(2)]
    assert sorted(e1_analysis.POSITION_BANDS) == list(e1_analysis.POSITION_BANDS)
    for bad in (-1, math.nan, True, "20"):
        with pytest.raises(stats.StatsError, match="position must be"):
            e1_analysis.position_band(bad)
    rows = e1_analysis.with_model_fields(
        [{"strategy": "sham", "compaction_count": 3, "position": 45, "violated": False}]
    )
    assert rows[0]["position_band"] == "[40,50)" and rows[0]["compaction_cell"] == "sham:2+"


def test_the_h1_count_slope_is_left_out_where_the_cells_determine_it() -> None:
    # Without a decision point after a third compaction the count is a function of the
    # cells (0, 1, and 2 within "2+"), so H1's fit leaves it out and says so; H2b's model
    # always keeps its count, the term it tests.
    def rows_with(counts: list[int]) -> list[dict[str, Any]]:
        return e1_analysis.with_model_fields(
            [
                {"strategy": "sham", "compaction_count": c, "position": 5 + 10 * i}
                for i, c in enumerate(counts)
            ]
        )

    low, high = rows_with([0, 1, 2, 2]), rows_with([0, 1, 2, 3])
    h1, h2b = e1_analysis.H1_MODEL, e1_analysis.H2B_MODEL
    assert e1_analysis.model_covariates(h1, low) == (("position",), ["compaction_count"])
    assert e1_analysis.model_covariates(h1, high) == (h1.covariates, [])
    assert e1_analysis.model_covariates(h2b, low) == (h2b.covariates, [])
    sessions, opps = synthetic(n_sessions=40, shared_plans=True)  # counts reach 3 rarely
    for row in opps:
        row["compaction_count"] = min(2, row["compaction_count"])
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    out = e1_analysis.hazard(rows, e1_analysis.Settings(n_boot=10, seed=1))
    for fit in out["h1_fits"]:
        assert fit["dropped_terms"] == ["compaction_count"] and "fit_failed" not in fit
    assert all(fit["dropped_terms"] == [] for fit in out["fits"])
    h1 = hypothesis_entry(out, "fam-a", e1_analysis.POSITION_TERM)
    assert h1["score_model"]["null_covariates"] == [] and h1["not_estimable"] is None
