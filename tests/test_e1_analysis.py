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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthetic sessions and opportunities from a known hazard model (not model data).

    With ``shared_plans`` both families run the same plans, so session ids repeat across
    families as they do in E1; ``type_effects`` add log odds per rule type, and ``slope``
    multiplies ``log1p(position)``.
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
    assert fam_b["log1p(position)"]["hypothesis"] == "H1"
    assert fam_b["compaction_count"]["hypothesis"] == "H2b"
    for term in ("log1p(position)", "compaction_count"):
        # Without a violation the score test has nothing to test, and says so.
        assert fam_b[term]["verdict"] == "not tested" and fam_b[term]["not_tested"] == "no_events"
        assert fam_b[term]["p_value"] is None and fam_b[term]["p_value_missing"] == "no_events"
    pooled = {e["term"]: e for e in out["estimates"] if e["family"] == "pooled"}
    assert pooled["family[fam-b]"]["not_estimable"] == "separation"
    assert pooled["log1p(position)"]["not_estimable"] is None
    fam_a = {e["term"]: e for e in out["estimates"] if e["family"] == "fam-a"}
    for term in ("log1p(position)", "compaction_count"):
        # Pooled tests are stratified by family (D-050 item 2): fam-b adds nothing, so the
        # pooled score test equals fam-a's, instead of resting on a separated family term.
        assert pooled[term]["verdict"] in {"supported", "not supported"}
        assert pooled[term]["p_value"] == pytest.approx(fam_a[term]["p_value"], rel=1e-9)
        assert pooled[term]["score"]["strata_skipped"] == ["fam-b"]
        assert pooled[term]["score_model"]["strata"] == "family"
        assert pooled[term]["score"]["p_value"] == pooled[term]["p_value"]
    assert fam_a["intercept"]["hypothesis"] is None and fam_a["intercept"]["verdict"] is None
    assert fam_a["intercept"]["decided_by"] is None and fam_a["intercept"]["not_tested"] is None
    json.dumps(out, allow_nan=False)


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
    # Synthetic: the hazard rises with log1p(position) (slope 1.5 per unit).
    sessions, opps = synthetic(n_sessions=80, shared_plans=True, slope=1.5)
    rows, _ = e1_analysis.analysis_set(sessions, opps)
    settings = e1_analysis.Settings(n_boot=10, seed=1)
    out = e1_analysis.hazard(rows, settings)
    assert out["decided_by"] == "score_test" and out["alpha"] == settings.alpha
    for family in ("fam-a", "fam-b", "pooled"):
        subset = [r for r in rows if family == "pooled" or r["family"] == family]
        for term, name in ((e1_analysis.POSITION_TERM, "position"), ("compaction_count", None)):
            entry = next(e for e in out["estimates"] if e["family"] == family and e["term"] == term)
            expected = stats.hazard_score_test(
                subset,
                name or term,
                e1_analysis.HAZARD_COVARIATES,
                e1_analysis.HAZARD_FACTORS,
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
    pooled_h1 = next(
        e for e in out["estimates"] if (e["family"], e["term"]) == ("pooled", "log1p(position)")
    )
    assert pooled_h1["verdict"] == "supported" and pooled_h1["direction"] == "positive"
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
