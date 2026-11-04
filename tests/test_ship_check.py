"""Tests for the gate logic in tools/ship_check.py.

Each check is exercised with a passing input and with inputs where the guarded
property is false, so a gate that could pass on bad data shows up here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_ship_check() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ship_check", REPO_ROOT / "tools" / "ship_check.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ship_check"] = module
    spec.loader.exec_module(module)
    return module


sc = _load_ship_check()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")


def _commit(root: Path, name: str, text: str, message: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _git(root, "add", name)
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


# ------------------------------------------------------------- statistics


def test_wilson_interval_properties() -> None:
    lo, hi = sc.wilson(95, 100)
    assert lo < 0.95 < hi
    assert lo == pytest.approx(0.8882, abs=1e-3)
    lo2, hi2 = sc.wilson(5, 100)
    assert lo2 == pytest.approx(1 - hi)
    assert hi2 == pytest.approx(1 - lo)
    assert sc.wilson(0, 0) == (0.0, 1.0)
    assert sc.wilson(10, 10)[1] == pytest.approx(1.0)


# ------------------------------------------------------------- G2


def test_bare_type_ignores(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.py").write_text(
        'x: int = "a"  # type: ignore[assignment]  # fixture models a bad payload\n'
        'note = "type: ignore in a string is not a comment"\n',
        encoding="utf-8",
    )
    (src / "bad.py").write_text("y: int = 'b'  # type: ignore\n", encoding="utf-8")
    (src / "bad2.py").write_text("z: int = 'c'  # type: ignore[assignment]\n", encoding="utf-8")
    assert sc.bare_type_ignores([src, tmp_path / "absent"], tmp_path) == [
        "src/bad.py:1",
        "src/bad2.py:1",
    ]


def test_weakened_static_config(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.ruff]\nextend-exclude = ['src/halflife/x.py', 'docs/a.md']\n"
        "[tool.ruff.lint.per-file-ignores]\n'src/**' = ['ALL']\n'tests/*' = ['E501']\n"
        "[tool.mypy]\nstrict = true\n"
        "[[tool.mypy.overrides]]\nmodule = 'halflife.x'\nignore_errors = true\n",
        encoding="utf-8",
    )
    problems = sc.weakened_static_config(tmp_path)
    assert len(problems) == 3
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n", encoding="utf-8")
    assert sc.weakened_static_config(tmp_path) == []


def test_repo_config_is_not_weakened() -> None:
    assert sc.weakened_static_config(REPO_ROOT) == []


# ------------------------------------------------------------- G1


def _coverage_json(path: Path, files: list[str], covered: int, statements: int) -> None:
    data = {
        "files": {name: {} for name in files},
        "totals": {"covered_lines": covered, "num_statements": statements},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _fake_package(root: Path, omit: list[str]) -> None:
    pkg = root / "src" / "halflife"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "core.py").write_text("", encoding="utf-8")
    (pkg / "adapter.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        f"[tool.coverage.run]\nomit = {json.dumps(omit)}\n", encoding="utf-8"
    )


def test_core_coverage_passes_and_fails_on_threshold(tmp_path: Path) -> None:
    _fake_package(tmp_path, [])
    files = ["src/halflife/__init__.py", "src/halflife/core.py", "src/halflife/adapter.py"]
    report = tmp_path / "cov.json"
    _coverage_json(report, files, 90, 100)
    ok, detail = sc.core_coverage(tmp_path, report)
    assert ok, detail
    _coverage_json(report, files, 84, 100)
    assert not sc.core_coverage(tmp_path, report)[0]


def test_core_coverage_requires_every_module_measured(tmp_path: Path) -> None:
    _fake_package(tmp_path, [])
    report = tmp_path / "cov.json"
    _coverage_json(report, ["src/halflife/core.py"], 100, 100)
    ok, detail = sc.core_coverage(tmp_path, report)
    assert not ok and "missing from coverage" in detail


def test_core_coverage_omit_needs_contract_tests(tmp_path: Path) -> None:
    _fake_package(tmp_path, ["src/halflife/adapter.py"])
    files = ["src/halflife/__init__.py", "src/halflife/core.py"]
    report = tmp_path / "cov.json"
    _coverage_json(report, files, 100, 100)
    ok, detail = sc.core_coverage(tmp_path, report)
    assert not ok and "contract tests" in detail
    (tmp_path / "tests" / "contract").mkdir(parents=True)
    (tmp_path / "tests" / "contract" / "test_adapter.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "fixtures" / "adapter").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "adapter" / "rec.json").write_text("{}", encoding="utf-8")
    assert sc.core_coverage(tmp_path, report)[0]


def test_core_coverage_rejects_glob_omit(tmp_path: Path) -> None:
    _fake_package(tmp_path, ["src/halflife/*"])
    ok, detail = sc.core_coverage(tmp_path, tmp_path / "cov.json")
    assert not ok and "explicit module path" in detail


# ------------------------------------------------------------- G3


def test_workflow_problems(tmp_path: Path) -> None:
    assert sc.workflow_problems(tmp_path) == ["missing .github/workflows/ci.yml"]
    wf = tmp_path / ".github" / "workflows" / "ci.yml"
    wf.parent.mkdir(parents=True)
    wf.write_text('python-version: ["3.12"]\nsteps:\n  - run: make test\n', encoding="utf-8")
    assert sc.workflow_problems(tmp_path) == ["no 'run: make ci' step", "Python 3.11 not in matrix"]
    assert sc.workflow_problems(REPO_ROOT) == []


# ------------------------------------------------------------- G5


def _detector_artifact(cases: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for c in cases if c["label"] and c["predicted"])
    fp = sum(1 for c in cases if not c["label"] and c["predicted"])
    fn = sum(1 for c in cases if c["label"] and not c["predicted"])
    out: dict[str, Any] = {"cases": cases}
    for name, (k, n) in {"precision": (tp, tp + fp), "recall": (tp, tp + fn)}.items():
        lo, hi = sc.wilson(k, n)
        out[name] = {"value": k / n, "lo": lo, "hi": hi}
    return out


def _cases(n_pos: int, n_neg: int, misses: int = 0) -> list[dict[str, Any]]:
    kinds = [*sorted(sc.SEEDED_INDIRECT_KINDS), "direct"]
    cases = [
        {"case_id": f"p{i}", "kind": kinds[i % len(kinds)], "label": True, "predicted": i >= misses}
        for i in range(n_pos)
    ]
    cases += [
        {"case_id": f"n{i}", "kind": "benign", "label": False, "predicted": False}
        for i in range(n_neg)
    ]
    return cases


def test_check_detectors_pass_and_recompute(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_detector_artifact(_cases(100, 100))), encoding="utf-8")
    ok, detail = sc.check_detectors(path)
    assert ok, detail
    assert "P 1.000" in detail and "R 1.000" in detail


def test_check_detectors_rejects_tampered_numbers(tmp_path: Path) -> None:
    artifact = _detector_artifact(_cases(100, 100, misses=10))
    artifact["recall"]["value"] = 0.99
    path = tmp_path / "d.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    ok, detail = sc.check_detectors(path)
    assert not ok and "does not match" in detail


def test_check_detectors_rejects_low_recall(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_detector_artifact(_cases(100, 100, misses=10))), encoding="utf-8")
    ok, detail = sc.check_detectors(path)
    assert not ok and "recall" in detail


def test_check_detectors_requires_indirect_kinds(tmp_path: Path) -> None:
    cases = [c for c in _cases(100, 10) if c["kind"] != "cp"]
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_detector_artifact(cases)), encoding="utf-8")
    ok, detail = sc.check_detectors(path)
    assert not ok and "cp" in detail


def test_check_detectors_requires_unique_ids_and_booleans(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    cases = _cases(100, 100)
    path.write_text(json.dumps(_detector_artifact([*cases, dict(cases[0])])), encoding="utf-8")
    assert sc.check_detectors(path) == (False, "duplicate case_id p0 in seeded cases")
    cases[150]["predicted"] = None
    path.write_text(json.dumps(_detector_artifact(cases)), encoding="utf-8")
    assert sc.check_detectors(path) == (False, "case n50: predicted is not a boolean")
    cases[150]["predicted"] = False
    cases[150]["label"] = 0
    path.write_text(json.dumps(_detector_artifact(cases)), encoding="utf-8")
    assert sc.check_detectors(path) == (False, "case n50: label is not a boolean")


def test_check_detectors_on_the_banked_validation() -> None:
    """The banked E0 detector validation still passes the hardened check."""
    banked = REPO_ROOT / "artifacts" / "e0" / "detector_validation.json"
    if not banked.is_file():
        pytest.skip("artifacts/e0/detector_validation.json is not banked")
    ok, detail = sc.check_detectors(banked)
    assert ok, detail


def test_check_leakage_requires_booleans_and_unique_runs(tmp_path: Path) -> None:
    path = tmp_path / "l.json"

    def check(runs: list[dict[str, Any]], validation_runs: Any, leaks: Any = 0) -> str:
        data = {"validation_runs": validation_runs, "leaks": leaks, "runs": runs}
        path.write_text(json.dumps(data), encoding="utf-8")
        result: str = sc.check_leakage(path)[1]
        return result

    good = {"run_id": "r", "session_id": "s1", "leaked": False}
    assert check([good, {**good, "session_id": "s2"}], 2) == "leakage 0/2"
    assert check([good, dict(good)], 2) == "duplicate probe leakage run r"
    assert check([{**good, "leaked": None}], 1) == "probe leakage run r: leaked is not a boolean"
    assert check([{**good, "leaked": "no"}], 1) == "probe leakage run r: leaked is not a boolean"
    assert check([{"leaked": False}], 1) == "probe leakage run without a run_id"
    assert check([good], True, False) == "probe leakage artifact inconsistent or empty"


def test_check_leakage(tmp_path: Path) -> None:
    path = tmp_path / "l.json"
    runs = [{"run_id": "a", "leaked": False}, {"run_id": "b", "leaked": False}]
    path.write_text(json.dumps({"validation_runs": 2, "leaks": 0, "runs": runs}), encoding="utf-8")
    assert sc.check_leakage(path)[0]
    runs[1]["leaked"] = True
    path.write_text(json.dumps({"validation_runs": 2, "leaks": 0, "runs": runs}), encoding="utf-8")
    assert not sc.check_leakage(path)[0]
    path.write_text(json.dumps({"validation_runs": 0, "leaks": 0, "runs": []}), encoding="utf-8")
    assert not sc.check_leakage(path)[0]


COMPLETE_LOG: dict[str, Any] = {
    "event_id": "c1",
    "strategy": "full_summary",
    "tool_call_index": 12,
    "scheduled": True,
    "tokens_before": 4000,
    "tokens_after": 900,
    "before": [{"role": "system", "content": "s"}, {"role": "user", "content": "rules"}],
    "after": [{"role": "system", "content": "s"}, {"role": "user", "content": "summary"}],
    "summary": "Fixed the parser.",
    "constraint_text_retained": {"protect-tests": False},
}


def test_check_compaction_logs(tmp_path: Path) -> None:
    # This fixture used to bank "{}" as the log, which passed only because completeness
    # was checked as presence plus hash; it now banks a complete log.
    log = tmp_path / "artifacts" / "e0" / "c1.json"
    log.parent.mkdir(parents=True)
    text = json.dumps(COMPLETE_LOG)
    log.write_text(text, encoding="utf-8")
    event = {
        "run_id": "r",
        "event_id": "c1",
        "strategy": "full_summary",
        "tool_call_index": 12,
        "tokens_before": 4000,
        "tokens_after": 900,
        "log_path": "artifacts/e0/c1.json",
        "log_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"events": [event]}), encoding="utf-8")
    assert sc.check_compaction_logs(path, tmp_path) == (True, "1 compaction logs complete")
    event["log_sha256"] = "0" * 64
    path.write_text(json.dumps({"events": [event]}), encoding="utf-8")
    assert not sc.check_compaction_logs(path, tmp_path)[0]
    del event["strategy"]
    path.write_text(json.dumps({"events": [event]}), encoding="utf-8")
    assert "missing strategy" in sc.check_compaction_logs(path, tmp_path)[1]
    path.write_text(json.dumps({"events": []}), encoding="utf-8")
    assert not sc.check_compaction_logs(path, tmp_path)[0]


def test_check_compaction_logs_reads_the_log_content(tmp_path: Path) -> None:
    log = tmp_path / "artifacts" / "e0" / "c1.json"
    log.parent.mkdir(parents=True)
    path = tmp_path / "c.json"

    def bank(content: str, **event: Any) -> tuple[bool, str]:
        log.write_text(content, encoding="utf-8")
        record = {
            "run_id": "r",
            "event_id": "c1",
            "strategy": "full_summary",
            "tool_call_index": 12,
            "tokens_before": 4000,
            "tokens_after": 900,
            "log_path": "artifacts/e0/c1.json",
            "log_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            **event,
        }
        path.write_text(json.dumps({"events": [record]}), encoding="utf-8")
        result: tuple[bool, str] = sc.check_compaction_logs(path, tmp_path)
        return result

    assert bank(json.dumps(COMPLETE_LOG))[0]
    assert "is not valid JSON" in bank("")[1]
    assert "lacks event_id" in bank("{}")[1]
    assert "has no summary text" in bank(json.dumps({**COMPLETE_LOG, "summary": None}))[1]
    sliding = {**COMPLETE_LOG, "strategy": "sliding_window", "summary": None}
    assert bank(json.dumps(sliding), strategy="sliding_window")[0]
    assert (
        "disagrees with its event on tokens_after"
        in bank(json.dumps(COMPLETE_LOG), tokens_after=901)[1]
    )
    assert (
        "log_path is not a clean path"
        in bank(json.dumps(COMPLETE_LOG), log_path=str(log.resolve()))[1]
    )
    assert "marked incomplete" in bank(json.dumps(COMPLETE_LOG), complete=False)[1]


def test_compaction_log_problems() -> None:
    assert sc.compaction_log_problems(COMPLETE_LOG) == []
    assert sc.compaction_log_problems([]) == ["is not a JSON object"]
    assert sc.compaction_log_problems({}) == [f"lacks {n}" for n in sc.COMPACTION_LOG_FIELDS]
    lost = {**COMPLETE_LOG, "after": [{"role": "user", "content": "x"}]}
    assert sc.compaction_log_problems(lost) == ["lost the system prompt"]
    assert sc.compaction_log_problems({**COMPLETE_LOG, "before": []}) == [
        "has an empty before or after context"
    ]
    assert sc.compaction_log_problems({**COMPLETE_LOG, "summary": "  "}) == ["has no summary text"]
    assert sc.compaction_log_problems({**COMPLETE_LOG, "constraint_text_retained": {}}) == [
        "does not log constraint text retention"
    ]
    assert sc.compaction_log_problems({**COMPLETE_LOG, "tokens_before": -1}) == [
        "has a count that is not a non-negative integer"
    ]
    assert sc.compaction_log_problems({**COMPLETE_LOG, "scheduled": 1}) == [
        "has a scheduled flag that is not a boolean"
    ]


# ------------------------------------------------------------- paths, numbers, templates


def test_clone_file_confines_paths(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    (clone / "a").mkdir(parents=True)
    (clone / "a" / "b.json").write_text("{}", encoding="utf-8")
    assert sc.clone_file(clone, "a/b.json") == clone / "a" / "b.json"
    assert sc.clone_file(clone, "a/missing.json") == clone / "a" / "missing.json"
    for bad in ("", "/etc/passwd", "../x.json", "a/../b.json", "./a/b.json", "a//b.json", "a\\b"):
        assert sc.clone_file(clone, bad) is None, bad
    assert sc.clone_file(clone, None) is None
    assert sc.clone_file(clone, str(clone / "a" / "b.json")) is None
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (clone / "link.json").symlink_to(outside)
    assert sc.clone_file(clone, "link.json") is None


def test_number_helpers() -> None:
    assert sc.is_count(0) and sc.is_count(7)
    for bad in (-1, True, False, 1.0, "3", None):
        assert not sc.is_count(bad), bad
    assert sc.finite_nonnegative(0) and sc.finite_nonnegative(0.4) and sc.finite_nonnegative(12)
    for bad in (-0.1, float("nan"), float("inf"), True, None, "1", 10**400):
        assert not sc.finite_nonnegative(bad), bad


def test_template_facts_parsing() -> None:
    facts = sc.template_facts({"id": "t", "split": "main", "constraint_types": ["procedure"]})
    assert facts == sc.TemplateFacts("t", "main", frozenset({"procedure"}))
    assert sc.template_facts({"error": "SchemaError: bad\nline"}) == "SchemaError: bad line"
    assert sc.template_facts(None) == "no result from the template loader"
    assert sc.template_facts({"id": "t", "split": "main"}) == (
        "template loader returned malformed facts"
    )


def _loader_context(tmp_path: Path, clone: Path, python: tuple[str, ...]) -> Any:
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    return sc.Context(
        repo=clone,
        clone=clone,
        logs=logs,
        sha="0" * 40,
        fast=True,
        env=dict(os.environ),
        python=python,
    )


def test_load_templates_reads_the_files_in_a_subprocess(tmp_path: Path) -> None:
    from halflife import bench

    rel = "src/halflife/bench/templates/invoice-ledger.yaml"
    ctx = _loader_context(tmp_path, REPO_ROOT, (sys.executable,))
    facts = sc.load_templates(ctx, [rel, "README.md"])
    loaded = facts[rel]
    assert isinstance(loaded, sc.TemplateFacts) and loaded.id == "invoice-ledger"
    assert loaded.constraint_types and loaded.constraint_types <= sc.CONSTRAINT_TYPES
    rules = {c.id for c in bench.load(REPO_ROOT / rel).template.constraints}
    assert loaded.constraint_ids == rules and rules
    assert isinstance(facts["README.md"], str)
    assert ctx.shared["template_facts"][rel] == loaded


def test_load_templates_reports_a_broken_loader(tmp_path: Path) -> None:
    ctx = _loader_context(tmp_path, tmp_path, ("false",))
    facts = sc.load_templates(ctx, ["t.yaml"])
    assert facts == {"t.yaml": "template loader failed in the clone (rc=1)"}


def test_template_facts_carry_constraint_ids() -> None:
    entry = {
        "id": "t",
        "split": "main",
        "constraint_types": ["procedure"],
        "constraint_ids": ["keep-api", "protect-tests"],
    }
    assert sc.template_facts(entry) == sc.TemplateFacts(
        "t", "main", frozenset({"procedure"}), frozenset({"keep-api", "protect-tests"})
    )
    malformed: tuple[object, ...] = ("keep-api", [1], None)
    for bad in malformed:
        assert sc.template_facts({**entry, "constraint_ids": bad}) == (
            "template loader returned malformed facts"
        ), bad


def test_inside_and_scratch_parent(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    (clone / "logs").mkdir(parents=True)
    assert sc.inside(clone / "logs", clone) and sc.inside(clone, clone)
    assert not sc.inside(tmp_path / "logs", clone)
    (clone / "out").symlink_to(tmp_path)
    assert not sc.inside(clone / "out" / "x.json", clone)
    ctx = _loader_context(tmp_path, clone, (sys.executable,))
    assert sc.scratch_parent(ctx) == tmp_path / "logs"
    ctx.logs = clone / "logs"
    assert sc.scratch_parent(ctx) is None


def test_source_sessions_maps_runs_and_sessions_to_templates(tmp_path: Path) -> None:
    rows = [
        {"run_id": "r1", "session_id": "s1", "template_id": "t1"},
        {"run_id": "r2", "session_id": "s1", "template_id": "t2"},
        {"session_id": "s3", "template_id": "t3"},
        {"run_id": "r1", "session_id": "", "template_id": "t4"},
    ]
    source = tmp_path / "artifacts" / "pilot"
    source.mkdir(parents=True)
    text = "".join(json.dumps(row) + "\n" for row in rows)
    (source / "sessions.jsonl").write_text(text, encoding="utf-8")
    assert sc.source_sessions(tmp_path, "artifacts/pilot") == (
        {("r1", "s1"): ["t1"], ("r2", "s1"): ["t2"]},
        "",
    )
    unnamed = "compaction logs name no source (the banked experiment directory)"
    assert sc.source_sessions(tmp_path, None) == ({}, unnamed)
    assert sc.source_sessions(tmp_path, "") == ({}, unnamed)
    assert sc.source_sessions(tmp_path, "../pilot")[1] == (
        "compaction log source '../pilot' is not a clean path inside the clone"
    )
    assert sc.source_sessions(tmp_path, "artifacts")[1] == (
        "compaction log source artifacts has no sessions.jsonl"
    )
    assert sc.source_sessions(tmp_path, "artifacts/pilot/sessions.jsonl")[1].endswith(
        "has no sessions.jsonl"
    )


def test_bench_records_on_the_banked_bench_pass() -> None:
    """Every banked route, reference run, and action log passes the hardened G6 checks."""
    manifest = REPO_ROOT / "artifacts" / "bench" / "manifest.json"
    if not manifest.is_file():
        pytest.skip("artifacts/bench/manifest.json is not banked")
    templates = json.loads(manifest.read_text(encoding="utf-8"))["templates"]
    problems = {t["id"]: sc.bench_records_problem(REPO_ROOT, t) for t in templates}
    assert {tid: problem for tid, problem in problems.items() if problem} == {}


# ------------------------------------------------------------- G7 and G12 helpers


def test_check_interval() -> None:
    assert sc.check_interval({"lo": 1, "estimate": 2, "hi": 3}, "estimate")
    assert not sc.check_interval({"lo": 2.5, "estimate": 2, "hi": 3}, "estimate")
    assert not sc.check_interval({"lo": 1, "estimate": float("nan"), "hi": 3}, "estimate")
    assert not sc.check_interval({"estimate": 2}, "estimate")


def test_json_difference_names_the_first_differing_key() -> None:
    banked: dict[str, Any] = {
        "estimates": [{"family": "a", "hi": 0.3, "lo": 0.1}],
        "settings": {"n_boot": 20},
    }
    assert sc.json_difference(banked, json.loads(json.dumps(banked))) is None
    changed = json.loads(json.dumps(banked))
    changed["estimates"][0]["hi"] = 0.31
    changed["settings"]["n_boot"] = 21
    assert sc.json_difference(banked, changed) == "estimates[0].hi"
    assert sc.json_difference(banked, {**banked, "extra": 1}) == "extra"
    assert sc.json_difference({**banked, "extra": 1}, banked) == "extra"
    assert sc.json_difference({"n": 1}, {"n": 1.0}) == "n"
    assert sc.json_difference({"n": 1}, {"n": True}) == "n"
    assert sc.json_difference({"x": float("nan")}, {"x": float("nan")}) == "x"
    assert sc.json_difference({"a": [1]}, {"a": []}) == "a[0]"
    assert sc.json_difference([1, 2], [1, 2, 3]) == "[2]"
    assert sc.json_difference([], {}) == "(top level)"
    assert sc.json_difference(None, None) is None


def test_analysis_settings_reads_n_boot_and_seed() -> None:
    settings = {"n_boot": 2000, "seed": 20261104, "level": 0.95, "max_position": 150}
    assert sc.analysis_settings({"settings": settings}) == (2000, 20261104)
    unusable: tuple[object, ...] = (
        {},
        {"settings": None},
        {"settings": {"seed": 1}},
        {"settings": {"n_boot": 20}},
        {"settings": {"n_boot": 0, "seed": 1}},
        {"settings": {"n_boot": True, "seed": 1}},
        {"settings": {"n_boot": 20.0, "seed": 1}},
        {"settings": {"n_boot": 20, "seed": "1"}},
        {"settings": {"n_boot": 20, "seed": False}},
        [],
    )
    for bad in unusable:
        assert sc.analysis_settings(bad) is None, bad


def test_prereg_link_requires_strictly_earlier_commit(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, "prereg/0001-e1.md", "plan", "prereg")
    _commit(tmp_path, "artifacts/e1/sessions.jsonl", "{}", "data")
    assert (
        sc.check_prereg_link(tmp_path, "prereg/0001-e1.md", "artifacts/e1/sessions.jsonl") is None
    )
    assert "missing" in str(
        sc.check_prereg_link(tmp_path, "notes.md", "artifacts/e1/sessions.jsonl")
    )

    # This fixture used "prereg/0001.md", a name make prereg-verify rejects; the link
    # check now requires a prereg/NNNN-slug.md path, so it uses one.
    same = tmp_path / "same"
    same.mkdir()
    _init_repo(same)
    (same / "prereg").mkdir()
    (same / "prereg" / "0001-e1.md").write_text("plan", encoding="utf-8")
    (same / "data.jsonl").write_text("{}", encoding="utf-8")
    _git(same, "add", ".")
    _git(same, "commit", "-q", "-m", "both at once")
    assert "does not predate" in str(sc.check_prereg_link(same, "prereg/0001-e1.md", "data.jsonl"))


def test_prereg_link_requires_a_clean_prereg_path(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, "README.md", "# half-life\n", "bootstrap")
    _commit(tmp_path, "prereg/.gitkeep", "", "prereg directory")
    _commit(tmp_path, "prereg/0001.md", "plan", "unslugged plan")
    _commit(tmp_path, "artifacts/e1/sessions.jsonl", "{}", "data")
    for prereg in ("prereg/.gitkeep", "prereg/../README.md", "prereg/0001.md"):
        problem = sc.check_prereg_link(tmp_path, prereg, "artifacts/e1/sessions.jsonl")
        assert problem is not None and "not a prereg/NNNN-slug.md path" in problem, prereg


def test_first_commit_and_ancestry(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    a = _commit(tmp_path, "a.txt", "1", "a")
    b = _commit(tmp_path, "b.txt", "1", "b")
    _commit(tmp_path, "a.txt", "2", "edit a")
    assert sc.first_commit(tmp_path, "a.txt") == a
    assert sc.first_commit(tmp_path, "never.txt") is None
    assert sc.strictly_predates(tmp_path, a, b)
    assert not sc.strictly_predates(tmp_path, b, a)
    assert not sc.strictly_predates(tmp_path, a, a)


# ------------------------------------------------------------- G13


GOOD_README = """# half-life

The headline finding with its 95% CI goes here.

![headline](docs/figures/headline.svg)

| arm | value |
|---|---|
| a | b |

## Quickstart

Run it offline.

## What it is and is not
## Key results
## How it works
## Use it on your own benchmark or agent
## Reproduce every figure
## Limitations and threats to validity
## Related work
## Citation
## License
"""


def test_readme_problems_accepts_section_11_layout(tmp_path: Path) -> None:
    (tmp_path / "docs" / "figures").mkdir(parents=True)
    (tmp_path / "docs" / "figures" / "headline.svg").write_text("<svg/>", encoding="utf-8")
    assert sc.readme_problems(tmp_path, GOOD_README) == []


def test_readme_problems_flags_brief_and_style(tmp_path: Path) -> None:
    brief = (REPO_ROOT / "docs" / "BUILD_BRIEF.md").read_text(encoding="utf-8")
    assert "README.md is still the build brief" in sc.readme_problems(tmp_path, brief)
    styled = GOOD_README + "\nA groundbreaking result \u2014 wow.\n"
    problems = sc.readme_problems(tmp_path, styled)
    assert "README.md contains an em dash" in problems
    assert "README.md uses 'groundbreaking'" in problems
    assert "headline figure docs/figures/headline.svg missing" in problems


def test_headings_skip_fenced_code() -> None:
    text = "# Title\n```\n# not a heading\n```\n## Real\n"
    assert sc.headings(text) == ["title", "real"]


def test_report_problems_word_count_and_sections(tmp_path: Path) -> None:
    sections = "".join(f"## {name}\n\n" for name in sc.REPORT_SECTIONS)
    short = "# Report\n" + sections + "words " * 10
    problems = sc.report_problems(tmp_path, short)
    assert any("words, needs" in p for p in problems)
    full = "# Report\n" + sections + "word " * 4500
    assert sc.report_problems(tmp_path, full) == []
    assert "docs/REPORT.md has placeholders" in sc.report_problems(tmp_path, full + " TODO")


def test_style_problems_catch_entities_and_split_phrases() -> None:
    for dash in ("\N{EM DASH}", "&mdash;", "&#8212;", "&#x2014;", "&#X2014;", "&#08212;"):
        assert sc.style_problems("R", f"a {dash} b") == ["R contains an em dash"], dash
    for not_dash in ("&#82120;", "&ndash;", "\N{EN DASH}", "--", "&#x20141;"):
        assert sc.style_problems("R", f"a {not_dash} b") == [], not_dash
    assert sc.style_problems("R", "A state of\n  the art result") == ["R uses 'state of the art'"]
    assert sc.style_problems("R", "state-of-the-\nart") == ["R uses 'state-of-the-art'"]
    assert sc.style_problems("R", "A Ground\nbreaking idea") == []
    assert sc.style_problems("R", "groundbreaking") == ["R uses 'groundbreaking'"]


def test_has_interval_needs_a_real_interval() -> None:
    for text in (
        "half-life 35 (95% CI 22 to 61)",
        "a 90 % credible interval",
        "the 95% interval",
        "with its confidence interval",
        "35 tool calls, CI [22, 61]",
        "interval (0.12-0.31)",
    ):
        assert sc.has_interval(text), text
    for text in (
        "run make ci first",
        "the interval is wide",
        "an interval of time",
        "CI passes",
        "make ci [22, 61]",
        "95% of sessions",
    ):
        assert not sc.has_interval(text), text


def test_report_placeholders(tmp_path: Path) -> None:
    sections = "".join(f"## {name}\n\n" for name in sc.REPORT_SECTIONS)
    full = "# Report\n" + sections + "word " * 4500
    for marker in ("Pending results.", "pending", "TBD", "FIXME", "XXX", "to be\nwritten"):
        assert "docs/REPORT.md has placeholders" in sc.report_problems(tmp_path, full + marker)
    for fine in ("the todo-cli template", "Coming next: E2", "xxx-large", "spending"):
        assert sc.report_problems(tmp_path, full + fine) == [], fine
    assert "docs/REPORT.md has placeholders" in sc.report_problems(tmp_path, full + "Coming soon")


def test_report_mentions_whole_tokens(tmp_path: Path) -> None:
    report = tmp_path / "docs" / "REPORT.md"
    report.parent.mkdir(parents=True)
    report.write_text("**E2**'s arm (E3) is ecological.", encoding="utf-8")
    assert sc.report_mentions(tmp_path, "E2", "E3", "ecological") == []
    report.write_text("E20, E3-like, non-ecological, ecologically", encoding="utf-8")
    assert sc.report_mentions(tmp_path, "E2", "E3", "ecological") == ["E2", "E3", "ecological"]


def test_has_install_heading() -> None:
    assert sc.has_install_heading("# Tools\n\n## Install\n")
    assert sc.has_install_heading("### Installation and setup\n")
    assert sc.has_install_heading("## How to install\n")
    assert not sc.has_install_heading("## Uninstall\n")
    assert not sc.has_install_heading("## Reinstalling\n")
    assert not sc.has_install_heading("```\n## Install\n```\n")
    assert not sc.has_install_heading("Install it with pip.\n")
    assert not sc.has_install_heading("#Install\n")


def test_console_scripts_parsing() -> None:
    text = (
        "[console_scripts]\nhalflife-guard = halflife.kit.guard:main\n"
        "  continuation = x\n# comment = y\n[gui_scripts]\nhalflife-pin = a:b\n"
    )
    assert sc.console_scripts(text) == {"halflife-guard"}
    assert sc.console_scripts("[console_scripts]\nx=a:b\n") == {"x"}
    assert sc.console_scripts("[ console_scripts ]\nx=a:b\n") == set()


# ------------------------------------------------------------- G14


def test_original_checkout_problems(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, "README.md", "x", "init")
    assert sc.original_checkout_problems(tmp_path) == []
    (tmp_path / "scratch.txt").write_text("untracked", encoding="utf-8")
    assert sc.original_checkout_problems(tmp_path) == ["working tree not clean (1 paths)"]
    (tmp_path / "scratch.txt").unlink()
    _commit(tmp_path, "private/secret.txt", "s", "oops")
    assert "files under private/ are tracked" in sc.original_checkout_problems(tmp_path)


def test_private_manifest_hashes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("private/\n", encoding="utf-8")
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "data.txt").write_text("raw", encoding="utf-8")
    good = hashlib.sha256(b"raw").hexdigest()
    manifest = {"files": [{"path": "private/data.txt", "sha256": good}]}
    _commit(tmp_path, "artifacts/private_hashes.json", json.dumps(manifest), "manifest")
    _git(tmp_path, "add", ".gitignore")
    _git(tmp_path, "commit", "-q", "-m", "ignore")
    assert sc.original_checkout_problems(tmp_path) == []
    (tmp_path / "private" / "data.txt").write_text("changed", encoding="utf-8")
    assert sc.original_checkout_problems(tmp_path) == [
        "private file private/data.txt hash mismatch"
    ]


# ------------------------------------------------------------- output format


def test_one_line_collapses_whitespace() -> None:
    assert sc.one_line("a\n  b\tc ") == "a b c"
    assert len(sc.one_line("x" * 500)) == 240


def test_run_gates_turns_errors_into_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(ctx: Any) -> Any:
        raise KeyError("cases")

    gates = {n: (lambda ctx: sc.GateResult(True, "ok")) for n in sc.GATES}
    gates[5] = boom
    monkeypatch.setattr(sc, "GATES", gates)
    results = sc.run_gates(None)
    assert not results[5].passed
    assert results[5].evidence.startswith("error: KeyError")
    assert sum(r.passed for r in results.values()) == 13


def test_check_half_life_forms() -> None:
    assert sc.check_half_life({"half_life": 40.0, "lo": 30.0, "hi": 55.0})
    assert sc.check_half_life(
        {"half_life": 140.5, "lo": 106.1, "hi": None, "hi_not_reached_within": 150}
    )
    assert sc.check_half_life(
        {"half_life": None, "not_reached_within": 150, "lo": 120.0, "hi": None}
    )
    assert not sc.check_half_life({"half_life": 140.5, "lo": 106.1, "hi": None})
    assert not sc.check_half_life(
        {"half_life": 160.0, "lo": 106.1, "hi": None, "hi_not_reached_within": 150}
    )
    assert not sc.check_half_life(
        {"half_life": 40.0, "lo": 50.0, "hi": None, "hi_not_reached_within": 150}
    )
    assert not sc.check_half_life({"half_life": None, "not_reached_within": True})
    assert not sc.check_half_life({"half_life": None, "not_reached_within": 0})
    assert not sc.check_half_life({"half_life": 40.0, "hi": None, "hi_not_reached_within": 150})
    assert not sc.check_half_life({"half_life": 40.0, "lo": 30.0, "hi": 35.0})
