# Adapted from raise-v1 tests/mutation_test.py (https://github.com/raj200501/raise-v1-public),
# Copyright 2026 Raj Kashikar, licensed under the Apache License, Version 2.0
# (LICENSES/Apache-2.0.txt; see NOTICE). Modified for half-life: the cases are pytest
# parameters against tools/ship_check.py, each detected mutation must fail for the reason
# the gate guards, and a surviving mutation is recorded as a strict expected failure that
# names the hole, so fixing the gate turns the test red until the marker is removed.
"""Mutation tests for the data gates in tools/ship_check.py: every gate must be able to fail.

For each data gate (G5 to G9, G12, and the G13 document helpers) a passing fixture is
built in tmp_path and confirmed to pass; then each mutation makes the property the gate
guards false, and the test asserts that the gate fails with evidence naming that
property (a failure for an unrelated reason does not count as a detection).
tools/ship_check.py is imported with importlib and never edited. Gates run through
``run_gates``, so an exception counts as a failure exactly as in a real ship check.

A mutation that the gate does not catch is a hole in the instrument. It is kept as a
strict expected failure (``xfail(strict=True, raises=AssertionError)``) whose reason
describes the hole: the suite stays green while the hole is open and goes red when the
gate is fixed. Every fixture here is synthetic.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from halflife.integrity import prereg as prereg_tool

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_ship_check() -> ModuleType:
    name = "ship_check_under_mutation"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / "ship_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sc: Any = _load_ship_check()

_IDENTITY = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}
SHA = "0123456789abcdef0123456789abcdef01234567"


# --------------------------------------------------------------------------- harness


class FixtureBroken(RuntimeError):
    """The unmutated fixture does not pass, so a mutation result would mean nothing."""


@dataclass(frozen=True)
class Case:
    """One mutation.

    Attributes:
        name: Test id.
        expect: Regular expression the failing gate's evidence must match.
        build: Makes the mutated fixture under a directory and returns its Context.
        hole: Non-empty for a known surviving mutation: why the gate still passes.
    """

    name: str
    expect: str
    build: Callable[[Path], Any]
    hole: str = ""


def params(items: list[Any]) -> list[Any]:
    """Pytest parameters; known holes become strict expected failures with their reason."""
    out: list[Any] = []
    for item in items:
        marks = []
        if item.hole:
            marks.append(
                pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"HOLE: {item.hole}")
            )
        out.append(pytest.param(item, id=item.name, marks=marks))
    return out


def evaluate(number: int, ctx: Any) -> Any:
    """Run one gate through ``run_gates`` (errors become failures, as in the real check)."""
    saved = sc.EVALUATION_ORDER
    sc.EVALUATION_ORDER = (number,)
    try:
        return sc.run_gates(ctx)[number]
    finally:
        sc.EVALUATION_ORDER = saved


def context(root: Path, sha: str = SHA) -> Any:
    root.mkdir(parents=True, exist_ok=True)
    logs = root / ".ship-logs"
    logs.mkdir(exist_ok=True)
    return sc.Context(repo=root, clone=root, logs=logs, sha=sha, fast=True, env=dict(os.environ))


def assert_caught(number: int, case: Case, base: Callable[[Path], Any], tmp: Path) -> None:
    """The control passes; the mutant fails with evidence matching ``case.expect``."""
    control = evaluate(number, base(tmp / "control"))
    if not control.passed:
        raise FixtureBroken(f"G{number} control fails: {control.evidence}")
    result = evaluate(number, case.build(tmp / "mutant"))
    assert not result.passed, f"G{number} mutation {case.name!r} survived: {result.evidence}"
    if not re.search(case.expect, result.evidence):
        raise FixtureBroken(
            f"G{number} mutation {case.name!r} failed for another reason: {result.evidence}"
        )


def run_git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
            "-C",
            str(root),
            *args,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, **_IDENTITY},
        check=True,
    )
    return proc.stdout.strip()


def serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        return "".join(json.dumps(v) + "\n" for v in value)
    return json.dumps(value, indent=1)


def write_files(root: Path, files: dict[str, Any]) -> None:
    for rel, value in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialize(value), encoding="utf-8")


def commit(root: Path, files: dict[str, Any], message: str) -> str:
    write_files(root, files)
    run_git(root, "add", "-A")
    run_git(root, "commit", "-q", "--allow-empty", "-m", message)
    return run_git(root, "rev-parse", "HEAD")


def edit_json(root: Path, rel: str, change: Callable[[Any], object]) -> None:
    """Apply ``change`` to the parsed file in place and write it back (return value ignored)."""
    path = root / rel
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mutated(base: Callable[[Path], Any], change: Callable[[Path], object]) -> Callable[[Path], Any]:
    def build(root: Path) -> Any:
        ctx = base(root)
        change(root)
        return ctx

    return build


# --------------------------------------------------------------------------- G5

INDIRECT = ("sed_inplace", "scripted_write", "git_checkout", "cp")
DV = "artifacts/e0/detector_validation.json"
PL = "artifacts/e0/probe_leakage.json"
CL = "artifacts/e0/compaction_logs.json"


def seeded_cases(tp: int, fp: int, fn: int, tn: int) -> list[dict[str, Any]]:
    kinds = [*INDIRECT, "direct_write"]
    rows: list[dict[str, Any]] = []
    for i in range(tp):
        rows.append({"case_id": f"tp{i}", "kind": kinds[i % 5], "label": True, "predicted": True})
    for i in range(fn):
        rows.append(
            {"case_id": f"fn{i}", "kind": "direct_write", "label": True, "predicted": False}
        )
    for i in range(fp):
        rows.append(
            {"case_id": f"fp{i}", "kind": "direct_write", "label": False, "predicted": True}
        )
    for i in range(tn):
        rows.append(
            {"case_id": f"tn{i}", "kind": "direct_write", "label": False, "predicted": False}
        )
    for row in rows:
        row["constraint_id"] = "protect-tests"
    return rows


def validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Banked numbers computed the way the gate reads the cases (synthetic)."""
    tp = sum(1 for c in rows if c["label"] and c["predicted"])
    fp = sum(1 for c in rows if not c["label"] and c["predicted"])
    fn = sum(1 for c in rows if c["label"] and not c["predicted"])

    def stat(k: int, n: int) -> dict[str, float]:
        lo, hi = sc.wilson(k, n)
        return {"value": k / n, "lo": lo, "hi": hi}

    return {
        "synthetic": True,
        "cases": rows,
        "precision": stat(tp, tp + fp),
        "recall": stat(tp, tp + fn),
    }


def build_g5(root: Path) -> Any:
    logs: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    for i in range(3):
        rel = f"artifacts/e0/compactions/r{i}/c0.json"
        logs[rel] = {"kept": ["system", "rules"], "dropped": ["turn 1"], "strategy": "full_summary"}
        events.append(
            {
                "run_id": f"r{i}",
                "event_id": "c0",
                "strategy": "full_summary",
                "tool_call_index": 12,
                "tokens_before": 3000,
                "tokens_after": 900,
                "log_path": rel,
            }
        )
    write_files(root, logs)
    for event in events:
        event["log_sha256"] = sha256(root / event["log_path"])
    write_files(
        root,
        {
            DV: validation(seeded_cases(60, 1, 1, 38)),
            PL: {
                "validation_runs": 3,
                "leaks": 0,
                "runs": [{"run_id": f"r{i}", "leaked": False} for i in range(3)],
            },
            CL: {"events": events},
        },
    )
    return context(root)


def g5(name: str, expect: str, change: Callable[[Path], object], hole: str = "") -> Case:
    return Case(name, expect, mutated(build_g5, change), hole)


def set_cases(rows: list[dict[str, Any]]) -> Callable[[Path], None]:
    return lambda root: write_files(root, {DV: validation(rows)})


def set_leakage(runs: list[dict[str, Any]], validation_runs: int, leaks: int) -> Any:
    data = {"validation_runs": validation_runs, "leaks": leaks, "runs": runs}
    return lambda root: write_files(root, {PL: data})


def first_event(root: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = json.loads((root / CL).read_text(encoding="utf-8"))["events"]
    return events[0]


def move_log_outside(root: Path, absolute: bool) -> None:
    outside = root.parent / "outside-the-clone"
    outside.mkdir(exist_ok=True)
    source = root / first_event(root)["log_path"]
    target = outside / "c0.json"
    target.write_bytes(source.read_bytes())
    source.unlink()
    path = str(target) if absolute else "../outside-the-clone/c0.json"
    edit_json(root, CL, lambda d: d["events"][0].update({"log_path": path}))


def empty_log(root: Path) -> None:
    log = root / first_event(root)["log_path"]
    log.write_text("", encoding="utf-8")
    edit_json(root, CL, lambda d: d["events"][0].update({"log_sha256": sha256(log)}))


def duplicate_true_positives(root: Path) -> None:
    rows = seeded_cases(45, 5, 0, 50)
    padded = rows + [dict(r) for r in rows if r["label"] and r["predicted"]] * 2
    write_files(root, {DV: validation(padded)})


def string_labels(root: Path) -> None:
    rows = seeded_cases(55, 0, 0, 40)
    rows += [
        {"case_id": f"neg{i}", "kind": "direct_write", "label": "false", "predicted": True}
        for i in range(5)
    ]
    write_files(root, {DV: validation(rows)})


def without_cp() -> list[dict[str, Any]]:
    return [
        {**c, "kind": "direct_write"} if c["kind"] == "cp" else c
        for c in seeded_cases(60, 1, 1, 38)
    ]


def cp_only_negative() -> list[dict[str, Any]]:
    rows = [c for c in seeded_cases(60, 1, 1, 38) if c["kind"] != "cp"]
    negative = {"case_id": "cp-neg", "kind": "cp", "label": False, "predicted": False}
    return [*rows, {**negative, "constraint_id": "protect-tests"}]


LEAK = [{"run_id": "r0", "leaked": True}]

G5_CASES = params(
    [
        g5(
            "precision-below-threshold",
            r"^precision .* < 0\.95",
            set_cases(seeded_cases(60, 6, 1, 33)),
        ),
        g5("recall-below-threshold", r"^recall .* < 0\.95", set_cases(seeded_cases(60, 1, 6, 33))),
        g5(
            "banked-precision-tampered",
            "banked precision does not match",
            lambda r: edit_json(r, DV, lambda d: d["precision"].update({"value": 0.999})),
        ),
        g5(
            "banked-interval-tampered",
            "banked recall does not match",
            lambda r: edit_json(r, DV, lambda d: d["recall"].update({"lo": 0.99})),
        ),
        g5("indirect-kind-cp-missing", "lack indirect kinds: cp", set_cases(without_cp())),
        g5(
            "indirect-kind-only-on-negative-cases",
            "lack indirect kinds: cp",
            set_cases(cp_only_negative()),
        ),
        g5("probe-leak", r"probe leakage 1/1", set_leakage(LEAK, 1, 1)),
        g5("leak-count-understated", "inconsistent or empty", set_leakage(LEAK, 1, 0)),
        g5(
            "run-count-mismatch",
            "inconsistent or empty",
            lambda r: edit_json(r, PL, lambda d: d.update({"validation_runs": 4})),
        ),
        g5("no-validation-runs", "inconsistent or empty", set_leakage([], 0, 0)),
        g5(
            "no-compaction-events",
            "no compaction events banked",
            lambda r: write_files(r, {CL: {"events": []}}),
        ),
        g5(
            "compaction-log-missing",
            "missing or hash mismatch",
            lambda r: (r / first_event(r)["log_path"]).unlink(),
        ),
        g5(
            "compaction-log-edited",
            "missing or hash mismatch",
            lambda r: (r / first_event(r)["log_path"]).write_text("{}", encoding="utf-8"),
        ),
        g5(
            "compaction-event-field-missing",
            "missing tokens_after",
            lambda r: edit_json(r, CL, lambda d: d["events"][0].pop("tokens_after")),
        ),
        g5(
            "probe-leakage-artifact-missing",
            "missing artifacts/e0/probe_leakage.json",
            lambda r: (r / PL).unlink(),
        ),
        g5(
            "duplicated-case-rows-inflate-precision",
            "",
            duplicate_true_positives,
            hole="G5 counts case rows, not distinct case_ids: 45 TP and 5 FP (precision 0.90) "
            "pass once the TP rows are repeated, since check_detectors never checks that "
            "case_id is unique",
        ),
        g5(
            "non-boolean-labels-count-as-violations",
            "",
            string_labels,
            hole="G5 reads label and predicted by truthiness, so label 'false' (a string) "
            "counts as a violation: 5 false positives become true positives and precision "
            "0.92 passes as 1.0",
        ),
        g5(
            "unknown-leak-status-counts-as-no-leak",
            "",
            set_leakage([{"run_id": "r0", "leaked": None}], 1, 0),
            hole="check_leakage treats leaked: null (leakage never checked) as no leak; it "
            "should require a boolean per run",
        ),
        g5(
            "compaction-log-outside-the-clone-absolute",
            "",
            lambda r: move_log_outside(r, absolute=True),
            hole="check_compaction_logs joins log_path to the clone without confining it, so "
            "an absolute path (into the original checkout or runs/, say) passes although the "
            "log is not banked",
        ),
        g5(
            "compaction-log-outside-the-clone-dotdot",
            "",
            lambda r: move_log_outside(r, absolute=False),
            hole="a log_path with '..' segments escapes the clone and still passes; paths "
            "should be repository-relative, normalized, and committed",
        ),
        g5(
            "empty-compaction-log",
            "",
            empty_log,
            hole="an empty compaction log with a matching SHA-256 passes: completeness is "
            "checked as presence plus hash, not that the log records what compaction kept",
        ),
    ]
)


def test_g5_control_passes(tmp_path: Path) -> None:
    result = evaluate(5, build_g5(tmp_path))
    assert result.passed, result.evidence
    assert "leakage 0/3" in result.evidence and "3 compaction logs complete" in result.evidence


@pytest.mark.parametrize("case", G5_CASES)
def test_g5_mutation_fails(case: Case, tmp_path: Path) -> None:
    assert_caught(5, case, build_g5, tmp_path)


# --------------------------------------------------------------------------- G6

TYPES = ("prohibition", "invariant", "procedure", "resource_limit")
MANIFEST = "artifacts/bench/manifest.json"
TEMPLATES = "src/halflife/bench/templates"


def template_text(index: int, kind: str) -> str:
    return (
        f"id: t{index:02d}\nsplit: main\nsynthetic: true\n"
        f"constraints:\n  - id: c{index}\n    type: {kind}\n"
    )


def bank_records(root: Path, tid: str, digest: str) -> tuple[str, str]:
    route = f"artifacts/bench/routes/{tid}.json"
    ref = f"artifacts/bench/reference/{tid}.json"
    common = {"template_id": tid, "template_sha256": digest}
    write_files(
        root,
        {
            route: {**common, "tasks_total": 6, "tasks_passed": 6, "violations": 0},
            ref: {**common, "completed": True, "synthetic": True},
        },
    )
    return route, ref


def build_g6(root: Path) -> Any:
    entries = []
    for i in range(24):
        tid, kind = f"t{i:02d}", TYPES[i % 4]
        rel = f"{TEMPLATES}/{tid}.yaml"
        write_files(root, {rel: template_text(i, kind)})
        digest = sha256(root / rel)
        route, ref = bank_records(root, tid, digest)
        entries.append(
            {
                "id": tid,
                "path": rel,
                "sha256": digest,
                "constraint_types": [kind],
                "compliant_route": route,
                "reference_run": ref,
            }
        )
    write_files(root, {MANIFEST: {"templates": entries}})
    return context(root)


def g6(name: str, expect: str, change: Callable[[Path], object], hole: str = "") -> Case:
    return Case(name, expect, mutated(build_g6, change), hole)


def edit_entry(index: int, **changes: Any) -> Callable[[Path], None]:
    return lambda root: edit_json(root, MANIFEST, lambda d: d["templates"][index].update(changes))


def edit_record(kind: str, index: int, **changes: Any) -> Callable[[Path], None]:
    rel = f"artifacts/bench/{kind}/t{index:02d}.json"
    return lambda root: edit_json(root, rel, lambda d: d.update(changes))


def drop_resource_limit(data: dict[str, Any]) -> None:
    for entry in data["templates"]:
        if entry["constraint_types"] == ["resource_limit"]:
            entry["constraint_types"] = ["prohibition"]


def same_template_twice(root: Path) -> None:
    data = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    entries = data["templates"]
    for i in range(12, 24):
        source, entry = entries[i - 12], entries[i]
        bank_records(root, entry["id"], source["sha256"])
        (root / entry["path"]).unlink()
        entry.update(
            path=source["path"],
            sha256=source["sha256"],
            constraint_types=source["constraint_types"],
        )
    write_files(root, {MANIFEST: data})


def types_only_declared(root: Path) -> None:
    data = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    for i, entry in enumerate(data["templates"]):
        path = root / entry["path"]
        path.write_text(template_text(i, "prohibition"), encoding="utf-8")
        entry["sha256"] = sha256(path)
        bank_records(root, entry["id"], entry["sha256"])
    write_files(root, {MANIFEST: data})


def template_outside_the_clone(root: Path) -> None:
    outside = root.parent / "outside-the-clone" / "t00.yaml"
    outside.parent.mkdir(exist_ok=True)
    source = root / f"{TEMPLATES}/t00.yaml"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    edit_json(root, MANIFEST, lambda d: d["templates"][0].update({"path": str(outside)}))


def route_is_reference(root: Path) -> None:
    route = "artifacts/bench/routes/t00.json"
    edit_json(root, route, lambda r: r.update({"completed": True}))
    (root / "artifacts/bench/reference/t00.json").unlink()
    edit_json(root, MANIFEST, lambda d: d["templates"][0].update({"reference_run": route}))


G6_CASES = params(
    [
        g6(
            "23-templates",
            "23 templates < 24",
            lambda r: edit_json(r, MANIFEST, lambda d: d["templates"].pop()),
        ),
        g6("duplicate-template-id", "duplicate template ids", edit_entry(1, id="t00")),
        g6(
            "three-constraint-types",
            "3 constraint types < 4",
            lambda r: edit_json(r, MANIFEST, drop_resource_limit),
        ),
        g6(
            "unknown-constraint-type", "unknown or empty", edit_entry(0, constraint_types=["style"])
        ),
        g6("empty-constraint-types", "unknown or empty", edit_entry(0, constraint_types=[])),
        g6(
            "template-edited-after-banking",
            "file missing or hash mismatch",
            lambda r: (r / f"{TEMPLATES}/t00.yaml").write_text("edited\n", encoding="utf-8"),
        ),
        g6(
            "template-file-missing",
            "file missing or hash mismatch",
            lambda r: (r / f"{TEMPLATES}/t00.yaml").unlink(),
        ),
        g6("route-has-a-violation", "route not verified", edit_record("routes", 0, violations=1)),
        g6("route-fails-a-task", "route not verified", edit_record("routes", 0, tasks_passed=5)),
        g6(
            "route-has-no-tasks",
            "route not verified",
            edit_record("routes", 0, tasks_total=0, tasks_passed=0),
        ),
        g6(
            "route-for-another-template-version",
            "compliant route is for a different template version",
            edit_record("routes", 0, template_sha256="0" * 64),
        ),
        g6(
            "reference-run-for-another-template",
            "reference run is for a different template version",
            edit_record("reference", 0, template_id="t01"),
        ),
        g6(
            "reference-run-incomplete",
            "reference run incomplete",
            edit_record("reference", 0, completed=False),
        ),
        g6(
            "reference-run-missing",
            "not banked",
            lambda r: (r / "artifacts/bench/reference/t00.json").unlink(),
        ),
        g6(
            "manifest-missing",
            "missing artifacts/bench/manifest.json",
            lambda r: (r / MANIFEST).unlink(),
        ),
        g6(
            "same-template-counted-twice",
            "",
            same_template_twice,
            hole="G6 counts manifest ids, not distinct templates: 12 template files listed "
            "twice under 24 ids pass, since neither path nor sha256 must be unique",
        ),
        g6(
            "constraint-types-only-declared-in-the-manifest",
            "",
            types_only_declared,
            hole="constraint_types come from the manifest, not the template files: every "
            "template holds only a prohibition, yet the four declared types pass",
        ),
        g6(
            "template-outside-the-clone",
            "",
            template_outside_the_clone,
            hole="a template path outside the clone (absolute, or with '..') passes, so the "
            "benchmark file need not be committed",
        ),
        g6(
            "route-record-reused-as-reference-run",
            "",
            route_is_reference,
            hole="compliant_route and reference_run may name the same record, so no separate "
            "reference run is needed (minor)",
        ),
    ]
)


def test_g6_control_passes(tmp_path: Path) -> None:
    result = evaluate(6, build_g6(tmp_path))
    assert result.passed, result.evidence
    assert result.evidence.startswith("24 templates, 4 constraint types")


@pytest.mark.parametrize("case", G6_CASES)
def test_g6_mutation_fails(case: Case, tmp_path: Path) -> None:
    assert_caught(6, case, build_g6, tmp_path)


# --------------------------------------------------------------------------- G7

PREREG = "prereg/0001-e1.md"
FAMILIES = ("qwen2.5", "llama3.2")
E1 = "artifacts/e1"


def e1_files(prereg: str = PREREG) -> dict[str, Any]:
    sessions = []
    opportunities = []
    for family in FAMILIES:
        for i in range(42):
            sid = f"{family}-{i:03d}"
            sessions.append(
                {
                    "session_id": sid,
                    "family": family,
                    "model": f"{family}-instruct",
                    "prereg": prereg,
                    "confirmatory": True,
                    "reached_first_dp": i < 40,
                }
            )
            if i < 40:
                opportunities.append(
                    {
                        "session_id": sid,
                        "family": family,
                        "constraint_type": "prohibition",
                        "position": 30 + i,
                        "compaction_count": i % 3,
                        "violated": i % 4 == 0,
                    }
                )
    hazard = [{"family": f, "estimate": 0.02, "lo": 0.01, "hi": 0.04} for f in FAMILIES]
    half_life = [
        {"family": FAMILIES[0], "half_life": 35.0, "lo": 22.0, "hi": 61.0},
        {"family": FAMILIES[1], "half_life": None, "not_reached_within": 150},
    ]
    curves = [{"family": f, "points": [[0, 1.0], [50, 0.7]]} for f in FAMILIES]
    provenance = {
        "prereg": prereg,
        "confirmatory": True,
        "command": "python -m halflife.experiment run",
        "git_sha": SHA,
    }
    return {
        f"{E1}/provenance.json": provenance,
        f"{E1}/sessions.jsonl": sessions,
        f"{E1}/opportunities.jsonl": opportunities,
        f"{E1}/hazard.json": {"estimates": hazard},
        f"{E1}/half_life.json": {"estimates": half_life},
        f"{E1}/km.json": {"descriptive": True, "curves": curves},
    }


Edit = Callable[[dict[str, Any]], object]


def build_g7(
    root: Path, edit: Edit | None = None, history: str = "prereg-first", prereg: str = PREREG
) -> Any:
    root.mkdir(parents=True, exist_ok=True)
    run_git(root, "init", "-q")
    commit(root, {"README.md": "# half-life\n", "prereg/.gitkeep": ""}, "bootstrap")
    files = e1_files(prereg)
    if edit is not None:
        edit(files)
    frozen = {PREREG: "# E1 preregistration\n\nDate: 2026-11-04\n"}
    if history == "prereg-first":
        commit(root, frozen, "freeze E1")
        sha = commit(root, files, "bank E1")
    elif history == "same-commit":
        sha = commit(root, {**frozen, **files}, "freeze and bank together")
    elif history == "artifacts-first":
        commit(root, files, "bank E1")
        sha = commit(root, frozen, "freeze E1 afterwards")
    elif history == "opportunities-first":
        early = {f"{E1}/opportunities.jsonl": files.pop(f"{E1}/opportunities.jsonl")}
        commit(root, early, "opportunities before the preregistration")
        commit(root, frozen, "freeze E1")
        sha = commit(root, files, "bank the rest")
    else:  # pragma: no cover - a typo in a case must not pass silently
        raise ValueError(history)
    return context(root, sha)


def g7(
    name: str,
    expect: str,
    edit: Edit | None = None,
    *,
    history: str = "prereg-first",
    prereg: str = PREREG,
    hole: str = "",
) -> Case:
    return Case(name, expect, lambda root: build_g7(root, edit, history, prereg), hole)


def table(files: dict[str, Any], name: str) -> list[dict[str, Any]]:
    value: list[dict[str, Any]] = files[f"{E1}/{name}"]
    return value


def estimate(files: dict[str, Any], name: str, index: int) -> dict[str, Any]:
    value: dict[str, Any] = files[f"{E1}/{name}"]["estimates"][index]
    return value


def family_short(files: dict[str, Any]) -> None:
    table(files, "sessions.jsonl")[0]["reached_first_dp"] = False


def one_family(files: dict[str, Any]) -> None:
    for name in ("sessions.jsonl", "opportunities.jsonl"):
        files[f"{E1}/{name}"] = [r for r in table(files, name) if r["family"] == FAMILIES[0]]


def duplicate_sessions(files: dict[str, Any]) -> None:
    sessions = table(files, "sessions.jsonl")
    first = [r for r in sessions if r["family"] == FAMILIES[0]]
    second = [r for r in sessions if r["family"] == FAMILIES[1]][:20]
    files[f"{E1}/sessions.jsonl"] = first + second + copy.deepcopy(second)
    keep = {r["session_id"] for r in first + second}
    opportunities = table(files, "opportunities.jsonl")
    files[f"{E1}/opportunities.jsonl"] = [o for o in opportunities if o["session_id"] in keep]


def reached_without_opportunities(files: dict[str, Any]) -> None:
    opportunities = table(files, "opportunities.jsonl")
    files[f"{E1}/opportunities.jsonl"] = [
        o for o in opportunities if o["session_id"].endswith(("000", "001"))
    ]


def open_upper_bound(files: dict[str, Any]) -> None:
    estimate(files, "half_life.json", 0).update({"hi": None, "hi_not_reached_within": 150})


def set_estimate(name: str, index: int, **changes: Any) -> Edit:
    return lambda files: estimate(files, name, index).update(changes)


HL_INVALID = r"half_life\.json: estimate for .* lacks a valid interval"

G7_CASES = params(
    [
        g7("prereg-committed-with-the-artifacts", "does not predate", history="same-commit"),
        g7("prereg-committed-after-the-artifacts", "does not predate", history="artifacts-first"),
        g7(
            "cites-a-missing-preregistration",
            "cites missing preregistration",
            prereg="prereg/0002-x.md",
        ),
        g7("cites-a-file-outside-prereg", "cites missing preregistration", prereg="docs/PLAN.md"),
        g7("family-with-39-sessions-reaching-a-dp", "families below 40 sessions", family_short),
        g7("one-model-family", "1 model families < 2", one_family),
        g7(
            "session-not-confirmatory",
            "not tagged with the E1 preregistration",
            lambda f: table(f, "sessions.jsonl")[3].update({"confirmatory": False}),
        ),
        g7(
            "session-cites-another-preregistration",
            "not tagged with the E1 preregistration",
            lambda f: table(f, "sessions.jsonl")[3].update({"prereg": "prereg/0002-x.md"}),
        ),
        g7(
            "opportunity-for-an-unknown-session",
            "references unknown sessions",
            lambda f: table(f, "opportunities.jsonl")[0].update({"session_id": "ghost"}),
        ),
        g7(
            "no-opportunities",
            "opportunities.jsonl empty",
            lambda f: f.update({f"{E1}/opportunities.jsonl": ""}),
        ),
        g7(
            "hazard-missing-a-family",
            "hazard.json lacks estimates",
            lambda f: f[f"{E1}/hazard.json"]["estimates"].pop(),
        ),
        g7(
            "hazard-interval-excludes-the-estimate",
            r"hazard\.json: estimate for qwen2\.5 lacks a valid interval",
            set_estimate("hazard.json", 0, lo=0.05),
        ),
        g7(
            "hazard-not-finite",
            r"hazard\.json: estimate for qwen2\.5 lacks a valid interval",
            set_estimate("hazard.json", 0, hi=float("inf")),
        ),
        g7(
            "half-life-interval-excludes-the-estimate",
            HL_INVALID,
            set_estimate("half_life.json", 0, hi=30.0),
        ),
        g7(
            "half-life-not-reached-without-a-cap",
            HL_INVALID,
            lambda f: estimate(f, "half_life.json", 1).pop("not_reached_within"),
        ),
        g7(
            "not-reached-within-zero",
            HL_INVALID,
            set_estimate("half_life.json", 1, not_reached_within=0),
        ),
        g7(
            "not-reached-within-is-a-boolean",
            HL_INVALID,
            set_estimate("half_life.json", 1, not_reached_within=True),
        ),
        g7(
            "open-bound-below-the-estimate",
            HL_INVALID,
            set_estimate("half_life.json", 0, hi=None, hi_not_reached_within=30),
        ),
        g7(
            "open-bound-with-lower-limit-above-the-estimate",
            HL_INVALID,
            set_estimate("half_life.json", 0, hi=None, hi_not_reached_within=150, lo=40.0),
        ),
        g7(
            "km-not-labeled-descriptive",
            "not labeled descriptive",
            lambda f: f[f"{E1}/km.json"].update({"descriptive": False}),
        ),
        g7("km-missing", "missing artifacts/e1/km.json", lambda f: f.pop(f"{E1}/km.json")),
        g7(
            "duplicated-session-rows-pad-a-family",
            "",
            duplicate_sessions,
            hole="G7 counts session rows, not distinct session_ids: 20 sessions listed twice "
            "count as 40 that reached the first decision point",
        ),
        g7(
            "reached-first-dp-without-any-opportunity",
            "",
            reached_without_opportunities,
            hole="reached_first_dp is not cross-checked with opportunities.jsonl: 80 sessions "
            "claim to reach a decision point while only 4 opportunity rows exist",
        ),
        g7(
            "prereg-path-escapes-with-dotdot",
            "",
            prereg="prereg/../README.md",
            hole="check_prereg_link accepts 'prereg/../README.md' (git normalizes it to "
            "README.md, committed at bootstrap); make prereg-verify (G11) rejects it",
        ),
        g7(
            "prereg-is-the-gitkeep",
            "",
            prereg="prereg/.gitkeep",
            hole="check_prereg_link accepts any file under prereg/, including the bootstrap "
            ".gitkeep; make prereg-verify (G11) rejects it",
        ),
        g7(
            "opportunities-committed-before-the-prereg",
            "",
            history="opportunities-first",
            hole="only sessions.jsonl is checked against the preregistration, so "
            "opportunities.jsonl (the outcome data) may predate it; make prereg-verify (G11) "
            "checks every file the E1 provenance governs",
        ),
    ]
)


def test_g7_control_passes(tmp_path: Path) -> None:
    result = evaluate(7, build_g7(tmp_path))
    assert result.passed, result.evidence
    assert "llama3.2=40, qwen2.5=40" in result.evidence


def test_g7_open_upper_bound_passes(tmp_path: Path) -> None:
    assert evaluate(7, build_g7(tmp_path, open_upper_bound)).passed


@pytest.mark.parametrize("case", G7_CASES)
def test_g7_mutation_fails(case: Case, tmp_path: Path) -> None:
    assert_caught(7, case, build_g7, tmp_path)


@pytest.mark.parametrize(
    ("history", "prereg", "message"),
    [
        ("prereg-first", "prereg/../README.md", "is not a prereg/NNNN-slug.md path"),
        ("prereg-first", "prereg/.gitkeep", "is not a prereg/NNNN-slug.md path"),
        ("opportunities-first", PREREG, "artifacts/e1/opportunities.jsonl was first committed"),
        ("same-commit", PREREG, "does not come after prereg/0001-e1.md"),
    ],
)
def test_g11_catches_g7_mutations(tmp_path: Path, history: str, prereg: str, message: str) -> None:
    build_g7(tmp_path, history=history, prereg=prereg)
    problems = [line for line in prereg_tool.verify(tmp_path).lines if line.startswith("FAIL")]
    assert any(message in line for line in problems), problems


def test_g11_passes_the_g7_control(tmp_path: Path) -> None:
    build_g7(tmp_path)
    assert prereg_tool.verify(tmp_path).lines[-1] == "PREREG-VERIFY: PASS"


# --------------------------------------------------------------------------- G8

REPORT_ARMS = (
    "# Report\n\n## Results\n\nE2 is the ecological arm; E3 varies placement. "
    "Both are reported, null results included.\n"
)
E2 = "artifacts/e2/summary.json"


def build_g8(root: Path) -> Any:
    sessions = [
        {"session_id": f"e2-{i}", "model": "haiku" if i < 3 else "sonnet"} for i in range(6)
    ]
    summary = {
        "arm": "E2",
        "ran": True,
        "n_sessions": 6,
        "label": "ecological",
        "models": ["haiku", "sonnet"],
    }
    write_files(
        root,
        {
            E2: summary,
            "artifacts/e2/sessions.jsonl": sessions,
            "artifacts/e3/summary.json": {"arm": "E3", "ran": True, "n_sessions": 8},
            "docs/REPORT.md": REPORT_ARMS,
        },
    )
    return context(root)


def g8(name: str, expect: str, change: Callable[[Path], object], hole: str = "") -> Case:
    return Case(name, expect, mutated(build_g8, change), hole)


def edit_e2(**changes: Any) -> Callable[[Path], None]:
    return lambda root: edit_json(root, E2, lambda d: d.update(changes))


def report(text: str) -> Callable[[Path], None]:
    return lambda root: write_files(root, {"docs/REPORT.md": text})


G8_CASES = params(
    [
        g8("e2-did-not-run", "E2 summary does not show a completed run", edit_e2(ran=False)),
        g8("e2-has-no-sessions", "E2 summary does not show a completed run", edit_e2(n_sessions=0)),
        g8("e2-summary-names-another-arm", "E2 summary does not show", edit_e2(arm="E3")),
        g8("e2-not-labeled-ecological", "E2 not labeled ecological", edit_e2(label="main")),
        g8("e2-one-model-tier", "fewer than 2 model tiers", edit_e2(models=["haiku"])),
        g8("e2-same-tier-twice", "fewer than 2 model tiers", edit_e2(models=["haiku", "haiku"])),
        g8(
            "e3-summary-missing",
            "missing artifacts/e3/summary.json",
            lambda r: (r / "artifacts/e3/summary.json").unlink(),
        ),
        g8("report-omits-ecological", "does not report ecological$", report("# R\n\nE2 and E3.\n")),
        g8(
            "report-missing",
            "does not report E2, E3, ecological",
            lambda r: (r / "docs/REPORT.md").unlink(),
        ),
        g8(
            "report-mentions-the-arms-only-inside-other-words",
            "",
            report("# Report\n\nTemplates E20 and E31 are non-ecological.\n"),
            hole="report_mentions is a substring test, so 'E20', 'E31', and 'non-ecological' "
            "satisfy 'E2', 'E3', and 'ecological' without the arms being reported",
        ),
        g8(
            "e2-summary-without-banked-sessions",
            "",
            lambda r: (r / "artifacts/e2/sessions.jsonl").unlink(),
            hole="G8 trusts summary.json alone: with no E2 session data banked, a summary "
            "claiming ran: true and n_sessions: 6 passes, and n_sessions is never compared "
            "with banked data",
        ),
    ]
)


def test_g8_control_passes(tmp_path: Path) -> None:
    result = evaluate(8, build_g8(tmp_path))
    assert result.passed, result.evidence


@pytest.mark.parametrize("case", G8_CASES)
def test_g8_mutation_fails(case: Case, tmp_path: Path) -> None:
    assert_caught(8, case, build_g8, tmp_path)


# --------------------------------------------------------------------------- G9

PYPROJECT = (
    '[project]\nname = "halflife"\nversion = "0.0.1"\n\n[project.scripts]\n'
    'halflife-guard = "halflife.kit.guard:main"\nhalflife-pin = "halflife.kit.pin:main"\n'
)
MITIGATIONS = (
    "# Mitigations\n\n## Install\n\nInstall halflife-guard as a PreToolUse hook and run "
    "halflife-pin to write CLAUDE.md.\n"
)
E4 = "artifacts/e4/summary.json"
OVERHEAD = "artifacts/e4/overhead.json"


def build_g9(root: Path) -> Any:
    cost = {"tokens": 1200, "latency_s": 0.4, "task_success": 0.83}
    summary = {
        "arm": "E4",
        "ran": True,
        "n_sessions": 12,
        "held_out_templates": ["h01", "h02"],
        "mitigations": [{"name": "guard", "cost": cost}, {"name": "pin", "cost": dict(cost)}],
    }
    overhead = {
        "halflife-guard": {"latency_ms": 3.5, "tokens": 0},
        "halflife-pin": {"latency_ms": 0.0, "tokens": 240},
    }
    write_files(
        root,
        {
            E4: summary,
            OVERHEAD: overhead,
            "artifacts/e1/sessions.jsonl": [{"session_id": "s1", "template_id": "t01"}],
            "pyproject.toml": PYPROJECT,
            "docs/MITIGATIONS.md": MITIGATIONS,
        },
    )
    return context(root)


def g9(name: str, expect: str, change: Callable[[Path], object], hole: str = "") -> Case:
    return Case(name, expect, mutated(build_g9, change), hole)


def edit_e4(change: Callable[[dict[str, Any]], object]) -> Callable[[Path], None]:
    return lambda root: edit_json(root, E4, change)


def edit_overhead(change: Callable[[dict[str, Any]], object]) -> Callable[[Path], None]:
    return lambda root: edit_json(root, OVERHEAD, change)


def mitigations_doc(old: str, new: str) -> Callable[[Path], None]:
    return lambda root: write_files(root, {"docs/MITIGATIONS.md": MITIGATIONS.replace(old, new)})


NULL_COST = {"tokens": None, "latency_s": None, "task_success": None}

G9_CASES = params(
    [
        g9("e4-did-not-run", "E4 summary does not show", edit_e4(lambda d: d.update(ran=False))),
        g9(
            "no-held-out-templates",
            "does not list held-out templates",
            edit_e4(lambda d: d.update(held_out_templates=[])),
        ),
        g9(
            "mitigation-without-token-cost",
            "mitigation guard lacks cost beside benefit",
            edit_e4(lambda d: d["mitigations"][0]["cost"].pop("tokens")),
        ),
        g9("no-mitigations", "reports no mitigations", edit_e4(lambda d: d.update(mitigations=[]))),
        g9(
            "pin-not-packaged",
            "not packaged as console scripts: halflife-pin",
            lambda r: write_files(
                r, {"pyproject.toml": PYPROJECT.replace("halflife-pin =", "x =")}
            ),
        ),
        g9("no-install-heading", "lacks install docs", mitigations_doc("## Install", "## Usage")),
        g9(
            "install-docs-omit-the-guard",
            "lacks install docs",
            mitigations_doc("halflife-guard", "it"),
        ),
        g9(
            "mitigations-doc-missing",
            "lacks install docs",
            lambda r: (r / "docs/MITIGATIONS.md").unlink(),
        ),
        g9(
            "overhead-missing",
            "missing artifacts/e4/overhead.json",
            lambda r: (r / OVERHEAD).unlink(),
        ),
        g9(
            "overhead-missing-for-pin",
            "overhead for halflife-pin not measured",
            edit_overhead(lambda d: d.pop("halflife-pin")),
        ),
        g9(
            "overhead-latency-not-a-number",
            "overhead for halflife-guard not measured",
            edit_overhead(lambda d: d["halflife-guard"].update(latency_ms="fast")),
        ),
        g9(
            "cost-fields-are-null",
            "",
            edit_e4(lambda d: d["mitigations"][0].update(cost=NULL_COST)),
            hole="G9 checks that the cost keys exist, not that they hold numbers: tokens, "
            "latency_s, and task_success all null pass (rule 15, costs beside benefits)",
        ),
        g9(
            "held-out-templates-were-used-in-e1",
            "",
            edit_e4(lambda d: d.update(held_out_templates=["t01"])),
            hole="held_out_templates only has to be non-empty; it is not checked against the "
            "bench split or the E1 sessions, so a template used in E1 passes as held out",
        ),
        g9(
            "uninstall-heading-counts-as-install",
            "",
            mitigations_doc("## Install", "## Uninstall"),
            hole="the install-heading regex matches 'Uninstall' (minor)",
        ),
        g9(
            "overhead-latency-is-nan",
            "",
            edit_overhead(lambda d: d["halflife-guard"].update(latency_ms=float("nan"))),
            hole="overhead values are only type-checked: NaN, negative numbers, and booleans "
            "pass as measured latency",
        ),
    ]
)


def test_g9_control_passes(tmp_path: Path) -> None:
    result = evaluate(9, build_g9(tmp_path))
    assert result.passed, result.evidence


@pytest.mark.parametrize("case", G9_CASES)
def test_g9_mutation_fails(case: Case, tmp_path: Path) -> None:
    assert_caught(9, case, build_g9, tmp_path)


def wheel(dist: Path, entry_points: str | None) -> None:
    dist.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dist / "halflife-0.0.1-py3-none-any.whl", "w") as archive:
        archive.writestr("halflife/__init__.py", "")
        if entry_points is not None:
            archive.writestr("halflife-0.0.1.dist-info/entry_points.txt", entry_points)


TOOLS = ("halflife-guard", "halflife-pin")


def test_wheel_entry_points_pass_and_fail(tmp_path: Path) -> None:
    wheel(tmp_path / "ok", "[console_scripts]\nhalflife-guard = a:b\nhalflife-pin = a:c\n")
    assert sc.entry_points_ok(tmp_path / "ok", TOOLS)
    wheel(tmp_path / "no-pin", "[console_scripts]\nhalflife-guard = a:b\n")
    assert not sc.entry_points_ok(tmp_path / "no-pin", TOOLS)
    wheel(tmp_path / "none", None)
    assert not sc.entry_points_ok(tmp_path / "none", TOOLS)
    (tmp_path / "empty").mkdir()
    assert not sc.entry_points_ok(tmp_path / "empty", TOOLS)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="HOLE: entry_points_ok accepts the tools under any section (gui_scripts here), "
    "not only console_scripts (minor)",
)
def test_wheel_entry_points_must_be_console_scripts(tmp_path: Path) -> None:
    wheel(tmp_path, "[gui_scripts]\nhalflife-guard = a:b\nhalflife-pin = a:c\n")
    assert not sc.entry_points_ok(tmp_path, TOOLS)


# --------------------------------------------------------------------------- G12

REVIEW = "artifacts/review/latest.json"


def build_g12(
    root: Path, review: dict[str, Any] | None = None, after: dict[str, Any] | None = None
) -> Any:
    root.mkdir(parents=True, exist_ok=True)
    run_git(root, "init", "-q")
    code = {"src/halflife/x.py": "X = 1\n", "docs/REVIEW_LOG.md": "# Review log\n"}
    reviewed = commit(root, code, "code")
    record = {
        "reviewed_commit": reviewed,
        "reviewer": "reviewer-subagent",
        "timestamp": "2026-11-04T00:00:00Z",
        "verdict": "SHIP",
        "blocking": [],
        "non_blocking": ["docs could be shorter"],
        "final_line": "REVIEW: SHIP",
    }
    record.update(review or {})
    banked = {REVIEW: record, "STATUS.md": "goal_turn: 1\n", "docs/LOG.md": "log\n"}
    head = commit(root, banked, "bank review")
    if after:
        head = commit(root, after, "later change")
    return context(root, head)


def g12(
    name: str,
    expect: str,
    review: dict[str, Any] | None = None,
    *,
    after: dict[str, Any] | None = None,
    change: Callable[[Path], object] | None = None,
    hole: str = "",
) -> Case:
    def build(root: Path) -> Any:
        ctx = build_g12(root, review, after)
        if change is not None:
            change(root)
        return ctx

    return Case(name, expect, build, hole)


def review_of_a_side_branch(root: Path) -> None:
    run_git(root, "checkout", "-q", "-b", "side")
    side = commit(root, {"src/halflife/y.py": "Y = 1\n"}, "side")
    run_git(root, "checkout", "-q", "main")
    edit_json(root, REVIEW, lambda d: d.update({"reviewed_commit": side}))


def short_sha(root: Path) -> None:
    edit_json(root, REVIEW, lambda d: d.update({"reviewed_commit": d["reviewed_commit"][:12]}))


G12_CASES = params(
    [
        g12(
            "verdict-block",
            "ends with 'REVIEW: BLOCK'",
            {"verdict": "BLOCK", "final_line": "REVIEW: BLOCK"},
        ),
        g12(
            "final-line-disagrees-with-verdict",
            "ends with 'REVIEW: BLOCK'",
            {"final_line": "REVIEW: BLOCK"},
        ),
        g12("verdict-disagrees-with-final-line", "latest review ends with", {"verdict": "BLOCK"}),
        g12("open-blocking-issue", "1 open blocking issue", {"blocking": ["G5 recomputation"]}),
        g12("short-commit-sha", "does not record a full commit SHA", change=short_sha),
        g12("unknown-commit", "not an ancestor of HEAD", {"reviewed_commit": "f" * 40}),
        g12(
            "code-changed-since-review",
            "code changed since review: src/halflife/x.py",
            after={"src/halflife/x.py": "X = 2\n"},
        ),
        g12(
            "new-file-since-review",
            "code changed since review: tools/new.py",
            after={"tools/new.py": "print(1)\n"},
        ),
        g12(
            "review-log-missing",
            "missing docs/REVIEW_LOG.md",
            change=lambda r: (r / "docs/REVIEW_LOG.md").unlink(),
        ),
        g12("review-missing", "no review at", change=lambda r: (r / REVIEW).unlink()),
        g12(
            "reviewed-commit-not-an-ancestor",
            "not an ancestor of HEAD",
            change=review_of_a_side_branch,
        ),
        g12(
            "blocking-field-missing",
            "",
            change=lambda r: edit_json(r, REVIEW, lambda d: d.pop("blocking")),
            hole="a review with no 'blocking' field passes as zero blocking issues; the field "
            "should be required to be a list",
        ),
    ]
)


def test_g12_control_passes(tmp_path: Path) -> None:
    result = evaluate(12, build_g12(tmp_path))
    assert result.passed, result.evidence


def test_g12_review_of_head_and_exempt_changes_pass(tmp_path: Path) -> None:
    after = {"STATUS.md": "goal_turn: 2\n", "artifacts/review/notes.json": "{}"}
    assert evaluate(12, build_g12(tmp_path / "a", after=after)).passed
    root = tmp_path / "b"
    root.mkdir()
    run_git(root, "init", "-q")
    head = commit(root, {"docs/REVIEW_LOG.md": "# Review log\n"}, "base")
    record = {
        "reviewed_commit": head,
        "verdict": "SHIP",
        "blocking": [],
        "final_line": "REVIEW: SHIP",
    }
    write_files(root, {REVIEW: record})
    assert evaluate(12, context(root, head)).passed


@pytest.mark.parametrize("case", G12_CASES)
def test_g12_mutation_fails(case: Case, tmp_path: Path) -> None:
    assert_caught(12, case, build_g12, tmp_path)


# --------------------------------------------------------------------------- G13 helpers

README = """# half-life

The rules survive a median of 35 tool calls (95% confidence interval 22 to 61); synthetic.

![Headline figure](docs/figures/headline.svg)

| Family | Half-life (tool calls) | 95% interval |
|---|---|---|
| qwen2.5 | 35 | 22 to 61 |

## Quickstart

```bash
make demo
```

## What it is and is not

A measurement.

## Key results

Results.

## How it works

Pieces.

## Use it on your own benchmark or agent

Steps.

## Reproduce every figure

One command.

## Limitations and threats to validity

Some.

## Related work

Papers.

## Citation

Cite it.

## License

MIT.
"""

REPORT_HEADINGS = (
    "Abstract",
    "1. Introduction",
    "2. Related work",
    "3. Method",
    "4. Experiments",
    "5. Results",
    "6. Limitations",
    "7. Ethics and disclosure",
    "8. Reproducibility",
)


def report_text(words: int = 4500) -> str:
    per = words // len(REPORT_HEADINGS)
    parts = ["# Report", "![Survival curves](figures/headline.svg)"]
    for heading in REPORT_HEADINGS:
        parts.append(f"## {heading}")
        parts.append(" ".join(["measured"] * per))
    return "\n\n".join(parts) + "\n"


def g13_problems(root: Path, readme: str, report_md: str) -> list[str]:
    figure = root / "docs" / "figures" / "headline.svg"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_text("<svg/>", encoding="utf-8")
    problems: list[str] = sc.readme_problems(root, readme) + sc.report_problems(root, report_md)
    return problems


def same(text: str) -> str:
    return text


@dataclass(frozen=True)
class DocCase:
    """A G13 mutation of the README.md or docs/REPORT.md text."""

    name: str
    expect: str
    readme: Callable[[str], str] = same
    report: Callable[[str], str] = same
    hole: str = ""


def no_interval(text: str) -> str:
    return text.replace(" (95% confidence interval 22 to 61)", "").replace("95% interval", "range")


def quickstart_last(text: str) -> str:
    text = text.replace("## Quickstart\n\n```bash\nmake demo\n```\n\n", "")
    return text.replace("## License", "## Quickstart\n\nmake demo\n\n## License")


G13_CASES = params(
    [
        DocCase(
            "readme-is-still-the-brief",
            "still the build brief",
            lambda t: t + "This README is a build brief.\n",
        ),
        DocCase(
            "readme-opens-with-another-name",
            "does not open with the project name",
            lambda t: t.replace("# half-life", "# Project", 1),
        ),
        DocCase(
            "readme-lacks-limitations",
            "lacks a 'limitations' section",
            lambda t: t.replace("## Limitations and threats to validity", "## Caveats"),
        ),
        DocCase(
            "readme-lacks-quickstart",
            "lacks a quickstart",
            lambda t: t.replace("## Quickstart", "## Try"),
        ),
        DocCase(
            "readme-quickstart-below-the-first-screen",
            "quickstart is not on the README first screen",
            quickstart_last,
        ),
        DocCase(
            "readme-without-headline-svg",
            "lacks the headline SVG",
            lambda t: t.replace("![Headline figure](docs/figures/headline.svg)\n", ""),
        ),
        DocCase(
            "readme-headline-svg-missing",
            "absent.svg missing",
            lambda t: t.replace("headline.svg", "absent.svg"),
        ),
        DocCase(
            "readme-without-headline-table",
            "lacks the headline table",
            lambda t: t.replace("|---|---|---|\n", ""),
        ),
        DocCase("readme-first-screen-without-interval", "gives no interval", no_interval),
        DocCase(
            "readme-em-dash",
            "contains an em dash",
            lambda t: t.replace("; synthetic", " \N{EM DASH} synthetic"),
        ),
        DocCase(
            "readme-hype-word",
            "uses 'groundbreaking'",
            lambda t: t.replace("A measurement", "A groundbreaking one"),
        ),
        DocCase("report-too-short", "words, needs 4000-8000", report=lambda t: report_text(3000)),
        DocCase("report-too-long", "words, needs 4000-8000", report=lambda t: report_text(9000)),
        DocCase(
            "report-lacks-ethics",
            "lacks 'ethics and disclosure'",
            report=lambda t: t.replace("7. Ethics and disclosure", "7. Other"),
        ),
        DocCase(
            "report-has-todo", "has placeholders", report=lambda t: t + "\nTODO: discussion.\n"
        ),
        DocCase(
            "report-figure-missing",
            "figure figures/absent.svg missing",
            report=lambda t: t.replace("figures/headline.svg", "figures/absent.svg"),
        ),
        DocCase(
            "report-hype-phrase",
            "uses 'state-of-the-art'",
            report=lambda t: t + "\nIt is state-of-the-art.\n",
        ),
        DocCase(
            "readme-em-dash-as-html-entity",
            "",
            lambda t: t.replace("; synthetic", " &mdash; synthetic"),
            hole="style_problems looks for the U+2014 character only; '&mdash;' and '&#8212;' "
            "render as em dashes and pass (rule 12)",
        ),
        DocCase(
            "readme-hype-phrase-split-across-lines",
            "",
            lambda t: t.replace("A measurement.", "A state of the\nart measurement."),
            hole="hype phrases are matched as exact substrings, so 'state of the' plus a "
            "newline plus 'art' (one phrase when rendered) passes",
        ),
        DocCase(
            "readme-interval-check-met-by-make-ci",
            "",
            lambda t: no_interval(t).replace("make demo", "make ci"),
            hole="the first-screen interval check matches \\bCI\\b case-insensitively, so the "
            "command 'make ci' satisfies it with no interval anywhere on the first screen",
        ),
        DocCase(
            "report-pending-placeholder",
            "",
            report=lambda t: t.replace("## Abstract\n\n", "## Abstract\n\nPending results.\n\n"),
            hole="only TODO, TBD, and FIXME count as placeholders; 'Pending results.' (the "
            "wording of the current docs/REPORT.md draft) passes as complete",
        ),
    ]
)


def test_g13_control_has_no_problems(tmp_path: Path) -> None:
    assert g13_problems(tmp_path, README, report_text()) == []


@pytest.mark.parametrize("case", G13_CASES)
def test_g13_mutation_is_flagged(case: DocCase, tmp_path: Path) -> None:
    if g13_problems(tmp_path / "control", README, report_text()):
        raise FixtureBroken("G13 control has problems")
    problems = g13_problems(tmp_path / "mutant", case.readme(README), case.report(report_text()))
    assert problems, f"G13 mutation {case.name!r} survived"
    if not any(re.search(case.expect, p) for p in problems):
        raise FixtureBroken(f"G13 mutation {case.name!r} flagged for another reason: {problems}")


def write_docs(root: Path) -> Any:
    docs = {"README.md": README, "docs/REPORT.md": report_text(), "docs/figures/headline.svg": "x"}
    write_files(root, docs)
    return context(root)


def test_g13_gate_needs_citecheck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def citecheck_ok(ctx: Any, targets: list[str]) -> Any:
        calls.append(list(targets))
        return sc.GateResult(True, "make citecheck ok")

    monkeypatch.setattr(sc, "make_targets_pass", citecheck_ok)
    result = evaluate(13, write_docs(tmp_path / "ok"))
    assert result.passed and calls == [["citecheck"]]

    monkeypatch.setattr(sc, "make_targets_pass", lambda ctx, targets: sc.fail("citecheck rc=1"))
    result = evaluate(13, write_docs(tmp_path / "bad"))
    assert not result.passed and result.evidence == "citecheck rc=1"


def test_g13_gate_fails_without_the_report(tmp_path: Path) -> None:
    ctx = write_docs(tmp_path)
    (tmp_path / "docs" / "REPORT.md").unlink()
    result = evaluate(13, ctx)
    assert not result.passed and result.evidence == "missing docs/REPORT.md"


def test_g10_and_g11_fail_until_setup_succeeds(tmp_path: Path) -> None:
    ctx = write_docs(tmp_path)
    for number in (10, 11):
        result = evaluate(number, ctx)
        assert not result.passed and "make setup failed" in result.evidence


def test_g10_fails_without_the_documents(tmp_path: Path) -> None:
    result = evaluate(10, context(tmp_path))
    assert not result.passed and "missing README.md" in result.evidence
