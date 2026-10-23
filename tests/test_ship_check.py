"""Tests for the gate logic in tools/ship_check.py.

Each check is exercised with a passing input and with inputs where the guarded
property is false, so a gate that could pass on bad data shows up here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
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


def test_check_compaction_logs(tmp_path: Path) -> None:
    log = tmp_path / "artifacts" / "e0" / "c1.json"
    log.parent.mkdir(parents=True)
    log.write_text("{}", encoding="utf-8")
    event = {
        "run_id": "r",
        "event_id": "c1",
        "strategy": "full_summary",
        "tool_call_index": 12,
        "tokens_before": 4000,
        "tokens_after": 900,
        "log_path": "artifacts/e0/c1.json",
        "log_sha256": hashlib.sha256(b"{}").hexdigest(),
    }
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"events": [event]}), encoding="utf-8")
    assert sc.check_compaction_logs(path, tmp_path)[0]
    event["log_sha256"] = "0" * 64
    path.write_text(json.dumps({"events": [event]}), encoding="utf-8")
    assert not sc.check_compaction_logs(path, tmp_path)[0]
    del event["strategy"]
    path.write_text(json.dumps({"events": [event]}), encoding="utf-8")
    assert "missing strategy" in sc.check_compaction_logs(path, tmp_path)[1]
    path.write_text(json.dumps({"events": []}), encoding="utf-8")
    assert not sc.check_compaction_logs(path, tmp_path)[0]


# ------------------------------------------------------------- G7 and G12 helpers


def test_check_interval() -> None:
    assert sc.check_interval({"lo": 1, "estimate": 2, "hi": 3}, "estimate")
    assert not sc.check_interval({"lo": 2.5, "estimate": 2, "hi": 3}, "estimate")
    assert not sc.check_interval({"lo": 1, "estimate": float("nan"), "hi": 3}, "estimate")
    assert not sc.check_interval({"estimate": 2}, "estimate")


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

    same = tmp_path / "same"
    same.mkdir()
    _init_repo(same)
    (same / "prereg").mkdir()
    (same / "prereg" / "0001.md").write_text("plan", encoding="utf-8")
    (same / "data.jsonl").write_text("{}", encoding="utf-8")
    _git(same, "add", ".")
    _git(same, "commit", "-q", "-m", "both at once")
    assert "does not predate" in str(sc.check_prereg_link(same, "prereg/0001.md", "data.jsonl"))


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
