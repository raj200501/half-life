"""Tests for the E0 seeded case suite.

These tests check structure, coverage, schema validity, and determinism. They never
execute the cases (that needs the executor and sandbox); the owner runs them.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from halflife import e0_cases
from halflife.e0_cases import (
    DIRECT_KINDS,
    MAX_ACTIONS,
    MAX_FILES,
    MIN_FILES,
    MIN_INDIRECT_CASES,
    REQUIRED_INDIRECT_KINDS,
    TOOL_REQUIRED_ARGS,
    SeededCase,
    SeededStep,
    build_cases,
    cases_from_json,
    cases_to_json,
    summarize,
)
from halflife.schema import (
    AGENT_TOOLS,
    CONSTRAINT_TYPES,
    DETECTOR_KINDS,
    Action,
    Constraint,
    DetectorSpec,
    HarnessCheckSpec,
    SchemaError,
)

EM_DASH = chr(0x2014)


@pytest.fixture(scope="module")
def cases() -> list[SeededCase]:
    return build_cases()


def _by_id(cases: list[SeededCase], case_id: str) -> SeededCase:
    return next(c for c in cases if c.id == case_id)


def _commands(case: SeededCase) -> list[str]:
    return [str(a.args["cmd"]) for a in case.actions if a.tool == "run"]


def _simulate(case: SeededCase) -> tuple[dict[str, str | None], list[str]]:
    """Apply direct tools to the seed files; a ``run`` makes every file unknown (None).

    Returns the final known contents and a list of problems found on the way.
    """
    state: dict[str, str | None] = dict(case.files)
    unknown = False
    problems: list[str] = []
    for action in case.actions:
        path = str(action.args.get("path", ""))
        if action.tool == "run":
            unknown = True
            state = {k: None for k in state}
        elif action.tool == "write_file":
            state[path] = str(action.args["content"])
        elif action.tool in ("replace_in_file", "read_file"):
            current = state.get(path)
            if current is None:
                if not unknown:
                    problems.append(f"{action.tool} on missing file {path}")
                continue
            if action.tool == "replace_in_file":
                old = str(action.args["old"])
                if current.count(old) != 1:
                    problems.append(f"old text occurs {current.count(old)} times in {path}")
                else:
                    state[path] = current.replace(old, str(action.args["new"]))
    return state, problems


# --------------------------------------------------------------------------- coverage


def test_ids_unique_and_well_formed(cases: list[SeededCase]) -> None:
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", i) for i in ids)


def test_label_balance(cases: list[SeededCase]) -> None:
    positives = sum(1 for c in cases if c.label)
    negatives = sum(1 for c in cases if not c.label)
    assert positives >= 120
    assert negatives >= 120


def test_every_detector_kind_has_both_labels(cases: list[SeededCase]) -> None:
    counts: Counter[tuple[str, bool]] = Counter((c.family, c.label) for c in cases)
    for kind in DETECTOR_KINDS:
        assert counts[(kind, True)] >= 8, kind
        assert counts[(kind, False)] >= 8, kind


def test_every_constraint_type_has_both_labels(cases: list[SeededCase]) -> None:
    counts: Counter[tuple[str, bool]] = Counter((c.constraint.type, c.label) for c in cases)
    for ctype in CONSTRAINT_TYPES:
        assert counts[(ctype, True)] > 0, ctype
        assert counts[(ctype, False)] > 0, ctype


@pytest.mark.parametrize("kind", REQUIRED_INDIRECT_KINDS)
def test_required_indirect_kinds(cases: list[SeededCase], kind: str) -> None:
    rows = [c for c in cases if c.kind == kind]
    assert all(c.label for c in rows), "indirect kinds label seeded violations"
    assert all(c.family == "protected_paths" for c in rows)
    assert len(rows) >= MIN_INDIRECT_CASES
    commands = {cmd for c in rows for cmd in _commands(c)}
    assert len(commands) >= MIN_INDIRECT_CASES, "commands must vary"
    protected = {g for c in rows for g in c.constraint.detector.params["globs"]}
    assert len(protected) >= 2, "protected paths must vary"


def test_indirect_kinds_cover_requested_shapes(cases: list[SeededCase]) -> None:
    def joined(kind: str) -> str:
        return "\n".join(cmd for c in cases if c.kind == kind for cmd in _commands(c))

    sed = [cmd for c in cases if c.kind == "sed_inplace" for cmd in _commands(c)]
    assert all("sed" in cmd and ("-i" in cmd or "--in-place" in cmd) for cmd in sed)
    assert '"s/' in "\n".join(sed) and "'s/" in "\n".join(sed), "both quoting styles"
    scripted = joined("scripted_write")
    for shape in ("python3 -c", "python3 -", " > ", " >> ", "tee ", "printf "):
        assert shape in scripted, shape
    git = joined("git_checkout")
    for shape in (
        "git checkout -- ",
        "git checkout HEAD -- ",
        "git restore ",
        "git checkout wip -- ",
        "rev-list --max-parents=0",
    ):
        assert shape in git, shape
    cp = joined("cp")
    for shape in ("cp ", "mv ", "cp -r ", "cp -f "):
        assert shape in cp, shape


def test_direct_kinds_present(cases: list[SeededCase]) -> None:
    for kind in DIRECT_KINDS:
        rows = [c for c in cases if c.kind == kind]
        assert len(rows) >= 3, kind
        assert all(c.label and c.family == "protected_paths" for c in rows)


@pytest.mark.parametrize(
    ("kind", "label"),
    [
        ("net_curl", True),
        ("net_wget", True),
        ("net_python", True),
        ("net_pip", True),
        ("net_git", True),
        ("dep_requirements", True),
        ("dep_pyproject", True),
        ("dep_optional", True),
        ("api_rename", True),
        ("api_param_change", True),
        ("api_removal", True),
        ("api_method_change", True),
        ("forbidden_command", True),
        ("changelog_skipped", True),
        ("tests_skipped", True),
        ("too_many_new_files", True),
        ("file_too_long", True),
        ("too_many_test_runs", True),
        ("check_fails", True),
        ("invariant_broken", True),
        ("read_protected", False),
        ("similar_path", False),
        ("identical_content", False),
        ("echo_only", False),
        ("comment_only", False),
        ("new_public_function", False),
        ("reformat", False),
        ("allowed_runs", False),
        ("procedure_followed", False),
        ("at_limit", False),
        ("check_passes", False),
        ("check_restored", False),
        ("invariant_kept", False),
    ],
)
def test_case_shapes_present(cases: list[SeededCase], kind: str, label: bool) -> None:
    rows = [c for c in cases if c.kind == kind]
    assert rows, kind
    assert all(c.label is label for c in rows), kind


def test_specific_hard_negatives(cases: list[SeededCase]) -> None:
    similar = [c for c in cases if c.kind == "similar_path" and c.constraint.id == "protect-tests"]
    touched = {
        str(a.args.get("path", a.args.get("cmd", "")))
        for c in similar
        for a in c.actions
        if a.tool != "read_file"
    }
    assert any("tests_helpers/" in t for t in touched)
    assert any("src/test_utils.py" in t for t in touched)
    echo = [c for c in cases if c.family == "no_network" and c.kind == "echo_only"]
    assert echo and all(
        cmd.startswith("echo") and "curl" in cmd for c in echo for cmd in _commands(c)
    )
    assert all(not c.label for c in echo)


def test_identical_content_negatives_write_baseline_bytes(cases: list[SeededCase]) -> None:
    rows = [c for c in cases if c.kind == "identical_content"]
    assert rows
    for case in rows:
        for action in case.actions:
            if action.tool == "write_file" and action.args["path"] in case.files:
                assert action.args["content"] == case.files[str(action.args["path"])], case.id


# --------------------------------------------------------------------------- schema validity


def test_cases_validate_against_schema(cases: list[SeededCase]) -> None:
    for case in cases:
        constraint = case.constraint
        assert Constraint.from_dict(constraint.to_dict()) == constraint
        DetectorSpec(constraint.detector.kind, dict(constraint.detector.params))
        assert case.family == constraint.detector.kind
        assert case.synthetic is True
        assert isinstance(case.label, bool)
        assert case.rationale.strip().startswith("DESIGN "), case.id
        assert MIN_FILES <= len(case.files) <= MAX_FILES
        assert all(text.endswith("\n") for text in case.files.values()), case.id
        for check in case.checks:
            assert HarnessCheckSpec.from_dict(check.to_dict()) == check
        if case.family == "invariant_command":
            assert constraint.detector.params["check_id"] in {c.id for c in case.checks}
        assert 1 <= len(case.actions) <= MAX_ACTIONS, case.id
        assert any(s.kind == "decision_point" for s in case.steps), case.id
        for step in case.steps:
            if step.kind == "decision_point":
                assert step.constraint == constraint.id
            else:
                assert step.kind == "work" and step.constraint is None
            for action in step.actions:
                assert action.tool in AGENT_TOOLS
                assert Action.from_dict(action.to_dict()) == action
                for arg in TOOL_REQUIRED_ARGS[action.tool]:
                    assert isinstance(action.args[arg], str), (case.id, arg)


def test_static_replays_find_their_targets(cases: list[SeededCase]) -> None:
    for case in cases:
        _, problems = _simulate(case)
        assert not problems, (case.id, problems)


@pytest.mark.parametrize(
    ("case_id", "path", "lines"),
    [
        ("ml-one-over", "src/units.py", 41),
        ("ml-much-longer", "src/units.py", 60),
        ("ml-exact-limit", "src/units.py", 40),
        ("ml-below-limit", "src/units.py", 39),
        ("ml-replace-grow", "src/calc.py", 43),
        ("ml-grow-under-limit", "src/calc.py", 39),
        ("ml-split-modules", "src/units_a.py", 30),
        ("ml-long-line", "src/calc.py", 11),
        ("ml-docs-long", "docs/units.md", 80),
    ],
)
def test_line_counts_of_seeded_files(
    cases: list[SeededCase], case_id: str, path: str, lines: int
) -> None:
    state, _ = _simulate(_by_id(cases, case_id))
    text = state[path]
    assert text is not None
    assert len(text.splitlines()) == lines


def test_line_limit_labels_follow_counts(cases: list[SeededCase]) -> None:
    for case in cases:
        if case.family != "max_file_lines" or any(a.tool == "run" for a in case.actions):
            continue
        if any(s.kind == "work" for s in case.steps):
            continue
        state, _ = _simulate(case)
        params = case.constraint.detector.params
        over = any(
            text is not None
            and len(text.splitlines()) > params["limit"]
            and re.fullmatch(r"src/[^/]*\.py", path) is not None
            for path, text in state.items()
        )
        assert over is case.label, case.id


def test_python_commands_skip_bytecode(cases: list[SeededCase]) -> None:
    commands = [cmd for c in cases for cmd in _commands(c)]
    commands += [check.cmd for c in cases for check in c.checks]
    workspace_import = re.compile(r"import (calc|shapes|textstats)|from (fetch|shapes|textstats)")
    for cmd in commands:
        if "-m unittest" in cmd:
            assert "python3 -B -m unittest" in cmd, cmd
        if workspace_import.search(cmd):
            assert "python3 -B" in cmd, cmd


def test_commands_avoid_bash_only_syntax(cases: list[SeededCase]) -> None:
    for case in cases:
        for cmd in _commands(case):
            assert not re.search(r"\{[^}\s]*,[^}\s]*\}", cmd), cmd
            assert "[[" not in cmd and "echo -e" not in cmd, cmd


def test_network_commands_only_in_no_network_family(cases: list[SeededCase]) -> None:
    risky = re.compile(r"https?://|\bwget\b|\bcurl\b|pip download|ls-remote|git clone")
    for case in cases:
        if case.family == "no_network":
            continue
        for cmd in _commands(case):
            assert not risky.search(cmd), (case.id, cmd)
            if "pip" in cmd and "install" in cmd:
                assert "--no-index" in cmd, (case.id, cmd)


def test_no_em_dashes(cases: list[SeededCase]) -> None:
    assert EM_DASH not in cases_to_json(cases)
    for path in (Path(e0_cases.__file__), Path(__file__)):
        assert EM_DASH not in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- serialization


def test_build_is_deterministic(cases: list[SeededCase]) -> None:
    again = build_cases()
    assert [c.to_dict() for c in again] == [c.to_dict() for c in cases]
    assert cases_to_json(again) == cases_to_json(cases)


def test_round_trip(cases: list[SeededCase]) -> None:
    for case in cases:
        assert SeededCase.from_dict(json.loads(json.dumps(case.to_dict()))) == case
    text = cases_to_json(cases)
    assert json.loads(text)["synthetic"] is True
    assert cases_from_json(text) == cases


def test_step_round_trip() -> None:
    step = SeededStep("s1", "work", None, (Action("run", {"cmd": "true"}),))
    assert SeededStep.from_dict(step.to_dict()) == step
    assert SeededStep.from_dict({"id": "s2", "kind": "work"}).actions == ()


def test_properties(cases: list[SeededCase]) -> None:
    case = cases[0]
    assert case.constraint_id == case.constraint.id
    assert case.actions == tuple(a for s in case.steps for a in s.actions)


def test_summarize(cases: list[SeededCase]) -> None:
    summary = summarize(cases)
    assert summary["synthetic"] is True
    assert summary["total"] == len(cases) == summary["positive"] + summary["negative"]
    assert set(summary["by_family"]) == set(DETECTOR_KINDS)
    assert set(summary["by_constraint_type"]) == set(CONSTRAINT_TYPES)
    assert sum(v["positive"] for v in summary["by_kind"].values()) == summary["positive"]
    assert summary["by_kind"]["sed_inplace"]["negative"] == 0


def test_from_dict_missing_field(cases: list[SeededCase]) -> None:
    data = cases[0].to_dict()
    del data["steps"]
    with pytest.raises(SchemaError, match="missing field"):
        SeededCase.from_dict(data)


@pytest.mark.parametrize("text", ['{"cases": {}}', "[]", '{"synthetic": true}'])
def test_cases_from_json_rejects_bad_payload(text: str) -> None:
    with pytest.raises(SchemaError):
        cases_from_json(text)


def test_build_cases_rejects_duplicate_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    original = e0_cases._file_invariant_cases
    monkeypatch.setattr(e0_cases, "_file_invariant_cases", lambda: original() + original()[:1])
    with pytest.raises(SchemaError, match="duplicate case ids"):
        build_cases()


# --------------------------------------------------------------------------- validation errors


def _valid(cases: list[SeededCase]) -> SeededCase:
    return _by_id(cases, "pp-sed-assert")


def _step(case: SeededCase, **changes: Any) -> tuple[SeededStep, ...]:
    return (case.steps[0]._replace(**changes),)


_RUN = Action("run", {"cmd": "true"})
_MUTATIONS: dict[str, Callable[[SeededCase], dict[str, Any]]] = {
    "bad id": lambda c: {"id": "Bad Id"},
    "non-string id": lambda c: {"id": cast(Any, 5)},
    "unknown family": lambda c: {"family": "nope"},
    "family mismatch": lambda c: {"family": "no_network"},
    "bad kind label": lambda c: {"kind": "Sed In Place"},
    "non-bool label": lambda c: {"label": 1},
    "not synthetic": lambda c: {"synthetic": False},
    "empty rationale": lambda c: {"rationale": "  "},
    "too few files": lambda c: {"files": {"a.py": "x = 1\n"}},
    "too many files": lambda c: {"files": {f"f{i}.py": "\n" for i in range(MAX_FILES + 1)}},
    "absolute path": lambda c: {"files": {"/etc/passwd": "x\n", "a.py": "\n"}},
    "parent path": lambda c: {"files": {"../x.py": "x\n", "a.py": "\n"}},
    "git path": lambda c: {"files": {".git/config": "x\n", "a.py": "\n"}},
    "empty path": lambda c: {"files": {"": "x\n", "a.py": "\n"}},
    "no steps": lambda c: {"steps": ()},
    "duplicate step ids": lambda c: {"steps": (c.steps[0], c.steps[0])},
    "bad step id": lambda c: {"steps": _step(c, id="S 1")},
    "bad step kind": lambda c: {"steps": _step(c, kind="phase")},
    "dp names other constraint": lambda c: {"steps": _step(c, constraint="other")},
    "work names constraint": lambda c: {"steps": _step(c, kind="work")},
    "empty actions": lambda c: {"steps": _step(c, actions=())},
    "no decision point": lambda c: {"steps": _step(c, kind="work", constraint=None)},
    "missing arg": lambda c: {"steps": _step(c, actions=(Action("run", {}),))},
    "non-string arg": lambda c: {"steps": _step(c, actions=(Action("run", {"cmd": 3}),))},
    "too many actions": lambda c: {"steps": _step(c, actions=(_RUN,) * (MAX_ACTIONS + 1))},
    "duplicate checks": lambda c: {
        "checks": (HarnessCheckSpec("unit", "true"), HarnessCheckSpec("unit", "false"))
    },
}


@pytest.mark.parametrize("name", sorted(_MUTATIONS))
def test_validation_rejects(cases: list[SeededCase], name: str) -> None:
    case = _valid(cases)
    with pytest.raises(SchemaError):
        dataclasses.replace(case, **_MUTATIONS[name](case))


def test_invariant_case_needs_named_check(cases: list[SeededCase]) -> None:
    case = _by_id(cases, "ic-break-add")
    with pytest.raises(SchemaError, match="unknown harness check"):
        dataclasses.replace(case, checks=(HarnessCheckSpec("other", "true"),))
