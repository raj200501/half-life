"""Tests for halflife.kit.spec: the portable constraint spec (YAML version 1)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from halflife import bench
from halflife.kit import spec as spec_mod
from halflife.kit.spec import (
    DEFAULT_SPEC_NAME,
    NEGATION_GROUP,
    SpecError,
    default_recall_keywords,
    dump_spec,
    find_spec,
    load_spec,
    parse_spec,
    spec_to_dict,
)
from halflife.schema import Constraint, DetectorSpec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "halflife.yaml"


def _constraint(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": "tests-frozen",
        "type": "prohibition",
        "text": "Never modify files under tests/.",
        "reason": "The tests are the acceptance criteria.",
        "detector": {"kind": "protected_paths", "params": {"globs": ["tests/**"]}},
    }
    item.update(overrides)
    return item


def _doc(*items: dict[str, Any], **top: Any) -> dict[str, Any]:
    document: dict[str, Any] = {"version": 1, "constraints": list(items or [_constraint()])}
    document.update(top)
    return document


def _write(tmp_path: Path, text: str, name: str = DEFAULT_SPEC_NAME) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- loading


def test_load_spec_reads_every_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
        version: 1
        constraints:
          - id: tests-frozen
            type: prohibition
            text: Never modify files under tests/.
            reason: The tests are the acceptance criteria.
            detector:
              kind: protected_paths
              params: {globs: ["tests/**"]}
            recall_keywords: [["test"], ["never", "not "]]
          - id: offline
            type: prohibition
            text: Do not access the network.
            reason: The build machines are offline.
            detector: {kind: no_network}
        """,
    )
    constraints = load_spec(path)
    assert [c.id for c in constraints] == ["tests-frozen", "offline"]
    first, second = constraints
    assert first.detector == DetectorSpec("protected_paths", {"globs": ["tests/**"]})
    assert first.recall_keywords == (("test",), ("never", "not "))
    assert second.detector.kind == "no_network" and dict(second.detector.params) == {}
    assert second.recall_keywords[-1] == NEGATION_GROUP


def test_load_spec_accepts_str_path_and_null_params(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
        version: 1
        constraints:
          - {id: offline, type: prohibition, text: No network., reason: Offline CI.,
             detector: {kind: no_network, params: null}}
        """,
    )
    (constraint,) = load_spec(str(path))
    assert constraint.id == "offline"


def test_example_spec_loads_and_covers_guard_kinds() -> None:
    constraints = load_spec(EXAMPLE)
    kinds = {c.detector.kind for c in constraints}
    assert {
        "protected_paths",
        "no_network",
        "no_new_dependencies",
        "forbidden_commands",
        "max_file_lines",
        "file_invariant",
        "stable_api",
    } <= kinds
    assert len({c.id for c in constraints}) == len(constraints)


def test_round_trip_through_dump_spec() -> None:
    constraints = load_spec(EXAMPLE)
    text = dump_spec(constraints)
    assert parse_spec(yaml.safe_load(text), source="dumped") == constraints


def test_bench_template_constraints_round_trip() -> None:
    for loaded in bench.load_all():
        constraints = loaded.template.constraints
        document = yaml.safe_load(dump_spec(constraints))
        assert parse_spec(document, source=loaded.template.id) == constraints


def test_spec_to_dict_can_leave_out_recall_keywords() -> None:
    constraints = load_spec(EXAMPLE)
    document = spec_to_dict(constraints, include_recall_keywords=False)
    assert document["version"] == 1
    assert all("recall_keywords" not in item for item in document["constraints"])
    reloaded = parse_spec(document)
    assert [c.text for c in reloaded] == [c.text for c in constraints]
    assert all(c.recall_keywords == default_recall_keywords(c.text, c.type) for c in reloaded)


# --------------------------------------------------------------------------- default keywords


@pytest.mark.parametrize(
    ("text", "kind", "expected"),
    [
        ("Never modify files under tests/.", "prohibition", (("modif",), ("test",))),
        (
            "Never modify, create, or delete anything under snapshots/.",
            "prohibition",
            (("modif",), ("snapshot",)),
        ),
        (
            "TOLERANCE in matlite/solve.py must stay exactly 1e-10.",
            "invariant",
            (("toler",), ("solve",)),
        ),
        (
            "Every change to eventbus/ must add a line to CHANGELOG.md.",
            "procedure",
            (("eventbus",), ("changelog",)),
        ),
        ("Keep mypkg/api.py stable.", "invariant", (("api.py",), ("stabl",))),
        ("Run pytest no more than twice in any task.", "resource_limit", (("pytes",), ("twice",))),
        ("Keep it.", "invariant", (("keep it.",),)),
    ],
)
def test_default_recall_keywords(
    text: str, kind: str, expected: tuple[tuple[str, ...], ...]
) -> None:
    groups = default_recall_keywords(text, kind)
    if kind == "prohibition":
        assert groups[-1] == NEGATION_GROUP
        groups = groups[:-1]
    assert groups == expected


def test_default_recall_keywords_are_usable_by_constraints() -> None:
    groups = default_recall_keywords("!!!", "prohibition")
    assert groups == (("!!!",), NEGATION_GROUP)
    Constraint(
        id="x",
        type="prohibition",
        text="!!!",
        reason="r",
        detector=DetectorSpec("no_network"),
        recall_keywords=groups,
    )


def test_missing_recall_keywords_get_the_default() -> None:
    (constraint,) = parse_spec(_doc())
    assert constraint.recall_keywords == default_recall_keywords(
        "Never modify files under tests/.", "prohibition"
    )


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([1, 2], "top level must be a mapping"),
        (_doc(extra=1), "unknown top-level field(s) ['extra']"),
        ({"constraints": [_constraint()]}, "missing version"),
        (_doc(version=2), "unsupported version 2"),
        (_doc(version="1"), "unsupported version '1'"),
        (_doc(version=True), "unsupported version True"),
        ({"version": 1}, "constraints must be a non-empty list"),
        ({"version": 1, "constraints": []}, "constraints must be a non-empty list"),
        ({"version": 1, "constraints": {"a": 1}}, "constraints must be a non-empty list"),
        ({"version": 1, "constraints": ["x"]}, "constraints[0]: each constraint must be a mapping"),
        (_doc(_constraint(colour="red")), "('tests-frozen'): unknown field(s) ['colour']"),
        (
            _doc({"id": "a", "type": "prohibition"}),
            "missing field(s) ['text', 'reason', 'detector']",
        ),
        (_doc(_constraint(text=3)), "text must be a string"),
        (_doc(_constraint(id=5)), "constraints[0]: id must be a string"),
        (_doc(_constraint(detector="protected_paths")), "detector must be a mapping"),
        (
            _doc(_constraint(detector={"kind": "no_network", "parms": {}})),
            "unknown detector field(s) ['parms']",
        ),
        (_doc(_constraint(detector={"params": {}})), "detector.kind must be a string"),
        (
            _doc(_constraint(detector={"kind": "no_network", "params": ["x"]})),
            "detector.params must be a mapping",
        ),
        (_doc(_constraint(detector={"kind": "network"})), "detector kind 'network' not in"),
        (
            _doc(_constraint(detector={"kind": "protected_paths", "params": {}})),
            "missing params ['globs']",
        ),
        (
            _doc(_constraint(detector={"kind": "protected_paths", "params": {"globs": [""]}})),
            "empty glob",
        ),
        (
            _doc(
                _constraint(detector={"kind": "forbidden_commands", "params": {"patterns": ["("]}})
            ),
            "invalid regex",
        ),
        (
            _doc(
                _constraint(
                    detector={"kind": "max_file_lines", "params": {"globs": ["*.py"], "limit": -1}}
                )
            ),
            "limit must be an integer >= 0",
        ),
        (_doc(_constraint(recall_keywords=[])), "recall_keywords must be a non-empty list"),
        (_doc(_constraint(recall_keywords="test")), "recall_keywords must be a non-empty list"),
        (_doc(_constraint(recall_keywords=[["test"], []])), "recall_keywords[1] must be"),
        (_doc(_constraint(recall_keywords=[["test", ""]])), "recall_keywords[0] must be"),
        (_doc(_constraint(recall_keywords=[[3]])), "recall_keywords[0] must be"),
        (_doc(_constraint(id="Bad Id")), "constraint id 'Bad Id' is invalid"),
        (_doc(_constraint(type="rule")), "constraint type 'rule' not in"),
        (_doc(_constraint(text="  ")), "has empty text"),
        (_doc(_constraint(reason="")), "has empty reason"),
        (_doc(_constraint(), _constraint()), "constraints[1]: duplicate id 'tests-frozen'"),
    ],
)
def test_parse_spec_errors(document: Any, message: str) -> None:
    with pytest.raises(SpecError) as excinfo:
        parse_spec(document, source="halflife.yaml")
    assert message in str(excinfo.value)
    assert str(excinfo.value).startswith("halflife.yaml")


def test_load_spec_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="spec file not found"):
        load_spec(tmp_path / "nope.yaml")


def test_load_spec_unreadable(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="cannot read the spec"):
        load_spec(tmp_path)
    bad = tmp_path / "latin1.yaml"
    bad.write_bytes(b"version: 1\ntext: caf\xe9\n")
    with pytest.raises(SpecError, match="cannot read the spec"):
        load_spec(bad)


def test_load_spec_invalid_yaml_reports_the_line(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\nconstraints: [\n")
    with pytest.raises(SpecError, match=r"invalid YAML at line 3, column 1"):
        load_spec(path)


def test_load_spec_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
        version: 1
        constraints:
          - id: a
            id: b
        """,
    )
    with pytest.raises(SpecError, match=r"duplicate key 'id' at line 4"):
        load_spec(path)


def test_load_spec_rejects_unhashable_keys(tmp_path: Path) -> None:
    path = _write(tmp_path, "? [a, b]\n: 1\n")
    with pytest.raises(SpecError, match="unhashable mapping key"):
        load_spec(path)


def test_load_spec_supports_yaml_merge_keys(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
        version: 1
        constraints:
          - &base
            id: one
            type: prohibition
            text: Never modify files under tests/.
            reason: Fixed tests.
            detector: {kind: protected_paths, params: {globs: ["tests/**"]}}
          - <<: *base
            id: two
        """,
    )
    assert [c.id for c in load_spec(path)] == ["one", "two"]


def test_yaml_error_without_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise yaml.YAMLError("no mark here")

    monkeypatch.setattr(spec_mod.yaml, "load", boom)
    with pytest.raises(SpecError, match="invalid YAML: no mark here"):
        spec_mod._load_yaml("x", "s.yaml")


# --------------------------------------------------------------------------- lookup


def test_find_spec_order(tmp_path: Path) -> None:
    project = tmp_path / "project"
    here = tmp_path / "here"
    project.mkdir()
    here.mkdir()
    env = {"CLAUDE_PROJECT_DIR": str(project)}
    assert find_spec(env=env, cwd=here) is None
    (project / DEFAULT_SPEC_NAME).write_text("version: 1\n", encoding="utf-8")
    assert find_spec(env=env, cwd=here) == project / DEFAULT_SPEC_NAME
    (here / DEFAULT_SPEC_NAME).write_text("version: 1\n", encoding="utf-8")
    assert find_spec(env=env, cwd=here) == here / DEFAULT_SPEC_NAME
    named = {**env, "HALFLIFE_SPEC": str(tmp_path / "named.yaml")}
    assert find_spec(env=named, cwd=here) == tmp_path / "named.yaml"
    assert find_spec("explicit.yaml", env=named, cwd=here) == Path("explicit.yaml")


def test_find_spec_defaults_to_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HALFLIFE_SPEC", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    assert find_spec() is None
    (tmp_path / DEFAULT_SPEC_NAME).write_text("version: 1\n", encoding="utf-8")
    assert find_spec() == tmp_path / DEFAULT_SPEC_NAME
