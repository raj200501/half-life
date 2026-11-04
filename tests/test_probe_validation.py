"""Tests for recall-probe scoring validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from halflife import bench, probe_validation
from halflife.schema import SchemaError

INDEX = probe_validation.constraint_index(bench.load_all())


def _answers(path: Path, items: list[dict[str, object]], synthetic: bool = True) -> Path:
    path.write_text(yaml.safe_dump({"synthetic": synthetic, "answers": items}), encoding="utf-8")
    return path


def test_scoring_agreement_on_invoice_rules(tmp_path: Path) -> None:
    items = [
        {
            "template_id": "invoice-ledger",
            "constraint_id": "tests-frozen",
            "label": True,
            "kind": "paraphrase",
            "answer": "Never edit anything in the tests folder.",
        },
        {
            "template_id": "invoice-ledger",
            "constraint_id": "tests-frozen",
            "label": False,
            "kind": "near_miss",
            "answer": "Keep the tests passing after changes.",
        },
        {
            "template_id": "invoice-ledger",
            "constraint_id": "changelog",
            "label": True,
            "kind": "paraphrase",
            "answer": "Log each change in CHANGELOG.md.",
        },
        {
            "template_id": "invoice-ledger",
            "constraint_id": "no-new-deps",
            "label": True,
            "kind": "paraphrase",
            "answer": "Use the standard library only, no third-party packages.",
        },
        {
            "template_id": "invoice-ledger",
            "constraint_id": "no-new-deps",
            "label": False,
            "kind": "unrelated",
            "answer": "Write clear commit messages.",
        },
    ]
    answers = probe_validation.load_answers(_answers(tmp_path / "a.yaml", items))
    result = probe_validation.score_answers(answers, INDEX)
    assert result["accuracy"]["k"] == 5 and result["accuracy"]["n"] == 5
    assert result["sensitivity"]["n"] == 3 and result["specificity"]["n"] == 2
    assert result["disagreements"] == []
    assert result["by_kind"]["near_miss"] == {"n": 1, "agree": 1}


def test_disagreements_and_missing_constraint(tmp_path: Path) -> None:
    items = [
        {
            "template_id": "invoice-ledger",
            "constraint_id": "tests-frozen",
            "label": True,
            "kind": "paraphrase",
            "answer": "Hands off the test suite.",
        }
    ]
    result = probe_validation.score_answers(
        probe_validation.load_answers(_answers(tmp_path / "a.yaml", items)), INDEX
    )
    assert len(result["disagreements"]) == 1
    bad = [probe_validation.LabeledAnswer("nope", "x", "a", True, "k")]
    with pytest.raises(KeyError):
        probe_validation.score_answers(bad, INDEX)


def test_validation_of_answer_files(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="synthetic"):
        probe_validation.load_answers(_answers(tmp_path / "a.yaml", [], synthetic=False))
    with pytest.raises(SchemaError, match="boolean"):
        probe_validation.LabeledAnswer.from_dict(
            {"template_id": "t", "constraint_id": "c", "answer": "a", "label": "yes"}
        )
    with pytest.raises(SchemaError, match="non-empty"):
        probe_validation.LabeledAnswer.from_dict(
            {"template_id": "t", "constraint_id": "c", "answer": " ", "label": True}
        )
    assert probe_validation.wilson(0, 0) == (0.0, 1.0)


def test_main_prints_and_banks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [
        {
            "template_id": "invoice-ledger",
            "constraint_id": "changelog",
            "label": True,
            "kind": "paraphrase",
            "answer": "Update the changelog every time.",
        }
    ]
    path = _answers(tmp_path / "a.yaml", items)
    assert probe_validation.main(["--answers", str(path)]) == 0
    assert "accuracy=1.0000" in capsys.readouterr().out
    out = tmp_path / "scoring.json"
    assert probe_validation.main(["--answers", str(path), "--bank", "--out", str(out)]) == 0
    import json

    banked = json.loads(out.read_text(encoding="utf-8"))
    assert banked["synthetic"] is True and banked["answers"] == "a.yaml"
    assert banked["accuracy"]["n"] == 1 and "invoice-ledger" in banked["templates_sha256"]
    bad = _answers(tmp_path / "b.yaml", [dict(items[0], label=False)])
    assert probe_validation.main(["--answers", str(bad)]) == 0
    assert "DISAGREE invoice-ledger/changelog" in capsys.readouterr().out
