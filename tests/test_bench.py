"""Tests for template discovery, coverage, and the manifest."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from halflife import bench

EXEMPLAR = bench.TEMPLATES_DIR / "invoice-ledger.yaml"


def test_committed_templates_load() -> None:
    loaded = bench.load_all()
    assert loaded, "no templates found"
    ids = [t.id for t in loaded]
    assert "invoice-ledger" in ids
    assert ids == sorted(ids)
    for item in loaded:
        assert len(item.sha256) == 64
        assert item.template.split in {"main", "pilot", "heldout"}


def test_split_filter_and_coverage(tmp_path: Path) -> None:
    shutil.copy(EXEMPLAR, tmp_path / "invoice-ledger.yaml")
    assert [t.id for t in bench.load_all(tmp_path, split="main")] == ["invoice-ledger"]
    assert bench.load_all(tmp_path, split="pilot") == []
    cov = bench.coverage(bench.load_all(tmp_path))
    assert cov["templates"] == 1
    assert cov["constraint_types"] == {"procedure": 1, "prohibition": 2}
    assert cov["detector_kinds"]["protected_paths"] == 1
    assert cov["decision_points_by_type"] == {"procedure": 1, "prohibition": 2}


def test_file_name_must_match_id(tmp_path: Path) -> None:
    shutil.copy(EXEMPLAR, tmp_path / "other-name.yaml")
    with pytest.raises(ValueError, match="declares id"):
        bench.load_all(tmp_path)


def test_manifest_layout(tmp_path: Path) -> None:
    root = tmp_path
    tdir = root / "src" / "halflife" / "bench" / "templates"
    tdir.mkdir(parents=True)
    shutil.copy(EXEMPLAR, tdir / "invoice-ledger.yaml")
    loaded = bench.load_all(tdir)
    data = bench.manifest(loaded, root)
    entry = data["templates"][0]
    assert data["synthetic"] is True
    assert entry["path"] == "src/halflife/bench/templates/invoice-ledger.yaml"
    assert entry["compliant_route"] == "artifacts/bench/routes/invoice-ledger.json"
    assert entry["reference_run"] == "artifacts/bench/reference/invoice-ledger.json"
    assert entry["sha256"] == loaded[0].sha256
    assert entry["constraint_types"] == ["procedure", "prohibition"]


def test_every_template_carries_the_canary() -> None:
    """Rule 9 of the brief: a canary string in every file of a dataset meant for release
    (DECISIONS.md D-055), so the templates can be filtered out of training corpora."""
    paths = bench.template_paths()
    assert paths
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[1] == bench.CANARY, path.name
    assert bench.CANARY_GUID in bench.CANARY and len(bench.CANARY_GUID) == 36
