"""half-life-bench: session templates and their verification (gate G6).

Templates are YAML files in ``templates/``. This module finds and loads them,
summarizes coverage (constraint types, detector kinds, splits), and builds the
manifest that ``tools/ship_check.py`` checks (docs/ARTIFACTS.md, G6). Running
the routes lives in :mod:`halflife.bench.verify`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from halflife.schema import Template, load_template

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class LoadedTemplate:
    """A template with the file it came from and that file's SHA-256."""

    template: Template
    path: Path
    sha256: str

    @property
    def id(self) -> str:
        """Template id."""
        return self.template.id


def template_paths(directory: Path = TEMPLATES_DIR) -> list[Path]:
    """Template files in a stable order."""
    return sorted(directory.glob("*.yaml"))


def load(path: Path) -> LoadedTemplate:
    """Load one template file and hash it."""
    return LoadedTemplate(
        template=load_template(path),
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def load_all(directory: Path = TEMPLATES_DIR, split: str | None = None) -> list[LoadedTemplate]:
    """Load every template, optionally only one split.

    Raises:
        ValueError: if two files declare the same template id, or a file name
            does not match its id.
    """
    loaded = [load(p) for p in template_paths(directory)]
    seen: dict[str, Path] = {}
    for item in loaded:
        if item.path.stem != item.id:
            raise ValueError(f"{item.path.name} declares id {item.id!r}")
        if item.id in seen:
            raise ValueError(f"duplicate template id {item.id!r}")
        seen[item.id] = item.path
    return [t for t in loaded if split is None or t.template.split == split]


def coverage(templates: Sequence[LoadedTemplate]) -> dict[str, Any]:
    """Counts of splits, constraint types, detector kinds, and decision points."""
    splits = Counter(t.template.split for t in templates)
    types = Counter(c.type for t in templates for c in t.template.constraints)
    kinds = Counter(c.detector.kind for t in templates for c in t.template.constraints)
    dps = Counter(
        t.template.constraint(s.constraint).type
        for t in templates
        for s in t.template.subtasks
        if s.is_dp and s.constraint is not None
    )
    return {
        "templates": len(templates),
        "splits": dict(sorted(splits.items())),
        "constraint_types": dict(sorted(types.items())),
        "detector_kinds": dict(sorted(kinds.items())),
        "decision_points_by_type": dict(sorted(dps.items())),
    }


def relative(path: Path, root: Path) -> str:
    """POSIX path of ``path`` relative to ``root``."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def manifest(
    templates: Sequence[LoadedTemplate],
    root: Path,
    artifacts_dir: str = "artifacts/bench",
) -> dict[str, Any]:
    """The G6 manifest: one entry per template with its hash and banked runs.

    Paths are relative to the repository ``root``; the compliant route and
    reference run files are written by :mod:`halflife.bench.verify`.
    """
    entries = [
        {
            "id": t.id,
            "path": relative(t.path, root),
            "sha256": t.sha256,
            "split": t.template.split,
            "constraint_types": list(t.template.constraint_types),
            "compliant_route": f"{artifacts_dir}/routes/{t.id}.json",
            "reference_run": f"{artifacts_dir}/reference/{t.id}.json",
        }
        for t in templates
    ]
    return {"synthetic": True, "coverage": coverage(templates), "templates": entries}
