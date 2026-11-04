"""Validation of recall-probe scoring (DECISIONS.md D-023).

Recall probes are scored with keyword groups (:func:`halflife.probe.recalls`).
Recall is an outcome for RQ2, so the scorer is checked against a labeled set of
probe answers written without seeing the keyword groups: faithful paraphrases
(label True), near misses that mention the topic but not the rule, partial
answers, and unrelated answers (label False). The answers are synthetic and
labeled as such. Agreement (sensitivity, specificity, accuracy with Wilson 95%
intervals) is banked as ``artifacts/e0/probe_scoring.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from halflife import bench
from halflife.probe import recalls
from halflife.schema import Constraint, SchemaError

Z95 = 1.959963984540054
DEFAULT_ANSWERS = Path(__file__).resolve().parent / "data" / "probe_answers.yaml"


@dataclass(frozen=True)
class LabeledAnswer:
    """One synthetic probe answer with a human-style label for one constraint."""

    template_id: str
    constraint_id: str
    answer: str
    label: bool
    kind: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LabeledAnswer:
        """Parse and validate one record."""
        if not isinstance(data.get("label"), bool):
            raise SchemaError("label must be a boolean")
        if not str(data.get("answer", "")).strip():
            raise SchemaError("answer must be non-empty")
        return cls(
            template_id=str(data["template_id"]),
            constraint_id=str(data["constraint_id"]),
            answer=str(data["answer"]),
            label=bool(data["label"]),
            kind=str(data.get("kind", "unspecified")),
        )


def load_answers(path: Path = DEFAULT_ANSWERS) -> list[LabeledAnswer]:
    """Read the labeled answer set (YAML with a top-level ``answers`` list)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("synthetic") is not True:
        raise SchemaError(f"{path} must be a mapping with synthetic: true")
    return [LabeledAnswer.from_dict(item) for item in data.get("answers", [])]


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score 95% interval."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(k: int, n: int) -> dict[str, float | int]:
    lo, hi = wilson(k, n)
    return {"value": k / n if n else 0.0, "lo": lo, "hi": hi, "k": k, "n": n}


def score_answers(
    answers: Sequence[LabeledAnswer], constraints: Mapping[tuple[str, str], Constraint]
) -> dict[str, Any]:
    """Score every labeled answer and summarize agreement with the labels.

    Raises:
        KeyError: if an answer names a template constraint that does not exist.
    """
    rows: list[dict[str, Any]] = []
    for item in answers:
        constraint = constraints[(item.template_id, item.constraint_id)]
        predicted = recalls(item.answer, constraint)
        rows.append(
            {
                "template_id": item.template_id,
                "constraint_id": item.constraint_id,
                "kind": item.kind,
                "label": item.label,
                "predicted": predicted,
            }
        )
    tp = sum(1 for r in rows if r["label"] and r["predicted"])
    tn = sum(1 for r in rows if not r["label"] and not r["predicted"])
    pos = sum(1 for r in rows if r["label"])
    neg = len(rows) - pos
    by_kind: dict[str, dict[str, int]] = {}
    for r in rows:
        entry = by_kind.setdefault(r["kind"], {"n": 0, "agree": 0})
        entry["n"] += 1
        entry["agree"] += int(r["label"] == r["predicted"])
    return {
        "rows": rows,
        "sensitivity": _rate(tp, pos),
        "specificity": _rate(tn, neg),
        "accuracy": _rate(tp + tn, len(rows)),
        "by_kind": dict(sorted(by_kind.items())),
        "disagreements": [r for r in rows if r["label"] != r["predicted"]],
    }


def constraint_index(
    templates: Sequence[bench.LoadedTemplate],
) -> dict[tuple[str, str], Constraint]:
    """(template id, constraint id) to constraint for every bench template."""
    return {(t.id, c.id): c for t in templates for c in t.template.constraints}


def main(argv: Sequence[str] | None = None) -> int:
    """Score the labeled set and optionally bank the artifact."""
    parser = argparse.ArgumentParser(description="Validate recall-probe scoring (D-023).")
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--bank", action="store_true", help="write the scoring artifact")
    parser.add_argument("--out", type=Path, default=None, help="artifact path for --bank")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    templates = bench.load_all()
    result = score_answers(load_answers(args.answers), constraint_index(templates))
    for name in ("sensitivity", "specificity", "accuracy"):
        r = result[name]
        print(f"{name}={r['value']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] ({r['k']}/{r['n']})")
    for row in result["disagreements"]:
        where = f"{row['template_id']}/{row['constraint_id']}"
        print(f"DISAGREE {where} kind={row['kind']} label={row['label']}")
    if args.bank:
        out = args.out or root / "artifacts" / "e0" / "probe_scoring.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        answers_path = args.answers.resolve()
        artifact = {
            "synthetic": True,
            "description": (
                "Agreement of keyword recall scoring with labels on synthetic probe answers."
            ),
            "answers": (
                answers_path.relative_to(root).as_posix()
                if answers_path.is_relative_to(root)
                else answers_path.name
            ),
            "templates_sha256": {t.id: t.sha256 for t in templates},
        } | {k: v for k, v in result.items() if k != "disagreements"}
        out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
