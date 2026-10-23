"""Experiment designs: from a design file to session plans, and from runs to artifacts.

A design file (YAML) fixes everything before data exists, so it is committed
with the preregistration it belongs to:

```yaml
experiment: e1
split: main                 # or template_ids: [a, b]
n_sessions: 60              # plans generated (the same plans run for every model family)
seed_base: 1000
confirmatory: true
conditions:                 # assigned by blocked randomization
  - {placement: first_user, phrasing: bare, strategy: full_summary, mitigation: none}
  - {placement: first_user, phrasing: bare, strategy: sham, mitigation: none}
settings: {max_tool_calls: 150, budget_tokens: 3000}
```

Templates and conditions are assigned to sessions by blocked randomization
(each block is a shuffled copy of the options), so both stay balanced.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from halflife import bench
from halflife.schema import SchemaError, SessionPlan
from halflife.sessions import Condition, PlanSettings, generate_plan

T = TypeVar("T")


@dataclass(frozen=True)
class Design:
    """A parsed design file."""

    experiment: str
    n_sessions: int
    seed_base: int
    conditions: tuple[Condition, ...]
    settings: PlanSettings
    split: str | None = None
    template_ids: tuple[str, ...] = ()
    confirmatory: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Design:
        """Build and validate a design from parsed YAML."""
        if not data.get("conditions"):
            raise SchemaError("design needs at least one condition")
        if bool(data.get("split")) == bool(data.get("template_ids")):
            raise SchemaError("design needs exactly one of split or template_ids")
        settings = dict(data.get("settings") or {})
        for key in ("compactions", "compaction_window"):
            if key in settings:
                settings[key] = tuple(settings[key])
        conditions = tuple(Condition(**c) for c in data["conditions"])
        n = int(data["n_sessions"])
        if n <= 0:
            raise SchemaError("n_sessions must be positive")
        return cls(
            experiment=str(data["experiment"]),
            n_sessions=n,
            seed_base=int(data.get("seed_base", 0)),
            conditions=conditions,
            settings=PlanSettings(**settings),
            split=data.get("split"),
            template_ids=tuple(data.get("template_ids") or ()),
            confirmatory=bool(data.get("confirmatory", False)),
        )


def load_design(path: Path) -> Design:
    """Read a design YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SchemaError(f"{path} does not hold a mapping")
    return Design.from_dict(data)


def blocked(options: Sequence[T], n: int, rng: random.Random) -> list[T]:
    """``n`` draws by blocked randomization: concatenated shuffled copies of ``options``."""
    out: list[T] = []
    while len(out) < n:
        block = list(options)
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def make_plans(design: Design, templates: Sequence[bench.LoadedTemplate]) -> list[SessionPlan]:
    """All plans for a design, in session order."""
    chosen = [
        t
        for t in templates
        if (design.split is not None and t.template.split == design.split)
        or t.id in design.template_ids
    ]
    if design.template_ids and {t.id for t in chosen} != set(design.template_ids):
        raise SchemaError("design names templates that do not exist")
    if not chosen:
        raise SchemaError("design selects no templates")
    rng = random.Random(f"{design.experiment}|{design.seed_base}")
    picks = blocked(chosen, design.n_sessions, rng)
    conds = blocked(list(design.conditions), design.n_sessions, rng)
    return [
        generate_plan(
            item.template,
            item.sha256,
            seed=design.seed_base + i,
            condition=cond,
            settings=design.settings,
            experiment=design.experiment,
            confirmatory=design.confirmatory,
        )
        for i, (item, cond) in enumerate(zip(picks, conds, strict=True))
    ]


# --------------------------------------------------------------------------- banking


def _git_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(run_dirs: Sequence[Path]) -> list[tuple[Path, dict[str, Any]]]:
    """Every finished session's directory and result record, sorted by session id."""
    found = []
    for run_dir in run_dirs:
        for result in sorted(run_dir.glob("*/result.json")):
            found.append((result.parent, json.loads(result.read_text(encoding="utf-8"))))
    return sorted(
        found, key=lambda item: str(item[1]["session"]["session_id"]) + str(item[1]["run_id"])
    )


def bank_runs(
    run_dirs: Sequence[Path],
    out_dir: Path,
    root: Path,
    *,
    experiment: str,
    prereg: str | None,
    design_path: Path | None,
    models: Sequence[Mapping[str, Any]],
    command: str,
) -> dict[str, Any]:
    """Aggregate finished sessions into artifact files under ``out_dir``.

    Writes ``sessions.jsonl``, ``opportunities.jsonl``, ``probes.jsonl``,
    ``compaction_logs.json`` with every compaction log copied under
    ``compactions/``, and ``provenance.json``. Returns the provenance record.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions: list[dict[str, Any]] = []
    opportunities: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for session_dir, record in collect(run_dirs):
        sid = record["session"]["session_id"]
        row = dict(record["session"]) | {"run_id": record["run_id"]}
        if prereg is not None:
            row["prereg"] = prereg
        sessions.append(row)
        opportunities.extend(
            dict(o) | {"run_id": record["run_id"]} for o in record["opportunities"]
        )
        for probe in json.loads((session_dir / "probes.json").read_text(encoding="utf-8")):
            probes.append({"session_id": sid, "run_id": record["run_id"], **probe})
        for event in record["compactions"]:
            source = session_dir / event["log_path"]
            target_rel = f"compactions/{record['run_id']}/{sid}/{event['event_id']}.json"
            target = out_dir / target_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            events.append(
                dict(event)
                | {
                    "log_path": (out_dir / target_rel).relative_to(root).as_posix(),
                    "log_sha256": _sha256(target),
                }
            )

    def dump_jsonl(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        with (out_dir / name).open("w", encoding="utf-8") as handle:
            for r in rows:
                handle.write(json.dumps(r, sort_keys=True) + "\n")

    dump_jsonl("sessions.jsonl", sessions)
    dump_jsonl("opportunities.jsonl", opportunities)
    dump_jsonl("probes.jsonl", probes)
    (out_dir / "compaction_logs.json").write_text(
        json.dumps({"events": events}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "experiment": experiment,
        "prereg": prereg,
        "confirmatory": prereg is not None,
        "git_sha": _git_sha(root),
        "design": design_path.relative_to(root).as_posix() if design_path else None,
        "design_sha256": _sha256(design_path) if design_path else None,
        "runs": sorted({str(r["run_id"]) for r in sessions}),
        "models": list(models),
        "n_sessions": len(sessions),
        "n_opportunities": len(opportunities),
        "command": command,
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance
