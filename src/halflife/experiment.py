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
import sys
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
        row["probe_leaks"] = list(record.get("summary", {}).get("leaks", []))
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


# --------------------------------------------------------------------------- E0 validation runs

LOG_FIELDS = (
    "event_id",
    "strategy",
    "tool_call_index",
    "scheduled",
    "tokens_before",
    "tokens_after",
    "before",
    "after",
    "summary",
    "constraint_text_retained",
)


def log_problems(log: Mapping[str, Any], constraint_ids: Sequence[str]) -> list[str]:
    """Ways a compaction log falls short of complete (every field, every constraint)."""
    problems = [f"missing {name}" for name in LOG_FIELDS if name not in log]
    if problems:
        return problems
    if not log["before"] or not log["after"]:
        problems.append("empty before or after context")
    elif log["after"][0].get("role") != "system":
        problems.append("compacted context lost the system prompt")
    if log["strategy"] != "sliding_window" and not log["summary"]:
        problems.append("no summary text")
    missing = sorted(set(constraint_ids) - set(log["constraint_text_retained"]))
    if missing:
        problems.append(f"retention not logged for {', '.join(missing)}")
    return problems


def e0_validation(
    banked: Path, out_dir: Path, root: Path, constraint_ids: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """Probe-leakage and compaction-log artifacts for G5 from a banked experiment.

    Args:
        banked: An experiment directory written by :func:`bank_runs` (its sessions
            are the validation runs).
        out_dir: Where to write ``probe_leakage.json`` and ``compaction_logs.json``.
        root: Repository root (paths in the artifacts are relative to it).
        constraint_ids: Template id to the constraint ids whose retention every
            compaction log must record.

    Returns:
        A summary with the counts written.
    """
    sessions = [
        json.loads(line)
        for line in (banked / "sessions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runs = [
        {
            "run_id": s["run_id"],
            "session_id": s["session_id"],
            "leaked": bool(s.get("probe_leaks")),
            "leaks": list(s.get("probe_leaks", [])),
        }
        for s in sessions
    ]
    templates = {s["session_id"]: s["template_id"] for s in sessions}
    events = json.loads((banked / "compaction_logs.json").read_text(encoding="utf-8"))["events"]
    checked = []
    for event in events:
        log = json.loads((root / event["log_path"]).read_text(encoding="utf-8"))
        problems = log_problems(log, constraint_ids.get(templates.get(event["session_id"], ""), ()))
        checked.append(dict(event) | {"complete": not problems, "problems": problems})
    source = banked.relative_to(root).as_posix()
    leakage = {
        "description": (
            "Probe leakage across validation runs: probe text found in any main-run model call."
        ),
        "source": source,
        "validation_runs": len(runs),
        "leaks": sum(1 for r in runs if r["leaked"]),
        "runs": runs,
    }
    logs = {
        "description": "Compaction logs from validation runs, each checked for completeness.",
        "source": source,
        "complete": sum(1 for e in checked if e["complete"]),
        "incomplete": sum(1 for e in checked if not e["complete"]),
        "events": checked,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_leakage.json").write_text(
        json.dumps(leakage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "compaction_logs.json").write_text(
        json.dumps(logs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "validation_runs": len(runs),
        "leaks": leakage["leaks"],
        "events": len(checked),
        "incomplete": logs["incomplete"],
    }


# --------------------------------------------------------------------------- command line


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _subject_python_version() -> str:
    from halflife.sandbox import subject_bin

    python = subject_bin() / "python"
    proc = subprocess.run(
        [str(python), "-c", "import sys; print(sys.version.split()[0])"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() or "unknown"


def check_plans(
    plans: Sequence[SessionPlan], templates: Mapping[str, bench.LoadedTemplate]
) -> list[str]:
    """Problems that make plans unsafe to run: unknown templates or changed template files."""
    problems = []
    for plan in plans:
        item = templates.get(plan.template_id)
        if item is None:
            problems.append(f"{plan.session_id}: unknown template {plan.template_id}")
        elif item.sha256 != plan.template_sha256:
            problems.append(
                f"{plan.session_id}: template {plan.template_id} changed since planning"
            )
    return problems


def cmd_plan(design_path: Path, out: Path) -> int:
    """Write the plans of a design file as JSON lines."""
    from halflife.runner import write_plans

    design = load_design(design_path)
    plans = make_plans(design, bench.load_all())
    write_plans(out, plans)
    print(f"wrote {len(plans)} plans for {design.experiment} to {out}")
    return 0


def cmd_run(args: Any, root: Path) -> int:
    """Run plans with a pinned local model; resumable; writes runs/<run_id>/."""
    import os

    from halflife import models as models_mod
    from halflife.agent import AgentSettings
    from halflife.client import CapExceeded, Client, ModelCallsDisabled
    from halflife.detect import detect_all
    from halflife.llm import ClientModel
    from halflife.runner import RunContext, load_plans, run_session
    from halflife.sandbox import detect_method

    plans = load_plans(args.plans)
    if args.index is not None:
        plans = plans[args.index : args.index + 1]
    if args.limit is not None:
        plans = plans[: args.limit]
    templates = {t.id: t for t in bench.load_all()}
    problems = check_plans(plans, templates)
    if problems:
        for problem in problems:
            print(f"refusing: {problem}")
        return 2
    run_dir = root / args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pid_file = run_dir / "pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    spec = models_mod.get_spec(args.model)
    settings = AgentSettings(
        temperature=args.temperature,
        max_step_tokens=args.max_step_tokens,
        family=spec.family,
        model=spec.key,
    )
    record: dict[str, Any] = {
        "run_id": args.run_id,
        "plans": args.plans.as_posix(),
        "plans_sha256": _sha256(args.plans),
        "model": spec.to_dict(),
        "n_ctx": args.n_ctx,
        "settings": {
            "temperature": settings.temperature,
            "max_step_tokens": settings.max_step_tokens,
        },
        "prereg": args.prereg,
        "git_sha": _git_sha(root),
        "sandbox_method": detect_method(),
        "subject_python": _subject_python_version(),
        "command": " ".join(sys.argv),
        "started": _now(),
        "cpu_count": os.cpu_count(),
    }
    with (run_dir / "invocations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    (run_dir / "run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    backend = models_mod.load_backend(spec.key, n_ctx=args.n_ctx)
    experiment = plans[0].experiment if plans else "dev"
    client = Client(
        backend,
        ledger_path=root / "artifacts" / "usage" / "ledger.jsonl",
        cache_dir=root / "runs" / "cache",
        run_id=args.run_id,
        experiment=experiment,
        budget_path=root / "budget.toml",
    )
    model = ClientModel(client)
    ctx = RunContext(
        run_id=args.run_id,
        family=spec.family,
        model=spec.key,
        prereg=args.prereg,
        settings=settings,
    )
    progress = run_dir / "progress.jsonl"
    status = 0
    for plan in plans:
        model.session_id = plan.session_id
        start = dt.datetime.now(dt.UTC)
        try:
            result = run_session(
                plan,
                templates[plan.template_id].template,
                model,
                detect_all,
                ctx,
                root / args.out_root,
            )
        except (CapExceeded, ModelCallsDisabled) as exc:
            print(f"stopping: {type(exc).__name__}: {exc}")
            status = 3
            break
        line = {
            "session_id": plan.session_id,
            "ended_reason": result["summary"]["ended_reason"],
            "tool_calls": result["summary"]["tool_calls"],
            "violations": result["session"]["n_violations"],
            "opportunities": len(result["opportunities"]),
            "wall_s": round((dt.datetime.now(dt.UTC) - start).total_seconds(), 1),
            "finished": _now(),
        }
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")
        print(json.dumps(line, sort_keys=True), flush=True)
    record["finished"] = _now()
    record["status"] = status
    (run_dir / "run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pid_file.unlink(missing_ok=True)
    return status


def cmd_bank(args: Any, root: Path) -> int:
    """Bank finished runs into an artifact directory."""
    run_dirs = [root / p for p in args.runs]
    models = []
    for run_dir in run_dirs:
        record_path = run_dir / "run.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            models.append(
                {
                    "run_id": record["run_id"],
                    **{
                        k: record["model"].get(k)
                        for k in (
                            "key",
                            "family",
                            "repo",
                            "revision",
                            "filename",
                            "sha256",
                            "quantization",
                            "license",
                        )
                    },
                }
            )
    provenance = bank_runs(
        run_dirs,
        root / args.out,
        root,
        experiment=args.experiment,
        prereg=args.prereg,
        design_path=root / args.design if args.design else None,
        models=models,
        command=" ".join(sys.argv),
    )
    counts = f"{provenance['n_sessions']} sessions, {provenance['n_opportunities']} opportunities"
    print(f"banked {counts} into {args.out}")
    return 0


def cmd_e0_validate(args: Any, root: Path) -> int:
    """Write the G5 probe-leakage and compaction-log artifacts from a banked experiment."""
    ids = {t.id: [c.id for c in t.template.constraints] for t in bench.load_all()}
    summary = e0_validation(root / args.banked, root / "artifacts" / "e0", root, ids)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["leaks"] == 0 and summary["incomplete"] == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m halflife.experiment {plan,run,bank,e0-validate}``."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m halflife.experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    p_plan = sub.add_parser("plan", help="write the plans of a design file")
    p_plan.add_argument("--design", type=Path, required=True)
    p_plan.add_argument("--out", type=Path, required=True)
    p_run = sub.add_parser("run", help="run plans with a pinned local model (resumable)")
    p_run.add_argument("--plans", type=Path, required=True)
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--model", required=True, help="model registry key")
    p_run.add_argument("--n-ctx", type=int, default=8192)
    p_run.add_argument("--temperature", type=float, default=0.7)
    p_run.add_argument("--max-step-tokens", type=int, default=384)
    p_run.add_argument("--prereg", default=None)
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--index", type=int, default=None, help="run only the plan at this index")
    p_run.add_argument("--out-root", default="runs")
    p_bank = sub.add_parser("bank", help="bank finished runs into artifacts")
    p_bank.add_argument("--experiment", required=True)
    p_bank.add_argument("--runs", nargs="+", required=True)
    p_bank.add_argument("--out", required=True)
    p_bank.add_argument("--prereg", default=None)
    p_bank.add_argument("--design", default=None)
    p_val = sub.add_parser("e0-validate", help="probe leakage and compaction logs from banked runs")
    p_val.add_argument("--banked", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.command == "plan":
        return cmd_plan(args.design, args.out)
    if args.command == "run":
        return cmd_run(args, root)
    if args.command == "bank":
        return cmd_bank(args, root)
    return cmd_e0_validate(args, root)


if __name__ == "__main__":
    sys.exit(main())
