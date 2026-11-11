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
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml

from halflife import bench
from halflife.schema import SchemaError, SessionPlan
from halflife.sessions import Condition, PlanSettings, generate_plan

if TYPE_CHECKING:
    from halflife.agent import Guard
    from halflife.schema import Template
    from halflife.workspace import Workspace

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


def _command() -> str:
    """The command line in ``python -m`` form: no absolute script path, so no home directory."""
    return "python -m halflife.experiment " + " ".join(sys.argv[1:])


class BankError(ValueError):
    """Raised when finished runs cannot be banked as asked (nothing is written then)."""


SESSION_FILES = ("plan.json", "action_log.json", "violations.json")
"""Session outputs whose SHA-256 every banked session row records."""


def _line_sha256(data: Any) -> str:
    """SHA-256 of one canonical JSON line (the plans-file form of a plan)."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def _log_hashes(path: Path) -> set[str]:
    """The hashes a run may have recorded for a compaction log: of the file's bytes (the E2
    driver) and of its canonical JSON without the final newline
    (``halflife.compaction.CompactionEvent.sha256``)."""
    raw = path.read_bytes()
    found = {hashlib.sha256(raw).hexdigest()}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return found
    canonical = json.dumps(parsed, sort_keys=True, indent=2).encode("utf-8")
    return found | {hashlib.sha256(canonical).hexdigest()}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_record(run_dir: Path) -> dict[str, Any]:
    """A run's ``run.json`` (None when missing) and every ``invocations.jsonl`` line."""
    invocations = run_dir / "invocations.jsonl"
    lines = invocations.read_text(encoding="utf-8").splitlines() if invocations.is_file() else []
    return {
        "run": _read_json(run_dir / "run.json"),
        "invocations": [json.loads(line) for line in lines if line.strip()],
    }


def _run_plans(root: Path, record: Mapping[str, Any]) -> tuple[dict[str, str] | None, str]:
    """``session_id -> plan line SHA-256`` of the plans file a run's ``run.json`` names.

    The second value names a problem (an empty string when there is none): no run record,
    a missing plans file, or a plans file whose hash differs from the one the run recorded.
    """
    run = record.get("run")
    if not isinstance(run, Mapping):
        return None, "no run.json, so its sessions cannot be checked against their plans file"
    path = Path(str(run.get("plans", "")))
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        return None, f"its plans file is missing ({run.get('plans')!r})"
    if _sha256(path) != run.get("plans_sha256"):
        return None, f"its plans file changed since the run ({run.get('plans')!r})"
    plans: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            plans[str(data["session_id"])] = _line_sha256(data)
    return plans, ""


def bank_problems(
    found: Sequence[tuple[Path, dict[str, Any]]],
    records: Mapping[str, Mapping[str, Any]],
    root: Path,
    *,
    experiment: str,
    prereg: str | None,
) -> list[str]:
    """Why these finished sessions cannot be banked under ``experiment`` and ``prereg``.

    Each session must have run under the preregistration it is banked under (the tag
    ``experiment run --prereg`` recorded, never rewritten here) and belong to the banked
    experiment; its stored plan must be the plan the run's plans file holds for it (a
    resumed run cannot mix plan versions); and every compaction log must still have the
    hash its session recorded. A confirmatory bank also needs each run's ``run.json``.
    """
    problems: list[str] = []
    plans: dict[str, tuple[dict[str, str] | None, str]] = {
        run_id: _run_plans(root, record) for run_id, record in records.items()
    }
    for session_dir, record in found:
        session = record["session"]
        run_id, sid = str(record["run_id"]), str(session["session_id"])
        where = f"{run_id}/{sid}"
        recorded = session.get("prereg")
        if recorded != prereg:
            problems.append(
                f"{where} ran under preregistration {recorded!r}, not {prereg!r} "
                "(experiment run --prereg records it; banking never changes it)"
            )
        if session.get("experiment") != experiment:
            problems.append(f"{where} is a session of experiment {session.get('experiment')!r}")
        stored = _read_json(session_dir / "plan.json")
        plan_hash = _line_sha256(stored) if stored is not None else None
        if plan_hash is None:
            problems.append(f"{where} has no plan.json")
        elif session.get("plan_sha256") not in (None, plan_hash):
            problems.append(f"{where}: plan.json changed after the session ran")
        expected, problem = plans.get(run_id, (None, "no run record"))
        if expected is None:
            if prereg is not None:
                problems.append(f"run {run_id}: {problem}")
        elif sid not in expected:
            problems.append(f"{where} is not in the run's plans file")
        elif plan_hash is not None and expected[sid] != plan_hash:
            problems.append(f"{where} ran a different plan than the run's plans file holds")
        for event in record["compactions"]:
            path = session_dir / event["log_path"]
            if not path.is_file() or event.get("log_sha256") not in _log_hashes(path):
                problems.append(f"{where}: compaction log changed since its session")
    return sorted(set(problems))


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

    Nothing is written when :func:`bank_problems` finds a session that did not run as it
    would be banked (its run-time preregistration tag, experiment, or plan) or a
    compaction log that changed. Rows keep the run-time preregistration tag (opportunity
    rows get their session's when they lack one), and each session row records the
    SHA-256 of its plan, action log, and violations, its run settings, and the run-time
    ``run_meta.json`` (git SHA, dirty flag, times) when ``experiment run`` wrote one;
    provenance records every run's ``run.json`` and ``invocations.jsonl``.

    Raises:
        BankError: with every problem found.
    """
    found = collect(run_dirs)
    records = {run_dir.name: run_record(run_dir) for run_dir in run_dirs}
    problems = bank_problems(found, records, root, experiment=experiment, prereg=prereg)
    if problems:
        raise BankError(f"refusing to bank {len(problems)} problem(s): " + "; ".join(problems))
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions: list[dict[str, Any]] = []
    opportunities: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for session_dir, record in found:
        sid = record["session"]["session_id"]
        row = dict(record["session"]) | {"run_id": record["run_id"]}
        row["probe_leaks"] = list(record.get("summary", {}).get("leaks", []))
        row["usage"] = _read_json(session_dir / "usage.json")
        row["run_meta"] = _read_json(session_dir / "run_meta.json")
        row["run_settings"] = record.get("settings")
        row["files_sha256"] = {
            name: _sha256(session_dir / name) if (session_dir / name).is_file() else None
            for name in SESSION_FILES
        }
        sessions.append(row)
        for opportunity in record["opportunities"]:
            banked = dict(opportunity) | {"run_id": record["run_id"]}
            banked.setdefault("prereg", record["session"].get("prereg"))
            opportunities.append(banked)
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
                    "run_log_sha256": event.get("log_sha256"),
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
        "run_records": {run_id: records[run_id] for run_id in sorted(records)},
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


PREREG_PATH_RE = re.compile(r"prereg/\d{4}-[A-Za-z0-9][A-Za-z0-9_-]*\.md")
"""The naming rule of ``make prereg-verify`` (and gate G7) for a preregistration file."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )


def prereg_problem(root: Path, prereg: str | None, plans: Sequence[SessionPlan]) -> str | None:
    """Why these plans may not run under ``prereg`` (rule 2), or None.

    Confirmatory plans need ``--prereg`` naming a ``prereg/NNNN-slug.md`` file that is
    committed and unchanged at HEAD, so every confirmatory session carries the tag of a
    preregistration that existed before it ran; a tag on plans that are not confirmatory
    is refused too.
    """
    confirmatory = [plan.session_id for plan in plans if plan.confirmatory]
    if prereg is None:
        if confirmatory:
            return (
                f"confirmatory plans need --prereg naming a committed preregistration "
                f"(first: {confirmatory[0]})"
            )
        return None
    loose = [plan.session_id for plan in plans if not plan.confirmatory]
    if loose:
        return f"--prereg {prereg} was given, but plan {loose[0]} is not confirmatory"
    if not PREREG_PATH_RE.fullmatch(prereg):
        return f"--prereg {prereg!r} is not a prereg/NNNN-slug.md path"
    path = root / prereg
    if path.is_symlink() or not path.is_file():
        return f"--prereg {prereg} does not exist as a regular file"
    if _git(root, "ls-files", "--error-unmatch", "--", prereg).returncode != 0:
        return f"--prereg {prereg} is not committed"
    if _git(root, "diff", "--quiet", "HEAD", "--", prereg).returncode != 0:
        return f"--prereg {prereg} has uncommitted changes"
    return None


def stored_plan_problems(run_dir: Path, plans: Sequence[SessionPlan]) -> list[str]:
    """Sessions of ``run_dir`` whose stored ``plan.json`` differs from the plan to run.

    The runner skips a session whose result exists, so resuming a run after its plans
    were regenerated (a template edit or a design change keeps the session id) would mix
    plan versions; such a run must use a new run id instead.
    """
    out = []
    for plan in plans:
        stored = _read_json(run_dir / plan.session_id / "plan.json")
        if stored is not None and _line_sha256(stored) != _line_sha256(plan.to_dict()):
            out.append(
                f"{plan.session_id}: the stored plan differs from the plan being run "
                f"(use a new run id, or remove {run_dir.name}/{plan.session_id})"
            )
    return out


def _source_dirty(root: Path) -> bool:
    """True when tracked files under ``src/`` or ``designs/`` differ from HEAD."""
    status = _git(root, "status", "--porcelain", "--untracked-files=no", "--", "src", "designs")
    return bool(status.stdout.strip())


def cmd_plan(design_path: Path, out: Path) -> int:
    """Write the plans of a design file as JSON lines."""
    from halflife.runner import write_plans

    design = load_design(design_path)
    plans = make_plans(design, bench.load_all())
    write_plans(out, plans)
    print(f"wrote {len(plans)} plans for {design.experiment} to {out}")
    return 0


@dataclass(frozen=True)
class KitHooks:
    """The E4 mitigation hooks from :mod:`halflife.kit` and their provenance.

    The agent uses each hook only in its own arm (``guard``, ``preserve``,
    ``reinject``), so passing them to every run leaves other arms unchanged.
    """

    summary_prompt: str
    reinject_text: Callable[[Sequence[str]], str]
    guard_factory: Callable[[Template, Workspace], Guard]
    provenance: dict[str, str]


def kit_hooks() -> KitHooks:
    """Build the hooks that ``python -m halflife.experiment run`` passes to every session."""
    from halflife.kit.compaction_prompt import preserve_prompt
    from halflife.kit.guard import minimal_agent_guard
    from halflife.kit.reinject import reinject_text

    prompt = preserve_prompt()

    def guard_factory(template: Template, workspace: Workspace) -> Guard:
        return minimal_agent_guard(template.constraints, workspace.root)

    return KitHooks(
        summary_prompt=prompt,
        reinject_text=reinject_text,
        guard_factory=guard_factory,
        provenance={
            "guard": "halflife.kit.guard.minimal_agent_guard",
            "preserve_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "reinject": "halflife.kit.reinject.reinject_text",
            "reinject_example_sha256": hashlib.sha256(
                reinject_text(["<rule one>", "<rule two>"]).encode("utf-8")
            ).hexdigest(),
        },
    )


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
    run_dir = root / args.out_root / args.run_id
    problems = check_plans(plans, templates)
    problems += [p for p in [prereg_problem(root, args.prereg, plans)] if p is not None]
    problems += stored_plan_problems(run_dir, plans)
    if problems:
        for problem in problems:
            print(f"refusing: {problem}")
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    pid_file = run_dir / "pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    spec = models_mod.get_spec(args.model)
    hooks = kit_hooks()
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
        "git_dirty": _source_dirty(root),
        "sandbox_method": detect_method(),
        "subject_python": _subject_python_version(),
        "command": _command(),
        "started": _now(),
        "cpu_count": os.cpu_count(),
        "kit": hooks.provenance,
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
        summary_prompt=hooks.summary_prompt,
        reinject_text=hooks.reinject_text,
    )
    progress = run_dir / "progress.jsonl"
    status = 0
    for plan in plans:
        model.session_id = plan.session_id
        model.reset_usage()
        start = dt.datetime.now(dt.UTC)
        try:
            result = run_session(
                plan,
                templates[plan.template_id].template,
                model,
                detect_all,
                ctx,
                root / args.out_root,
                guard_factory=hooks.guard_factory,
            )
        except (CapExceeded, ModelCallsDisabled) as exc:
            print(f"stopping: {type(exc).__name__}: {exc}")
            status = 3
            break
        usage = model.reset_usage()
        if usage:  # empty when the session was already finished and only loaded
            session_dir = root / args.out_root / args.run_id / plan.session_id
            (session_dir / "usage.json").write_text(
                json.dumps(usage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            # The code version and times of the invocation that ran this session, so the
            # banked row can show it ran after its preregistration and under which code.
            meta = {
                "run_id": args.run_id,
                "prereg": args.prereg,
                "git_sha": record["git_sha"],
                "git_dirty": record["git_dirty"],
                "invocation_started": record["started"],
                "started": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "finished": _now(),
            }
            (session_dir / "run_meta.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        line = {
            "session_id": plan.session_id,
            "ended_reason": result["summary"]["ended_reason"],
            "tool_calls": result["summary"]["tool_calls"],
            "violations": result["session"]["n_violations"],
            "opportunities": len(result["opportunities"]),
            "wall_s": round((dt.datetime.now(dt.UTC) - start).total_seconds(), 1),
            "finished": _now(),
            # False for a session whose result an earlier invocation wrote (only loaded).
            "ran_now": bool(usage),
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
    try:
        provenance = bank_runs(
            run_dirs,
            root / args.out,
            root,
            experiment=args.experiment,
            prereg=args.prereg,
            design_path=root / args.design if args.design else None,
            models=models,
            command=_command(),
        )
    except BankError as exc:
        print(f"refusing: {exc}")
        return 2
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
