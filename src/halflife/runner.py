"""Run session plans with the minimal agent and bank everything they produce.

Each session runs in a fresh sandboxed workspace. Outputs go to
``runs/<run_id>/<session_id>/`` (git-ignored); a session whose ``result.json``
exists is skipped, so an interrupted run resumes where it stopped. Banking a
finished run into ``artifacts/`` is a separate, explicit step.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from halflife.agent import AgentSettings, Guard, MinimalAgent, SessionResult
from halflife.analysis import RunFacts, opportunity_rows, session_row
from halflife.compaction import ModelPort
from halflife.sandbox import SandboxOptions
from halflife.schema import Action, SessionPlan, Template, ToolEvent, Violation
from halflife.scripted import DEFAULT_CHECK_TIMEOUT_S, run_check
from halflife.tools import execute
from halflife.workspace import Workspace

Detector = Callable[..., list[Violation]]
GuardFactory = Callable[[Template, Workspace], Guard]


class WorkspaceRunner:
    """The agent's :class:`~halflife.agent.ToolRunner` over a real sandboxed workspace."""

    def __init__(
        self,
        workspace: Workspace,
        options: SandboxOptions | None = None,
        check_options: SandboxOptions | None = None,
    ) -> None:
        self.workspace = workspace
        self.options = options or SandboxOptions()
        self.check_options = check_options or SandboxOptions(
            timeout_s=DEFAULT_CHECK_TIMEOUT_S, method=self.options.method
        )

    def execute(self, action: Action, *, index: int, subtask_id: str) -> ToolEvent:
        """Run one action in the workspace."""
        return execute(
            action, self.workspace, index=index, subtask_id=subtask_id, options=self.options
        )

    def snapshot(self) -> dict[str, str]:
        """Current file hashes."""
        return self.workspace.snapshot()

    def blobs(self) -> dict[str, str]:
        """Every text version seen."""
        return dict(sorted(self.workspace.blobs.items()))

    def run_check(self, cmd: str) -> int:
        """Run a harness command on a scratch copy; return its exit code."""
        return run_check(self.workspace, cmd, self.check_options).exit_code


@dataclass(frozen=True)
class RunContext:
    """Labels and settings shared by every session in a run."""

    run_id: str
    family: str
    model: str
    prereg: str | None = None
    settings: AgentSettings = field(default_factory=AgentSettings)
    sandbox: SandboxOptions = field(default_factory=SandboxOptions)
    summary_prompt: str | None = None
    reinject_text: Callable[[Sequence[str]], str] | None = None


SOURCE_ROOT = Path(__file__).resolve().parents[2]
"""The checkout this package runs from (not a git checkout when installed as a wheel)."""


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str | None:
    """Output of a git command in :data:`SOURCE_ROOT`, or None outside a checkout."""
    try:
        done = subprocess.run(
            ["git", "-C", str(SOURCE_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def run_provenance(ctx: RunContext, started: str | None) -> dict[str, Any]:
    """The run provenance stamped on every saved session (DECISIONS.md D-050, item 6).

    It names the run and its preregistration tag, the code version (the git SHA, and
    whether tracked files under ``src/`` or ``designs/`` differed from it), and when the
    session started and finished, so a banked session shows under which code and when it
    ran even when no model call was billed to it. Fields that cannot be known are null.
    """
    status = _git("status", "--porcelain", "--untracked-files=no", "--", "src", "designs")
    return {
        "run_id": ctx.run_id,
        "prereg": ctx.prereg,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "started": started,
        "finished": _utc_now(),
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_session(
    out_dir: Path,
    result: SessionResult,
    violations: Sequence[Violation],
    ctx: RunContext,
    started: str | None = None,
) -> dict[str, Any]:
    """Write every output file for one session and return the result record.

    The record carries :func:`run_provenance` under ``run_provenance``; ``started`` is
    the session's start time (UTC, ISO 8601) when the caller knows it.
    """
    plan = result.plan
    _write_json(out_dir / "plan.json", plan.to_dict())
    (out_dir / "action_log.json").write_text(result.log.dumps() + "\n", encoding="utf-8")
    compactions = []
    for event in result.compactions:
        rel = f"compactions/{event.event_id}.json"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / rel).write_bytes(event.to_json().encode("utf-8"))
        compactions.append(
            {
                "run_id": ctx.run_id,
                "session_id": plan.session_id,
                "event_id": event.event_id,
                "strategy": event.strategy,
                "tool_call_index": event.tool_call_index,
                "scheduled": event.scheduled,
                "tokens_before": event.tokens_before,
                "tokens_after": event.tokens_after,
                "constraint_text_retained": dict(event.constraint_text_retained),
                "log_path": rel,
                "log_sha256": event.sha256(),
            }
        )
    _write_json(out_dir / "probes.json", [p.to_dict() for p in result.probes])
    with (out_dir / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        for entry in result.transcript:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    _write_json(out_dir / "violations.json", [v.to_dict() for v in violations])
    facts = RunFacts(
        family=ctx.family,
        model=ctx.model,
        compaction_indices=tuple(e.tool_call_index for e in result.compactions),
        probes=tuple(p.to_dict() for p in result.probes),
        ended_reason=result.ended_reason,
        prereg=ctx.prereg,
    )
    record = {
        "run_id": ctx.run_id,
        "summary": result.summary(),
        "session": session_row(plan, result.log, violations, facts),
        "opportunities": opportunity_rows(plan, result.log, violations, facts),
        "compactions": compactions,
        "settings": {
            "temperature": ctx.settings.temperature,
            "max_step_tokens": ctx.settings.max_step_tokens,
            "keep_last": ctx.settings.keep_last,
            "selfcheck_every": ctx.settings.selfcheck_every,
        },
        "run_provenance": run_provenance(ctx, started),
    }
    _write_json(out_dir / "result.json", record)
    return record


def run_session(
    plan: SessionPlan,
    template: Template,
    model: ModelPort,
    detect_all: Detector,
    ctx: RunContext,
    out_root: Path,
    guard_factory: GuardFactory | None = None,
) -> dict[str, Any]:
    """Run one plan in a fresh workspace, detect violations, and save outputs."""
    out_dir = out_root / ctx.run_id / plan.session_id
    done = out_dir / "result.json"
    if done.exists():
        loaded: dict[str, Any] = json.loads(done.read_text(encoding="utf-8"))
        return loaded
    started = _utc_now()
    with Workspace(template.files) as workspace:
        runner = WorkspaceRunner(workspace, ctx.sandbox)
        guard = (
            guard_factory(template, workspace)
            if (guard_factory and plan.mitigation == "guard")
            else None
        )
        agent = MinimalAgent(
            plan,
            template,
            model,
            runner,
            settings=ctx.settings,
            guard=guard,
            reinject_text=ctx.reinject_text,
            summary_prompt=ctx.summary_prompt,
        )
        result = agent.run()
    violations = detect_all(result.log)
    return save_session(out_dir, result, violations, ctx, started)


def run_plans(
    plans: Iterable[SessionPlan],
    templates: Mapping[str, Template],
    model: ModelPort,
    detect_all: Detector,
    ctx: RunContext,
    out_root: Path,
    guard_factory: GuardFactory | None = None,
    on_done: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run plans in order (resumable) and return their result records."""
    records = []
    for plan in plans:
        record = run_session(
            plan, templates[plan.template_id], model, detect_all, ctx, out_root, guard_factory
        )
        records.append(record)
        if on_done is not None:
            on_done(record)
    return records


def load_plans(path: Path) -> list[SessionPlan]:
    """Read plans from a JSON-lines file."""
    return [
        SessionPlan.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_plans(path: Path, plans: Sequence[SessionPlan]) -> None:
    """Write plans as JSON lines (one canonical object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for plan in plans:
            handle.write(json.dumps(plan.to_dict(), sort_keys=True) + "\n")
