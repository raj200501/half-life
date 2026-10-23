"""Run session plans with the minimal agent and bank everything they produce.

Each session runs in a fresh sandboxed workspace. Outputs go to
``runs/<run_id>/<session_id>/`` (git-ignored); a session whose ``result.json``
exists is skipped, so an interrupted run resumes where it stopped. Banking a
finished run into ``artifacts/`` is a separate, explicit step.
"""

from __future__ import annotations

import json
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


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_session(
    out_dir: Path,
    result: SessionResult,
    violations: Sequence[Violation],
    ctx: RunContext,
) -> dict[str, Any]:
    """Write every output file for one session and return the result record."""
    plan = result.plan
    _write_json(out_dir / "plan.json", plan.to_dict())
    (out_dir / "action_log.json").write_text(result.log.dumps() + "\n", encoding="utf-8")
    compactions = []
    for event in result.compactions:
        rel = f"compactions/{event.event_id}.json"
        _write_json(out_dir / rel, event.to_dict())
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
    return save_session(out_dir, result, violations, ctx)


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
