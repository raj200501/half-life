"""Scripted sessions: execute fixed action sequences for real and log them.

:func:`run_scripted_session` is what E0 seeded logs, bench verification, and
the offline demo use. Every action runs through :func:`halflife.tools.execute`
in a real :class:`~halflife.workspace.Workspace`, so the resulting
:class:`~halflife.schema.ActionLog` has real ``fs_after`` hashes, outputs, and
sandbox network logs. Harness checks run in a throwaway copy of the workspace
(so they can never change what the agent sees) at baseline (subtask
``baseline``, ``after_index`` -1) and at the end of every subtask.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from halflife.sandbox import SandboxOptions, SandboxResult, run_with
from halflife.schema import (
    Action,
    ActionLog,
    Constraint,
    HarnessCheck,
    HarnessCheckSpec,
    Subtask,
    SubtaskKind,
    SubtaskSpan,
    SuccessCheck,
    Template,
)
from halflife.tools import execute
from halflife.workspace import Workspace

BASELINE_SUBTASK = "baseline"
"""``subtask_id`` of harness checks run before tool call 0."""

DEFAULT_CHECK_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class StepSpec:
    """A lightweight subtask record for scripted steps (instead of a full Subtask).

    Attributes:
        id: Subtask id used in events and spans.
        kind: ``work`` or ``decision_point``.
        constraint: Constraint id a decision point tests.
        success: Optional completion check run at the subtask end.
    """

    id: str
    kind: SubtaskKind = "work"
    constraint: str | None = None
    success: SuccessCheck | None = None


StepLike = Subtask | StepSpec
Step = tuple[StepLike, Sequence[Action]]


def run_check(
    workspace: Workspace, cmd: str, options: SandboxOptions | None = None
) -> SandboxResult:
    """Run a harness command in a scratch copy of the workspace through the sandbox."""
    opts = options or SandboxOptions(timeout_s=DEFAULT_CHECK_TIMEOUT_S)
    with workspace.scratch_copy() as copy:
        return run_with(cmd, copy, opts)


def _harness_checks(
    workspace: Workspace,
    checks: Sequence[HarnessCheckSpec],
    subtask_id: str,
    after_index: int,
    options: SandboxOptions,
) -> list[HarnessCheck]:
    return [
        HarnessCheck(
            check_id=spec.id,
            subtask_id=subtask_id,
            after_index=after_index,
            exit_code=run_check(workspace, spec.cmd, options).exit_code,
        )
        for spec in checks
    ]


def run_scripted_session(
    files: Mapping[str, str],
    constraints: Sequence[Constraint],
    checks: Sequence[HarnessCheckSpec],
    steps: Sequence[Step],
    *,
    session_id: str = "scripted",
    template_id: str = "scripted",
    options: SandboxOptions | None = None,
    check_options: SandboxOptions | None = None,
    uncommitted: Mapping[str, str] | None = None,
) -> ActionLog:
    """Execute scripted steps for real and return the complete action log.

    Args:
        files: Seed repository (path to text).
        constraints: Constraints in force (copied into the log).
        checks: Harness checks run at baseline and after every subtask.
        steps: ``(subtask, actions)`` pairs in order; ``subtask`` is a
            :class:`~halflife.schema.Subtask` or a :class:`StepSpec`.
        session_id: Session id for the log.
        template_id: Template id for the log.
        options: Sandbox settings for ``run`` tool calls.
        check_options: Sandbox settings for harness and success checks.
        uncommitted: Files written after the baseline commit (local edits that
            ``git checkout --`` would discard); see
            :class:`~halflife.workspace.Workspace`.

    Returns:
        An :class:`~halflife.schema.ActionLog` with events, spans (``success``
        set when the subtask has a success check), harness checks, the
        baseline snapshot, and every text blob seen.
    """
    tool_opts = options or SandboxOptions()
    check_opts = check_options or SandboxOptions(
        timeout_s=DEFAULT_CHECK_TIMEOUT_S,
        method=tool_opts.method,
    )
    with Workspace(files, uncommitted=uncommitted) as workspace:
        log = ActionLog(
            session_id=session_id,
            template_id=template_id,
            constraints=tuple(constraints),
            baseline_fs=dict(workspace.baseline),
        )
        log.checks.extend(_harness_checks(workspace, checks, BASELINE_SUBTASK, -1, check_opts))
        index = 0
        for subtask, actions in steps:
            start = index
            for action in actions:
                event = execute(
                    action, workspace, index=index, subtask_id=subtask.id, options=tool_opts
                )
                log.events.append(event)
                index += 1
            success: bool | None = None
            if subtask.success is not None:
                result = run_check(workspace, subtask.success.cmd, check_opts)
                success = result.exit_code == subtask.success.expect_exit
            log.spans.append(
                SubtaskSpan(
                    subtask_id=subtask.id,
                    kind=subtask.kind,
                    start=start,
                    end=index,
                    constraint=subtask.constraint,
                    success=success,
                )
            )
            log.checks.extend(_harness_checks(workspace, checks, subtask.id, index - 1, check_opts))
        log.blobs = dict(sorted(workspace.blobs.items()))
    return log


def template_steps(template: Template, *, tempting: Collection[str] = ()) -> list[Step]:
    """Steps for a template's core subtasks in order.

    Each subtask uses its compliant route (``solution``), except decision points
    whose id is in ``tempting``, which use their tempting route.

    Raises:
        KeyError: if ``tempting`` names an id that is not a decision point.
    """
    dp_ids = {s.id for s in template.subtasks if s.is_dp}
    unknown = sorted(set(tempting) - dp_ids)
    if unknown:
        raise KeyError(f"not decision points of {template.id}: {', '.join(unknown)}")
    return [
        (subtask, subtask.tempting if subtask.id in tempting else subtask.solution)
        for subtask in template.subtasks
    ]


def run_template(
    template: Template,
    *,
    tempting: Collection[str] = (),
    session_id: str | None = None,
    options: SandboxOptions | None = None,
    check_options: SandboxOptions | None = None,
) -> ActionLog:
    """Run a template's core subtasks (see :func:`template_steps`) as a scripted session."""
    suffix = "tempting" if tempting else "compliant"
    return run_scripted_session(
        template.files,
        template.constraints,
        template.checks,
        template_steps(template, tempting=tempting),
        session_id=session_id or f"{template.id}-{suffix}",
        template_id=template.id,
        options=options,
        check_options=check_options,
    )
