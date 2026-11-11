"""Opportunity and session rows from finished runs (docs/DESIGN.md, Key definitions).

An opportunity is one decision point (DP) that a session reached. Its position
is the tool call index at which the DP subtask starts, and its outcome is
whether a detector for the DP's constraint fired during the DP subtask. These
rows are the input to :mod:`halflife.stats`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from halflife.schema import ActionLog, SessionPlan, SubtaskSpan, Violation

SESSION_KEY: tuple[str, ...] = ("family", "session_id")
"""The row fields that identify one session.

Every model family runs the same plans, and a row's ``session_id`` is its plan's id, so
one ``session_id`` appears once per family; only the pair names a session. Risk sets,
times to first violation, and session counts group rows by this key. E2 rows carry the
model tier instead (``("model", "session_id")``) and name that key where they group.
"""


@dataclass(frozen=True)
class RunFacts:
    """What a driver reports about a finished session besides its action log.

    Attributes:
        family: Model family label (for example ``qwen2.5``).
        model: Exact model id.
        compaction_indices: Tool call index of every compaction event, in order.
            A compaction at index ``i`` happened after ``i`` tool calls, before call ``i``.
        probes: Probe results; each has ``checkpoint`` (``before_dp:<subtask id>`` or
            ``after_compaction:<event id>``), ``tool_call_index``, and ``recalled``
            (constraint id to bool).
        ended_reason: ``completed``, ``tool_cap``, ``error``, or similar.
        prereg: Preregistration path for confirmatory sessions, else None.
    """

    family: str
    model: str
    compaction_indices: tuple[int, ...] = ()
    probes: tuple[Mapping[str, Any], ...] = ()
    ended_reason: str = "completed"
    prereg: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def _violated_in(span: SubtaskSpan, constraint_id: str, violations: Sequence[Violation]) -> bool:
    """True if a violation of ``constraint_id`` was detected inside the span."""
    return any(
        v.constraint_id == constraint_id
        and (span.start <= v.index < span.end or v.subtask_id == span.subtask_id)
        for v in violations
    )


ACTING_TOOLS = frozenset({"write_file", "replace_in_file", "run"})


def _activity(log: ActionLog, span: SubtaskSpan) -> tuple[int, bool, bool]:
    """Tool calls in the span, whether the agent acted (wrote, edited, or ran a
    command that was not blocked), and whether any workspace file changed."""
    events = [e for e in log.events if span.start <= e.index < span.end]
    acted = any(e.tool in ACTING_TOOLS and not e.blocked for e in events)
    before = log.fs_before(span.start) if span.start <= len(log.events) else {}
    after = events[-1].fs_after if events else before
    return len(events), acted, dict(before) != dict(after)


def plan_sha256(plan: SessionPlan) -> str:
    """SHA-256 of the plan's canonical JSON line (as ``runner.write_plans`` writes it)."""
    line = json.dumps(plan.to_dict(), sort_keys=True)
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _reminder_on_request(plan: SessionPlan, log: ActionLog, start: int) -> bool:
    """True when a reminder of the rules rode on the decision-point request.

    The minimal agent appends a reminder after every ``reminder_every``-th call; one that
    falls due on the unblocked ``done`` call that ends the previous subtask is sent with
    the next request, after the before-DP probe has run, so the probe did not see it.
    """
    every = plan.reminder_every
    if plan.placement != "reminder" or every <= 0 or start <= 0 or start % every != 0:
        return False
    previous = next((e for e in log.events if e.index == start - 1), None)
    return previous is not None and previous.tool == "done" and not previous.blocked


def _probe_before(
    probes: Sequence[Mapping[str, Any]], dp_id: str, constraint_id: str
) -> bool | None:
    """Recall of ``constraint_id`` in the probe run just before DP ``dp_id``, if any."""
    for probe in probes:
        if probe.get("checkpoint") == f"before_dp:{dp_id}":
            recalled = probe.get("recalled", {})
            value = recalled.get(constraint_id)
            return None if value is None else bool(value)
    return None


def opportunity_rows(
    plan: SessionPlan,
    log: ActionLog,
    violations: Sequence[Violation],
    facts: RunFacts,
) -> list[dict[str, Any]]:
    """One row per decision point reached, in session order.

    A DP counts as reached when the log has a span for it that starts before the
    session ended; a span that starts but records no tool calls still counts,
    because the agent saw the request.

    ``compaction_count`` counts compactions at indices up to the DP start: one at the
    start index runs before the DP's first call, so the agent decides in the compacted
    context (the hazard covariate). The before-DP probe runs before that compaction, so
    ``compactions_before_probe`` (indices strictly before the start) is the count the
    probe's ``recalled_before`` belongs with. ``reminder_on_request`` marks a DP whose
    request carried a reminder of the rules that the probe did not see.
    """
    types = {c.id: c.type for c in log.constraints}
    rows: list[dict[str, Any]] = []
    for span in log.spans:
        if span.kind != "decision_point" or span.constraint is None:
            continue
        compactions_before = [i for i in facts.compaction_indices if i <= span.start]
        seen_by_probe = sum(1 for i in facts.compaction_indices if i < span.start)
        last = compactions_before[-1] if compactions_before else None
        prior = sum(1 for v in violations if v.index < span.start)
        calls, acted, changed = _activity(log, span)
        rows.append(
            {
                "session_id": plan.session_id,
                "experiment": plan.experiment,
                "confirmatory": plan.confirmatory,
                "prereg": facts.prereg,
                "family": facts.family,
                "model": facts.model,
                "template_id": plan.template_id,
                "seed": plan.seed,
                "dp_id": span.subtask_id,
                "constraint_id": span.constraint,
                "constraint_type": types[span.constraint],
                "position": span.start,
                "compaction_count": len(compactions_before),
                "compactions_before_probe": seen_by_probe,
                "reminder_on_request": _reminder_on_request(plan, log, span.start),
                "since_last_compaction": span.start - last if last is not None else span.start,
                "placement": plan.placement,
                "phrasing": plan.phrasing,
                "strategy": plan.compaction.strategy,
                "mitigation": plan.mitigation,
                "violated": _violated_in(span, span.constraint, violations),
                "recalled_before": _probe_before(facts.probes, span.subtask_id, span.constraint),
                "task_success": span.success,
                "span_calls": calls,
                "acted": acted,
                "changed_workspace": changed,
                "prior_violations": prior,
            }
        )
    return rows


def session_row(
    plan: SessionPlan,
    log: ActionLog,
    violations: Sequence[Violation],
    facts: RunFacts,
) -> dict[str, Any]:
    """One summary row per session (the ``sessions.jsonl`` layout in docs/ARTIFACTS.md).

    ``subtasks_checked`` counts the success checks that ran (subtasks the session
    reached); ``subtasks_planned_checked`` counts the plan's subtasks with a check, so a
    checked subtask the session never reached (for example after the tool-call cap) can
    count as not passed. ``plan_sha256`` and ``template_sha256`` tie the row to the plan
    it ran.
    """
    dp_spans = [s for s in log.spans if s.kind == "decision_point"]
    successes = [s.success for s in log.spans if s.success is not None]
    return {
        "session_id": plan.session_id,
        "experiment": plan.experiment,
        "confirmatory": plan.confirmatory,
        "prereg": facts.prereg,
        "plan_sha256": plan_sha256(plan),
        "template_sha256": plan.template_sha256,
        "family": facts.family,
        "model": facts.model,
        "template_id": plan.template_id,
        "seed": plan.seed,
        "placement": plan.placement,
        "phrasing": plan.phrasing,
        "strategy": plan.compaction.strategy,
        "mitigation": plan.mitigation,
        "reached_first_dp": bool(dp_spans),
        "dps_reached": len(dp_spans),
        "dps_planned": len(plan.decision_points),
        "n_tool_calls": len(log.events),
        "n_compactions": len(facts.compaction_indices),
        "n_violations": len(violations),
        "first_violation_index": min((v.index for v in violations), default=None),
        "subtasks_succeeded": sum(1 for s in successes if s),
        "subtasks_checked": len(successes),
        "subtasks_planned_checked": sum(1 for s in plan.subtasks if s.success is not None),
        "ended_reason": facts.ended_reason,
    }


def time_to_first_violation(
    rows: Sequence[Mapping[str, Any]], by: Sequence[str] = SESSION_KEY
) -> list[tuple[tuple[str, ...], int, bool]]:
    """Per session: (session key, time, event) for descriptive Kaplan-Meier curves.

    Sessions are identified by the ``by`` fields (:data:`SESSION_KEY` by default), so the
    sessions two families ran from one plan stay apart. Time is the position of the first
    violated opportunity, or of the last opportunity reached when none was violated
    (censored).
    """
    by_session: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_session.setdefault(tuple(str(row[name]) for name in by), []).append(row)
    out: list[tuple[tuple[str, ...], int, bool]] = []
    for key in sorted(by_session):
        ordered = sorted(by_session[key], key=lambda r: int(r["position"]))
        hit = next((r for r in ordered if r["violated"]), None)
        if hit is not None:
            out.append((key, int(hit["position"]), True))
        else:
            out.append((key, int(ordered[-1]["position"]), False))
    return out
