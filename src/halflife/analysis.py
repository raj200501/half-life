"""Opportunity and session rows from finished runs (docs/DESIGN.md, Key definitions).

An opportunity is one decision point (DP) that a session reached. Its position
is the tool call index at which the DP subtask starts, and its outcome is
whether a detector for the DP's constraint fired during the DP subtask. These
rows are the input to :mod:`halflife.stats`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from halflife.schema import ActionLog, SessionPlan, SubtaskSpan, Violation


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
    """
    types = {c.id: c.type for c in log.constraints}
    rows: list[dict[str, Any]] = []
    for span in log.spans:
        if span.kind != "decision_point" or span.constraint is None:
            continue
        compactions_before = [i for i in facts.compaction_indices if i <= span.start]
        last = compactions_before[-1] if compactions_before else None
        prior = sum(1 for v in violations if v.index < span.start)
        calls, acted, changed = _activity(log, span)
        rows.append(
            {
                "session_id": plan.session_id,
                "experiment": plan.experiment,
                "confirmatory": plan.confirmatory,
                "family": facts.family,
                "model": facts.model,
                "template_id": plan.template_id,
                "seed": plan.seed,
                "dp_id": span.subtask_id,
                "constraint_id": span.constraint,
                "constraint_type": types[span.constraint],
                "position": span.start,
                "compaction_count": len(compactions_before),
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
    """One summary row per session (the ``sessions.jsonl`` layout in docs/ARTIFACTS.md)."""
    dp_spans = [s for s in log.spans if s.kind == "decision_point"]
    successes = [s.success for s in log.spans if s.success is not None]
    return {
        "session_id": plan.session_id,
        "experiment": plan.experiment,
        "confirmatory": plan.confirmatory,
        "prereg": facts.prereg,
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
        "ended_reason": facts.ended_reason,
    }


def time_to_first_violation(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, int, bool]]:
    """Per session: (session_id, time, event) for descriptive Kaplan-Meier curves.

    Time is the position of the first violated opportunity, or of the last
    opportunity reached when none was violated (censored).
    """
    by_session: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_session.setdefault(str(row["session_id"]), []).append(row)
    out: list[tuple[str, int, bool]] = []
    for sid in sorted(by_session):
        ordered = sorted(by_session[sid], key=lambda r: int(r["position"]))
        hit = next((r for r in ordered if r["violated"]), None)
        if hit is not None:
            out.append((sid, int(hit["position"]), True))
        else:
            out.append((sid, int(ordered[-1]["position"]), False))
    return out
