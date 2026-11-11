"""Tests for opportunity and session rows."""

from __future__ import annotations

from halflife import analysis
from halflife.schema import (
    ActionLog,
    CompactionPlan,
    Constraint,
    DetectorSpec,
    SessionPlan,
    Subtask,
    SubtaskSpan,
    Violation,
)


def _constraint(cid: str, ctype: str) -> Constraint:
    return Constraint(
        id=cid,
        type=ctype,  # type: ignore[arg-type]  # literal chosen by the test
        text=f"Rule {cid}.",
        reason="Because.",
        detector=DetectorSpec(kind="no_network"),
        recall_keywords=((cid,),),
    )


def _dp(sid: str, cid: str) -> Subtask:
    return Subtask.from_dict(
        {
            "id": sid,
            "kind": "decision_point",
            "prompt": "p",
            "constraint": cid,
            "solution": [{"tool": "done", "args": {}}],
            "tempting": [{"tool": "done", "args": {}}],
        }
    )


def _fixture() -> tuple[SessionPlan, ActionLog]:
    work = Subtask.from_dict({"id": "w0", "kind": "work", "prompt": "p", "solution": []})
    subtasks = (work, _dp("dp1", "c1"), _dp("dp2", "c2"), _dp("dp3", "c1"))
    plan = SessionPlan(
        session_id="s1",
        template_id="t1",
        template_sha256="0" * 64,
        seed=4,
        placement="first_user",
        phrasing="reason",
        compaction=CompactionPlan(strategy="full_summary", schedule=(5,), budget_tokens=100),
        mitigation="none",
        subtasks=subtasks,
        experiment="e1",
        confirmatory=True,
    )
    log = ActionLog(
        session_id="s1",
        template_id="t1",
        constraints=(_constraint("c1", "prohibition"), _constraint("c2", "procedure")),
        baseline_fs={},
        spans=[
            SubtaskSpan("w0", "work", 0, 4),
            SubtaskSpan("dp1", "decision_point", 4, 9, constraint="c1", success=True),
            SubtaskSpan(
                "dp2", "decision_point", 9, 12, constraint="c2", success=False, ended="subtask_cap"
            ),
        ],
    )
    return plan, log


def test_opportunity_rows() -> None:
    plan, log = _fixture()
    violations = [
        Violation("c1", "no_network", 2, "w0", "spontaneous"),
        Violation("c2", "changelog_procedure", 11, "dp2", "missing changelog"),
    ]
    facts = analysis.RunFacts(
        family="fam",
        model="m",
        compaction_indices=(5, 10),
        probes=({"checkpoint": "before_dp:dp1", "tool_call_index": 4, "recalled": {"c1": True}},),
    )
    rows = analysis.opportunity_rows(plan, log, violations, facts)
    assert [r["dp_id"] for r in rows] == ["dp1", "dp2"]
    first, second = rows
    assert first["position"] == 4 and first["compaction_count"] == 0
    assert first["since_last_compaction"] == 4
    assert first["violated"] is False and first["recalled_before"] is True
    assert first["prior_violations"] == 1 and first["constraint_type"] == "prohibition"
    assert second["position"] == 9 and second["compaction_count"] == 1
    assert second["since_last_compaction"] == 4
    assert second["violated"] is True and second["recalled_before"] is None
    assert second["task_success"] is False and second["placement"] == "first_user"
    # D-050 item 5: how the span ended (null for logs written before spans recorded it)
    assert first["span_end"] is None and second["span_end"] == "subtask_cap"


def test_span_end_round_trips_and_stays_out_of_old_logs() -> None:
    """A span written before ``ended`` existed serializes exactly as before."""
    old = SubtaskSpan("dp1", "decision_point", 4, 9, constraint="c1", success=True)
    assert "ended" not in old.to_dict() and SubtaskSpan.from_dict(old.to_dict()) == old
    cut = SubtaskSpan("dp1", "decision_point", 4, 9, constraint="c1", ended="tool_cap")
    assert cut.to_dict()["ended"] == "tool_cap" and SubtaskSpan.from_dict(cut.to_dict()) == cut


def test_subtask_level_violation_counts_for_its_span() -> None:
    plan, log = _fixture()
    violations = [Violation("c1", "changelog_procedure", 8, "dp1", "end of subtask")]
    rows = analysis.opportunity_rows(plan, log, violations, analysis.RunFacts("f", "m"))
    assert rows[0]["violated"] is True


def test_probe_without_constraint_score() -> None:
    plan, log = _fixture()
    facts = analysis.RunFacts("f", "m", probes=({"checkpoint": "before_dp:dp1", "recalled": {}},))
    rows = analysis.opportunity_rows(plan, log, [], facts)
    assert rows[0]["recalled_before"] is None


def test_session_row() -> None:
    plan, log = _fixture()
    violations = [Violation("c2", "changelog_procedure", 11, "dp2", "x")]
    facts = analysis.RunFacts("fam", "m", compaction_indices=(5,), prereg="prereg/0001-e1.md")
    row = analysis.session_row(plan, log, violations, facts)
    assert row["reached_first_dp"] is True
    assert row["dps_reached"] == 2 and row["dps_planned"] == 3
    assert row["n_violations"] == 1 and row["first_violation_index"] == 11
    assert row["subtasks_succeeded"] == 1 and row["subtasks_checked"] == 2
    assert row["prereg"] == "prereg/0001-e1.md" and row["confirmatory"] is True


def test_session_row_without_dps() -> None:
    plan, log = _fixture()
    log.spans = [SubtaskSpan("w0", "work", 0, 3)]
    row = analysis.session_row(plan, log, [], analysis.RunFacts("f", "m", ended_reason="tool_cap"))
    assert row["reached_first_dp"] is False
    assert row["first_violation_index"] is None
    assert row["ended_reason"] == "tool_cap"


def test_recall_is_paired_with_the_compactions_the_probe_saw() -> None:
    # A compaction at index 9, where dp2 starts, runs after the before_dp:dp2 probe and
    # before dp2's first call: the hazard counts it (the agent decides in the compacted
    # context), but the probe that gives recalled_before never saw it.
    plan, log = _fixture()
    facts = analysis.RunFacts("fam", "m", compaction_indices=(5, 9), prereg="prereg/0001-e1.md")
    first, second = analysis.opportunity_rows(plan, log, [], facts)
    assert first["compaction_count"] == first["compactions_before_probe"] == 0
    assert second["position"] == 9 and second["compaction_count"] == 2
    assert second["compactions_before_probe"] == 1
    assert first["prereg"] == second["prereg"] == "prereg/0001-e1.md"


def test_a_reminder_that_rides_on_the_decision_point_request_is_flagged() -> None:
    import dataclasses

    from halflife.schema import ToolEvent

    # Under reminder placement, a reminder due on the done call that ends the previous
    # subtask is sent with the decision-point request, after the before_dp probe ran.
    plan, log = _fixture()
    plan = dataclasses.replace(plan, placement="reminder", reminder_every=9)
    log.events = [
        ToolEvent(i, "x", "done" if i in (3, 8) else "read_file", {}, "", None, {})  # type: ignore[arg-type]  # tool names chosen by the test
        for i in range(12)
    ]
    first, second = analysis.opportunity_rows(plan, log, [], analysis.RunFacts("f", "m"))
    assert (first["reminder_on_request"], second["reminder_on_request"]) == (False, True)
    log.events[8] = dataclasses.replace(log.events[8], blocked=True)  # a refused done
    rows = analysis.opportunity_rows(plan, log, [], analysis.RunFacts("f", "m"))
    assert rows[1]["reminder_on_request"] is False


def test_session_row_records_its_plan_and_every_planned_check() -> None:
    import dataclasses
    import hashlib
    import json

    from halflife.schema import SuccessCheck

    plan, log = _fixture()
    check = SuccessCheck("true")
    subtasks = tuple(
        dataclasses.replace(s, success=check) if s.id in {"dp1", "dp3"} else s
        for s in plan.subtasks
    )
    plan = dataclasses.replace(plan, subtasks=subtasks)
    row = analysis.session_row(plan, log, [], analysis.RunFacts("f", "m"))
    canonical = json.dumps(plan.to_dict(), sort_keys=True)
    assert row["plan_sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert row["plan_sha256"] == analysis.plan_sha256(plan)
    assert row["template_sha256"] == plan.template_sha256
    # dp3 has a check but the session never reached it: it is planned, not checked.
    assert row["subtasks_planned_checked"] == 2 and row["subtasks_checked"] == 2


def test_time_to_first_violation() -> None:
    rows = [
        {"family": "f", "session_id": "a", "position": 10, "violated": False},
        {"family": "f", "session_id": "a", "position": 30, "violated": True},
        {"family": "f", "session_id": "a", "position": 50, "violated": True},
        {"family": "f", "session_id": "b", "position": 12, "violated": False},
        {"family": "f", "session_id": "b", "position": 40, "violated": False},
    ]
    assert analysis.time_to_first_violation(rows) == [
        (("f", "a"), 30, True),
        (("f", "b"), 40, False),
    ]


def test_time_to_first_violation_is_per_family_session() -> None:
    # Both families ran plan p (the same session_id); each session has its own time.
    rows = [
        {"family": "fa", "session_id": "p", "position": 5, "violated": True},
        {"family": "fb", "session_id": "p", "position": 9, "violated": False},
        {"family": "fb", "session_id": "p", "position": 20, "violated": True},
        {"family": "fb", "session_id": "q", "position": 7, "violated": False},
    ]
    assert analysis.SESSION_KEY == ("family", "session_id")
    assert analysis.time_to_first_violation(rows) == [
        (("fa", "p"), 5, True),
        (("fb", "p"), 20, True),
        (("fb", "q"), 7, False),
    ]
    merged = analysis.time_to_first_violation(rows, by=("session_id",))
    assert merged == [(("p",), 5, True), (("q",), 7, False)]


def test_activity_fields() -> None:
    from halflife.schema import ToolEvent

    plan, log = _fixture()
    log.baseline_fs = {"a.py": "h0"}

    def event(i: int, tool: str, fs: str, blocked: bool = False) -> ToolEvent:
        return ToolEvent(i, "x", tool, {}, "", None, {"a.py": fs}, blocked=blocked)  # type: ignore[arg-type]  # tool names chosen by the test

    log.events = [event(i, "read_file", "h0") for i in range(4)]
    log.events += [event(4, "read_file", "h0"), event(5, "write_file", "h1")]
    log.events += [event(i, "read_file", "h1") for i in range(6, 9)]
    log.events += [
        event(9, "run", "h1", blocked=True),
        event(10, "done", "h1"),
        event(11, "done", "h1"),
    ]
    rows = analysis.opportunity_rows(plan, log, [], analysis.RunFacts("f", "m"))
    first, second = rows
    assert (first["span_calls"], first["acted"], first["changed_workspace"]) == (5, True, True)
    assert (second["span_calls"], second["acted"], second["changed_workspace"]) == (3, False, False)
