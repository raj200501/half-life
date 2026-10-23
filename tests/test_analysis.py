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
            SubtaskSpan("dp2", "decision_point", 9, 12, constraint="c2", success=False),
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


def test_time_to_first_violation() -> None:
    rows = [
        {"session_id": "a", "position": 10, "violated": False},
        {"session_id": "a", "position": 30, "violated": True},
        {"session_id": "a", "position": 50, "violated": True},
        {"session_id": "b", "position": 12, "violated": False},
        {"session_id": "b", "position": 40, "violated": False},
    ]
    assert analysis.time_to_first_violation(rows) == [("a", 30, True), ("b", 40, False)]
