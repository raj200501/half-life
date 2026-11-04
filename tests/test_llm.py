"""Tests for halflife.llm: ClientModel is the agent's ModelPort over the client."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from halflife.agent import TOOL_CALL_SCHEMA
from halflife.backends.base import ChatBackend, ChatRequest, ChatResponse, Message
from halflife.backends.scripted import ScriptedBackend
from halflife.budget import read_ledger
from halflife.client import NO_MODEL_CALLS_ENV, Client, ModelCallsDisabled
from halflife.compaction import STRUCTURED_SCHEMA, ModelPort, compact
from halflife.llm import ClientModel
from halflife.probe import PROBE_QUESTION, run_probe
from halflife.schema import Constraint, DetectorSpec

BUDGET = (
    "cash_usd = 0\nclaude_agent_runs_per_day = 30\nclaude_agent_max_turns = 40\n"
    "claude_single_calls_per_day = 300\nclaude_review_runs_per_day = 6\n"
    "local_model_max_params_b = 8\ndisk_floor_gb = 8\n"
)


class LocalFake:
    """A 'local model' that answers by purpose-shaped heuristics and records requests."""

    kind = "local"
    offline_safe = False
    params_b = 1.0
    name = "fake/local"

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        last = request.messages[-1].content
        if PROBE_QUESTION in last:
            text = "1. Never modify files under tests/."
        elif request.json_schema is STRUCTURED_SCHEMA or request.json_schema == STRUCTURED_SCHEMA:
            text = (
                '{"project": "p", "requests_done": [], "current_request": "c", '
                '"files_changed": [], "notes": ""}'
            )
        else:
            text = "Summary: worked on the parser."
        return ChatResponse(text, 10, 4, 0.01, backend=self.name)

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return sum(len(m.content) for m in messages)


def make(tmp_path: Path, backend: ChatBackend, env: dict[str, str] | None = None) -> Client:
    budget = tmp_path / "budget.toml"
    budget.write_text(BUDGET, encoding="utf-8")
    return Client(
        backend,
        tmp_path / "ledger.jsonl",
        tmp_path / "cache",
        run_id="r",
        experiment="test",
        budget_path=budget,
        env=env or {},
    )


def test_chat_builds_the_request_and_returns_text(tmp_path: Path) -> None:
    backend = ScriptedBackend(outputs=['{"thought": "t", "tool": "done", "args": {}}'])
    model = ClientModel(make(tmp_path, backend), session_id="s", stop=("</end>",))
    port: ModelPort = model
    text = port.chat(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}],
        max_tokens=50,
        temperature=0.7,
        seed=9,
        json_schema=TOOL_CALL_SCHEMA,
    )
    assert text.startswith('{"thought"')
    request = backend.requests[0]
    assert request.messages == (Message("system", "sys"), Message("user", "go"))
    assert (request.max_tokens, request.temperature, request.seed) == (50, 0.7, 9)
    assert request.json_schema == TOOL_CALL_SCHEMA
    assert request.stop == ("</end>",)
    assert model.last_response is not None and model.last_response.text == text
    assert port.count_tokens([{"role": "user", "content": "abcd"}]) == 1 + 4 + 2


def test_purposes_and_session_reach_the_ledger(tmp_path: Path) -> None:
    backend = LocalFake()
    model = ClientModel(make(tmp_path, backend))
    model.session_id = "sess-1"
    constraint = Constraint(
        id="no-tests",
        type="prohibition",
        text="Never modify files under tests/.",
        reason="The tests are the spec.",
        detector=DetectorSpec(kind="protected_paths", params={"globs": ["tests/**"]}),
        recall_keywords=(("tests",), ("never", "not")),
    )
    context = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Rules: Never modify files under tests/.\n\nFix the parser."},
        {"role": "assistant", "content": '{"thought": "x", "tool": "done", "args": {}}'},
    ]
    probe = run_probe(
        context, [constraint], model, checkpoint="before_dp:d1", tool_call_index=3, seed=4
    )
    assert probe.recalled == {"no-tests": True}
    _, event = compact(
        context,
        strategy="full_summary",
        model=model,
        event_id="c1",
        tool_call_index=3,
        scheduled=True,
        rendered_constraints={"no-tests": constraint.text},
        rules_message_index=1,
        continuation="Continue: fix the parser.",
        seed=5,
    )
    assert event.summary == "Summary: worked on the parser."
    _, structured = compact(
        context,
        strategy="structured_state",
        model=model,
        event_id="c2",
        tool_call_index=3,
        scheduled=False,
        rendered_constraints={},
        rules_message_index=None,
        continuation=None,
        seed=6,
    )
    assert structured.summary is not None and structured.summary.startswith("Project: p")
    rows = read_ledger(tmp_path / "ledger.jsonl")
    assert [row["purpose"] for row in rows] == ["probe", "compaction", "compaction"]
    assert {row["session_id"] for row in rows} == {"sess-1"}
    assert backend.requests[2].json_schema == STRUCTURED_SCHEMA
    assert list(backend.requests[2].json_schema or {}) == list(STRUCTURED_SCHEMA)


def test_model_port_respects_the_no_model_calls_switch(tmp_path: Path) -> None:
    model = ClientModel(make(tmp_path, LocalFake(), env={NO_MODEL_CALLS_ENV: "1"}))
    with pytest.raises(ModelCallsDisabled):
        model.chat([{"role": "user", "content": "x"}], max_tokens=5, temperature=0.0, seed=0)
    assert model.last_response is None
