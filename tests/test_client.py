"""Tests for halflife.client: the no-model-calls switch, cache, caps, and usage ledger."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from halflife import client as client_module
from halflife.backends.base import ChatRequest, ChatResponse, Message, estimate_context_tokens
from halflife.backends.scripted import ScriptedBackend
from halflife.billing_guard import GuardReport
from halflife.budget import BudgetError, evaluate, read_ledger, validate_caps
from halflife.client import (
    LEDGER_KINDS,
    NO_MODEL_CALLS_ENV,
    BillingGuardRefused,
    CacheCorrupt,
    CapExceeded,
    Client,
    ClientStats,
    ModelCallsDisabled,
    append_jsonl,
    cache_key,
    env_flag,
)

BUDGET = """cash_usd = 0
claude_agent_runs_per_day = 30
claude_agent_max_turns = 40
claude_single_calls_per_day = 300
claude_review_runs_per_day = 6
local_model_max_params_b = 8
disk_floor_gb = 8
cap_share = 0.2
"""


class FakeModel:
    """A backend that 'runs a model': counts calls and answers deterministically."""

    def __init__(self, kind: str = "local", params_b: float = 3.0, name: str = "fake/m") -> None:
        self.kind = kind
        self.params_b = params_b
        self.offline_safe = False
        self.name = name
        self.model = "fake-model"
        self.calls = 0

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        text = f"reply {self.calls} to {request.messages[-1].content}"
        return ChatResponse(text, 11, 5, 0.25, backend=self.name)

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return 7 * len(messages)


def req(text: str = "hello", seed: int = 1) -> ChatRequest:
    return ChatRequest(messages=(Message("user", text),), max_tokens=8, temperature=0.7, seed=seed)


@pytest.fixture()
def budget(tmp_path: Path) -> Path:
    path = tmp_path / "budget.toml"
    path.write_text(BUDGET, encoding="utf-8")
    return path


def make_client(
    tmp_path: Path, budget: Path, backend: Any, env: dict[str, str] | None = None, **kw: Any
) -> Client:
    return Client(
        backend,
        tmp_path / "ledger.jsonl",
        tmp_path / "cache",
        run_id="run-1",
        experiment="test",
        budget_path=budget,
        env=env if env is not None else {},
        **kw,
    )


def test_real_call_is_logged_and_cached(tmp_path: Path, budget: Path) -> None:
    backend = FakeModel()
    ticks = iter([10.0, 10.5])
    client = make_client(tmp_path, budget, backend, clock=lambda: next(ticks))
    first = client.complete(req(), purpose="probe", session_id="s1")
    assert (first.text, first.cached) == ("reply 1 to hello", False)
    rows = read_ledger(tmp_path / "ledger.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert {k: row[k] for k in ("kind", "backend", "model", "run_id", "experiment")} == {
        "kind": "local",
        "backend": "fake/m",
        "model": "fake-model",
        "run_id": "run-1",
        "experiment": "test",
    }
    assert (row["session_id"], row["purpose"], row["prompt_tokens"]) == ("s1", "probe", 11)
    assert (row["completion_tokens"], row["wall_s"], row["params_b"]) == (5, 0.5, 3.0)
    assert (row["cash_usd"], row["plan_equivalent_usd"], row["cached"]) == (0, 0, False)
    assert dt.datetime.fromisoformat(row["ts"]).tzinfo is not None
    key = cache_key("fake/m", req())
    assert row["cache_key"] == key
    entry = json.loads(client.cache_path(key).read_text())
    assert entry["response"]["text"] == "reply 1 to hello"
    assert (entry["run_id"], entry["purpose"], entry["session_id"]) == ("run-1", "probe", "s1")

    again = client.complete(req(), purpose="probe", session_id="s1")
    assert (again.text, again.cached) == ("reply 1 to hello", True)
    assert backend.calls == 1
    assert len(read_ledger(tmp_path / "ledger.jsonl")) == 1
    assert client.stats == ClientStats(
        calls=1, cache_hits=1, prompt_tokens=11, completion_tokens=5, wall_s=0.5
    )
    assert client.stats.to_dict()["wall_s"] == 0.5


def test_cache_is_shared_across_clients_and_keyed_by_backend(tmp_path: Path, budget: Path) -> None:
    make_client(tmp_path, budget, FakeModel()).complete(req())
    fresh = FakeModel()
    assert make_client(tmp_path, budget, fresh).complete(req()).cached
    assert fresh.calls == 0
    other = FakeModel(name="fake/other")
    assert not make_client(tmp_path, budget, other).complete(req()).cached
    assert cache_key("a", req()) != cache_key("b", req())
    assert cache_key("a", req(seed=1)) != cache_key("a", req(seed=2))


def test_no_model_calls_switch(tmp_path: Path, budget: Path) -> None:
    for value in ("1", "true", " YES "):
        client = make_client(tmp_path, budget, FakeModel(), env={NO_MODEL_CALLS_ENV: value})
        with pytest.raises(ModelCallsDisabled, match="forbids model calls"):
            client.complete(req())
    assert not (tmp_path / "ledger.jsonl").exists()
    allowed = make_client(tmp_path, budget, FakeModel(), env={NO_MODEL_CALLS_ENV: "0"})
    assert allowed.complete(req()).text.startswith("reply 1")


def test_offline_backends_skip_cache_and_ledger(tmp_path: Path) -> None:
    backend = ScriptedBackend(outputs=["a", "b"])
    client = Client(
        backend,
        tmp_path / "ledger.jsonl",
        tmp_path / "cache",
        budget_path=tmp_path / "no-budget.toml",
        env={NO_MODEL_CALLS_ENV: "1"},
    )
    assert client.complete(req()).text == "a"
    assert client.complete(req()).text == "b"
    assert client.stats.offline_calls == 2
    assert not (tmp_path / "ledger.jsonl").exists()
    assert not (tmp_path / "cache").exists()
    assert client.count_tokens([{"role": "user", "content": "abcd"}]) == estimate_context_tokens(
        [Message("user", "abcd")]
    )
    assert client.model == "scripted"


def test_local_size_cap(tmp_path: Path, budget: Path) -> None:
    with pytest.raises(CapExceeded, match="local_model_max_params_b is 8"):
        make_client(tmp_path, budget, FakeModel(params_b=9.0))
    backend = FakeModel(params_b=3.0)
    client = make_client(tmp_path, budget, backend)
    backend.params_b = 12.0
    with pytest.raises(CapExceeded):
        client.complete(req())
    assert backend.calls == 0


def test_unknown_kind_and_bad_budget(tmp_path: Path, budget: Path) -> None:
    with pytest.raises(ValueError, match="not a ledger kind"):
        make_client(tmp_path, budget, FakeModel(kind="scripted"))
    loose = tmp_path / "loose.toml"
    loose.write_text(
        BUDGET.replace("local_model_max_params_b = 8", "local_model_max_params_b = 80")
    )
    with pytest.raises(BudgetError):
        make_client(tmp_path, loose, FakeModel())
    with pytest.raises(ValueError, match="purpose"):
        make_client(tmp_path, budget, FakeModel()).complete(req(), purpose=" ")


def test_corrupt_cache_is_never_silently_rerun(tmp_path: Path, budget: Path) -> None:
    backend = FakeModel()
    client = make_client(tmp_path, budget, backend)
    path = client.cache_path(cache_key(backend.name, req()))
    path.parent.mkdir(parents=True)
    path.write_text("{broken")
    with pytest.raises(CacheCorrupt, match="unreadable"):
        client.complete(req())
    client.complete(req("other"))
    other_entry = client.cache_path(cache_key(backend.name, req("other")))
    path.write_text(other_entry.read_text())
    with pytest.raises(CacheCorrupt, match="does not match"):
        client.complete(req())
    assert backend.calls == 1


def test_claude_kind_needs_clean_guard_and_daily_room(
    tmp_path: Path, budget: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        client_module.billing_guard,
        "check",
        lambda env: GuardReport(env_vars=("ANTHROPIC_BASE_URL",)),
    )
    backend = FakeModel(kind="claude_single_call", params_b=0.0)
    client = make_client(tmp_path, budget, backend)
    with pytest.raises(BillingGuardRefused, match="ANTHROPIC_BASE_URL"):
        client.complete(req())
    monkeypatch.setattr(client_module.billing_guard, "check", lambda env: GuardReport())
    assert client.complete(req()).text.startswith("reply")
    assert client.used_today("claude_single_call") == 1
    caps = client.caps
    assert caps["claude_single_calls_per_day"] == 60  # 300 scaled by cap_share 0.2
    now = client_module.utc_now().isoformat(timespec="seconds")
    for _ in range(59):
        append_jsonl(tmp_path / "ledger.jsonl", {"ts": now, "kind": "claude_single_call"})
    append_jsonl(
        tmp_path / "ledger.jsonl", {"ts": "2000-01-01T00:00:00+00:00", "kind": "claude_single_call"}
    )
    append_jsonl(
        tmp_path / "ledger.jsonl", {"ts": now, "kind": "claude_single_call", "cached": True}
    )
    assert client.used_today("claude_single_call") == 60
    with pytest.raises(CapExceeded, match="claude_single_calls_per_day = 60"):
        client.complete(req("new"))


def test_cap_share_from_environment_only_lowers(tmp_path: Path, budget: Path) -> None:
    lower = make_client(
        tmp_path, budget, FakeModel(kind="claude_review_run"), env={"CAP_SHARE": "0.1"}
    )
    assert lower.caps["claude_review_runs_per_day"] == 0
    higher = make_client(
        tmp_path, budget, FakeModel(kind="claude_review_run"), env={"CAP_SHARE": "5"}
    )
    assert higher.caps["claude_review_runs_per_day"] == 1


def test_free_tier_needs_confirmation(tmp_path: Path, budget: Path) -> None:
    client = make_client(tmp_path, budget, FakeModel(kind="free_tier", params_b=0.0))
    with pytest.raises(CapExceeded, match="FREE_TIER_KEYS_CONFIRMED"):
        client.complete(req())
    confirmed = make_client(
        tmp_path,
        budget,
        FakeModel(kind="free_tier", params_b=0.0),
        env={"FREE_TIER_KEYS_CONFIRMED": "1"},
    )
    assert not confirmed.complete(req()).cached


def test_ledger_rows_pass_the_budget_check(tmp_path: Path, budget: Path) -> None:
    client = make_client(tmp_path, budget, FakeModel())
    for seed in range(3):
        client.complete(req(seed=seed))
    rows = read_ledger(tmp_path / "ledger.jsonl")
    assert {row["kind"] for row in rows} <= set(LEDGER_KINDS)
    report = evaluate(validate_caps({**client.caps}), rows)
    assert report.ok and report.max_params_b == 3.0 and report.cash_usd == 0


def test_count_tokens_and_env_defaults(
    tmp_path: Path, budget: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(NO_MODEL_CALLS_ENV, "1")
    client = Client(FakeModel(), tmp_path / "l.jsonl", tmp_path / "c", budget_path=budget)
    assert client.env[NO_MODEL_CALLS_ENV] == "1"
    assert client.count_tokens([Message("user", "x"), Message("user", "y")]) == 14
    with pytest.raises(ModelCallsDisabled):
        client.complete(req())
    assert (client.run_id, client.experiment) == ("adhoc", "dev")
    assert str(client_module.DEFAULT_CACHE_DIR) == "runs/cache"


def test_env_flag_and_append_jsonl(tmp_path: Path) -> None:
    assert env_flag({"X": "on"}, "X") and not env_flag({"X": "0"}, "X") and not env_flag({}, "X")
    path = tmp_path / "deep" / "l.jsonl"
    append_jsonl(path, {"b": 2, "a": "é"})
    append_jsonl(path, {"c": 3})
    assert path.read_text(encoding="utf-8").splitlines() == ['{"a": "é", "b": 2}', '{"c": 3}']
