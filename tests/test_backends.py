"""Tests for halflife.backends: request types, scripted, replay, and llama.cpp backends.

The llama.cpp backend is tested against an injected fake ``Llama``; the
contract tests at the end check the real library's API (when installed) and
replay replies recorded from the real backend (tests/fixtures/replay/).
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from halflife.agent import SYSTEM_PROMPT, TOOL_CALL_SCHEMA, parse_tool_call
from halflife.backends import base as base_module
from halflife.backends import llama_cpp as lc
from halflife.backends.base import (
    CALL_RECORD_FORMAT,
    ChatRequest,
    ChatResponse,
    Message,
    as_messages,
    call_record,
    canonical_json,
    estimate_context_tokens,
    estimate_tokens,
    messages_sha256,
    sha256_text,
    write_json_atomic,
)
from halflife.backends.replay import COUNTS_FILE, RecordingBackend, ReplayBackend, ReplayMiss
from halflife.backends.scripted import ScriptedBackend, ScriptExhausted
from halflife.schema import AGENT_TOOLS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "replay"
TOOL_REPLY = '{"thought": "look", "tool": "list_files", "args": {"path": "."}}'


def req(text: str = "hi", **kwargs: Any) -> ChatRequest:
    """A small valid request."""
    base: dict[str, Any] = {"max_tokens": 16, "temperature": 0.0, "seed": 1}
    base.update(kwargs)
    return ChatRequest(messages=(Message("user", text),), **base)


# --------------------------------------------------------------------------- base types


def test_message_validates_role_and_content() -> None:
    assert Message("user", "x").to_dict() == {"role": "user", "content": "x"}
    assert Message.from_dict({"role": "system", "content": "s", "extra": 1}) == Message(
        "system", "s"
    )
    with pytest.raises(ValueError, match="role"):
        Message("tool", "x")  # type: ignore[arg-type]  # invalid role on purpose
    with pytest.raises(ValueError, match="string"):
        Message("user", 3)  # type: ignore[arg-type]  # invalid content on purpose


def test_as_messages_accepts_mappings_and_messages() -> None:
    out = as_messages([{"role": "user", "content": "a"}, Message("assistant", "b")])
    assert out == (Message("user", "a"), Message("assistant", "b"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": True}, "max_tokens"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 9.0}, "temperature"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"json_schema": "nope"}, "json_schema"),
        ({"stop": ("",)}, "stop"),
    ],
)
def test_chat_request_validation(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        req(**kwargs)


def test_chat_request_needs_message_objects() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ChatRequest(messages=(), max_tokens=1, temperature=0.0, seed=0)
    with pytest.raises(ValueError, match="Message objects"):
        ChatRequest(
            messages=({"role": "user", "content": "x"},),  # type: ignore[arg-type]  # wrong type on purpose
            max_tokens=1,
            temperature=0.0,
            seed=0,
        )


def test_chat_request_round_trip_and_hash() -> None:
    request = req("hello", json_schema={"type": "object"}, stop=("END",))
    again = ChatRequest.from_dict(json.loads(json.dumps(request.to_dict())))
    assert again == request
    assert again.sha256() == request.sha256() == sha256_text(request.canonical())
    assert req("hello").sha256() != req("hello!").sha256()


def test_canonical_form_keeps_schema_property_order() -> None:
    first = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
    second = {"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "string"}}}
    assert req(json_schema=first).sha256() != req(json_schema=second).sha256()
    order = list(req(json_schema=first).to_dict()["json_schema"]["properties"])
    assert order == ["a", "b"]
    assert canonical_json({"b": 1, "a": "é"}) == '{"b":1,"a":"é"}'


def test_chat_response_round_trip_and_validation() -> None:
    response = ChatResponse("t", 3, 2, 0.5, backend="b", finish_reason="length")
    assert ChatResponse.from_dict(response.to_dict()) == response
    with pytest.raises(ValueError, match="non-negative"):
        ChatResponse("t", -1, 0, 0.0)
    with pytest.raises(ValueError, match="elapsed"):
        ChatResponse("t", 1, 0, -1.0)
    with pytest.raises(ValueError, match="finish_reason"):
        ChatResponse("t", 1, 0, 0.0, finish_reason="eos")


def test_estimates_are_deterministic() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    messages = [Message("system", "abcd"), Message("user", "abcdefgh")]
    assert estimate_context_tokens(messages) == (1 + 4) + (2 + 4) + 2
    assert messages_sha256(messages) == messages_sha256(list(messages))


def test_call_record_stores_uncached_reply() -> None:
    request = req()
    response = ChatResponse("x", 1, 1, 0.1, cached=True, backend="b")
    record = call_record(request, response, backend="b", model="m", extra={"run_id": "r"})
    assert record["format"] == CALL_RECORD_FORMAT
    assert record["request_sha256"] == request.sha256()
    assert record["response"]["cached"] is False
    assert record["run_id"] == "r"
    assert ChatRequest.from_dict(record["request"]) == request


def test_write_json_atomic_writes_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "sub" / "x.json"
    write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}

    def boom(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(base_module.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 1}
    assert [p.name for p in target.parent.iterdir()] == ["x.json"]


# --------------------------------------------------------------------------- scripted


def test_scripted_outputs_in_order_then_exhausted() -> None:
    backend = ScriptedBackend(outputs=["one", ""])
    assert backend.remaining == 2
    first = backend.complete(req("a"))
    assert (first.text, first.cached, first.elapsed_s, first.backend) == (
        "one",
        False,
        0.0,
        "scripted",
    )
    assert first.prompt_tokens == estimate_context_tokens([Message("user", "a")])
    assert backend.complete(req("b")).completion_tokens == 0
    with pytest.raises(ScriptExhausted, match="no reply left for request 3"):
        backend.complete(req("c"))
    assert len(backend.requests) == 3
    assert (backend.kind, backend.offline_safe, backend.params_b) == ("scripted", True, 0.0)


def test_scripted_policy_and_argument_checks() -> None:
    backend = ScriptedBackend(lambda r: r.messages[-1].content.upper(), name="upper")
    assert backend.complete(req("abc")).text == "ABC"
    assert backend.name == "upper"
    assert backend.remaining == 0
    with pytest.raises(ValueError, match="exactly one"):
        ScriptedBackend()
    with pytest.raises(ValueError, match="exactly one"):
        ScriptedBackend(lambda r: "", outputs=["x"])
    bad = ScriptedBackend(lambda r: 3)  # type: ignore[arg-type,return-value]  # wrong type on purpose
    with pytest.raises(TypeError, match="expected str"):
        bad.complete(req())


# --------------------------------------------------------------------------- replay


def test_recording_then_replay(tmp_path: Path) -> None:
    inner = ScriptedBackend(lambda r: f"reply to {r.messages[-1].content}", name="inner")
    recorder = RecordingBackend(inner, tmp_path / "fx")
    assert (recorder.name, recorder.kind, recorder.offline_safe, recorder.params_b) == (
        "inner",
        "scripted",
        True,
        0.0,
    )
    messages = [Message("user", "q")]
    assert recorder.complete(req("q")).text == "reply to q"
    assert recorder.count_tokens(messages) == recorder.count_tokens(messages)
    assert len((tmp_path / "fx" / COUNTS_FILE).read_text().splitlines()) == 1

    replay = ReplayBackend(tmp_path / "fx")
    assert len(replay) == 1
    out = replay.complete(req("q"))
    assert (out.text, out.backend, out.cached) == ("reply to q", "replay:fx", False)
    assert replay.served == 1
    assert replay.count_tokens(messages) == estimate_context_tokens(messages)
    assert (replay.kind, replay.offline_safe, replay.params_b) == ("replay", True, 0.0)


def test_replay_miss_is_clear(tmp_path: Path) -> None:
    (tmp_path / "fx").mkdir()
    replay = ReplayBackend(tmp_path / "fx", name="r")
    with pytest.raises(ReplayMiss) as info:
        replay.complete(req("never recorded"))
    assert req("never recorded").sha256() in str(info.value)
    assert "never recorded" in str(info.value)
    with pytest.raises(ReplayMiss, match="token count"):
        replay.count_tokens([Message("user", "x")])
    loose = ReplayBackend(tmp_path / "fx", strict_counts=False)
    assert loose.count_tokens([Message("user", "x")]) == estimate_context_tokens(
        [Message("user", "x")]
    )
    assert str(ReplayMiss()) == "replay miss"


def _record(path: Path, request: ChatRequest, text: str, backend: str = "b") -> None:
    response = ChatResponse(text, 1, 1, 0.0, backend=backend)
    write_json_atomic(path, call_record(request, response, backend=backend, model="m"))


def test_replay_reads_nested_records_and_filters_by_backend(tmp_path: Path) -> None:
    _record(tmp_path / "ab" / "one.json", req("1"), "first", backend="x")
    _record(tmp_path / "cd" / "two.json", req("2"), "second", backend="y")
    _record(tmp_path / "cd" / "dup.json", req("2"), "second", backend="y")
    assert len(ReplayBackend(tmp_path)) == 2
    only_x = ReplayBackend(tmp_path, source_backend="x")
    assert only_x.complete(req("1")).text == "first"
    with pytest.raises(ReplayMiss):
        only_x.complete(req("2"))


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{not json", "not valid JSON"),
        ('{"format": "other"}', "not a halflife-call/1 record"),
        (json.dumps({"format": CALL_RECORD_FORMAT, "request": {}}), "lacks"),
    ],
)
def test_replay_rejects_malformed_records(tmp_path: Path, content: str, match: str) -> None:
    (tmp_path / "bad.json").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        ReplayBackend(tmp_path)


def test_replay_rejects_conflicts(tmp_path: Path) -> None:
    _record(tmp_path / "a.json", req("q"), "one")
    _record(tmp_path / "b.json", req("q"), "two")
    with pytest.raises(ValueError, match="two different replies"):
        ReplayBackend(tmp_path)
    other = tmp_path / "counts"
    other.mkdir()
    (other / COUNTS_FILE).write_text(
        '{"messages_sha256": "aa", "count": 1}\n\n{"messages_sha256": "aa", "count": 2}\n'
    )
    with pytest.raises(ValueError, match="different count"):
        ReplayBackend(other)
    with pytest.raises(FileNotFoundError):
        ReplayBackend(tmp_path / "missing")


# --------------------------------------------------------------------------- llama.cpp (fake)


class FakeLlama:
    """Stands in for ``llama_cpp.Llama``: one token per character, scripted replies."""

    def __init__(self, *, adds_bos: bool = False, template: str = "chat", **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.metadata: dict[str, str] = {lc.TEMPLATE_KEY: template} if template else {}
        self.adds_bos_flag = adds_bos
        self.replies: list[dict[str, Any]] = []
        self.calls: list[tuple[list[int], dict[str, Any]]] = []
        self.resets = 0
        self.ctx = "ctx"

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        tokens = list(text)
        return [1, *tokens] if (add_bos and self.adds_bos_flag) else tokens

    def detokenize(
        self, tokens: list[int], prev_tokens: list[int] | None = None, special: bool = False
    ) -> bytes:
        return {1: b"<s>", 2: b"</s>"}.get(tokens[0], b"")

    def token_bos(self) -> int:
        return 1

    def token_eos(self) -> int:
        return 2 if self.adds_bos_flag else -1

    def reset(self) -> None:
        self.resets += 1

    def create_completion(self, prompt: list[int], **kwargs: Any) -> dict[str, Any]:
        self.calls.append((list(prompt), kwargs))
        if self.replies:
            return self.replies.pop(0)
        return {
            "choices": [{"text": TOOL_REPLY, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": 7},
        }


def simple_renderer(source: str, bos: str, eos: str) -> lc.TemplateRenderer:
    """A template stand-in: BOS (when the template says so), role tags, generation prompt."""

    def render(messages: list[dict[str, str]]) -> str:
        head = bos if source.startswith("bos") else ""
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return f"{head}{body}<assistant>"

    return render


class Runtime:
    """Builds a fake runtime around one FakeLlama and records what it was asked."""

    def __init__(self, llama: FakeLlama, *, perf: bool = True) -> None:
        self.llama = llama
        self.loaded: dict[str, Any] = {}
        self.gbnf_inputs: list[str] = []
        self.perf_resets = 0
        self.runtime = lc.LlamaRuntime(
            load=self.load,
            schema_to_gbnf=self.to_gbnf,
            grammar=lambda text: ("grammar", text),
            compile_template=simple_renderer,
            perf_reset=self.reset_perf if perf else None,
            perf=(lambda llama: {"prompt_eval_tokens": 5.0}) if perf else None,
            version="fake-0",
        )

    def load(self, **kwargs: Any) -> FakeLlama:
        self.loaded = kwargs
        return self.llama

    def to_gbnf(self, schema_text: str) -> str:
        self.gbnf_inputs.append(schema_text)
        return 'root ::= object\nchar ::= [^"\\\\] | "\\\\" (["\\\\/bfnrt])\n'

    def reset_perf(self, llama: Any) -> None:
        self.perf_resets += 1


def make_backend(
    tmp_path: Path, *, adds_bos: bool = False, template: str = "chat", perf: bool = True, **kw: Any
) -> tuple[lc.LlamaCppBackend, FakeLlama, Runtime]:
    llama = FakeLlama(adds_bos=adds_bos, template=template)
    runtime = Runtime(llama, perf=perf)
    options: dict[str, Any] = {"n_ctx": 4096, "n_threads": 3}
    options.update(kw)
    backend = lc.LlamaCppBackend(
        tmp_path / "m.gguf",
        model="fake-model",
        sha256="ab" * 32,
        params_b=1.5,
        runtime=runtime.runtime,
        **options,
    )
    return backend, llama, runtime


def test_llama_backend_loads_with_threads_and_names_itself(tmp_path: Path) -> None:
    backend, _llama, runtime = make_backend(tmp_path)
    assert runtime.loaded == {
        "model_path": str(tmp_path / "m.gguf"),
        "n_ctx": 4096,
        "n_batch": 512,
        "n_threads": 3,
        "n_threads_batch": 3,
        "verbose": False,
    }
    assert backend.name.startswith("llama_cpp/fake-model@abababababab/")
    assert (backend.kind, backend.offline_safe, backend.params_b) == ("local", False, 1.5)
    info = backend.info()
    assert info["llama_cpp_version"] == "fake-0"
    assert info["fixed_date"] == "2026-11-04"
    assert info["template_sha256"] == sha256_text("chat")
    other, _, _ = make_backend(tmp_path, n_ctx=2048)
    assert other.name != backend.name


def test_llama_backend_default_threads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc.os, "cpu_count", lambda: 6)
    backend, _, _ = make_backend(tmp_path, n_threads=None)
    assert backend.n_threads == 6


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_ctx": 0}, "positive"),
        ({"n_batch": 0}, "positive"),
        ({"system_mode": "weird"}, "system_mode"),
        ({"template": ""}, "no chat template"),
    ],
)
def test_llama_backend_rejects_bad_settings(
    tmp_path: Path, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        make_backend(tmp_path, **kwargs)


def test_prompt_has_exactly_one_bos(tmp_path: Path) -> None:
    messages = [Message("user", "hi")]
    backend, _, _ = make_backend(tmp_path, adds_bos=True, template="bos-template")
    assert backend.adds_bos
    assert backend.format_prompt(messages) == "<user>hi<assistant>"
    tokens = backend.prompt_tokens(messages)
    assert tokens[0] == 1 and tokens.count(1) == 1
    assert backend.count_tokens(messages) == len(tokens)
    plain, _, _ = make_backend(tmp_path)
    assert not plain.adds_bos
    assert plain.eos_text == ""
    assert plain.format_prompt(messages) == "<user>hi<assistant>"


def test_complete_sends_exact_tokens_and_grammar(tmp_path: Path) -> None:
    backend, llama, runtime = make_backend(tmp_path)
    request = ChatRequest(
        messages=(Message("system", "s"), Message("user", "u")),
        max_tokens=64,
        temperature=0.7,
        seed=2**32 + 5,
        json_schema=TOOL_CALL_SCHEMA,
        stop=("STOP",),
    )
    response = backend.complete(request)
    prompt, kwargs = llama.calls[0]
    assert prompt == backend.prompt_tokens(request.messages)
    assert kwargs["seed"] == (2**32 + 5) % lc.SEED_MODULUS == 6
    assert backend.last_raw is not None and backend.last_raw["choices"][0]["text"] == TOOL_REPLY
    assert kwargs["max_tokens"] == 64
    assert kwargs["temperature"] == 0.7
    assert kwargs["stop"] == ["STOP"]
    assert (kwargs["top_k"], kwargs["top_p"], kwargs["min_p"]) == (40, 0.95, 0.05)
    assert kwargs["grammar"][1].splitlines()[1] == lc.STRICT_CHAR_RULE
    assert json.loads(runtime.gbnf_inputs[0]) == TOOL_CALL_SCHEMA
    # The schema reaches the converter as JSON text in its own key order (the order
    # in which a constrained reply writes its fields).
    assert runtime.gbnf_inputs[0] == json.dumps(TOOL_CALL_SCHEMA, ensure_ascii=False)
    assert response.text == TOOL_REPLY
    assert response.prompt_tokens == len(prompt)
    assert response.completion_tokens == 7
    assert response.finish_reason == "stop"
    assert response.backend == backend.name
    assert backend.last_perf == {"prompt_eval_tokens": 5.0}
    assert runtime.perf_resets == 1
    backend.complete(request)
    assert len(runtime.gbnf_inputs) == 1  # grammar built once per schema


def test_seed_never_reaches_the_random_seed_sentinel(tmp_path: Path) -> None:
    """llama.cpp reads 0xFFFFFFFF as 'random seed'; no request seed may map to it."""
    backend, llama, _ = make_backend(tmp_path)
    for seed in (0, 7, 2**31 - 2, 2**32 - 2, 2**32 - 1, 2**32, 2**40 + 3):
        backend.complete(req(seed=seed))
    sent = [kwargs["seed"] for _, kwargs in llama.calls]
    assert sent[:4] == [0, 7, 2**31 - 2, 2**32 - 2]
    assert all(0 <= s < 0xFFFFFFFF for s in sent)


def test_complete_without_schema_and_length_finish(tmp_path: Path) -> None:
    backend, llama, _ = make_backend(tmp_path, perf=False, n_ctx=40)
    llama.replies.append({"choices": [{"text": "abc", "finish_reason": "length"}], "usage": {}})
    response = backend.complete(req("x" * 10, max_tokens=500))
    _, kwargs = llama.calls[0]
    assert kwargs["grammar"] is None
    assert kwargs["max_tokens"] == 40 - len(backend.prompt_tokens([Message("user", "x" * 10)]))
    assert (response.finish_reason, response.completion_tokens) == ("length", 0)
    assert backend.last_perf is None


def test_complete_refuses_prompts_that_do_not_fit(tmp_path: Path) -> None:
    backend, llama, _ = make_backend(tmp_path, n_ctx=20)
    with pytest.raises(lc.ContextOverflowError, match="context window is 20"):
        backend.complete(req("y" * 30))
    assert llama.calls == []


def test_reset_cache_forgets_prefix_and_close_frees(tmp_path: Path) -> None:
    backend, llama, _ = make_backend(tmp_path)
    backend.reset_cache()
    assert llama.resets == 1
    backend.close()  # FakeLlama has no close(): nothing to free
    closed: list[bool] = []
    llama.close = lambda: closed.append(True)  # type: ignore[attr-defined]  # add a close() to the fake
    backend.close()
    assert closed == [True]


def test_normalize_messages() -> None:
    merged = lc.normalize_messages(
        [Message("system", "s"), Message("user", "a"), Message("user", "b")], "native"
    )
    assert merged == [Message("system", "s"), Message("user", "a\n\nb")]
    folded = lc.normalize_messages(
        [Message("system", "s"), Message("user", "a")], "merge_first_user"
    )
    assert folded == [Message("user", "s\n\na")]
    lone = lc.normalize_messages([Message("system", "s")], "merge_first_user")
    assert lone == [Message("user", "s")]
    before_assistant = lc.normalize_messages(
        [Message("system", "s"), Message("assistant", "x")], "merge_first_user"
    )
    assert before_assistant == [Message("user", "s"), Message("assistant", "x")]
    with pytest.raises(ValueError, match="system_mode"):
        lc.normalize_messages([], "other")


def test_merge_first_user_backend(tmp_path: Path) -> None:
    backend, _, _ = make_backend(tmp_path, system_mode="merge_first_user")
    text = backend.format_prompt([Message("system", "rules"), Message("user", "go")])
    assert text == "<user>rules\n\ngo<assistant>"


def test_strict_json_gbnf() -> None:
    gbnf = 'root ::= string\nchar ::= [^"\\\\] | "\\\\" (["\\\\/bfnrt])\n'
    out = lc.strict_json_gbnf(gbnf)
    assert out.splitlines() == ["root ::= string", lc.STRICT_CHAR_RULE]
    assert out.endswith("\n")
    assert lc.strict_json_gbnf("root ::= number") == "root ::= number"
    assert r"\x00-\x1F" in lc.STRICT_CHAR_RULE


def test_tojson_and_template_helpers() -> None:
    assert lc._tojson({"a": "é"}) == '{"a": "é"}'
    assert lc._tojson([1], indent=2) == "[\n  1\n]"
    assert lc.fixed_strftime("%d %b %Y") == "04 Nov 2026"
    with pytest.raises(ValueError, match="chat template error: no"):
        lc._raise_template_error("no")


def test_default_runtime_without_llama_cpp(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(lc.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="make setup-local"):
        lc.default_runtime()


def test_default_runtime_wires_the_library(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class PerfData:
        n_p_eval, t_p_eval_ms, n_eval, t_eval_ms = 10, 100.0, 4, 50.0

    fake = types.ModuleType("llama_cpp")
    fake.Llama = FakeLlama  # type: ignore[attr-defined]  # fake module attribute
    fake.__version__ = "9.9"  # type: ignore[attr-defined]  # fake module attribute
    fake.llama_perf_context_reset = lambda ctx: calls.append(f"reset {ctx}")  # type: ignore[attr-defined]  # fake module attribute
    fake.llama_perf_context = lambda ctx: PerfData()  # type: ignore[attr-defined]  # fake module attribute
    grammar = types.ModuleType("llama_cpp.llama_grammar")
    grammar.json_schema_to_gbnf = lambda text: "root ::= x"  # type: ignore[attr-defined]  # fake module attribute
    grammar.LlamaGrammar = types.SimpleNamespace(from_string=lambda text: ("g", text))  # type: ignore[attr-defined]  # fake module attribute
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_grammar", grammar)
    runtime = lc.default_runtime()
    assert runtime.version == "9.9"
    assert runtime.load is FakeLlama
    assert runtime.schema_to_gbnf("{}") == "root ::= x"
    assert runtime.grammar("g1") == ("g", "g1")
    assert runtime.perf_reset is not None and runtime.perf is not None
    llama = types.SimpleNamespace(ctx="C")
    runtime.perf_reset(llama)
    assert calls == ["reset C"]
    assert runtime.perf(llama) == {
        "prompt_eval_tokens": 10.0,
        "prompt_eval_ms": 100.0,
        "eval_tokens": 4.0,
        "eval_ms": 50.0,
    }


LLAMA32_STYLE = (
    "{{- bos_token }}{%- if not date_string is defined %}{%- if strftime_now is defined %}"
    '{%- set date_string = strftime_now("%d %b %Y") %}{%- else %}'
    '{%- set date_string = "26 Jul 2024" %}{%- endif %}{%- endif %}'
    "{%- for m in messages %}{{ '<' + m['role'] + '>' + m['content'] | trim }}"
    "{%- if m['role'] == 'system' %}{{ ' Today Date: ' + date_string }}{%- endif %}"
    "{%- endfor %}{%- if add_generation_prompt %}{{ '<assistant>' }}{%- endif %}"
)


def test_jinja_template_is_deterministic() -> None:
    pytest.importorskip("jinja2")
    render = lc.compile_jinja_template(LLAMA32_STYLE, "<s>", "</s>")
    text = render([{"role": "system", "content": " rules "}, {"role": "user", "content": "go"}])
    assert text == "<s><system>rules Today Date: 04 Nov 2026<user>go<assistant>"
    failing = lc.compile_jinja_template("{{ raise_exception('nope') }}", "", "")
    with pytest.raises(ValueError, match="nope"):
        failing([])
    source = "{{ messages | tojson }}{% for m in messages %}{% break %}{% endfor %}"
    tool = lc.compile_jinja_template(source, "", "")
    assert tool([{"role": "user", "content": "é"}]) == '[{"role": "user", "content": "é"}]'


def test_jinja_template_wiring_with_a_fake_jinja2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Renderer options and context, checked without jinja2 (CI installs no ``local`` extra)."""
    seen: dict[str, Any] = {}

    class FakeTemplate:
        def render(self, **context: Any) -> str:
            seen["context"] = context
            return "rendered"

    class FakeEnvironment:
        def __init__(self, **options: Any) -> None:
            seen["options"] = options
            self.filters: dict[str, Any] = {}
            seen["filters"] = self.filters

        def from_string(self, source: str) -> FakeTemplate:
            seen["source"] = source
            return FakeTemplate()

    fakes = {
        "jinja2.sandbox": types.SimpleNamespace(ImmutableSandboxedEnvironment=FakeEnvironment),
        "jinja2.ext": types.SimpleNamespace(loopcontrols="loopcontrols-extension"),
    }
    monkeypatch.setattr(lc.importlib, "import_module", fakes.__getitem__)
    render = lc.compile_jinja_template("SOURCE", "<s>", "</s>")
    assert render([{"role": "user", "content": "x"}]) == "rendered"
    assert seen["source"] == "SOURCE"
    assert seen["options"] == {
        "trim_blocks": True,
        "lstrip_blocks": True,
        "extensions": ["loopcontrols-extension"],
    }
    assert seen["filters"]["tojson"] is lc._tojson
    context = seen["context"]
    assert context["messages"] == [{"role": "user", "content": "x"}]
    assert (context["bos_token"], context["eos_token"]) == ("<s>", "</s>")
    assert context["add_generation_prompt"] is True
    assert context["strftime_now"] is lc.fixed_strftime
    assert context["raise_exception"] is lc._raise_template_error


# --------------------------------------------------------------------------- contract tests


def test_contract_real_library_api() -> None:
    """The keyword arguments and helpers this backend uses exist in llama-cpp-python."""
    llama_cpp = pytest.importorskip("llama_cpp")
    grammar_module = importlib.import_module("llama_cpp.llama_grammar")
    params = inspect.signature(llama_cpp.Llama.create_completion).parameters
    used = ["prompt", "max_tokens", "temperature", "top_k", "top_p", "min_p", "typical_p"]
    used += ["repeat_penalty", "seed", "stop", "grammar"]
    assert all(name in params for name in used)
    init = inspect.signature(llama_cpp.Llama.__init__).parameters
    loaded = ["model_path", "n_ctx", "n_batch", "n_threads", "n_threads_batch", "verbose"]
    assert all(name in init for name in loaded)
    assert hasattr(llama_cpp, "llama_perf_context")
    assert hasattr(llama_cpp, "llama_perf_context_reset")
    gbnf = grammar_module.json_schema_to_gbnf(json.dumps(TOOL_CALL_SCHEMA))
    assert any(line.startswith("char ::=") for line in gbnf.splitlines())
    assert lc.STRICT_CHAR_RULE in lc.strict_json_gbnf(gbnf)
    ordered = {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "tool": {"type": "string"},
            "args": {"type": "object"},
        },
        "required": ["thought", "tool", "args"],
    }
    flat = grammar_module.json_schema_to_gbnf(json.dumps(ordered))
    root = next(line for line in flat.splitlines() if line.startswith("root ::="))
    assert root.index("thought") < root.index("tool-kv") < root.index("args")  # order kept


def _fixture_records() -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(FIXTURES.glob("*.json"))]


def test_contract_recorded_llama_replies_replay() -> None:
    """Replies recorded from the real backend replay exactly and parse as tool calls."""
    records = _fixture_records()
    assert records, "tests/fixtures/replay/ holds recorded llama.cpp calls"
    replay = ReplayBackend(FIXTURES)
    for record in records:
        assert record["backend"].startswith("llama_cpp/")
        request = ChatRequest.from_dict(record["request"])
        reply = replay.complete(request)
        assert reply.text == record["response"]["text"]
        assert reply.prompt_tokens == replay.count_tokens(request.messages)
        if request.json_schema is not None:
            assert parse_tool_call(reply.text) is not None
            json.loads(reply.text)
    assert replay.served == len(records)


def test_contract_recorded_raw_completion_shape(tmp_path: Path) -> None:
    """A raw ``create_completion`` result recorded from the library parses as expected."""
    lines = (FIXTURES / "raw_create_completion.jsonl").read_text().splitlines()
    raw = json.loads(lines[0])
    backend, llama, _ = make_backend(tmp_path)
    llama.replies.append(raw["result"])
    response = backend.complete(req("x"))
    assert response.text == raw["result"]["choices"][0]["text"]
    assert response.completion_tokens == raw["result"]["usage"]["completion_tokens"]
    assert response.finish_reason == raw["result"]["choices"][0]["finish_reason"]


def test_contract_fixture_covers_agent_tool_calls() -> None:
    """Recorded calls include agent steps: a system prompt, a tool-call schema, and a reply
    that the agent parses as one of its tools.

    The check is structural on purpose: CI has no model to re-record fixtures with, so it
    must not fail when the agent's prompt or schema is later edited (re-record with
    ``tools/bench_models.py --record-model KEY --record-dir tests/fixtures/replay``).
    """
    steps: Sequence[Mapping[str, Any]] = [
        r
        for r in _fixture_records()
        if r["request"]["json_schema"] is not None
        and r["request"]["messages"][0]["role"] == "system"
    ]
    assert steps
    for record in steps:
        action = parse_tool_call(record["response"]["text"])
        assert action is not None and action.tool in AGENT_TOOLS
    assert any(r["request"]["messages"][0]["content"] == SYSTEM_PROMPT for r in steps)
