"""Tests for halflife.models, tools/fetch_models.py, and tools/bench_models.py.

No network and no real model: HTTP goes through a fake opener, and the llama.cpp
backend runs on a fake ``Llama`` injected through its runtime.
"""

from __future__ import annotations

import datetime as dt
import email.message
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from halflife import models
from halflife.agent import RULES_INTRO, SYSTEM_PROMPT, TOOL_CALL_SCHEMA, parse_tool_call
from halflife.backends.base import ChatBackend, ChatRequest, Message, messages_sha256
from halflife.backends.llama_cpp import SYSTEM_MODES, LlamaCppBackend, LlamaRuntime
from halflife.backends.replay import COUNTS_FILE, ReplayBackend
from halflife.budget import DEFAULT_CAPS, read_ledger
from halflife.client import NO_MODEL_CALLS_ENV, Client
from halflife.models import (
    CANDIDATES,
    HF_BASE,
    MODELS_DIR_ENV,
    REGISTRY,
    ModelFileError,
    ModelSpec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTENT = b"GGUF synthetic model bytes for tests " * 40
TOOL_REPLY = '{"thought": "look", "tool": "list_files", "args": {"path": "."}}'
BUDGET = (
    "cash_usd = 0\nclaude_agent_runs_per_day = 30\nclaude_agent_max_turns = 40\n"
    "claude_single_calls_per_day = 300\nclaude_review_runs_per_day = 6\n"
    "local_model_max_params_b = 8\ndisk_floor_gb = 8\n"
)


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fetch = _load_tool("fetch_models")
bench = _load_tool("bench_models")


def tiny_spec(**overrides: Any) -> ModelSpec:
    """A pin for CONTENT (synthetic; never downloaded)."""
    data: dict[str, Any] = {
        "key": "tiny-q4km",
        "family": "tiny",
        "upstream": "org/tiny",
        "repo": "org/tiny-GGUF",
        "revision": "a" * 40,
        "filename": "tiny-Q4_K_M.gguf",
        "quantization": "Q4_K_M",
        "size_bytes": len(CONTENT),
        "sha256": hashlib.sha256(CONTENT).hexdigest(),
        "license": "apache-2.0",
        "license_url": "https://example.org/license",
        "params_b": 0.1,
        "architecture": "llama",
        "context_length": 2048,
        "chat_notes": "test only",
    }
    data.update(overrides)
    return ModelSpec(**data)


def write_model(directory: Path, spec: ModelSpec, content: bytes = CONTENT) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / spec.filename
    path.write_bytes(content)
    return path


# --------------------------------------------------------------------------- fakes


class FakeLlama:
    """Stands in for ``llama_cpp.Llama``: one token per byte, scripted replies."""

    def __init__(self) -> None:
        self.metadata = {"tokenizer.chat_template": "fake-template"}
        self.calls: list[tuple[list[int], dict[str, Any]]] = []
        self.replies: list[str] = []
        self.resets = 0
        self.closed = False

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        return list(text)

    def detokenize(
        self, tokens: list[int], prev_tokens: list[int] | None = None, special: bool = False
    ) -> bytes:
        return b""

    def token_bos(self) -> int:
        return -1

    def token_eos(self) -> int:
        return -1

    def reset(self) -> None:
        self.resets += 1

    def create_completion(self, prompt: list[int], **kwargs: Any) -> dict[str, Any]:
        self.calls.append((list(prompt), kwargs))
        text = self.replies.pop(0) if self.replies else TOOL_REPLY
        return {
            "id": "cmpl-fake",
            "model": "/home/someone/.cache/halflife/models/tiny-Q4_K_M.gguf",
            "choices": [{"text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": 9},
        }

    def close(self) -> None:
        self.closed = True


PERF = {"prompt_eval_tokens": 120.0, "prompt_eval_ms": 400.0, "eval_tokens": 9.0, "eval_ms": 300.0}


def fake_runtime(llama: FakeLlama, loaded: dict[str, Any] | None = None) -> LlamaRuntime:
    def load(**kwargs: Any) -> FakeLlama:
        if loaded is not None:
            loaded.update(kwargs)
        return llama

    def compile_template(source: str, bos: str, eos: str) -> Callable[[list[dict[str, str]]], str]:
        def render(messages: list[dict[str, str]]) -> str:
            return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

        return render

    return LlamaRuntime(
        load=load,
        schema_to_gbnf=lambda text: 'root ::= x\nchar ::= [^"]\n',
        grammar=lambda text: ("grammar", text),
        compile_template=compile_template,
        perf_reset=lambda llama: None,
        perf=lambda llama: dict(PERF),
        version="fake-1",
    )


class FakeResponse:
    """An HTTP response that serves ``body`` in small blocks, optionally failing midway."""

    def __init__(
        self, body: bytes, status: int = 200, *, block: int = 256, fail_at: int | None = None
    ) -> None:
        self.body = body
        self.status = status
        self.block = block
        self.fail_at = fail_at
        self.pos = 0

    def read(self, size: int = -1) -> bytes:
        if self.fail_at is not None and self.pos >= self.fail_at:
            raise OSError("connection reset by peer")
        end = len(self.body) if size < 0 else self.pos + min(size, self.block)
        if self.fail_at is not None:
            end = min(end, self.fail_at)
        chunk = self.body[self.pos : end]
        self.pos += len(chunk)
        return chunk

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeOpener:
    """Serves scripted responses (or raises scripted errors) and records the requests."""

    def __init__(self, items: list[Any]) -> None:
        self.items = list(items)
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request) -> Any:
        self.requests.append(request)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def json_response(data: Any) -> FakeResponse:
    return FakeResponse(json.dumps(data).encode("utf-8"), block=1 << 20)


def api_documents(spec: ModelSpec, **changes: Any) -> list[FakeResponse]:
    """The two API documents remote_info reads (model, then tree), agreeing with ``spec``."""
    model = {
        "sha": changes.get("head", spec.revision),
        "gated": False,
        "gguf": {
            "total": changes.get("params", int(spec.params_b * 1e9)),
            "architecture": spec.architecture,
            "context_length": spec.context_length,
        },
        "cardData": {
            "license": spec.license,
            "license_name": changes.get("license_name"),
            "license_link": changes.get("license_link"),
        },
    }
    tree = [
        {"path": "README.md", "size": 10},
        {
            "path": spec.filename,
            "size": changes.get("size", spec.size_bytes),
            "lfs": {"oid": changes.get("sha256", spec.sha256)},
        },
    ]
    return [json_response(model), json_response(tree)]


# --------------------------------------------------------------------------- registry


def test_registry_pins_five_small_families() -> None:
    assert len(CANDIDATES) == 5
    assert len({s.family for s in CANDIDATES}) == 5
    assert len({s.key for s in CANDIDATES}) == 5
    assert len({s.filename for s in CANDIDATES}) == 5
    for spec in CANDIDATES:
        assert REGISTRY[spec.key] is spec
        assert 0 < spec.params_b <= 4 <= DEFAULT_CAPS["local_model_max_params_b"]
        assert spec.quantization == "Q4_K_M" and spec.filename.endswith("Q4_K_M.gguf")
        assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256)
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)
        assert spec.repo.startswith("bartowski/")
        assert spec.size_bytes > 0 and spec.context_length > 0
        assert spec.license and spec.license_url.startswith("https://")
        assert spec.system_mode in SYSTEM_MODES
        assert spec.chat_notes.strip()
        assert "\u2014" not in spec.chat_notes  # no em dashes (rule 12)


def test_spec_urls_are_pinned_to_the_revision() -> None:
    spec = tiny_spec()
    assert spec.url == f"{HF_BASE}/org/tiny-GGUF/resolve/{'a' * 40}/tiny-Q4_K_M.gguf"
    assert spec.api_model_url == f"{HF_BASE}/api/models/org/tiny-GGUF"
    assert spec.api_tree_url == f"{HF_BASE}/api/models/org/tiny-GGUF/tree/{'a' * 40}"
    data = spec.to_dict()
    assert data["url"] == spec.url and data["sha256"] == spec.sha256
    assert json.loads(json.dumps(data)) == data
    for real in CANDIDATES:
        assert f"/resolve/{real.revision}/" in real.url


def test_get_spec() -> None:
    first = CANDIDATES[0]
    assert models.get_spec(first.key) is first
    with pytest.raises(KeyError, match="pinned models") as info:
        models.get_spec("gpt-unknown")
    assert first.key in str(info.value)


def test_models_dir_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    default = tmp_path / ".cache" / "halflife" / "models"
    assert models.models_dir({}) == default
    assert models.models_dir({MODELS_DIR_ENV: "  "}) == default
    assert models.models_dir({MODELS_DIR_ENV: "~/weights"}) == tmp_path / "weights"
    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path / "env-dir"))
    assert models.models_dir() == tmp_path / "env-dir"
    spec = tiny_spec()
    assert models.model_path(spec) == tmp_path / "env-dir" / spec.filename
    assert models.model_path(spec, tmp_path / "x") == tmp_path / "x" / spec.filename
    path = tmp_path / "x" / spec.filename
    assert models.provenance_path(path).name == f"{spec.filename}.provenance.json"


def test_read_provenance(tmp_path: Path) -> None:
    path = tmp_path / "m.gguf"
    assert models.read_provenance(path) is None
    sidecar = models.provenance_path(path)
    sidecar.write_text("{broken", encoding="utf-8")
    assert models.read_provenance(path) is None
    sidecar.write_text("[1, 2]", encoding="utf-8")
    assert models.read_provenance(path) is None
    sidecar.write_text('{"retrieved_at": "2026-11-04T00:00:00+00:00"}', encoding="utf-8")
    assert models.read_provenance(path) == {"retrieved_at": "2026-11-04T00:00:00+00:00"}


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(CONTENT)
    expected = hashlib.sha256(CONTENT).hexdigest()
    assert models.sha256_file(path) == expected
    assert models.sha256_file(path, chunk_size=7) == expected


def test_verify_file(tmp_path: Path) -> None:
    spec = tiny_spec()
    path = tmp_path / spec.filename
    with pytest.raises(ModelFileError, match=r"fetch_models\.py --models tiny-q4km"):
        models.verify_file(path, spec)
    path.write_bytes(CONTENT[:-1])
    with pytest.raises(ModelFileError, match="pinned size"):
        models.verify_file(path, spec)
    tampered = b"X" + CONTENT[1:]
    path.write_bytes(tampered)
    with pytest.raises(ModelFileError, match="SHA-256"):
        models.verify_file(path, spec)
    models.verify_file(path, spec, check_hash=False)  # same size: passes without hashing
    path.write_bytes(CONTENT)
    models.verify_file(path, spec)


def test_load_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = tiny_spec()
    monkeypatch.setitem(REGISTRY, spec.key, spec)
    write_model(tmp_path, spec)
    llama = FakeLlama()
    loaded: dict[str, Any] = {}
    backend = models.load_backend(
        spec.key, n_ctx=2048, n_threads=4, directory=tmp_path, runtime=fake_runtime(llama, loaded)
    )
    assert isinstance(backend, LlamaCppBackend)
    assert backend.name.startswith(f"llama_cpp/{spec.key}@{spec.sha256[:12]}/")
    assert (backend.model, backend.params_b, backend.n_ctx, backend.n_threads) == (
        spec.key,
        0.1,
        2048,
        4,
    )
    assert loaded["model_path"] == str(tmp_path / spec.filename)
    assert loaded["n_threads"] == 4 and loaded["verbose"] is False
    write_model(tmp_path, spec, b"Y" * len(CONTENT))
    with pytest.raises(ModelFileError, match="SHA-256"):
        models.load_backend(spec.key, n_ctx=2048, directory=tmp_path, runtime=fake_runtime(llama))
    unverified = models.load_backend(
        spec.key, n_ctx=1024, directory=tmp_path, verify=False, runtime=fake_runtime(llama)
    )
    assert unverified.n_ctx == 1024
    with pytest.raises(KeyError):
        models.load_backend("missing-model", n_ctx=8)


# --------------------------------------------------------------------------- fetch_models


def test_fetch_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert dt.datetime.fromisoformat(fetch.now_iso()).tzinfo is not None
    opener = FakeOpener([json_response({"a": 1})])
    assert fetch.get_json("https://example.org/x", opener) == {"a": 1}
    assert opener.requests[0].get_header("User-agent") == fetch.USER_AGENT
    seen: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> str:
        seen["url"], seen["timeout"] = request.full_url, timeout
        return "response"

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    assert fetch.urlopen(urllib.request.Request("https://example.org/y")) == "response"
    assert seen == {"url": "https://example.org/y", "timeout": fetch.TIMEOUT_S}
    assert fetch.free_bytes(tmp_path / "not" / "yet") > 0
    assert fetch.git_sha(tmp_path) == "unknown"
    assert re.fullmatch(r"[0-9a-f]{40}", fetch.git_sha(REPO_ROOT))


def http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://example.org/x", code, "error", headers, None)


def test_get_json_waits_and_retries_on_rate_limits() -> None:
    sleeps: list[float] = []
    opener = FakeOpener([http_error(429, "7"), http_error(503), json_response({"ok": True})])
    assert fetch.get_json("https://example.org/x", opener, sleep=sleeps.append) == {"ok": True}
    assert sleeps == [7.0, 10.0]  # Retry-After first, then the doubling default
    with pytest.raises(urllib.error.HTTPError) as info:
        fetch.get_json("https://example.org/x", FakeOpener([http_error(404)]), sleep=no_sleep)
    assert info.value.code == 404
    with pytest.raises(urllib.error.HTTPError):
        fetch.get_json(
            "https://example.org/x",
            FakeOpener([http_error(429), http_error(429)]),
            attempts=2,
            sleep=no_sleep,
        )
    with pytest.raises(ValueError, match="attempts"):
        fetch.get_json("https://example.org/x", FakeOpener([]), attempts=0)


def test_retry_after_s() -> None:
    assert fetch.retry_after_s(http_error(429, "3"), default=9.0) == 3.0
    assert fetch.retry_after_s(http_error(429, "Wed, 21 Oct 2026 07:28:00 GMT"), 9.0) == 9.0
    assert fetch.retry_after_s(http_error(429, "100000"), 1.0) == fetch.MAX_WAIT_S
    assert fetch.retry_after_s(http_error(429, "-5"), 4.0) == 0.0
    assert fetch.retry_after_s(http_error(429), 4.0) == 4.0
    bare = urllib.error.HTTPError("https://example.org/x", 429, "error", None, None)  # type: ignore[arg-type]  # no headers on purpose
    assert fetch.retry_after_s(bare, 2.0) == 2.0


def test_fetch_main_reports_api_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = tiny_spec()
    monkeypatch.setitem(REGISTRY, spec.key, spec)

    def offline(s: ModelSpec, opener: Any = None) -> dict[str, Any]:
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(fetch, "remote_info", offline)
    assert fetch.main(["--models", spec.key, "--check-api"]) == 1
    assert "FETCH: FAIL tiny-q4km" in capsys.readouterr().err

    def rate_limited(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise http_error(429)

    monkeypatch.setattr(fetch, "build_pins", rate_limited)
    assert fetch.main(["--models", spec.key, "--write-pins"]) == 1
    assert "FETCH: FAIL HTTP Error 429" in capsys.readouterr().err


def test_remote_info_and_api_problems() -> None:
    spec = tiny_spec()
    info = fetch.remote_info(spec, FakeOpener(api_documents(spec, head="b" * 40)))
    assert info["repo_head"] == "b" * 40
    assert (info["size_bytes"], info["lfs_sha256"]) == (spec.size_bytes, spec.sha256)
    assert (info["license"], info["params"], info["gated"]) == ("apache-2.0", 100_000_000, False)
    assert (info["license_name"], info["license_link"]) == (None, None)
    named = fetch.remote_info(
        spec,
        FakeOpener(api_documents(spec, license_name="custom", license_link="https://x.org/l")),
    )
    assert (named["license_name"], named["license_link"]) == ("custom", "https://x.org/l")
    assert fetch.api_problems(spec, info) == []
    bad = fetch.remote_info(
        spec, FakeOpener(api_documents(spec, size=1, sha256="c" * 64, params=2_000_000_000))
    )
    problems = fetch.api_problems(spec, bad)
    assert len(problems) == 3
    assert any("size" in p for p in problems) and any("params" in p for p in problems)
    missing = [json_response({}), json_response([{"path": "other.gguf"}])]
    with pytest.raises(ModelFileError, match="is not in org/tiny-GGUF"):
        fetch.remote_info(spec, FakeOpener(missing))


def test_ensure_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "free_bytes", lambda path: 10 * fetch.GIB)
    fetch.ensure_disk(tmp_path, 1 * fetch.GIB, 8)
    with pytest.raises(fetch.DiskFloorError, match="below the 8 GiB floor"):
        fetch.ensure_disk(tmp_path, 3 * fetch.GIB, 8)


def no_sleep(seconds: float) -> None:
    return None


def test_download_fresh_file(tmp_path: Path) -> None:
    spec = tiny_spec()
    messages: list[str] = []
    opener = FakeOpener([FakeResponse(CONTENT)])
    path = fetch.download(spec, tmp_path, floor_gb=0, opener=opener, log=messages.append)
    assert path.read_bytes() == CONTENT
    assert opener.requests[0].full_url == spec.url
    assert opener.requests[0].get_header("Range") is None
    provenance = models.read_provenance(path)
    assert provenance is not None
    assert (provenance["sha256"], provenance["revision"]) == (spec.sha256, spec.revision)
    assert not (tmp_path / f"{spec.filename}.part").exists()
    assert "downloaded and verified" in messages[-1]


def test_download_resumes_a_partial_file(tmp_path: Path) -> None:
    spec = tiny_spec()
    part = tmp_path / f"{spec.filename}.part"
    part.write_bytes(CONTENT[:300])
    opener = FakeOpener([FakeResponse(CONTENT[300:], status=206)])
    path = fetch.download(spec, tmp_path, floor_gb=0, opener=opener, log=lambda m: None)
    assert opener.requests[0].get_header("Range") == "bytes=300-"
    assert path.read_bytes() == CONTENT


def test_download_restarts_when_the_server_ignores_range(tmp_path: Path) -> None:
    spec = tiny_spec()
    (tmp_path / f"{spec.filename}.part").write_bytes(CONTENT[:300])
    opener = FakeOpener([FakeResponse(CONTENT, status=200)])
    path = fetch.download(spec, tmp_path, floor_gb=0, opener=opener, log=lambda m: None)
    assert path.read_bytes() == CONTENT


def test_download_retries_after_errors_and_short_reads(tmp_path: Path) -> None:
    spec = tiny_spec()
    sleeps: list[float] = []
    opener = FakeOpener(
        [
            urllib.error.URLError("proxy down"),
            FakeResponse(CONTENT, fail_at=400),
            FakeResponse(CONTENT[400:600], status=206),
            FakeResponse(CONTENT[600:], status=206),
        ]
    )
    path = fetch.download(
        spec, tmp_path, floor_gb=0, opener=opener, log=lambda m: None, sleep=sleeps.append
    )
    assert path.read_bytes() == CONTENT
    ranges = [r.get_header("Range") for r in opener.requests]
    assert ranges == [None, None, "bytes=400-", "bytes=600-"]
    assert sleeps == [2.0, 4.0, 8.0]


def test_download_gives_up_and_keeps_the_partial_file(tmp_path: Path) -> None:
    spec = tiny_spec()
    opener = FakeOpener(
        [FakeResponse(CONTENT, fail_at=100), urllib.error.URLError("down"), TimeoutError("slow")]
    )
    with pytest.raises(TimeoutError):
        fetch.download(
            spec,
            tmp_path,
            floor_gb=0,
            opener=opener,
            attempts=3,
            log=lambda m: None,
            sleep=no_sleep,
        )
    assert (tmp_path / f"{spec.filename}.part").read_bytes() == CONTENT[:100]
    assert not (tmp_path / spec.filename).exists()


def test_download_rejects_a_file_that_does_not_match(tmp_path: Path) -> None:
    spec = tiny_spec()
    opener = FakeOpener([FakeResponse(b"Z" * len(CONTENT))])
    with pytest.raises(ModelFileError, match="pinned"):
        fetch.download(spec, tmp_path, floor_gb=0, opener=opener, log=lambda m: None)
    assert not (tmp_path / f"{spec.filename}.part").exists()
    assert not (tmp_path / spec.filename).exists()


def test_download_part_edge_cases(tmp_path: Path) -> None:
    spec = tiny_spec()
    part = tmp_path / f"{spec.filename}.part"
    part.write_bytes(CONTENT + b"extra")  # longer than the pin: start over
    opener = FakeOpener([FakeResponse(CONTENT)])
    fetch.download(spec, tmp_path, floor_gb=0, opener=opener, log=lambda m: None)
    assert opener.requests[0].get_header("Range") is None
    other = tmp_path / "other"
    other.mkdir()
    (other / f"{spec.filename}.part").write_bytes(CONTENT)  # complete already: just verify
    untouched = FakeOpener([])
    path = fetch.download(spec, other, floor_gb=0, opener=untouched, log=lambda m: None)
    assert path.read_bytes() == CONTENT and untouched.requests == []


def test_download_verifies_an_existing_file(tmp_path: Path) -> None:
    spec = tiny_spec()
    path = write_model(tmp_path, spec)
    messages: list[str] = []
    untouched = FakeOpener([])
    assert fetch.download(spec, tmp_path, floor_gb=0, opener=untouched, log=messages.append) == path
    assert untouched.requests == [] and messages == ["tiny-q4km: present and verified"]
    assert models.read_provenance(path) is not None  # a missing sidecar is written
    path.write_bytes(b"Q" * len(CONTENT))
    with pytest.raises(ModelFileError, match="delete it and retry"):
        fetch.download(spec, tmp_path, floor_gb=0, opener=untouched, log=lambda m: None)


def test_download_refuses_below_the_disk_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "free_bytes", lambda path: len(CONTENT))
    opener = FakeOpener([])
    with pytest.raises(fetch.DiskFloorError):
        fetch.download(tiny_spec(), tmp_path, floor_gb=8, opener=opener, log=lambda m: None)
    assert opener.requests == []


def test_download_logs_to_stderr_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch.download(tiny_spec(), tmp_path, floor_gb=0, opener=FakeOpener([FakeResponse(CONTENT)]))
    assert "downloaded and verified" in capsys.readouterr().err


def test_build_pins(tmp_path: Path) -> None:
    spec = tiny_spec()
    with pytest.raises(ModelFileError, match="download it first"):
        fetch.build_pins([spec], tmp_path, opener=FakeOpener([]), command=["x"], commit="c")
    path = write_model(tmp_path, spec)
    fetch.download(spec, tmp_path, floor_gb=0, opener=FakeOpener([]), log=lambda m: None)
    pins = fetch.build_pins(
        [spec], tmp_path, opener=FakeOpener(api_documents(spec)), command=["a", "b"], commit="c1"
    )
    assert (pins["format"], pins["git_sha"], pins["command"]) == (
        fetch.PINS_FORMAT,
        "c1",
        ["a", "b"],
    )
    entry = pins["models"][0]
    assert entry["key"] == spec.key and entry["url"] == spec.url
    assert entry["file_sha256_verified"] == spec.sha256
    assert entry["file_size_verified"] == spec.size_bytes
    provenance = models.read_provenance(path)
    assert provenance is not None and entry["retrieved_at"] == provenance["retrieved_at"]
    assert entry["api"]["lfs_sha256"] == spec.sha256 and "checked_at" in entry["api"]
    with pytest.raises(ModelFileError, match="API disagrees"):
        fetch.build_pins(
            [spec],
            tmp_path,
            opener=FakeOpener(api_documents(spec, size=5)),
            command=[],
            commit="c",
        )
    path.write_bytes(b"W" * len(CONTENT))
    with pytest.raises(ModelFileError, match="does not match its pin"):
        fetch.build_pins([spec], tmp_path, opener=FakeOpener([]), command=[], commit="c")


def test_fetch_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = tiny_spec()
    monkeypatch.setitem(REGISTRY, spec.key, spec)
    budget = tmp_path / "budget.toml"
    budget.write_text(BUDGET, encoding="utf-8")
    assert fetch.main(["--models", "nope"]) == 2
    assert "FETCH: FAIL" in capsys.readouterr().err

    good = fetch.remote_info(spec, FakeOpener(api_documents(spec, head="d" * 40)))
    monkeypatch.setattr(fetch, "remote_info", lambda s, opener=None: good)
    assert fetch.main(["--models", spec.key, "--check-api"]) == 0
    assert "matches the API (repository head is now dddd" in capsys.readouterr().out
    same_head = {**good, "repo_head": spec.revision}
    monkeypatch.setattr(fetch, "remote_info", lambda s, opener=None: same_head)
    assert fetch.main(["--models", spec.key, "--check-api"]) == 0
    assert "pinned revision is the head" in capsys.readouterr().out
    monkeypatch.setattr(fetch, "remote_info", lambda s, opener=None: {**good, "size_bytes": 3})
    assert fetch.main(["--models", spec.key, "--check-api"]) == 1

    calls: list[tuple[str, float]] = []

    def fake_download(s: ModelSpec, dest: Path, *, floor_gb: float) -> Path:
        calls.append((s.key, floor_gb))
        write_model(dest, s)
        return dest / s.filename

    monkeypatch.setattr(fetch, "download", fake_download)
    dest = tmp_path / "models"
    argv = ["--models", spec.key, "--dest", str(dest), "--budget", str(budget)]
    assert fetch.main(argv) == 0
    assert calls == [(spec.key, 8)]
    assert "FETCH: OK 1 models" in capsys.readouterr().out

    monkeypatch.setattr(fetch, "remote_info", lambda s, opener=None: good)
    pins = tmp_path / "pins.json"
    assert (
        fetch.main(["--models", spec.key, "--dest", str(dest), "--write-pins", "--pins", str(pins)])
        == 0
    )
    written = json.loads(pins.read_text(encoding="utf-8"))
    assert written["models"][0]["key"] == spec.key
    assert written["command"][:2] == ["python", "tools/fetch_models.py"]

    def refuse(s: ModelSpec, dest: Path, *, floor_gb: float) -> Path:
        raise fetch.DiskFloorError("too little disk")

    monkeypatch.setattr(fetch, "download", refuse)
    assert fetch.main(argv) == 1
    assert "FETCH: FAIL tiny-q4km: too little disk" in capsys.readouterr().err


# --------------------------------------------------------------------------- bench_models


def test_bench_prompts_mirror_the_agent() -> None:
    prompts = bench.bench_prompts()
    assert [p.prompt_id for p in prompts] == [
        "first-request",
        "after-list-files",
        "after-read-file",
    ]
    assert len({p.seed for p in prompts}) == 3
    first, listed, read = (p.messages for p in prompts)
    assert first[0] == Message("system", SYSTEM_PROMPT)
    assert first[1].content.startswith(RULES_INTRO)
    assert all(rule in first[1].content for rule in bench.RULES)
    assert listed[: len(first)] == first and read[: len(listed)] == listed
    assert listed[-1].content.startswith("Result of list_files:\n")
    assert read[-1].content.startswith("Result of read_file:\n")
    tools = [parse_tool_call(m.content) for m in read if m.role == "assistant"]
    assert [t.tool for t in tools if t is not None] == ["list_files", "read_file"]
    assert bench.bench_prompts() == prompts  # fixed


def test_concurrent_ledger_runs(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    start, end = "2026-11-04T02:00:00+00:00", "2026-11-04T03:00:00+00:00"
    assert bench.concurrent_ledger_runs(ledger, start, end, "me") == {}
    rows = [
        {"ts": "2026-11-04T01:59:59+00:00", "run_id": "early"},
        {"ts": "2026-11-04T02:10:00+00:00", "run_id": "dry"},
        {"ts": "2026-11-04T02:20:00+00:00", "run_id": "dry"},
        {"ts": "2026-11-04T02:21:00+00:00", "run_id": "another"},
        {"ts": "2026-11-04T02:30:00+00:00", "run_id": "me"},
        {"ts": "2026-11-04T03:00:01+00:00", "run_id": "late"},
    ]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    found = bench.concurrent_ledger_runs(ledger, start, end, "me")
    assert found == {"another": 1, "dry": 2} and list(found) == ["another", "dry"]


def test_cpu_share_and_portable_raw() -> None:
    assert bench.cpu_share(6.0, 2.0, 4) == 0.75
    assert bench.cpu_share(1.0, 0.0, 4) is None and bench.cpu_share(1.0, 1.0, 0) is None
    raw = {"model": "/root/.cache/m/x.gguf", "choices": []}
    assert bench.portable_raw(raw) == {"model": "x.gguf", "choices": []}
    assert raw["model"] == "/root/.cache/m/x.gguf"  # the input is not changed
    assert bench.portable_raw({"model": 3}) == {"model": 3}
    assert bench.portable_raw({"id": "c"}) == {"id": "c"}


def test_bench_model_reports_contention_with_injected_clocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _, loader, budget = bench_setup(tmp_path, monkeypatch)
    walls = iter([0.0, 1.0, 11.0])  # start, loaded, prompts done: 10 s of prompts
    cpus = iter([5.0, 25.0])  # 20 CPU seconds over 4 threads in 10 s
    monkeypatch.setattr(bench, "load_average", lambda: [6.0, 5.0, 4.0])
    entry = bench.bench_model(
        spec,
        loader,
        client_factory(tmp_path, budget),
        bench.bench_prompts()[:1],
        max_tokens=8,
        temperature=0.0,
        clock=lambda: next(walls),
        process_clock=lambda: next(cpus),
    )
    assert entry["contention"] == {
        "load_average_start": [6.0, 5.0, 4.0],
        "load_average_end": [6.0, 5.0, 4.0],
        "cpu_share": 0.5,
    }
    assert entry["load_and_verify_s"] == 1.0


def test_check_reply_and_rate() -> None:
    assert bench.check_reply(TOOL_REPLY) == {
        "json_valid": True,
        "tool_call_parsed": True,
        "tool": "list_files",
    }
    assert bench.check_reply(f"Sure: {TOOL_REPLY}") == {
        "json_valid": False,
        "tool_call_parsed": True,
        "tool": "list_files",
    }
    assert bench.check_reply('{"answer": 1}') == {
        "json_valid": True,
        "tool_call_parsed": False,
        "tool": None,
    }
    assert bench.check_reply("no json")["tool_call_parsed"] is False
    assert bench.rate(10, 2) == 5.0
    assert bench.rate(0, 2) is None and bench.rate(10, 0) is None and bench.rate(3, -1) is None


def test_machine_facts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Fake CPU @ 1.00GHz\n", encoding="utf-8")
    assert bench.cpu_model(cpuinfo) == "Fake CPU @ 1.00GHz"
    monkeypatch.setattr(bench.platform, "processor", lambda: "")
    monkeypatch.setattr(bench.platform, "machine", lambda: "x86_64")
    assert bench.cpu_model(tmp_path / "missing") == "x86_64"
    monkeypatch.setattr(bench.platform, "machine", lambda: "")
    assert bench.cpu_model(tmp_path / "missing") == "unknown"

    def no_package(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(bench.importlib.metadata, "version", no_package)
    assert bench.llama_cpp_version() == "not installed"
    monkeypatch.setattr(bench.importlib.metadata, "version", lambda name: "0.3.36")
    assert bench.llama_cpp_version() == "0.3.36"
    assert dt.datetime.fromisoformat(bench.now_iso()).tzinfo is not None
    assert bench.git_sha(tmp_path) == "unknown"
    assert re.fullmatch(r"[0-9a-f]{40}", bench.git_sha(REPO_ROOT))
    (tmp_path / "a.py").write_bytes(b"x")
    assert bench.code_hashes(tmp_path, ["a.py", "b.py"]) == {
        "a.py": hashlib.sha256(b"x").hexdigest(),
        "b.py": None,
    }
    assert set(bench.code_hashes(REPO_ROOT)) == set(bench.CODE_FILES)
    load = bench.load_average()
    assert load is None or len(load) == 3

    def no_loadavg() -> tuple[float, float, float]:
        raise OSError("not supported")

    monkeypatch.setattr(bench.os, "getloadavg", no_loadavg)
    assert bench.load_average() is None


def bench_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ModelSpec, list[FakeLlama], Callable[[ModelSpec], LlamaCppBackend], Path]:
    """A registered tiny spec, a loader over fake models, and a budget file."""
    spec = tiny_spec()
    monkeypatch.setitem(REGISTRY, spec.key, spec)
    write_model(tmp_path / "models", spec)
    llamas: list[FakeLlama] = []

    def loader(s: ModelSpec) -> LlamaCppBackend:
        llama = FakeLlama()
        llamas.append(llama)
        return models.load_backend(
            s.key,
            n_ctx=4096,
            n_threads=4,
            directory=tmp_path / "models",
            runtime=fake_runtime(llama),
        )

    budget = tmp_path / "budget.toml"
    budget.write_text(BUDGET, encoding="utf-8")
    return spec, llamas, loader, budget


def client_factory(tmp_path: Path, budget: Path) -> Callable[[ChatBackend], Client]:
    def make(backend: ChatBackend) -> Client:
        return Client(
            backend,
            tmp_path / "ledger.jsonl",
            tmp_path / "cache",
            run_id="bench-test",
            experiment="bench",
            budget_path=budget,
            env={},
        )

    return make


def test_bench_model_measures_and_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, llamas, loader, budget = bench_setup(tmp_path, monkeypatch)
    entry = bench.bench_model(
        spec,
        loader,
        client_factory(tmp_path, budget),
        bench.bench_prompts(),
        max_tokens=384,
        temperature=0.7,
    )
    llama = llamas[0]
    assert llama.closed and llama.resets == 3
    assert [kwargs["seed"] for _, kwargs in llama.calls] == [1101, 1102, 1103]
    assert all(kwargs["grammar"] is not None for _, kwargs in llama.calls)
    assert all(kwargs["max_tokens"] == 384 for _, kwargs in llama.calls)
    summary = entry["summary"]
    assert summary["prompt_tokens_per_s"] == 300.0  # 360 tokens in 1.2 s (fake timers)
    assert summary["gen_tokens_per_s"] == 30.0  # 27 tokens in 0.9 s
    assert (summary["json_valid"], summary["tool_calls_parsed"], summary["prompts"]) == (3, 3, 3)
    assert summary["token_counts_exact"] is True
    call = entry["calls"][0]
    assert call["tool"] == "list_files" and call["finish_reason"] == "stop"
    assert call["messages_sha256"] == messages_sha256(bench.bench_prompts()[0].messages)
    assert (entry["key"], entry["n_threads"], entry["n_ctx"]) == (spec.key, 4, 4096)
    assert entry["recorded_fixtures"] is False and entry["client"]["calls"] == 3
    contention = entry["contention"]
    assert set(contention) == {"load_average_start", "load_average_end", "cpu_share"}
    assert contention["cpu_share"] is None or contention["cpu_share"] >= 0
    rows = read_ledger(tmp_path / "ledger.jsonl")
    assert [r["purpose"] for r in rows] == ["bench"] * 3
    assert {r["session_id"] for r in rows} == {f"bench/{spec.key}"}
    assert {r["experiment"] for r in rows} == {"bench"}


def test_bench_model_records_replay_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _, loader, budget = bench_setup(tmp_path, monkeypatch)
    record_dir = tmp_path / "fixtures"
    record_dir.mkdir()
    entry = bench.bench_model(
        spec,
        loader,
        client_factory(tmp_path, budget),
        bench.bench_prompts(),
        max_tokens=64,
        temperature=0.0,
        record_dir=record_dir,
    )
    assert entry["recorded_fixtures"] is True
    raw_lines = (record_dir / bench.RAW_FILE).read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 3
    first_raw = json.loads(raw_lines[0])["result"]
    assert first_raw["choices"][0]["text"] == TOOL_REPLY
    assert first_raw["model"] == "tiny-Q4_K_M.gguf"  # no local directory in fixtures
    assert len((record_dir / COUNTS_FILE).read_text(encoding="utf-8").splitlines()) == 3
    replay = ReplayBackend(record_dir)
    for prompt in bench.bench_prompts():
        request = ChatRequest(
            messages=prompt.messages,
            max_tokens=64,
            temperature=0.0,
            seed=prompt.seed,
            json_schema=TOOL_CALL_SCHEMA,
        )
        reply = replay.complete(request)
        assert reply.text == TOOL_REPLY
        assert reply.prompt_tokens == replay.count_tokens(prompt.messages)


def test_bench_model_with_a_cached_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, _, loader, budget = bench_setup(tmp_path, monkeypatch)
    make = client_factory(tmp_path, budget)
    prompts = bench.bench_prompts()[:1]
    bench.bench_model(spec, loader, make, prompts, max_tokens=8, temperature=0.0)
    again = bench.bench_model(spec, loader, make, prompts, max_tokens=8, temperature=0.0)
    call = again["calls"][0]
    assert call["cached"] is True and call["prompt_eval_tokens"] == 0
    assert call["prompt_tokens_per_s"] is None and again["client"]["cache_hits"] == 1


def test_summarize() -> None:
    calls = [
        {
            "prompt_eval_tokens": 100,
            "prompt_eval_ms": 500.0,
            "eval_tokens": 10,
            "eval_ms": 1000.0,
            "completion_tokens": 10,
            "elapsed_s": 2.5,
            "json_valid": True,
            "tool_call_parsed": True,
            "count_matches": True,
        },
        {
            "prompt_eval_tokens": 100,
            "prompt_eval_ms": 500.0,
            "eval_tokens": 10,
            "eval_ms": 1000.0,
            "completion_tokens": 10,
            "elapsed_s": 2.5,
            "json_valid": False,
            "tool_call_parsed": False,
            "count_matches": False,
        },
    ]
    summary = bench.summarize(calls)
    assert summary == {
        "prompts": 2,
        "prompt_tokens_per_s": 200.0,
        "gen_tokens_per_s": 10.0,
        "gen_tokens_per_s_wall": 5.0,
        "json_valid": 1,
        "tool_calls_parsed": 1,
        "token_counts_exact": False,
        "call_s": 5.0,
    }


def test_run_benchmark_respects_the_time_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _, loader, budget = bench_setup(tmp_path, monkeypatch)
    second = tiny_spec(key="tiny-two-q4km", family="tiny2")
    ticks = iter(range(0, 10_000, 100))
    messages: list[str] = []
    entries = bench.run_benchmark(
        [spec, second],
        loader,
        client_factory(tmp_path, budget),
        max_tokens=16,
        temperature=0.0,
        time_budget_s=150,
        clock=lambda: float(next(ticks)),
        log=messages.append,
    )
    assert entries[0]["key"] == spec.key and "summary" in entries[0]
    assert entries[1] == {"key": "tiny-two-q4km", "family": "tiny2", "skipped": "time budget"}
    assert any("skipped" in m for m in messages) and any("tok/s" in m for m in messages)


def test_build_artifact_labels_and_provenance() -> None:
    artifact = bench.build_artifact(
        [{"key": "m"}], settings={"threads": 4}, provenance={"git_sha": "abc"}
    )
    assert artifact["format"] == bench.BENCH_FORMAT
    assert artifact["synthetic_prompts"] is True
    assert "not a study result" in artifact["kind"]
    assert "lower bound" in artifact["concurrency_note"]
    assert artifact["settings"]["threads"] == 4
    assert [p["seed"] for p in artifact["settings"]["prompts"]] == [1101, 1102, 1103]
    assert len(artifact["settings"]["json_schema_sha256"]) == 64
    assert artifact["provenance"] == {"git_sha": "abc"} and artifact["models"] == [{"key": "m"}]
    text = json.dumps(artifact)
    assert "\u2014" not in text


def test_bench_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec, _, loader, budget = bench_setup(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(NO_MODEL_CALLS_ENV, raising=False)
    monkeypatch.delenv("CAP_SHARE", raising=False)
    assert bench.main(["--models", "nope"], loader=loader) == 2
    assert bench.main(["--record-model", spec.key], loader=loader) == 2
    assert (
        bench.main(
            ["--models", spec.key, "--record-model", CANDIDATES[0].key, "--record-dir", "fx"],
            loader=loader,
        )
        == 2
    )
    assert "is not benchmarked" in capsys.readouterr().err

    out = tmp_path / "bench.json"
    argv = ["--models", spec.key, "--out", str(out), "--budget", str(budget)]
    assert bench.main([*argv, "--record-model", spec.key, "--record-dir", "fx"], loader=loader) == 0
    assert "BENCH: OK 1 of 1 models" in capsys.readouterr().out
    artifact = json.loads(out.read_text(encoding="utf-8"))
    provenance = artifact["provenance"]
    assert provenance["git_sha"] == "unknown"  # tmp_path is not a git checkout
    assert provenance["code_hashed"] == "at the start of the run"
    assert provenance["concurrent_ledger_runs"] == {}  # only this run wrote to the ledger
    assert "contention_note" in artifact
    assert provenance["command"][:2] == ["python", "tools/bench_models.py"]
    assert provenance["run_id"].startswith("bench-models-")
    assert provenance["cache_dir"] == f"runs/bench_models/{provenance['run_id']}/cache"
    assert artifact["settings"]["threads"] == 4 and artifact["settings"]["verify_sha256"] is True
    assert artifact["models"][0]["summary"]["tool_calls_parsed"] == 3
    assert (tmp_path / "fx" / bench.RAW_FILE).exists()
    rows = read_ledger(tmp_path / "artifacts" / "usage" / "ledger.jsonl")
    assert len(rows) == 3 and rows[0]["run_id"] == provenance["run_id"]

    assert bench.main([*argv, "--record-model", spec.key, "--record-dir", "fx"], loader=loader) == 2
    assert "not empty" in capsys.readouterr().err

    monkeypatch.setenv(NO_MODEL_CALLS_ENV, "1")
    assert bench.main([*argv, "--run-id", "blocked", "--no-verify"], loader=loader) == 1
    assert "BENCH: FAIL ModelCallsDisabled" in capsys.readouterr().err
