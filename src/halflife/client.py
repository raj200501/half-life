"""The single model client (section 8 of the build brief, usage hygiene).

Every model call in experiment code goes through :meth:`Client.complete`:

1. With ``HALFLIFE_NO_MODEL_CALLS=1`` it refuses (:class:`ModelCallsDisabled`)
   unless the backend is offline-safe (scripted or replay), which is how the
   offline demo and the ship check guarantee that no model runs.
2. A real backend's reply is cached by the content hash of (backend name,
   request). A cache hit returns the stored reply with ``cached=True`` and
   never calls the backend: a banked call is never re-run.
3. Before a real call it enforces the caps in budget.toml: a local model must
   not exceed ``local_model_max_params_b``; a Claude kind needs a clean billing
   guard and room under its daily cap, scaled by ``cap_share`` (budget.toml
   and ``CAP_SHARE``, whichever is smaller); a free-tier kind needs
   ``FREE_TIER_KEYS_CONFIRMED=1``.
4. It appends one JSON line per real call to the usage ledger
   (``artifacts/usage/ledger.jsonl``) and then writes the cache entry atomically.

Offline-safe backends run no model, so their calls skip the cache and the
ledger: they are deterministic and free, and the demo must leave tracked
files (the ledger is one) unchanged.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from halflife import billing_guard
from halflife.backends.base import (
    CALL_RECORD_FORMAT,
    ChatBackend,
    ChatRequest,
    ChatResponse,
    Message,
    as_messages,
    call_record,
    canonical_json,
    sha256_text,
    write_json_atomic,
)
from halflife.budget import (
    DEFAULT_BUDGET_PATH,
    DEFAULT_LEDGER_PATH,
    cap_share,
    effective_caps,
    file_cap_share,
    load_caps,
    read_ledger,
)

DEFAULT_CACHE_DIR = Path("runs/cache")
NO_MODEL_CALLS_ENV = "HALFLIFE_NO_MODEL_CALLS"
LEDGER_KINDS: tuple[str, ...] = (
    "claude_agent_run",
    "claude_single_call",
    "claude_review_run",
    "local",
    "free_tier",
)
CLAUDE_DAILY_CAPS: dict[str, str] = {
    "claude_agent_run": "claude_agent_runs_per_day",
    "claude_single_call": "claude_single_calls_per_day",
    "claude_review_run": "claude_review_runs_per_day",
}
_TRUE = {"1", "true", "yes", "on"}


class ModelCallsDisabled(RuntimeError):
    """Raised for a real model call while ``HALFLIFE_NO_MODEL_CALLS=1`` is set."""


class CapExceeded(RuntimeError):
    """Raised when a call would break a cap in budget.toml."""


class BillingGuardRefused(RuntimeError):
    """Raised for a Claude call while the billing guard is not clean."""


class CacheCorrupt(RuntimeError):
    """Raised when a cache entry cannot be trusted (the call is not silently re-run)."""


def env_flag(env: Mapping[str, str], name: str) -> bool:
    """True when ``name`` is set to 1, true, yes, or on (any case)."""
    return env.get(name, "").strip().lower() in _TRUE


def cache_key(backend_name: str, request: ChatRequest) -> str:
    """SHA-256 of the canonical JSON of ``[backend name, request]``."""
    return sha256_text(canonical_json([backend_name, request.to_dict()]))


def utc_now() -> dt.datetime:
    """Current UTC time."""
    return dt.datetime.now(dt.UTC)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON line under an exclusive lock, flushed to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass
class ClientStats:
    """What one client did, for cost reporting (rule 15).

    Attributes:
        calls: Real backend calls (each one is a ledger line).
        cache_hits: Replies served from the cache.
        offline_calls: Calls to offline-safe backends.
        prompt_tokens: Prompt tokens of real calls.
        completion_tokens: Completion tokens of real calls.
        wall_s: Wall time of real calls.
    """

    calls: int = 0
    cache_hits: int = 0
    offline_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "offline_calls": self.offline_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_s": round(self.wall_s, 3),
        }


class Client:
    """Caps, usage ledger, and content-hash cache around one backend.

    Args:
        backend: The model backend.
        ledger_path: Usage ledger (JSON lines).
        cache_dir: Cache directory (``runs/cache`` by default, git-ignored).
        run_id: Run label written to every ledger line.
        experiment: Experiment label (``e1``, ``pilot``, ``bench``, ...).
        budget_path: budget.toml with the caps.
        env: Environment to read flags from; defaults to ``os.environ`` at
            call time.
        clock: Monotonic clock (tests inject one).

    Raises:
        ValueError: when a backend that runs a model has a kind the ledger
            does not know.
        halflife.budget.BudgetError: when budget.toml is invalid.
        CapExceeded: when a local model is larger than the cap.
    """

    def __init__(
        self,
        backend: ChatBackend,
        ledger_path: Path = DEFAULT_LEDGER_PATH,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        run_id: str = "adhoc",
        experiment: str = "dev",
        budget_path: Path = DEFAULT_BUDGET_PATH,
        *,
        env: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.ledger_path = Path(ledger_path)
        self.cache_dir = Path(cache_dir)
        self.run_id = run_id
        self.experiment = experiment
        self.budget_path = Path(budget_path)
        self._env_override = env
        self._clock = clock
        self.model = str(getattr(backend, "model", backend.name))
        self.stats = ClientStats()
        self.caps: dict[str, float] = {}
        if not backend.offline_safe:
            if backend.kind not in LEDGER_KINDS:
                raise ValueError(
                    f"backend kind {backend.kind!r} is not a ledger kind {list(LEDGER_KINDS)}"
                )
            share = cap_share(self.env, file_share=file_cap_share(self.budget_path))
            self.caps = effective_caps(load_caps(self.budget_path), share)
            self._check_size()

    @property
    def env(self) -> Mapping[str, str]:
        """The environment flags are read from."""
        return os.environ if self._env_override is None else self._env_override

    # ------------------------------------------------------------------ caps

    def _check_size(self) -> None:
        if self.backend.kind != "local":
            return
        cap = self.caps["local_model_max_params_b"]
        if self.backend.params_b > cap:
            raise CapExceeded(
                f"{self.model} has {self.backend.params_b:g}B parameters; "
                f"local_model_max_params_b is {cap:g}"
            )

    def used_today(self, kind: str) -> int:
        """Real calls of ``kind`` in the ledger on the current UTC day."""
        today = utc_now().date().isoformat()
        return sum(
            1
            for record in read_ledger(self.ledger_path)
            if record.get("kind") == kind
            and str(record.get("ts", ""))[:10] == today
            and not record.get("cached", False)
        )

    def _enforce_caps(self) -> None:
        kind = self.backend.kind
        self._check_size()
        if kind in CLAUDE_DAILY_CAPS:
            report = billing_guard.check(self.env)
            if not report.clean:
                raise BillingGuardRefused(
                    "Claude calls are refused: " + "; ".join(report.problems())
                )
            cap_key = CLAUDE_DAILY_CAPS[kind]
            used = self.used_today(kind)
            if used >= self.caps[cap_key]:
                raise CapExceeded(f"{cap_key} = {self.caps[cap_key]:g} already used ({used})")
        if kind == "free_tier" and not env_flag(self.env, "FREE_TIER_KEYS_CONFIRMED"):
            raise CapExceeded("free-tier calls need FREE_TIER_KEYS_CONFIRMED=1")

    # ------------------------------------------------------------------ cache

    def cache_path(self, key: str) -> Path:
        """Where the cache entry for ``key`` lives."""
        return self.cache_dir / key[:2] / f"{key}.json"

    def _read_cache(self, key: str, request: ChatRequest) -> ChatResponse | None:
        path = self.cache_path(key)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            ok = (
                isinstance(record, dict)
                and record.get("format") == CALL_RECORD_FORMAT
                and record.get("cache_key") == key
                and record.get("backend") == self.backend.name
                and canonical_json(record.get("request")) == request.canonical()
            )
            response = ChatResponse.from_dict(record["response"]) if ok else None
        except (ValueError, KeyError, TypeError) as exc:
            raise CacheCorrupt(
                f"cache entry {path} is unreadable ({exc}); delete it to allow a re-run"
            ) from exc
        if response is None:
            raise CacheCorrupt(
                f"cache entry {path} does not match its key; delete it to allow a re-run"
            )
        return replace(response, cached=True)

    # ------------------------------------------------------------------ calls

    def complete(
        self, request: ChatRequest, *, purpose: str = "step", session_id: str | None = None
    ) -> ChatResponse:
        """Answer ``request`` through the cache, caps, and ledger.

        Args:
            request: The chat request.
            purpose: Why the call is made (``step``, ``probe``, ``compaction``,
                ``bench``, ...), written to the ledger.
            session_id: Session the call belongs to, written to the ledger.

        Raises:
            ModelCallsDisabled: for a real backend under ``HALFLIFE_NO_MODEL_CALLS=1``.
            CapExceeded: when a cap would be broken.
            BillingGuardRefused: for a Claude kind with a dirty billing guard.
            CacheCorrupt: when the cache entry for this request is unusable.
        """
        if not purpose.strip():
            raise ValueError("purpose must be a non-empty label")
        backend = self.backend
        if backend.offline_safe:
            self.stats.offline_calls += 1
            return backend.complete(request)
        if env_flag(self.env, NO_MODEL_CALLS_ENV):
            raise ModelCallsDisabled(
                f"{NO_MODEL_CALLS_ENV}=1 forbids model calls; backend {backend.name} runs a model"
            )
        key = cache_key(backend.name, request)
        cached = self._read_cache(key, request)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        self._enforce_caps()
        start = self._clock()
        response = backend.complete(request)
        wall = self._clock() - start
        self.stats.calls += 1
        self.stats.prompt_tokens += response.prompt_tokens
        self.stats.completion_tokens += response.completion_tokens
        self.stats.wall_s += wall
        timestamp = utc_now().isoformat(timespec="seconds")
        append_jsonl(
            self.ledger_path,
            {
                "ts": timestamp,
                "kind": backend.kind,
                "backend": backend.name,
                "model": self.model,
                "run_id": self.run_id,
                "experiment": self.experiment,
                "session_id": session_id,
                "purpose": purpose,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "wall_s": round(wall, 3),
                "cash_usd": 0,
                "plan_equivalent_usd": 0,
                "cached": False,
                "params_b": backend.params_b,
                "cache_key": key,
            },
        )
        record = call_record(
            request,
            response,
            backend=backend.name,
            model=self.model,
            extra={
                "cache_key": key,
                "created": timestamp,
                "run_id": self.run_id,
                "experiment": self.experiment,
                "session_id": session_id,
                "purpose": purpose,
            },
        )
        write_json_atomic(self.cache_path(key), record)
        return response

    def count_tokens(self, messages: Sequence[Mapping[str, str]] | Sequence[Message]) -> int:
        """Token count of a context as the backend formats it (not a model call)."""
        return self.backend.count_tokens(as_messages(messages))
