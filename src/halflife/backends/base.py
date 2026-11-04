"""Chat request and response types shared by every model backend.

A backend turns a :class:`ChatRequest` into a :class:`ChatResponse` and counts
the tokens of a context exactly as it would send it. :class:`halflife.client.Client`
is the only caller in experiment code: it adds caps, the usage ledger, and the
content-hash cache on top of a backend.

Requests have a canonical JSON form (:meth:`ChatRequest.canonical`) so equal
requests hash equally across processes and Python versions. Canonical here
means compact and in a fixed key order, not sorted: request fields are always
written in the same order, and a JSON schema keeps its own property order,
because that order decides the order in which a constrained reply writes its
fields (so two schemas that differ only in order are different requests).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast, get_args

Role = Literal["system", "user", "assistant"]
ROLES: tuple[str, ...] = get_args(Role)
BackendKind = Literal["local", "scripted", "replay"]
BACKEND_KINDS: tuple[str, ...] = get_args(BackendKind)
FINISH_REASONS: tuple[str, ...] = ("stop", "length")


def canonical_json(data: Any) -> str:
    """Compact JSON that keeps key order and does not escape non-ASCII text.

    Callers build their dicts in a fixed order, so the text is stable across
    runs; see the module docstring for why keys are not sorted.
    """
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Message:
    """One chat message."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"message role {self.role!r} not in {list(ROLES)}")
        if not isinstance(self.content, str):
            raise ValueError("message content must be a string")

    def to_dict(self) -> dict[str, str]:
        """JSON-ready form."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Message:
        """Inverse of :meth:`to_dict`; extra keys are ignored."""
        return cls(role=cast(Role, data["role"]), content=data["content"])


def as_messages(messages: Sequence[Mapping[str, str]] | Sequence[Message]) -> tuple[Message, ...]:
    """Convert role/content mappings (the agent's context format) to messages."""
    return tuple(m if isinstance(m, Message) else Message.from_dict(m) for m in messages)


def messages_sha256(messages: Sequence[Message]) -> str:
    """Content hash of a context (used to record and replay token counts)."""
    return sha256_text(canonical_json([m.to_dict() for m in messages]))


@dataclass(frozen=True)
class ChatRequest:
    """Everything that determines a model reply.

    Attributes:
        messages: The context, oldest first.
        max_tokens: Most tokens the reply may have.
        temperature: Sampling temperature; 0 means greedy decoding.
        seed: Sampling seed (every call gets its own, see ``halflife.agent``).
        json_schema: When set, the reply is constrained to JSON matching it.
        stop: Extra stop strings (the model's end-of-turn tokens always stop).
    """

    messages: tuple[Message, ...]
    max_tokens: int
    temperature: float
    seed: int
    json_schema: Mapping[str, Any] | None = None
    stop: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a chat request needs at least one message")
        if not all(isinstance(m, Message) for m in self.messages):
            raise ValueError("messages must be Message objects (see as_messages)")
        if isinstance(self.max_tokens, bool) or self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens!r}")
        if not 0.0 <= float(self.temperature) <= 5.0:
            raise ValueError(f"temperature must be in [0, 5], got {self.temperature!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed!r}")
        if self.json_schema is not None and not isinstance(self.json_schema, Mapping):
            raise ValueError("json_schema must be a mapping or None")
        if any(not isinstance(s, str) or not s for s in self.stop):
            raise ValueError("stop strings must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form; the schema is copied through JSON (order kept) so it is plain data."""
        return {
            "messages": [m.to_dict() for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": float(self.temperature),
            "seed": self.seed,
            "json_schema": (
                None if self.json_schema is None else json.loads(canonical_json(self.json_schema))
            ),
            "stop": list(self.stop),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChatRequest:
        """Inverse of :meth:`to_dict`."""
        schema = data.get("json_schema")
        return cls(
            messages=tuple(Message.from_dict(m) for m in data["messages"]),
            max_tokens=int(data["max_tokens"]),
            temperature=float(data["temperature"]),
            seed=int(data["seed"]),
            json_schema=dict(schema) if schema is not None else None,
            stop=tuple(str(s) for s in data.get("stop") or ()),
        )

    def canonical(self) -> str:
        """Canonical JSON text of the request."""
        return canonical_json(self.to_dict())

    def sha256(self) -> str:
        """Content hash of the request alone (replay fixtures are keyed by it)."""
        return sha256_text(self.canonical())


@dataclass(frozen=True)
class ChatResponse:
    """A model reply and what it cost.

    Attributes:
        text: Reply text.
        prompt_tokens: Tokens in the formatted prompt (exact for local backends).
        completion_tokens: Tokens generated.
        elapsed_s: Wall time of the backend call (0 for scripted replies).
        cached: True when the client served the reply from its cache.
        backend: Name of the backend that produced the reply.
        finish_reason: ``stop`` (end of turn, stop string, or closed JSON) or
            ``length`` (hit ``max_tokens``).
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    cached: bool = False
    backend: str = ""
    finish_reason: str = "stop"

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.elapsed_s < 0:
            raise ValueError("elapsed_s must be non-negative")
        if self.finish_reason not in FINISH_REASONS:
            raise ValueError(f"finish_reason {self.finish_reason!r} not in {FINISH_REASONS}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_s": round(self.elapsed_s, 6),
            "cached": self.cached,
            "backend": self.backend,
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChatResponse:
        """Inverse of :meth:`to_dict`."""
        return cls(
            text=str(data["text"]),
            prompt_tokens=int(data["prompt_tokens"]),
            completion_tokens=int(data["completion_tokens"]),
            elapsed_s=float(data["elapsed_s"]),
            cached=bool(data.get("cached", False)),
            backend=str(data.get("backend", "")),
            finish_reason=str(data.get("finish_reason", "stop")),
        )


class ChatBackend(Protocol):
    """What the client needs from a model backend.

    Attributes:
        name: Identifies the backend, the exact model file, and any setting
            that changes replies; it is part of the client's cache key.
        kind: ``local``, ``scripted``, or ``replay`` (also the ledger ``kind``
            for backends that make real model calls).
        offline_safe: True when the backend never runs a model (scripted and
            replay), so it is allowed under ``HALFLIFE_NO_MODEL_CALLS=1``.
        params_b: Model size in billions of parameters (0 when no model runs).
    """

    @property
    def name(self) -> str:
        """Backend identity (part of the cache key)."""
        ...

    @property
    def kind(self) -> str:
        """``local``, ``scripted``, or ``replay``."""
        ...

    @property
    def offline_safe(self) -> bool:
        """True when no model ever runs."""
        ...

    @property
    def params_b(self) -> float:
        """Model size in billions of parameters."""
        ...

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Produce the reply to ``request``."""
        ...

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Tokens in ``messages`` as the backend would format them for a reply."""
        ...


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate (about four characters per token, at least 1)."""
    return max(1, (len(text) + 3) // 4)


def estimate_context_tokens(messages: Sequence[Message]) -> int:
    """Deterministic token estimate of a context, with 4 tokens of framing per message."""
    return sum(estimate_tokens(m.content) + 4 for m in messages) + 2


CALL_RECORD_FORMAT = "halflife-call/1"
"""Layout of one recorded call, shared by the client cache and replay fixtures."""


def call_record(
    request: ChatRequest,
    response: ChatResponse,
    *,
    backend: str,
    model: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One recorded call: the request, the reply as first produced, and labels.

    Args:
        request: The request.
        response: The backend's reply (``cached`` is stored as False).
        backend: Name of the backend that produced it.
        model: Model identifier.
        extra: Further labels (run id, purpose, timestamps).
    """
    stored = ChatResponse.from_dict({**response.to_dict(), "cached": False})
    return {
        "format": CALL_RECORD_FORMAT,
        "request_sha256": request.sha256(),
        "backend": backend,
        "model": model,
        **dict(extra or {}),
        "request": request.to_dict(),
        "response": stored.to_dict(),
    }


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON so that readers see either the old file or the complete new one.

    The text goes to a temporary file in the same directory, is flushed to
    disk, and then replaces ``path`` in one rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
