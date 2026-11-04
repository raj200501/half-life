"""Recorded replies looked up by request content hash (no model runs).

A fixtures directory holds recorded calls in the shared record layout
(:func:`halflife.backends.base.call_record`), one JSON file per call, plus an
optional ``counts.jsonl`` of recorded token counts::

    <dir>/<request_sha256>.json   {"format": "halflife-call/1", "request": ..., "response": ...}
    <dir>/counts.jsonl            {"messages_sha256": ..., "count": N} per line

Files may sit in subdirectories, so a client cache directory (same record
layout) can be replayed as it is. :class:`RecordingBackend` writes the layout
while forwarding calls to a real backend; :class:`ReplayBackend` serves it.
A request that was never recorded raises :class:`ReplayMiss` naming the
request hash, so a replay can never silently fall back to a live model.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from halflife.backends.base import (
    CALL_RECORD_FORMAT,
    ChatBackend,
    ChatRequest,
    ChatResponse,
    Message,
    call_record,
    estimate_context_tokens,
    messages_sha256,
    write_json_atomic,
)

COUNTS_FILE = "counts.jsonl"
PREVIEW_CHARS = 80


class ReplayMiss(KeyError):
    """Raised when a replay is asked for a call that was never recorded."""

    def __str__(self) -> str:
        """The message itself, without KeyError's quoting."""
        return str(self.args[0]) if self.args else "replay miss"


def _load_record(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"replay fixture {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != CALL_RECORD_FORMAT:
        raise ValueError(f"replay fixture {path} is not a {CALL_RECORD_FORMAT} record")
    if "request" not in data or "response" not in data:
        raise ValueError(f"replay fixture {path} lacks a request or a response")
    return data


class ReplayBackend:
    """Serves recorded replies.

    Args:
        fixtures_dir: Directory with recorded calls (searched recursively).
        name: Backend name; defaults to ``replay:<directory name>``.
        source_backend: Only use records produced by this backend name.
        strict_counts: When True (default), an unrecorded token count raises
            :class:`ReplayMiss`; when False it falls back to the scripted estimate.

    Raises:
        FileNotFoundError: when ``fixtures_dir`` does not exist.
        ValueError: on a malformed record, or two records that give the same
            request different replies.
    """

    kind = "replay"
    offline_safe = True
    params_b = 0.0

    def __init__(
        self,
        fixtures_dir: Path,
        *,
        name: str | None = None,
        source_backend: str | None = None,
        strict_counts: bool = True,
    ) -> None:
        self.fixtures_dir = Path(fixtures_dir)
        if not self.fixtures_dir.is_dir():
            raise FileNotFoundError(f"replay fixtures directory {self.fixtures_dir} does not exist")
        self._name = name or f"replay:{self.fixtures_dir.name}"
        self.strict_counts = strict_counts
        self.responses: dict[str, dict[str, Any]] = {}
        self.counts: dict[str, int] = {}
        self.served = 0
        for path in sorted(self.fixtures_dir.rglob("*.json")):
            record = _load_record(path)
            if source_backend is not None and record.get("backend") != source_backend:
                continue
            key = ChatRequest.from_dict(record["request"]).sha256()
            response = dict(record["response"])
            previous = self.responses.get(key)
            if previous is not None and previous["text"] != response["text"]:
                raise ValueError(f"replay fixtures give request {key[:16]} two different replies")
            self.responses[key] = response
        for path in sorted(self.fixtures_dir.rglob(COUNTS_FILE)):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                digest, count = str(entry["messages_sha256"]), int(entry["count"])
                if self.counts.get(digest, count) != count:
                    raise ValueError(f"{path}:{number} records a different count for {digest[:16]}")
                self.counts[digest] = count

    @property
    def name(self) -> str:
        """Backend name."""
        return self._name

    def __len__(self) -> int:
        """Number of recorded replies."""
        return len(self.responses)

    def complete(self, request: ChatRequest) -> ChatResponse:
        """The recorded reply to ``request``.

        Raises:
            ReplayMiss: when the request was never recorded.
        """
        key = request.sha256()
        stored = self.responses.get(key)
        if stored is None:
            last = request.messages[-1]
            preview = last.content[:PREVIEW_CHARS].replace("\n", " ")
            raise ReplayMiss(
                f"no recorded reply for request {key} ({len(request.messages)} messages, "
                f"last {last.role}: {preview!r}, seed {request.seed}) in {self.fixtures_dir}; "
                "record it with halflife.backends.replay.RecordingBackend"
            )
        self.served += 1
        return ChatResponse.from_dict({**stored, "cached": False, "backend": self.name})

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """The recorded token count of ``messages``.

        Raises:
            ReplayMiss: when the count was never recorded and ``strict_counts`` is set.
        """
        digest = messages_sha256(messages)
        if digest in self.counts:
            return self.counts[digest]
        if self.strict_counts:
            raise ReplayMiss(
                f"no recorded token count for a {len(messages)}-message context "
                f"{digest} in {self.fixtures_dir}"
            )
        return estimate_context_tokens(messages)


class RecordingBackend:
    """Forwards calls to ``inner`` and records them as replay fixtures.

    It presents itself as ``inner`` (same name, kind, safety, and size), so the
    client applies the same caps and ledger rules as for ``inner`` itself.
    """

    def __init__(self, inner: ChatBackend, fixtures_dir: Path) -> None:
        self.inner = inner
        self.fixtures_dir = Path(fixtures_dir)
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        self.model = str(getattr(inner, "model", inner.name))
        self._counted: set[str] = set()

    @property
    def name(self) -> str:
        """The inner backend's name."""
        return self.inner.name

    @property
    def kind(self) -> str:
        """The inner backend's kind."""
        return self.inner.kind

    @property
    def offline_safe(self) -> bool:
        """The inner backend's offline safety."""
        return self.inner.offline_safe

    @property
    def params_b(self) -> float:
        """The inner backend's size."""
        return self.inner.params_b

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Call ``inner`` and record the call."""
        response = self.inner.complete(request)
        record = call_record(request, response, backend=self.name, model=self.model)
        write_json_atomic(self.fixtures_dir / f"{request.sha256()}.json", record)
        return response

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Count with ``inner`` and record the count once per distinct context."""
        count = self.inner.count_tokens(messages)
        digest = messages_sha256(messages)
        if digest not in self._counted:
            self._counted.add(digest)
            line = json.dumps({"messages_sha256": digest, "count": count}, sort_keys=True)
            with (self.fixtures_dir / COUNTS_FILE).open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return count
