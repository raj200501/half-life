"""Model backends behind :class:`halflife.client.Client`.

* :mod:`halflife.backends.llama_cpp`: pinned local GGUF models (real model calls).
* :mod:`halflife.backends.scripted`: deterministic replies from a policy or a
  list (tests and the demo; no model runs).
* :mod:`halflife.backends.replay`: recorded replies looked up by request hash
  (contract tests and replays of banked sessions; no model runs).
"""

from halflife.backends.base import (
    ChatBackend,
    ChatRequest,
    ChatResponse,
    Message,
    as_messages,
)
from halflife.backends.replay import RecordingBackend, ReplayBackend, ReplayMiss
from halflife.backends.scripted import ScriptedBackend, ScriptExhausted

__all__ = [
    "ChatBackend",
    "ChatRequest",
    "ChatResponse",
    "Message",
    "RecordingBackend",
    "ReplayBackend",
    "ReplayMiss",
    "ScriptExhausted",
    "ScriptedBackend",
    "as_messages",
]
