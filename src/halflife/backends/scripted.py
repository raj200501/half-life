"""Deterministic scripted replies (tests and the offline demo; no model runs).

Replies come from a policy callable (``policy(request) -> str``) or from a list
served in order. Token counts are a fixed estimate (about four characters per
token) and elapsed time is 0, so everything a scripted session produces is
deterministic. Anything produced with this backend is synthetic and must be
labeled so wherever it is shown (rule 4).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from halflife.backends.base import (
    ChatRequest,
    ChatResponse,
    Message,
    estimate_context_tokens,
    estimate_tokens,
)

Policy = Callable[[ChatRequest], str]


class ScriptExhausted(RuntimeError):
    """Raised when a list-scripted backend is asked for more replies than it holds."""


class ScriptedBackend:
    """Replies from a policy or a fixed list.

    Args:
        policy: Called with each request; returns the reply text.
        outputs: Replies served in order (exactly one of ``policy`` and
            ``outputs`` must be given).
        name: Backend name reported in responses.

    Attributes:
        requests: Every request received, in order (handy in tests).
    """

    kind = "scripted"
    offline_safe = True
    params_b = 0.0

    def __init__(
        self,
        policy: Policy | None = None,
        *,
        outputs: Sequence[str] | None = None,
        name: str = "scripted",
    ) -> None:
        if (policy is None) == (outputs is None):
            raise ValueError("give exactly one of policy or outputs")
        self._policy = policy
        self._outputs = list(outputs or [])
        self._name = name
        self.requests: list[ChatRequest] = []

    @property
    def name(self) -> str:
        """Backend name."""
        return self._name

    @property
    def remaining(self) -> int:
        """Replies left in a list script (0 for a policy)."""
        return len(self._outputs)

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Return the scripted reply for ``request``.

        Raises:
            ScriptExhausted: when a list script has no reply left.
            TypeError: when a policy returns something other than a string.
        """
        self.requests.append(request)
        if self._policy is not None:
            text = self._policy(request)
            if not isinstance(text, str):
                raise TypeError(f"policy returned {type(text).__name__}, expected str")
        elif self._outputs:
            text = self._outputs.pop(0)
        else:
            raise ScriptExhausted(
                f"scripted backend {self.name!r} has no reply left for request {len(self.requests)}"
            )
        return ChatResponse(
            text=text,
            prompt_tokens=self.count_tokens(request.messages),
            completion_tokens=estimate_tokens(text) if text else 0,
            elapsed_s=0.0,
            cached=False,
            backend=self.name,
        )

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Deterministic estimate (see :func:`halflife.backends.base.estimate_context_tokens`)."""
        return estimate_context_tokens(messages)
