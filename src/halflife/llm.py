"""The agent's model port over the single client.

:class:`ClientModel` implements :class:`halflife.compaction.ModelPort`, the
interface the minimal agent, compaction, and probes call, and sends every call
through :meth:`halflife.client.Client.complete`, so caps, the usage ledger,
and the cache apply to agent steps, recall probes, and compaction summaries
alike. The runner sets :attr:`ClientModel.session_id` before each session so
ledger lines name the session.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from halflife.backends.base import ChatRequest, ChatResponse, as_messages
from halflife.client import Client

if TYPE_CHECKING:
    from halflife.compaction import ModelPort


class ClientModel:
    """A :class:`~halflife.compaction.ModelPort` backed by a :class:`~halflife.client.Client`.

    Args:
        client: The client every call goes through.
        session_id: Session label for ledger lines (the runner may change it
            between sessions).
        stop: Extra stop strings added to every request.

    Attributes:
        last_response: The most recent reply, with token counts and timing.
    """

    def __init__(
        self, client: Client, *, session_id: str | None = None, stop: Sequence[str] = ()
    ) -> None:
        self.client = client
        self.session_id = session_id
        self.stop = tuple(stop)
        self.last_response: ChatResponse | None = None

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        json_schema: Mapping[str, Any] | None = None,
        purpose: str = "step",
    ) -> str:
        """Return the model's reply text for ``messages``."""
        request = ChatRequest(
            messages=as_messages(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            json_schema=json_schema,
            stop=self.stop,
        )
        response = self.client.complete(request, purpose=purpose, session_id=self.session_id)
        self.last_response = response
        return response.text

    def count_tokens(self, messages: Sequence[Mapping[str, str]]) -> int:
        """Token count of a context as the backend formats it."""
        return self.client.count_tokens(messages)


if TYPE_CHECKING:

    def _implements_model_port(model: ClientModel) -> ModelPort:
        """Lets mypy check that ClientModel matches ModelPort exactly."""
        return model
