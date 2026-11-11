"""Local GGUF models through llama-cpp-python (section 8, backend preference 1).

One pinned model is loaded per process and the ``Llama`` object is kept, so
llama.cpp reuses the KV cache for the longest shared prefix between
consecutive calls (the agent's context grows by appending, so most of each
prompt is already evaluated).

Prompts are formatted by this module, not by llama-cpp-python's chat
handlers, so the formatting is fixed and the token count is exact:

* the GGUF's own chat template (``tokenizer.chat_template``) is rendered with
  Jinja2 in a sandbox, as Hugging Face transformers does, with
  ``add_generation_prompt`` set and ``strftime_now`` fixed to :data:`FIXED_DATE`
  (some templates print today's date, which would make prompts, and so cache
  keys, change from day to day);
* consecutive messages with the same role are merged (some templates require
  alternation; the agent already merges, so this only affects other callers);
* the BOS token is handled as llama.cpp's server does: a BOS rendered by the
  template is dropped when the tokenizer adds one itself;
* the text is tokenized with special tokens parsed, and those exact tokens are
  sent to ``create_completion`` and counted by :meth:`LlamaCppBackend.count_tokens`.

JSON-schema-constrained replies use a GBNF grammar built from the schema by
llama-cpp-python, with the string-character rule tightened so raw control
characters (which strict JSON parsers reject) cannot appear inside strings.

``llama_cpp`` and ``jinja2`` are imported lazily (they come with the optional
``local`` extra), so the rest of the package works without them; tests inject
a fake runtime.
"""

from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from halflife.backends.base import (
    ChatRequest,
    ChatResponse,
    Message,
    canonical_json,
    sha256_text,
)

SystemMode = Literal["native", "merge_first_user"]
SYSTEM_MODES: tuple[str, ...] = ("native", "merge_first_user")
TEMPLATE_KEY = "tokenizer.chat_template"
FIXED_DATE = datetime(2026, 11, 4)
"""The date ``strftime_now`` returns in chat templates (the date the models were pinned)."""
FORMAT_VERSION = 1
"""Bump when prompt formatting changes, so cached replies from the old format are not reused."""
SEED_MODULUS = 2**32 - 1
"""Request seeds are reduced modulo this before they reach llama.cpp, which reads the seed
0xFFFFFFFF (``LLAMA_DEFAULT_SEED``) as "pick a random seed"; the reduced seed never equals it."""

STRICT_CHAR_RULE = (
    r'char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] '
    r"[0-9a-fA-F] [0-9a-fA-F])"
)
"""JSON string characters without raw control characters (as in llama.cpp's own converter)."""

TemplateRenderer = Callable[[list[dict[str, str]]], str]


class ContextOverflowError(ValueError):
    """Raised when a prompt does not fit in the context window."""


class LlamaLike(Protocol):
    """The subset of ``llama_cpp.Llama`` the backend uses."""

    @property
    def metadata(self) -> Mapping[str, str]:
        """GGUF metadata as strings."""
        ...

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        """Tokenize UTF-8 bytes; ``add_bos`` adds BOS only if the vocabulary wants one."""
        ...

    def detokenize(
        self, tokens: list[int], prev_tokens: list[int] | None = None, special: bool = False
    ) -> bytes:
        """Turn tokens back into bytes."""
        ...

    def token_bos(self) -> int:
        """BOS token id (-1 if none)."""
        ...

    def token_eos(self) -> int:
        """EOS token id (-1 if none)."""
        ...

    def reset(self) -> None:
        """Forget the evaluated prefix."""
        ...

    def create_completion(self, prompt: list[int], **kwargs: Any) -> Any:
        """Generate a completion for prompt tokens."""
        ...


@dataclass(frozen=True)
class LlamaRuntime:
    """The pieces of llama-cpp-python the backend uses (tests inject fakes).

    Attributes:
        load: Builds a ``Llama`` from keyword arguments.
        schema_to_gbnf: JSON schema text to GBNF grammar text.
        grammar: GBNF text to a grammar object for ``create_completion``.
        compile_template: ``(template, bos, eos)`` to a renderer of messages.
        perf_reset: Resets llama.cpp's timing counters for a ``Llama``.
        perf: Reads llama.cpp's timing counters for a ``Llama``.
        version: llama-cpp-python version.
    """

    load: Callable[..., LlamaLike]
    schema_to_gbnf: Callable[[str], str]
    grammar: Callable[[str], Any]
    compile_template: Callable[[str, str, str], TemplateRenderer]
    perf_reset: Callable[[Any], None] | None = None
    perf: Callable[[Any], Mapping[str, float]] | None = None
    version: str = "unknown"


@dataclass(frozen=True)
class SamplingParams:
    """Sampler settings besides temperature and seed (llama-cpp-python's defaults, explicit)."""

    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    typical_p: float = 1.0
    repeat_penalty: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)


def _tojson(
    value: Any,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    """The ``tojson`` filter as Hugging Face chat templates expect it."""
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def _raise_template_error(message: str) -> None:
    """``raise_exception`` for chat templates."""
    raise ValueError(f"chat template error: {message}")


def fixed_strftime(fmt: str) -> str:
    """``strftime_now`` for chat templates, pinned to :data:`FIXED_DATE`."""
    return FIXED_DATE.strftime(fmt)


def compile_jinja_template(source: str, bos_token: str, eos_token: str) -> TemplateRenderer:
    """Compile a Hugging Face chat template with Jinja2 in a sandbox.

    The renderer always sets ``add_generation_prompt`` (the reply is the
    assistant's next turn) and pins ``strftime_now`` to :data:`FIXED_DATE`.
    """
    sandbox = importlib.import_module("jinja2.sandbox")
    ext = importlib.import_module("jinja2.ext")
    environment = sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[ext.loopcontrols]
    )
    environment.filters["tojson"] = _tojson
    template = environment.from_string(source)

    def render(messages: list[dict[str, str]]) -> str:
        return str(
            template.render(
                messages=messages,
                bos_token=bos_token,
                eos_token=eos_token,
                add_generation_prompt=True,
                raise_exception=_raise_template_error,
                strftime_now=fixed_strftime,
            )
        )

    return render


def strict_json_gbnf(gbnf: str) -> str:
    """Replace the grammar's JSON string-character rule with :data:`STRICT_CHAR_RULE`."""
    lines = [
        STRICT_CHAR_RULE if line.startswith("char ::=") else line for line in gbnf.splitlines()
    ]
    return "\n".join(lines) + ("\n" if gbnf.endswith("\n") else "")


def default_runtime() -> LlamaRuntime:
    """The real llama-cpp-python runtime.

    Raises:
        ImportError: when the optional ``local`` extra is not installed
            (``make setup-local``).
    """
    try:
        module = importlib.import_module("llama_cpp")
        grammar_module = importlib.import_module("llama_cpp.llama_grammar")
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is not installed; run `make setup-local` for local models"
        ) from exc

    def perf_reset(llama: Any) -> None:
        module.llama_perf_context_reset(llama.ctx)

    def perf(llama: Any) -> dict[str, float]:
        data = module.llama_perf_context(llama.ctx)
        return {
            "prompt_eval_tokens": float(data.n_p_eval),
            "prompt_eval_ms": float(data.t_p_eval_ms),
            "eval_tokens": float(data.n_eval),
            "eval_ms": float(data.t_eval_ms),
        }

    return LlamaRuntime(
        load=module.Llama,
        schema_to_gbnf=grammar_module.json_schema_to_gbnf,
        grammar=grammar_module.LlamaGrammar.from_string,
        compile_template=compile_jinja_template,
        perf_reset=perf_reset,
        perf=perf,
        version=str(getattr(module, "__version__", "unknown")),
    )


def normalize_messages(messages: Sequence[Message], system_mode: str) -> list[Message]:
    """Merge consecutive same-role messages; fold the system text into the first user turn.

    Args:
        messages: Context to normalize.
        system_mode: ``native`` keeps a system message; ``merge_first_user``
            prepends its text to the first user message (for templates with no
            system role).
    """
    if system_mode not in SYSTEM_MODES:
        raise ValueError(f"system_mode {system_mode!r} not in {list(SYSTEM_MODES)}")
    merged: list[Message] = []
    for message in messages:
        if merged and merged[-1].role == message.role:
            merged[-1] = Message(message.role, f"{merged[-1].content}\n\n{message.content}")
        else:
            merged.append(message)
    if system_mode == "merge_first_user" and merged and merged[0].role == "system":
        system = merged.pop(0)
        first = merged[0] if merged else None
        if first is not None and first.role == "user":
            merged[0] = Message("user", f"{system.content}\n\n{first.content}")
        else:
            merged.insert(0, Message("user", system.content))
    return merged


class LlamaCppBackend:
    """A pinned GGUF model served by llama.cpp on the CPU.

    Args:
        model_path: GGUF file.
        model: Registry key (the ledger's ``model``).
        sha256: Pinned SHA-256 of the file (part of :attr:`name`).
        params_b: Parameters in billions (checked against the budget cap).
        n_ctx: Context window in tokens.
        n_threads: Threads for generation and prompt processing; defaults to
            ``os.cpu_count()``.
        n_batch: Prompt-processing batch size.
        system_mode: See :func:`normalize_messages`.
        sampling: Sampler settings besides temperature and seed.
        runtime: llama-cpp-python runtime; defaults to :func:`default_runtime`.
        chat_template: Overrides the GGUF's template (tests only).
    """

    kind = "local"
    offline_safe = False

    def __init__(
        self,
        model_path: Path,
        *,
        model: str,
        sha256: str,
        params_b: float,
        n_ctx: int,
        n_threads: int | None = None,
        n_batch: int = 512,
        system_mode: SystemMode = "native",
        sampling: SamplingParams | None = None,
        runtime: LlamaRuntime | None = None,
        chat_template: str | None = None,
    ) -> None:
        if n_ctx <= 0 or n_batch <= 0:
            raise ValueError("n_ctx and n_batch must be positive")
        if system_mode not in SYSTEM_MODES:
            raise ValueError(f"system_mode {system_mode!r} not in {list(SYSTEM_MODES)}")
        self.runtime = runtime or default_runtime()
        self.model_path = Path(model_path)
        self.model = model
        self.sha256 = sha256
        self._params_b = float(params_b)
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count() or 1
        self.n_batch = n_batch
        self.system_mode: SystemMode = system_mode
        self.sampling = sampling or SamplingParams()
        start = time.monotonic()
        self.llama = self.runtime.load(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=self.n_threads,
            n_threads_batch=self.n_threads,
            verbose=False,
        )
        self.load_s = time.monotonic() - start
        template = chat_template or self.llama.metadata.get(TEMPLATE_KEY, "")
        if not template:
            raise ValueError(f"{self.model_path} has no chat template in its GGUF metadata")
        self.template_sha256 = sha256_text(template)
        self.bos_text = self._token_text(self.llama.token_bos())
        self.eos_text = self._token_text(self.llama.token_eos())
        self._render = self.runtime.compile_template(template, self.bos_text, self.eos_text)
        probe = b"a"
        self.adds_bos = len(self.llama.tokenize(probe, add_bos=True, special=True)) > len(
            self.llama.tokenize(probe, add_bos=False, special=True)
        )
        self._grammars: dict[str, Any] = {}
        self.last_perf: dict[str, float] | None = None
        self.last_raw: dict[str, Any] | None = None
        config_hash = sha256_text(canonical_json(self.config()))
        self._name = f"llama_cpp/{model}@{sha256[:12]}/{config_hash[:8]}"

    # ------------------------------------------------------------------ identity

    @property
    def name(self) -> str:
        """``llama_cpp/<model>@<sha256 prefix>/<config hash prefix>``."""
        return self._name

    @property
    def params_b(self) -> float:
        """Parameters in billions."""
        return self._params_b

    def config(self) -> dict[str, Any]:
        """Every setting that can change a reply (hashed into :attr:`name`)."""
        return {
            "backend": "llama_cpp",
            "model": self.model,
            "sha256": self.sha256,
            "n_ctx": self.n_ctx,
            "sampling": self.sampling.to_dict(),
            "template_sha256": self.template_sha256,
            "system_mode": self.system_mode,
            "fixed_date": FIXED_DATE.date().isoformat(),
            "format_version": FORMAT_VERSION,
        }

    def info(self) -> dict[str, Any]:
        """Provenance: the config plus runtime facts that do not change replies."""
        return {
            **self.config(),
            "name": self.name,
            "model_file": self.model_path.name,
            "params_b": self.params_b,
            "n_threads": self.n_threads,
            "n_batch": self.n_batch,
            "load_s": round(self.load_s, 3),
            "llama_cpp_version": self.runtime.version,
            "adds_bos": self.adds_bos,
            "bos_token": self.bos_text,
            "eos_token": self.eos_text,
        }

    # ------------------------------------------------------------------ formatting

    def _token_text(self, token: int) -> str:
        if token < 0:
            return ""
        return self.llama.detokenize([token], special=True).decode("utf-8", errors="replace")

    def format_prompt(self, messages: Sequence[Message]) -> str:
        """The prompt text for ``messages``, ready for tokenization."""
        normalized = normalize_messages(messages, self.system_mode)
        text = self._render([m.to_dict() for m in normalized])
        if self.adds_bos and self.bos_text and text.startswith(self.bos_text):
            text = text[len(self.bos_text) :]
        return text

    def prompt_tokens(self, messages: Sequence[Message]) -> list[int]:
        """The exact tokens sent to the model for ``messages``."""
        text = self.format_prompt(messages)
        return list(self.llama.tokenize(text.encode("utf-8"), add_bos=True, special=True))

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Exact prompt length of ``messages`` in tokens."""
        return len(self.prompt_tokens(messages))

    def _grammar(self, schema: Mapping[str, Any]) -> Any:
        key = json.dumps(schema, ensure_ascii=False)
        if key not in self._grammars:
            gbnf = strict_json_gbnf(self.runtime.schema_to_gbnf(key))
            self._grammars[key] = self.runtime.grammar(gbnf)
        return self._grammars[key]

    # ------------------------------------------------------------------ calls

    def reset_cache(self) -> None:
        """Forget the evaluated prefix, so the next call evaluates its whole prompt."""
        self.llama.reset()

    def close(self) -> None:
        """Free the model and its context now instead of at garbage collection."""
        close = getattr(self.llama, "close", None)
        if callable(close):
            close()

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Generate the reply to ``request``.

        Raises:
            ContextOverflowError: when the prompt does not fit in ``n_ctx``.
        """
        tokens = self.prompt_tokens(request.messages)
        if len(tokens) >= self.n_ctx:
            raise ContextOverflowError(
                f"prompt has {len(tokens)} tokens; the context window is {self.n_ctx}"
            )
        grammar = self._grammar(request.json_schema) if request.json_schema is not None else None
        if self.runtime.perf_reset is not None:
            self.runtime.perf_reset(self.llama)
        start = time.monotonic()
        result = self.llama.create_completion(
            prompt=tokens,
            max_tokens=min(request.max_tokens, self.n_ctx - len(tokens)),
            temperature=float(request.temperature),
            top_k=self.sampling.top_k,
            top_p=self.sampling.top_p,
            min_p=self.sampling.min_p,
            typical_p=self.sampling.typical_p,
            repeat_penalty=self.sampling.repeat_penalty,
            seed=request.seed % SEED_MODULUS,
            stop=list(request.stop),
            grammar=grammar,
        )
        elapsed = time.monotonic() - start
        self.last_perf = dict(self.runtime.perf(self.llama)) if self.runtime.perf else None
        self.last_raw = dict(result)
        choice = result["choices"][0]
        usage = result.get("usage") or {}
        finish = choice.get("finish_reason") or "stop"
        return ChatResponse(
            text=str(choice["text"]),
            prompt_tokens=len(tokens),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            elapsed_s=elapsed,
            cached=False,
            backend=self.name,
            finish_reason="length" if finish == "length" else "stop",
        )
