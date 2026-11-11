"""Pinned local models: the candidate registry and backend loading (section 8).

Every candidate is pinned by Hugging Face repository, revision commit SHA (from
``https://huggingface.co/api/models/<repo>``), file name, quantization, file
size, and SHA-256 (the LFS object id from ``/api/models/<repo>/tree/<revision>``,
confirmed against the downloaded file, which is authoritative). The weights live
outside the repository, in ``~/.cache/halflife/models`` (or the directory named
by ``HALFLIFE_MODELS_DIR``); ``tools/fetch_models.py`` downloads and verifies
them and writes ``artifacts/models/pins.json``. Licenses and terms are recorded
in docs/DATA.md (rule 9).

The five candidates come from five families, are at most 4B parameters, and
use the same quantizer (bartowski, imatrix ``Q4_K_M``) so the quantization is
held constant across families. They are E1 pilot candidates only; the pilot
(section 5) decides which families E1 keeps.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from halflife.backends.llama_cpp import LlamaCppBackend, LlamaRuntime, SystemMode

MODELS_DIR_ENV = "HALFLIFE_MODELS_DIR"
PINS_PATH = Path("artifacts/models/pins.json")
HF_BASE = "https://huggingface.co"
PROVENANCE_SUFFIX = ".provenance.json"


class ModelFileError(RuntimeError):
    """Raised when a pinned model file is missing or does not match its pin."""


@dataclass(frozen=True)
class ModelSpec:
    """One pinned GGUF file.

    Attributes:
        key: Registry key used everywhere else (run configs, ledger ``model``).
        family: Model family; E1 compares families (section 5).
        upstream: The original model repository the GGUF was converted from.
        repo: Hugging Face repository holding the GGUF file.
        revision: Commit SHA of ``repo`` the file is pinned to.
        filename: GGUF file name in ``repo``.
        quantization: llama.cpp quantization type.
        size_bytes: File size in bytes.
        sha256: SHA-256 of the file.
        license: License identifier as declared on the model card.
        license_url: Where the license text is published.
        params_b: Parameters in billions (``gguf.total`` from the Hugging Face API).
        architecture: GGUF architecture name.
        context_length: Trained context length from the GGUF metadata.
        chat_notes: How the chat template formats a conversation, and anything
            the backend does to keep prompts deterministic.
        system_mode: ``native`` when the template accepts a system message, or
            ``merge_first_user`` when the system text must be folded into the
            first user turn.
    """

    key: str
    family: str
    upstream: str
    repo: str
    revision: str
    filename: str
    quantization: str
    size_bytes: int
    sha256: str
    license: str
    license_url: str
    params_b: float
    architecture: str
    context_length: int
    chat_notes: str
    system_mode: SystemMode = "native"

    @property
    def url(self) -> str:
        """Download URL pinned to the revision (never a moving branch)."""
        return f"{HF_BASE}/{self.repo}/resolve/{self.revision}/{self.filename}"

    @property
    def api_model_url(self) -> str:
        """Hugging Face API URL with the repository's metadata."""
        return f"{HF_BASE}/api/models/{self.repo}"

    @property
    def api_tree_url(self) -> str:
        """Hugging Face API URL listing the files at the pinned revision."""
        return f"{HF_BASE}/api/models/{self.repo}/tree/{self.revision}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, including the derived URLs."""
        data = asdict(self)
        data["url"] = self.url
        return data


CANDIDATES: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="qwen2.5-3b-instruct-q4km",
        family="qwen",
        upstream="Qwen/Qwen2.5-3B-Instruct",
        repo="bartowski/Qwen2.5-3B-Instruct-GGUF",
        revision="f302c64a2269a69fb27b2f9473b362f5bb8e78d8",
        filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        quantization="Q4_K_M",
        size_bytes=1929903264,
        sha256="9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94",
        license="qwen-research",
        license_url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE",
        params_b=3.09,
        architecture="qwen2",
        context_length=32768,
        chat_notes=(
            "ChatML turns (<|im_start|>role ... <|im_end|>); native system role; the "
            "template adds a default Qwen system prompt only when no system message is "
            "given; the tokenizer adds no BOS token."
        ),
    ),
    ModelSpec(
        key="llama-3.2-3b-instruct-q4km",
        family="llama",
        upstream="meta-llama/Llama-3.2-3B-Instruct",
        repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        revision="5ab33fa94d1d04e903623ae72c95d1696f09f9e8",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        quantization="Q4_K_M",
        size_bytes=2019377696,
        sha256="6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff",
        license="llama3.2",
        license_url="https://www.llama.com/llama3_2/license/",
        params_b=3.21,
        architecture="llama",
        context_length=131072,
        chat_notes=(
            "Llama 3 header turns (<|start_header_id|>role<|end_header_id|> ... "
            "<|eot_id|>); the template writes 'Cutting Knowledge Date' and 'Today Date' "
            "lines into the system turn, so the backend fixes strftime_now to a constant "
            "date; the tokenizer adds the BOS token, so the BOS the template renders is "
            "dropped and appears once; message text is trimmed."
        ),
    ),
    ModelSpec(
        key="phi-3.5-mini-instruct-q4km",
        family="phi",
        upstream="microsoft/Phi-3.5-mini-instruct",
        repo="bartowski/Phi-3.5-mini-instruct-GGUF",
        revision="6d70da17e749a471ccb62ade694486011a75cda3",
        filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        quantization="Q4_K_M",
        size_bytes=2393232672,
        sha256="e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5",
        license="mit",
        license_url="https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/LICENSE",
        params_b=3.82,
        architecture="phi3",
        context_length=131072,
        chat_notes=(
            "<|system|>, <|user|>, <|assistant|> turns ending in <|end|>; native system "
            "role (skipped by the template when empty); no BOS token (the GGUF sets "
            "add_bos_token false and the template renders none)."
        ),
    ),
    ModelSpec(
        key="smollm2-1.7b-instruct-q4km",
        family="smollm",
        upstream="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        repo="bartowski/SmolLM2-1.7B-Instruct-GGUF",
        revision="1f03464768bfcc0319fc50da8ff5fb20b6417ba2",
        filename="SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        quantization="Q4_K_M",
        size_bytes=1055609824,
        sha256="77665ea4815999596525c636fbeb56ba8b080b46ae85efef4f0d986a139834d7",
        license="apache-2.0",
        license_url="https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct",
        params_b=1.71,
        architecture="llama",
        context_length=8192,
        chat_notes=(
            "ChatML turns; native system role; the template adds a default SmolLM system "
            "prompt only when no system message is given; <|im_start|> doubles as the BOS "
            "token and comes from the template; trained context is 8192 tokens."
        ),
    ),
    ModelSpec(
        key="granite-3.3-2b-instruct-q4km",
        family="granite",
        upstream="ibm-granite/granite-3.3-2b-instruct",
        repo="bartowski/ibm-granite_granite-3.3-2b-instruct-GGUF",
        revision="4e06dce39d4c5fcf65ca1cc69b92962e411d38e1",
        filename="ibm-granite_granite-3.3-2b-instruct-Q4_K_M.gguf",
        quantization="Q4_K_M",
        size_bytes=1545303616,
        sha256="555b91485955bc96eb445b57dd4bbf8809aa7d8cce7c313f4f8bc5b2340896b4",
        license="apache-2.0",
        license_url="https://huggingface.co/ibm-granite/granite-3.3-2b-instruct",
        params_b=2.53,
        architecture="granite",
        context_length=131072,
        chat_notes=(
            "<|start_of_role|>role<|end_of_role|> ... <|end_of_text|> turns; native "
            "system role; the template calls strftime_now only when no system message is "
            "given (the backend fixes it to a constant date)."
        ),
    ),
)

REGISTRY: dict[str, ModelSpec] = {spec.key: spec for spec in CANDIDATES}


def get_spec(key: str) -> ModelSpec:
    """Look up a candidate by key.

    Raises:
        KeyError: with the list of known keys when ``key`` is not pinned.
    """
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; pinned models: {sorted(REGISTRY)}") from None


def models_dir(env: Mapping[str, str] | None = None) -> Path:
    """Directory with the GGUF files: ``$HALFLIFE_MODELS_DIR`` or ``~/.cache/halflife/models``."""
    env = os.environ if env is None else env
    override = env.get(MODELS_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "halflife" / "models"


def model_path(spec: ModelSpec, directory: Path | None = None) -> Path:
    """Where the file for ``spec`` lives (or will live once downloaded)."""
    return (directory or models_dir()) / spec.filename


def provenance_path(path: Path) -> Path:
    """Sidecar file next to a downloaded model that records where it came from and when."""
    return path.with_name(path.name + PROVENANCE_SUFFIX)


def read_provenance(path: Path) -> dict[str, Any] | None:
    """The download sidecar for ``path``, or None when it is missing or unreadable."""
    try:
        data = json.loads(provenance_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, spec: ModelSpec, *, check_hash: bool = True) -> None:
    """Check that ``path`` is the pinned file: right size and, by default, right SHA-256.

    Raises:
        ModelFileError: when the file is missing or differs from the pin.
    """
    if not path.is_file():
        raise ModelFileError(
            f"model file {path} is missing; run "
            f"`uv run --frozen python tools/fetch_models.py --models {spec.key}`"
        )
    size = path.stat().st_size
    if size != spec.size_bytes:
        raise ModelFileError(f"{path} has {size} bytes, pinned size is {spec.size_bytes}")
    if check_hash:
        actual = sha256_file(path)
        if actual != spec.sha256:
            raise ModelFileError(f"{path} has SHA-256 {actual}, pinned {spec.sha256}")


def load_backend(
    key: str,
    *,
    n_ctx: int,
    n_threads: int | None = None,
    directory: Path | None = None,
    verify: bool = True,
    runtime: LlamaRuntime | None = None,
) -> LlamaCppBackend:
    """Load a pinned model into a llama.cpp backend (one model per process).

    Args:
        key: Registry key.
        n_ctx: Context window in tokens.
        n_threads: CPU threads; defaults to ``os.cpu_count()``.
        directory: Models directory; defaults to :func:`models_dir`.
        verify: Check the file's SHA-256 against the pin before loading.
        runtime: llama.cpp runtime override (tests inject a fake).

    Raises:
        KeyError: unknown key.
        ModelFileError: the file is missing or does not match its pin.
    """
    spec = get_spec(key)
    path = model_path(spec, directory)
    verify_file(path, spec, check_hash=verify)
    return LlamaCppBackend(
        path,
        model=spec.key,
        sha256=spec.sha256,
        params_b=spec.params_b,
        n_ctx=n_ctx,
        n_threads=n_threads,
        system_mode=spec.system_mode,
        runtime=runtime,
    )
