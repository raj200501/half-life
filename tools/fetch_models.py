#!/usr/bin/env python3
"""Download the pinned GGUF models and bank ``artifacts/models/pins.json``.

Files go to ``~/.cache/halflife/models`` (or ``$HALFLIFE_MODELS_DIR``), outside
the repository, and are never committed. Every download is pinned to a revision
commit, resumes a partial ``.part`` file, is verified against the pinned size
and SHA-256 before it is renamed into place, and is refused when free disk
would fall below ``disk_floor_gb`` from budget.toml.

Usage::

    uv run --frozen python tools/fetch_models.py                    # every candidate
    uv run --frozen python tools/fetch_models.py --models KEY [KEY ...]
    uv run --frozen python tools/fetch_models.py --check-api        # compare pins with the API
    uv run --frozen python tools/fetch_models.py --write-pins       # verify files, write pins.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from halflife.backends.base import write_json_atomic
from halflife.budget import DEFAULT_BUDGET_PATH, load_caps
from halflife.models import (
    CANDIDATES,
    PINS_PATH,
    ModelFileError,
    ModelSpec,
    get_spec,
    model_path,
    models_dir,
    provenance_path,
    read_provenance,
    sha256_file,
)

GIB = 1024**3
CHUNK = 1 << 20
TIMEOUT_S = 60.0
USER_AGENT = "halflife-fetch-models/1"
PINS_FORMAT = "halflife-pins/1"
RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_WAIT_S = 120.0


class DiskFloorError(RuntimeError):
    """Raised when a download would leave less free disk than the floor."""


class Response(Protocol):
    """The part of an HTTP response the downloader reads."""

    status: int

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes."""
        ...

    def __enter__(self) -> Response:
        """Context manager entry."""
        ...

    def __exit__(self, *exc: object) -> None:
        """Context manager exit."""
        ...


Opener = Callable[[urllib.request.Request], Response]


def urlopen(request: urllib.request.Request) -> Response:
    """Open a URL through the environment's proxy and CA settings."""
    response: Response = urllib.request.urlopen(request, timeout=TIMEOUT_S)
    return response


def now_iso() -> str:
    """Current UTC time, ISO-8601 to the second."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def retry_after_s(error: urllib.error.HTTPError, default: float) -> float:
    """Seconds to wait before retrying: the server's ``Retry-After`` seconds, else ``default``.

    The wait is capped at :data:`MAX_WAIT_S`; an HTTP-date ``Retry-After`` falls back to
    ``default``.
    """
    headers = error.headers
    raw = headers.get("Retry-After") if headers is not None else None
    try:
        wait = float(raw) if raw is not None else default
    except ValueError:
        wait = default
    return min(max(wait, 0.0), MAX_WAIT_S)


def get_json(
    url: str,
    opener: Opener = urlopen,
    *,
    attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET a JSON document, waiting and retrying on rate limits and server errors.

    Raises:
        urllib.error.HTTPError: for other HTTP errors, or when every attempt failed.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with opener(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUSES or attempt == attempts:
                raise
            sleep(retry_after_s(exc, default=min(MAX_WAIT_S, 5.0 * 2.0 ** (attempt - 1))))
    raise ValueError("attempts must be at least 1")


def remote_info(spec: ModelSpec, opener: Opener = urlopen) -> dict[str, Any]:
    """What the Hugging Face API reports for a pinned file.

    Returns the repository head commit, the file's size and LFS SHA-256 at the
    pinned revision, the declared license, and the parameter count.

    Raises:
        ModelFileError: when the file is not listed at the pinned revision.
    """
    model = get_json(spec.api_model_url, opener)
    tree = get_json(spec.api_tree_url, opener)
    entry = next((e for e in tree if e.get("path") == spec.filename), None)
    if entry is None:
        raise ModelFileError(f"{spec.filename} is not in {spec.repo} at {spec.revision}")
    lfs = entry.get("lfs") or {}
    gguf = model.get("gguf") or {}
    card = model.get("cardData") or {}
    return {
        "api_model_url": spec.api_model_url,
        "api_tree_url": spec.api_tree_url,
        "repo_head": model.get("sha"),
        "gated": model.get("gated"),
        "size_bytes": entry.get("size"),
        "lfs_sha256": lfs.get("oid"),
        "license": card.get("license"),
        "license_name": card.get("license_name"),
        "license_link": card.get("license_link"),
        "params": gguf.get("total"),
        "architecture": gguf.get("architecture"),
        "context_length": gguf.get("context_length"),
    }


def api_problems(spec: ModelSpec, info: dict[str, Any]) -> list[str]:
    """Differences between a pin and the API's report (an empty list means they agree)."""
    problems = []
    if info["size_bytes"] != spec.size_bytes:
        problems.append(f"size {info['size_bytes']} != pinned {spec.size_bytes}")
    if info["lfs_sha256"] != spec.sha256:
        problems.append(f"sha256 {info['lfs_sha256']} != pinned {spec.sha256}")
    params = info.get("params")
    if isinstance(params, int | float) and round(params / 1e9, 2) != spec.params_b:
        problems.append(f"params {params} != pinned {spec.params_b}B")
    return problems


def free_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding ``path`` (or its nearest existing parent)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def ensure_disk(path: Path, needed: int, floor_gb: float) -> None:
    """Refuse when writing ``needed`` more bytes would leave less than ``floor_gb`` GiB free.

    Raises:
        DiskFloorError: when the floor would be crossed.
    """
    free = free_bytes(path)
    left = free - needed
    if left < floor_gb * GIB:
        raise DiskFloorError(
            f"need {needed / GIB:.2f} GiB with {free / GIB:.2f} GiB free; that would leave "
            f"{left / GIB:.2f} GiB, below the {floor_gb:g} GiB floor in budget.toml"
        )


def _write_provenance(final: Path, spec: ModelSpec) -> None:
    write_json_atomic(
        provenance_path(final),
        {
            "url": spec.url,
            "repo": spec.repo,
            "revision": spec.revision,
            "filename": spec.filename,
            "size_bytes": spec.size_bytes,
            "sha256": spec.sha256,
            "retrieved_at": now_iso(),
            "tool": "tools/fetch_models.py",
        },
    )


def download(
    spec: ModelSpec,
    dest_dir: Path,
    *,
    floor_gb: float,
    opener: Opener = urlopen,
    attempts: int = 5,
    log: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download one pinned file, resuming a partial download, and verify it.

    Returns the final path. An existing final file is verified, not refetched.

    Raises:
        DiskFloorError: when the download would cross the disk floor.
        ModelFileError: when the finished file does not match the pin, or an
            existing final file is corrupt.
        urllib.error.URLError: when every attempt failed (the partial file is kept).
    """
    say = log or (lambda message: print(message, file=sys.stderr))
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = model_path(spec, dest_dir)
    if final.exists():
        if final.stat().st_size == spec.size_bytes and sha256_file(final) == spec.sha256:
            if read_provenance(final) is None:
                _write_provenance(final, spec)
            say(f"{spec.key}: present and verified")
            return final
        raise ModelFileError(f"{final} exists but does not match its pin; delete it and retry")
    part = final.with_name(final.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset > spec.size_bytes:
            part.unlink()
            offset = 0
        ensure_disk(dest_dir, spec.size_bytes - offset, floor_gb)
        if offset == spec.size_bytes:
            break
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with opener(urllib.request.Request(spec.url, headers=headers)) as response:
                mode = "ab"
                if offset and response.status != 206:
                    offset, mode = 0, "wb"
                say(f"{spec.key}: downloading from byte {offset} (attempt {attempt})")
                with part.open(mode) as handle:
                    while block := response.read(CHUNK):
                        handle.write(block)
            if part.stat().st_size == spec.size_bytes:
                break
            last_error = urllib.error.URLError(
                f"short read: {part.stat().st_size} of {spec.size_bytes} bytes"
            )
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
        say(f"{spec.key}: attempt {attempt} of {attempts} stopped early ({last_error})")
        if attempt < attempts:
            sleep(min(30.0, 2.0**attempt))
    else:
        assert last_error is not None
        raise last_error
    size = part.stat().st_size
    digest = sha256_file(part)
    if size != spec.size_bytes or digest != spec.sha256:
        part.unlink()
        raise ModelFileError(
            f"{spec.key}: downloaded {size} bytes with SHA-256 {digest}; pinned "
            f"{spec.size_bytes} bytes with {spec.sha256}"
        )
    part.replace(final)
    _write_provenance(final, spec)
    say(f"{spec.key}: downloaded and verified ({size} bytes)")
    return final


def git_sha(repo: Path = Path(".")) -> str:
    """Current commit, or ``unknown`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def build_pins(
    specs: Sequence[ModelSpec],
    dest_dir: Path,
    *,
    opener: Opener = urlopen,
    command: Sequence[str],
    commit: str,
) -> dict[str, Any]:
    """Verify every downloaded file and the API report, and return the pins artifact.

    Raises:
        ModelFileError: when a file is missing or does not match its pin, or
            the API disagrees with a pin.
    """
    entries = []
    for spec in specs:
        path = model_path(spec, dest_dir)
        if not path.is_file():
            raise ModelFileError(f"{path} is missing; download it first")
        digest = sha256_file(path)
        if digest != spec.sha256 or path.stat().st_size != spec.size_bytes:
            raise ModelFileError(f"{path} does not match its pin")
        info = remote_info(spec, opener)
        problems = api_problems(spec, info)
        if problems:
            raise ModelFileError(f"{spec.key}: API disagrees with the pin: {'; '.join(problems)}")
        provenance = read_provenance(path) or {}
        entries.append(
            {
                **spec.to_dict(),
                "retrieved_at": provenance.get("retrieved_at"),
                "file_sha256_verified": digest,
                "file_size_verified": path.stat().st_size,
                "api": {**info, "checked_at": now_iso()},
            }
        )
    return {
        "format": PINS_FORMAT,
        "generated_at": now_iso(),
        "git_sha": commit,
        "command": list(command),
        "models_dir": "~/.cache/halflife/models (outside the repository; never committed)",
        "note": (
            "Pinned E1 pilot candidates. The SHA-256 of each downloaded file is authoritative; "
            "the API's LFS object id must agree with it. Licenses and terms: docs/DATA.md."
        ),
        "models": entries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--models", nargs="+", metavar="KEY", help="registry keys (default: all)")
    parser.add_argument("--dest", type=Path, default=None, help="models directory")
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET_PATH)
    parser.add_argument("--check-api", action="store_true", help="compare pins with the API")
    parser.add_argument("--write-pins", action="store_true", help="write the pins artifact")
    parser.add_argument("--pins", type=Path, default=PINS_PATH, help="pins artifact path")
    args = parser.parse_args(argv)
    try:
        specs = [get_spec(k) for k in args.models] if args.models else list(CANDIDATES)
    except KeyError as exc:
        print(f"FETCH: FAIL {exc}", file=sys.stderr)
        return 2
    dest = args.dest or models_dir()
    if args.check_api:
        failed = 0
        for spec in specs:
            try:
                info = remote_info(spec)
            except (ModelFileError, urllib.error.URLError, OSError, ValueError) as exc:
                print(f"FETCH: FAIL {spec.key}: {exc}", file=sys.stderr)
                return 1
            problems = api_problems(spec, info)
            head = (
                "pinned revision is the head"
                if info["repo_head"] == spec.revision
                else (f"repository head is now {info['repo_head']}")
            )
            print(f"{spec.key}: {'; '.join(problems) or 'matches the API'} ({head})")
            failed += bool(problems)
        return 1 if failed else 0
    if args.write_pins:
        command = ["python", "tools/fetch_models.py", *(argv if argv is not None else sys.argv[1:])]
        try:
            pins = build_pins(specs, dest, command=command, commit=git_sha())
        except (ModelFileError, urllib.error.URLError, OSError, ValueError) as exc:
            print(f"FETCH: FAIL {exc}", file=sys.stderr)
            return 1
        write_json_atomic(args.pins, pins)
        print(f"wrote {args.pins} with {len(pins['models'])} models")
        return 0
    floor = float(load_caps(args.budget)["disk_floor_gb"])
    for spec in specs:
        try:
            download(spec, dest, floor_gb=floor)
        except (DiskFloorError, ModelFileError, urllib.error.URLError, OSError) as exc:
            print(f"FETCH: FAIL {spec.key}: {exc}", file=sys.stderr)
            return 1
    print(f"FETCH: OK {len(specs)} models in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
