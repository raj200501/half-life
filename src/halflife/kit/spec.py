"""The portable constraint spec (YAML, version 1) shared by every kit tool.

A spec file, conventionally ``halflife.yaml`` at the project root, lists the
rules an agent must keep. Each constraint has the fields of
:class:`halflife.schema.Constraint`::

    version: 1
    constraints:
      - id: tests-frozen
        type: prohibition
        text: Never modify, create, or delete files under tests/.
        reason: The tests encode the behavior the maintainers signed off on.
        detector:
          kind: protected_paths
          params: {globs: ["tests/**"]}
        recall_keywords: [["test"], ["never", "not "]]   # optional

Rules:

* ``version`` must be the integer 1 and ``constraints`` a non-empty list.
  Unknown keys are errors at every level, and so are duplicate keys (a repeated
  key would otherwise silently replace the first).
* ``id``, ``type``, ``text``, ``reason``, and ``detector`` are required;
  ``detector.params`` may be omitted for kinds without parameters
  (``no_network``). Parameters are checked with
  :func:`halflife.detect.check_spec`, so globs and regexes mean exactly what
  they mean to the detectors.
* ``recall_keywords`` is optional. The default
  (:func:`default_recall_keywords`) takes the two longest content words of the
  text, cut to their first five characters, and adds a negation group for
  prohibitions. It only matters for recall probes; set it explicitly when you
  score probes against a spec.

Every problem raises :class:`SpecError` with the file and the constraint it
concerns, for example ``halflife.yaml: constraints[1] ('offline'): detector kind
'network' not in [...]``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from halflife.detect import DetectorError, check_spec
from halflife.schema import Constraint, DetectorKind, DetectorSpec, SchemaError

__all__ = [
    "DEFAULT_SPEC_NAME",
    "NEGATION_GROUP",
    "SPEC_ENV",
    "SPEC_VERSION",
    "SpecError",
    "default_recall_keywords",
    "dump_spec",
    "find_spec",
    "load_spec",
    "parse_spec",
    "spec_to_dict",
]

SPEC_VERSION = 1
"""The only spec format version this package reads and writes."""
DEFAULT_SPEC_NAME = "halflife.yaml"
"""File name the kit looks for when no spec path is given."""
SPEC_ENV = "HALFLIFE_SPEC"
"""Environment variable naming the spec file."""
PROJECT_DIR_ENV = "CLAUDE_PROJECT_DIR"
"""Set by Claude Code for hook commands: the project root where the session started."""

NEGATION_GROUP: tuple[str, ...] = (
    "never",
    "not ",
    "n't",
    "n\u2019t",
    "no ",
    "avoid",
    "without",
    "forbid",
    "prohibit",
)
"""Negation words added to default recall keywords of prohibitions (DECISIONS.md D-030)."""

KEYWORD_PREFIX = 5
"""Default recall keywords keep this many leading characters, so ``modify`` matches ``modified``."""
MIN_KEYWORD_LENGTH = 4
"""Shorter words are never chosen as default recall keywords."""

STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "always",
        "anything",
        "before",
        "being",
        "below",
        "between",
        "both",
        "could",
        "does",
        "doing",
        "done",
        "during",
        "each",
        "every",
        "everything",
        "exactly",
        "from",
        "have",
        "into",
        "just",
        "keep",
        "make",
        "more",
        "most",
        "must",
        "need",
        "never",
        "nothing",
        "only",
        "other",
        "over",
        "same",
        "should",
        "single",
        "some",
        "something",
        "stay",
        "such",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "until",
        "using",
        "very",
        "what",
        "when",
        "where",
        "which",
        "while",
        "will",
        "with",
        "within",
        "without",
        "would",
        "your",
    }
)
"""Words the default recall keywords skip: function words and rule boilerplate."""

_TOP_KEYS = frozenset({"version", "constraints"})
_CONSTRAINT_KEYS = frozenset({"id", "type", "text", "reason", "detector", "recall_keywords"})
_REQUIRED = ("id", "type", "text", "reason", "detector")
_DETECTOR_KEYS = frozenset({"kind", "params"})
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]*")
_NOT_PLURAL = ("ss", "us", "is")
_MERGE_TAG = "tag:yaml.org,2002:merge"


class SpecError(ValueError):
    """Raised for a spec file that cannot be read or does not fit format version 1."""


# --------------------------------------------------------------------------- recall keywords


def _keyword_form(raw: str) -> tuple[str, bool]:
    """The matching form of one word of a rule, and whether it names a path or file.

    A path keeps its last segment without the extension (``matlite/solve.py``
    gives ``solve``) and drops a plural ``s`` (``snapshots/`` gives
    ``snapshot``); other words keep their first :data:`KEYWORD_PREFIX` characters.
    """
    word = raw.lower().strip("./-_")
    is_path = "/" in raw or "." in raw.strip(".")
    if not is_path:
        return word[:KEYWORD_PREFIX] if len(word) >= MIN_KEYWORD_LENGTH else word, False
    word = word.rsplit("/", 1)[-1]
    stem = word.split(".", 1)[0]
    if len(stem) >= MIN_KEYWORD_LENGTH:
        word = stem
    if len(word) > MIN_KEYWORD_LENGTH and word.endswith("s") and not word.endswith(_NOT_PLURAL):
        word = word[:-1]
    return word, True


def default_recall_keywords(text: str, constraint_type: str) -> tuple[tuple[str, ...], ...]:
    """Recall keyword groups for a constraint whose spec gives none.

    Content words have at least four characters and are not in
    :data:`STOP_WORDS`. Two of them become one group each: words written as a
    path or file name (with ``/`` or ``.``) first, in text order, then the
    longest other words (ties go to the earlier word); the groups keep text
    order. A path counts as its last segment without the extension and without
    a plural ``s`` (``matlite/solve.py`` gives ``solve``, ``snapshots/`` gives
    ``snapshot``); any other word is cut to its first five characters, so
    ``modify`` also matches ``modified``. Prohibitions add
    :data:`NEGATION_GROUP`, so "run the tests" never counts as recalling "never
    modify the tests" (DECISIONS.md D-023). A text without content words falls
    back to one group holding the whole lower-cased text (first 40 characters).

    Args:
        text: The rule text.
        constraint_type: The constraint type (only ``prohibition`` changes the result).

    Returns:
        Keyword groups in the form :class:`halflife.schema.Constraint` expects.
    """
    found: list[tuple[str, bool, int]] = []
    for raw in _WORD.findall(text):
        full = raw.lower().strip("./-_")
        word, is_path = _keyword_form(raw)
        if (
            len(full) >= MIN_KEYWORD_LENGTH
            and full not in STOP_WORDS
            and word not in [w for w, _, _ in found]
        ):
            found.append((word, is_path, len(full)))
    ranked = sorted(range(len(found)), key=lambda i: (not found[i][1], -found[i][2], i))[:2]
    groups: list[tuple[str, ...]] = [(found[i][0],) for i in sorted(ranked)]
    if not groups:
        groups = [(" ".join(text.lower().split())[:40] or "rule",)]
    if constraint_type == "prohibition":
        groups.append(NEGATION_GROUP)
    return tuple(groups)


# --------------------------------------------------------------------------- YAML


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.Node) -> dict[Any, Any]:
    if not isinstance(node, yaml.MappingNode):  # pragma: no cover - the tag guarantees it
        raise SpecError(f"expected a mapping at line {node.start_mark.line + 1}")
    seen: set[Any] = set()
    for key_node, _value in node.value:
        if key_node.tag == _MERGE_TAG:
            continue
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, Hashable):
            raise SpecError(f"unhashable mapping key at line {key_node.start_mark.line + 1}")
        if key in seen:
            raise SpecError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    # Keys merged in with ``<<`` come first, so the mapping's own keys override them.
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        mapping[loader.construct_object(key_node, deep=True)] = loader.construct_object(
            value_node, deep=True
        )
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _load_yaml(text: str, source: str) -> object:
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)  # a SafeLoader subclass
    except SpecError as exc:
        raise SpecError(f"{source}: {exc}") from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise SpecError(f"{source}: invalid YAML{where}: {problem}") from exc


# --------------------------------------------------------------------------- parsing


def _keyword_groups(value: object, where: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list) or not value:
        raise SpecError(f"{where}: recall_keywords must be a non-empty list of lists of strings")
    groups: list[tuple[str, ...]] = []
    for number, group in enumerate(value):
        if (
            not isinstance(group, list)
            or not group
            or not all(isinstance(w, str) and w for w in group)
        ):
            raise SpecError(
                f"{where}: recall_keywords[{number}] must be a non-empty list of non-empty strings"
            )
        groups.append(tuple(group))
    return tuple(groups)


def _detector(value: object, where: str) -> DetectorSpec:
    if not isinstance(value, Mapping):
        raise SpecError(f"{where}: detector must be a mapping with kind and params")
    unknown = sorted(str(k) for k in set(value) - _DETECTOR_KEYS)
    if unknown:
        raise SpecError(f"{where}: unknown detector field(s) {unknown}")
    kind = value.get("kind")
    if not isinstance(kind, str):
        raise SpecError(f"{where}: detector.kind must be a string")
    params = value.get("params")
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise SpecError(f"{where}: detector.params must be a mapping")
    try:
        spec = DetectorSpec(kind=cast(DetectorKind, kind), params=dict(params))
        check_spec(spec)
    except (SchemaError, DetectorError) as exc:
        raise SpecError(f"{where}: {exc}") from exc
    return spec


def _constraint(item: object, where: str) -> Constraint:
    if not isinstance(item, Mapping):
        raise SpecError(f"{where}: each constraint must be a mapping")
    cid = item.get("id")
    label = f"{where} ({cid!r})" if isinstance(cid, str) else where
    unknown = sorted(str(k) for k in set(item) - _CONSTRAINT_KEYS)
    if unknown:
        raise SpecError(f"{label}: unknown field(s) {unknown}")
    missing = [key for key in _REQUIRED if key not in item]
    if missing:
        raise SpecError(f"{label}: missing field(s) {missing}")
    for key in ("id", "type", "text", "reason"):
        if not isinstance(item[key], str):
            raise SpecError(f"{label}: {key} must be a string")
    detector = _detector(item["detector"], label)
    keywords = item.get("recall_keywords")
    groups = (
        default_recall_keywords(item["text"], item["type"])
        if keywords is None
        else _keyword_groups(keywords, label)
    )
    try:
        return Constraint(
            id=item["id"],
            type=item["type"],
            text=item["text"],
            reason=item["reason"],
            detector=detector,
            recall_keywords=groups,
        )
    except SchemaError as exc:
        raise SpecError(f"{label}: {exc}") from exc


def parse_spec(data: object, *, source: str = "spec") -> tuple[Constraint, ...]:
    """Validate parsed spec data and build its constraints.

    Args:
        data: The parsed YAML (or JSON) document.
        source: Name used in error messages, usually the file path.

    Returns:
        The constraints in file order.

    Raises:
        SpecError: for anything that does not fit format version 1.
    """
    if not isinstance(data, Mapping):
        raise SpecError(f"{source}: the top level must be a mapping with version and constraints")
    unknown = sorted(str(k) for k in set(data) - _TOP_KEYS)
    if unknown:
        raise SpecError(f"{source}: unknown top-level field(s) {unknown}")
    if "version" not in data:
        raise SpecError(f"{source}: missing version (write version: {SPEC_VERSION})")
    version = data["version"]
    if isinstance(version, bool) or version != SPEC_VERSION:
        raise SpecError(
            f"{source}: unsupported version {version!r}; this package reads version {SPEC_VERSION}"
        )
    raw = data.get("constraints")
    if not isinstance(raw, list) or not raw:
        raise SpecError(f"{source}: constraints must be a non-empty list")
    constraints: list[Constraint] = []
    seen: set[str] = set()
    for number, item in enumerate(raw):
        constraint = _constraint(item, f"{source}: constraints[{number}]")
        if constraint.id in seen:
            raise SpecError(f"{source}: constraints[{number}]: duplicate id {constraint.id!r}")
        seen.add(constraint.id)
        constraints.append(constraint)
    return tuple(constraints)


def load_spec(path: str | os.PathLike[str]) -> tuple[Constraint, ...]:
    """Read and validate a spec file.

    Args:
        path: The YAML spec file.

    Returns:
        The constraints in file order, as :class:`halflife.schema.Constraint` values.

    Raises:
        SpecError: if the file is missing, unreadable, not YAML, or invalid.
    """
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SpecError(f"{file}: spec file not found") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise SpecError(f"{file}: cannot read the spec: {exc}") from exc
    return parse_spec(_load_yaml(text, str(file)), source=str(file))


# --------------------------------------------------------------------------- writing


def spec_to_dict(
    constraints: Sequence[Constraint], *, include_recall_keywords: bool = True
) -> dict[str, Any]:
    """The version 1 document for ``constraints`` (inverse of :func:`parse_spec`)."""
    items: list[dict[str, Any]] = []
    for constraint in constraints:
        item = constraint.to_dict()
        if not include_recall_keywords:
            del item["recall_keywords"]
        items.append(item)
    return {"version": SPEC_VERSION, "constraints": items}


def dump_spec(constraints: Sequence[Constraint], *, include_recall_keywords: bool = True) -> str:
    """YAML text of a version 1 spec, for example to write a template's rules as halflife.yaml."""
    document = spec_to_dict(constraints, include_recall_keywords=include_recall_keywords)
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100)


# --------------------------------------------------------------------------- lookup


def find_spec(
    explicit: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Where the kit's command-line tools read the spec from.

    Order: ``explicit`` (the ``--spec`` option), then ``$HALFLIFE_SPEC``, then
    ``halflife.yaml`` in ``cwd``, then ``halflife.yaml`` in
    ``$CLAUDE_PROJECT_DIR``. An explicit path or the variable is returned even
    when the file is missing, so the caller reports that error; the two
    implicit locations count only when the file exists.

    Args:
        explicit: Path given on the command line, if any.
        env: Environment to read (default: ``os.environ``).
        cwd: Directory for the implicit lookup (default: the process directory).

    Returns:
        The spec path, or None when no spec was named and none exists.
    """
    environ = os.environ if env is None else env
    if explicit is not None:
        return Path(explicit)
    named = environ.get(SPEC_ENV)
    if named:
        return Path(named)
    here = Path(cwd) if cwd is not None else Path.cwd()
    candidates = [here / DEFAULT_SPEC_NAME]
    project = environ.get(PROJECT_DIR_ENV)
    if project:
        candidates.append(Path(project) / DEFAULT_SPEC_NAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
