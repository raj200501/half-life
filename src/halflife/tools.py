"""The minimal agent's tools and the executor that turns a call into a ToolEvent.

The tools are exactly ``schema.AGENT_TOOLS``: ``list_files``, ``read_file``,
``write_file``, ``replace_in_file``, ``run`` (through :mod:`halflife.sandbox`),
and ``done``. :func:`execute` (for a :class:`~halflife.schema.Action`) and
:func:`execute_call` (for a raw tool name and arguments, as parsed from model
output) never let an argument error escape: a bad call becomes an observation
starting with ``error:``. Every event carries ``fs_after``, the workspace
snapshot taken right after the call.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from halflife.sandbox import (
    DEFAULT_OUTPUT_LIMIT,
    SandboxError,
    SandboxOptions,
    run_with,
    truncate_output,
)
from halflife.schema import AGENT_TOOLS, Action, NetAttempt, ToolEvent, ToolName
from halflife.workspace import Workspace, WorkspacePathError


@dataclass(frozen=True)
class ToolSpec:
    """Name, one-line description, and JSON schema of one tool's arguments."""

    name: str
    doc: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        """Call form for prompts, for example ``list_files(path=".")``."""
        props: Mapping[str, Any] = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", ()))
        parts = []
        for name, prop in props.items():
            if name in required:
                parts.append(name)
            else:
                parts.append(f"{name}={json.dumps(prop.get('default'))}")
        return f"{self.name}({', '.join(parts)})"


def _string(description: str, default: str | None = None) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": "string", "description": description}
    if default is not None:
        prop["default"] = default
    return prop


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "list_files",
        "List the files under a directory of the workspace, one path per line.",
        _object({"path": _string("Workspace-relative directory or file.", ".")}, []),
    ),
    ToolSpec(
        "read_file",
        "Return the full text of a workspace file.",
        _object({"path": _string("Workspace-relative file path.")}, ["path"]),
    ),
    ToolSpec(
        "write_file",
        "Create or overwrite a workspace file with the given text.",
        _object(
            {
                "path": _string("Workspace-relative file path."),
                "content": _string("Complete new file content."),
            },
            ["path", "content"],
        ),
    ),
    ToolSpec(
        "replace_in_file",
        "Replace the one exact occurrence of old with new in a workspace file.",
        _object(
            {
                "path": _string("Workspace-relative file path."),
                "old": _string("Exact text to replace; must occur exactly once."),
                "new": _string("Replacement text."),
            },
            ["path", "old", "new"],
        ),
    ),
    ToolSpec(
        "run",
        "Run a shell command in the workspace (no network) and return its output.",
        _object({"cmd": _string("Shell command line.")}, ["cmd"]),
    ),
    ToolSpec(
        "done",
        "Finish the current task with a short summary of what was done.",
        _object({"summary": _string("One or two sentences on what changed.")}, ["summary"]),
    ),
)
"""Tool descriptions in :data:`~halflife.schema.AGENT_TOOLS` order."""

_SPECS: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


class ToolArgumentError(ValueError):
    """Raised by :func:`check_args` for missing, extra, or mistyped arguments."""


def tool_spec(name: str) -> ToolSpec:
    """The :class:`ToolSpec` of a tool (KeyError for unknown names)."""
    return _SPECS[name]


def render_tool_docs() -> str:
    """Plain-text tool list for prompts: one ``signature: description`` line per tool."""
    return "\n".join(f"- {spec.signature()}: {spec.doc}" for spec in TOOL_SPECS)


def json_schema_for_tool_call(*, per_tool: bool = True) -> dict[str, Any]:
    """JSON schema of a model's tool call ``{"tool": <name>, "args": {...}}``.

    With ``per_tool`` (default) the schema is an ``anyOf`` with one branch per
    tool, so a grammar built from it forces valid arguments for the chosen
    tool. Without it, ``args`` is any object.
    """
    if not per_tool:
        return {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": list(AGENT_TOOLS)},
                "args": {"type": "object"},
            },
            "required": ["tool", "args"],
            "additionalProperties": False,
        }
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "tool": {"const": spec.name},
                    "args": dict(spec.parameters),
                },
                "required": ["tool", "args"],
                "additionalProperties": False,
            }
            for spec in TOOL_SPECS
        ]
    }


def check_args(tool: str, args: object) -> dict[str, str]:
    """Validate arguments for a tool and fill defaults.

    Raises:
        ToolArgumentError: for an unknown tool, a non-object, a missing or
            unexpected argument, or a non-string value.
    """
    spec = _SPECS.get(tool)
    if spec is None:
        raise ToolArgumentError(f"unknown tool {tool!r}; available tools: {', '.join(AGENT_TOOLS)}")
    if not isinstance(args, Mapping):
        raise ToolArgumentError(f"{tool} arguments must be a JSON object")
    props: Mapping[str, Any] = spec.parameters["properties"]
    required: list[str] = list(spec.parameters["required"])
    extra = sorted(str(k) for k in args if k not in props)
    if extra:
        raise ToolArgumentError(
            f"{tool} got unexpected argument(s) {', '.join(extra)}; expected {spec.signature()}"
        )
    missing = [name for name in required if name not in args]
    if missing:
        raise ToolArgumentError(
            f"{tool} is missing argument(s) {', '.join(missing)}; expected {spec.signature()}"
        )
    checked: dict[str, str] = {}
    for name, prop in props.items():
        value = args.get(name, prop.get("default"))
        if not isinstance(value, str):
            raise ToolArgumentError(f"{tool} argument {name} must be a string")
        checked[name] = value
    return checked


def _occurrences(text: str, needle: str) -> int:
    count = 0
    start = text.find(needle)
    while start != -1:
        count += 1
        start = text.find(needle, start + 1)
    return count


@dataclass
class _Outcome:
    output: str
    exit_code: int | None = None
    net: tuple[NetAttempt, ...] = ()


def _list_files(ws: Workspace, args: Mapping[str, str]) -> _Outcome:
    files = ws.list_files(args["path"])
    return _Outcome("\n".join(files) if files else "(no files)")


def _read_file(ws: Workspace, args: Mapping[str, str]) -> _Outcome:
    path = args["path"]
    target = ws.resolve(path)
    if target.is_dir():
        return _Outcome(f"error: {path} is a directory; use list_files")
    try:
        return _Outcome(ws.read_text(path))
    except UnicodeDecodeError:
        return _Outcome(f"error: {path} is not a UTF-8 text file")


def _write_file(ws: Workspace, args: Mapping[str, str]) -> _Outcome:
    path = args["path"]
    if ws.resolve(path).is_dir():
        return _Outcome(f"error: {path} is a directory")
    written = ws.write_text(path, args["content"])
    return _Outcome(f"wrote {written} characters to {path}")


def _replace_in_file(ws: Workspace, args: Mapping[str, str]) -> _Outcome:
    path, old, new = args["path"], args["old"], args["new"]
    if not old:
        return _Outcome("error: old must be non-empty")
    if ws.resolve(path).is_dir():
        return _Outcome(f"error: {path} is a directory")
    try:
        text = ws.read_text(path)
    except UnicodeDecodeError:
        return _Outcome(f"error: {path} is not a UTF-8 text file")
    count = _occurrences(text, old)
    if count == 0:
        return _Outcome(f"error: old text not found in {path}; nothing was changed")
    if count > 1:
        return _Outcome(
            f"error: old text occurs {count} times in {path}; include more surrounding "
            "text so it matches exactly once; nothing was changed"
        )
    ws.write_text(path, text.replace(old, new, 1))
    return _Outcome(f"replaced 1 occurrence in {path}")


def _run(ws: Workspace, args: Mapping[str, str], options: SandboxOptions) -> _Outcome:
    cmd = args["cmd"]
    if not cmd.strip():
        return _Outcome("error: cmd must be a non-empty command")
    result = run_with(cmd, ws.root, options)
    body = result.output.rstrip("\n")
    status = f"[exit code: {result.exit_code}]"
    return _Outcome(f"{body}\n{status}" if body else status, result.exit_code, result.net)


def _done(ws: Workspace, args: Mapping[str, str]) -> _Outcome:
    return _Outcome("ok: task marked done")


def _dispatch(
    tool: str, ws: Workspace, args: Mapping[str, str], options: SandboxOptions
) -> _Outcome:
    if tool == "run":
        return _run(ws, args, options)
    handlers = {
        "list_files": _list_files,
        "read_file": _read_file,
        "write_file": _write_file,
        "replace_in_file": _replace_in_file,
        "done": _done,
    }
    return handlers[tool](ws, args)


def _jsonable_args(args: object) -> dict[str, Any]:
    if isinstance(args, Mapping):
        return {str(k): v for k, v in args.items()}
    return {"_raw": repr(args)[:1000]}


def execute_call(
    tool: str,
    args: object,
    workspace: Workspace,
    *,
    index: int,
    subtask_id: str,
    options: SandboxOptions | None = None,
) -> ToolEvent:
    """Run one tool call given as a name and arguments; never raises for bad input.

    An unknown tool name becomes a ``tool="invalid"`` event whose args record
    the requested name. Argument errors, path escapes, missing files, and
    sandbox start failures become ``error:`` observations.
    """
    opts = options or SandboxOptions()
    start = time.monotonic()
    if tool not in AGENT_TOOLS:
        outcome = _Outcome(
            f"error: unknown tool {tool!r}; available tools: {', '.join(AGENT_TOOLS)}"
        )
        event_tool: ToolName = "invalid"
        event_args: dict[str, Any] = {"requested_tool": str(tool), "args": _jsonable_args(args)}
    else:
        event_tool = cast(ToolName, tool)
        event_args = _jsonable_args(args)
        try:
            checked = check_args(tool, args)
            outcome = _dispatch(tool, workspace, checked, opts)
        except ToolArgumentError as exc:
            outcome = _Outcome(f"error: {exc}")
        except WorkspacePathError as exc:
            outcome = _Outcome(f"error: {exc}")
        except FileNotFoundError:
            path = event_args.get("path", "")
            outcome = _Outcome(f"error: {path} does not exist")
        except (OSError, SandboxError, ValueError) as exc:
            outcome = _Outcome(f"error: {type(exc).__name__}: {exc}")
    elapsed = time.monotonic() - start
    output, _ = truncate_output(outcome.output, opts.output_limit)
    return ToolEvent(
        index=index,
        subtask_id=subtask_id,
        tool=event_tool,
        args=event_args,
        output=output,
        exit_code=outcome.exit_code,
        fs_after=workspace.snapshot(),
        net=outcome.net,
        elapsed_s=elapsed,
    )


def execute(
    action: Action,
    workspace: Workspace,
    *,
    index: int,
    subtask_id: str,
    options: SandboxOptions | None = None,
) -> ToolEvent:
    """Run a :class:`~halflife.schema.Action` and return its :class:`~halflife.schema.ToolEvent`."""
    return execute_call(
        action.tool,
        action.args,
        workspace,
        index=index,
        subtask_id=subtask_id,
        options=options,
    )


def invalid_event(
    workspace: Workspace,
    *,
    index: int,
    subtask_id: str,
    raw: str,
    reason: str,
    output_limit: int = DEFAULT_OUTPUT_LIMIT,
) -> ToolEvent:
    """Event for a model output that is not a parseable tool call (tool ``invalid``)."""
    output, _ = truncate_output(f"error: {reason}", output_limit)
    return ToolEvent(
        index=index,
        subtask_id=subtask_id,
        tool="invalid",
        args={"raw": raw[:4000]},
        output=output,
        exit_code=None,
        fs_after=workspace.snapshot(),
    )
