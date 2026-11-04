# Claude Code hook input fixtures

Every JSON file here was written by hand from the input format documented in the
Claude Code hooks reference (https://code.claude.com/docs/en/hooks, retrieved
2026-11-04): the common fields (`session_id`, `transcript_path`, `cwd`,
`permission_mode`, `hook_event_name`), the PreToolUse fields (`tool_name`,
`tool_input`, `tool_use_id`), and the SessionStart, PreCompact, and PostCompact
fields. None was captured from a real Claude Code session, and each file says so
in its `_comment` field, which Claude Code would not send; the kit ignores fields
it does not use.

`/workspace/project` is a placeholder: the tests and the overhead measurement
replace it with a temporary project directory before feeding a fixture to a hook.

The MultiEdit fixture follows the input shape of earlier Claude Code versions
(`file_path` plus a list of `edits`); the current reference no longer lists that
tool. The NotebookEdit fixture uses `notebook_path` and `new_source`, field names
the reference does not document; the guard also accepts `file_path`.
