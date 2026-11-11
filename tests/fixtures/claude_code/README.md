# Claude Code driver fixtures (synthetic)

These files are authored by hand from the documented Claude Code formats. They are not recorded from a real Claude Code session, and no model produced any of their text: every reply, tool call, token count, cost, and session id is made up for the contract tests in `tests/contract/test_claude_code.py`. Nothing here is a model result.

Sources of the formats:

- stream-json output of `claude -p --output-format stream-json --verbose --include-hook-events` (the `system` `init`, `compact_boundary`, `status`, and `hook_started`/`hook_response` events, `assistant` and `user` messages with `tool_use` and `tool_result` blocks, `rate_limit_event`, and the final `result` event), as documented for Claude Code headless mode and the Agent SDK message types;
- hook input JSON for `PostToolUse`, `PostToolUseFailure`, and `InstructionsLoaded`, as documented in the Claude Code hooks reference;
- the option list of `claude --help` for version 2.1.309, cut down to the options the driver checks and a few it refuses.

The event and field names used here and in the tests (`compact_boundary` with `compact_metadata.trigger` and `pre_tokens`, `hook_started` and `hook_response` with `exit_code` and `outcome`, `rate_limit_event` with `rate_limit_info` and its overage fields, the assistant `error` values, the `init` and `result` fields, the `InstructionsLoaded` and `SessionStart` hook inputs, and the transcript's `prompt_snapshot` and `isCompactSummary` entries) were also checked by reading the message schemas embedded in the Claude Code 2.1.318 build installed in the build environment, without running it or making any model call. Checking names is all that was done; no output of that build is stored here.

Some inputs are built inline by the tests instead of stored here: `SessionStart` hook input, transcript entries (`prompt_snapshot` and compaction summaries), `rate_limit_event` lines with overage fields, and assistant messages carrying API errors. They are just as synthetic.

Placeholders in braces are filled in by the tests before use: `{ws}` (the workspace root), `{session_id}` (the conversation UUID the driver passed), `{fork_id}` (the id of a forked probe session), `{model}` (a made-up model id), and in the smoke fixture `{smoke_command}`, `{subject_python}`, `{shim_curl}`, and `{probe_file}`.

| File | What it stands for |
|---|---|
| `help.txt` | `claude --help` options the driver needs, plus `--bare` and `--dangerously-skip-permissions`, which it never passes |
| `version.txt` | `claude --version` output |
| `stream_tool_calls.jsonl` | one subtask: a failing test run, an edit of a test file, a passing run, a final answer |
| `stream_compaction.jsonl` | one subtask with an auto-compaction (`compact_boundary`, `trigger: auto`) and the summary message that follows it |
| `stream_usage_limit.jsonl` | a plan usage limit: a rejected `rate_limit_event`, a synthetic assistant message with `error: rate_limit`, an error result |
| `stream_billing_error.jsonl` | a billing error (`error: billing_error`, "Credit balance is too low") |
| `stream_contaminated_init.jsonl` | an `init` event showing configuration the driver did not set: an MCP server and its tool, a user plugin, a custom output style |
| `stream_probe.jsonl` | a forked, tool-less recall probe answering with a fenced JSON rule list |
| `stream_smoke.jsonl` | the smoke session's single Bash call and its output |
| `hook_post_tool_use.json` | `PostToolUse` hook input for a Write call |
| `hook_post_tool_use_failure.json` | `PostToolUseFailure` hook input for a failing Bash call |
| `hook_instructions_loaded.json` | `InstructionsLoaded` hook input for the workspace `CLAUDE.md` |
| `fake_claude.py` | a stand-in executable for the subprocess launcher test; it is not Claude Code and makes no model calls |
