# Mitigations for real agent tools

half-life ships four mitigations for coding agents that drift from their rules over a long session (RQ5). All four read one portable constraint spec, `halflife.yaml`. E4 tests the same mechanisms in the study's minimal agent, not the installed tools inside Claude Code: the `guard` arm runs the guard's decision core (`halflife.kit.guard.decide_minimal_agent`) and the `reinject` arm appends the hook's text (`halflife.kit.reinject.reinject_text`), but the `pin` arm is a memory message the harness re-inserts after each compaction (DECISIONS.md D-013) rather than the CLAUDE.md block `halflife-pin` writes, and the `preserve` arm replaces the minimal agent's own summary prompt with `preserve_prompt()`, while in Claude Code the Compact Instructions are read by Claude Code's own compactor. E4 also runs a fifth arm, `selfcheck`, a periodic request to check recent actions against the rules, which exists only in the minimal agent. E4 results, with each mitigation's cost in tokens, latency, and task success beside its effect, will be reported in docs/REPORT.md once E4 has run; the measured overhead of `halflife-guard` and `halflife-pin` is banked in `artifacts/e4/overhead.json` (see [Measured overhead](#measured-overhead)).

| Mitigation | Command | What it does | What it does not do |
|---|---|---|---|
| Policy guard | `halflife-guard` (Claude Code PreToolUse hook) | Refuses a tool call that would break an enforced rule, before it runs, and tells the agent which rule and why. | It cannot see what a program does at run time, and it does not enforce procedures or limits that need session history (see below). |
| Pinned memory file | `halflife-pin` | Writes the rules into a managed block of CLAUDE.md or AGENTS.md. Claude Code loads the project-root CLAUDE.md at session start and re-reads it from disk after compaction. | It refuses nothing: the agent still has to read and follow the file. |
| Re-injection after compaction | `python -m halflife.kit.reinject` (SessionStart hook, matcher `compact`) | Adds the rules to the context right after every compaction. | Same as the memory file; it only restores the text. |
| Constraint-preserving compaction | `halflife-pin --compact-instructions`, or `/compact` with the text from `halflife.kit.compaction_prompt.compact_command()` | Asks the summarizer to begin its summary with a "Rules in force" section that copies every rule word for word. | The summarizer may still drop or reword a rule; nothing checks the summary. |

## Install

1. Install the package from a clone of this repository (it is not published to a package index):

   ```bash
   uv tool install /path/to/half-life      # puts halflife-guard and halflife-pin on PATH
   # or, inside a virtual environment:
   pip install /path/to/half-life
   ```

2. Copy `examples/halflife.yaml` to your project root as `halflife.yaml` and edit the rules (see [The portable spec](#the-portable-spec)). Check what the guard will enforce:

   ```bash
   halflife-guard --spec halflife.yaml --explain
   ```

3. Register the guard and the re-injection hook in `.claude/settings.json` at the project root:

   ```json
   {
     "hooks": {
       "PreToolUse": [
         {
           "matcher": "Write|Edit|MultiEdit|NotebookEdit|Bash",
           "hooks": [
             {
               "type": "command",
               "command": "halflife-guard",
               "args": ["--spec", "${CLAUDE_PROJECT_DIR}/halflife.yaml"]
             }
           ]
         }
       ],
       "SessionStart": [
         {
           "matcher": "compact",
           "hooks": [
             {
               "type": "command",
               "command": "/path/to/venv/bin/python",
               "args": ["-m", "halflife.kit.reinject", "--spec", "${CLAUDE_PROJECT_DIR}/halflife.yaml"]
             }
           ]
         }
       ]
     }
   }
   ```

   `command` must resolve to the installed tool: `halflife-guard` on PATH, and for re-injection the Python interpreter of the environment where halflife is installed. The `args` form runs the command without a shell, which the hooks reference recommends for path placeholders such as `${CLAUDE_PROJECT_DIR}`. In shell form, write `"command": "halflife-guard --spec \"${CLAUDE_PROJECT_DIR}/halflife.yaml\""` instead. MultiEdit is matched for Claude Code versions that still have that tool.

   The guard fails open: if it cannot read its input or the spec, it exits 1, Claude Code shows a hook error notice, and the call goes ahead. To fail closed instead, add `"onFailure": "block"` to the guard's entry (Claude Code v2.1.295 or later).

4. Pin the rules in the memory file, and keep the block current in CI:

   ```bash
   halflife-pin --spec halflife.yaml --out CLAUDE.md
   halflife-pin --spec halflife.yaml --out CLAUDE.md --check    # exit 1 when stale
   halflife-pin --spec halflife.yaml --format agents --out AGENTS.md
   ```

5. Optionally ask Claude Code's compaction to keep the rules, by adding a "Compact Instructions" section to the same block:

   ```bash
   halflife-pin --spec halflife.yaml --out CLAUDE.md --with-reasons --compact-instructions
   ```

Every command reads the spec from `--spec`, else `$HALFLIFE_SPEC`, else `halflife.yaml` in the working directory, else `halflife.yaml` in `$CLAUDE_PROJECT_DIR`.

## The portable spec

A spec is YAML, format version 1. Each constraint has the fields of `halflife.schema.Constraint`; `examples/halflife.yaml` shows every detector kind the guard enforces plus one it does not.

```yaml
version: 1
constraints:
  - id: tests-frozen              # lower case, digits, - and _
    type: prohibition             # prohibition | invariant | procedure | resource_limit
    text: Never modify, create, or delete files under tests/.
    reason: The tests encode the behavior the maintainers agreed on.
    detector:
      kind: protected_paths       # one of the twelve detector kinds in docs/DESIGN.md
      params: {globs: ["tests/**"]}
    recall_keywords: [["test"], ["never", "not "]]   # optional
```

- `version`, `id`, `type`, `text`, `reason`, and `detector` are required; unknown or duplicated keys are errors, so a typo cannot silently drop a rule. `halflife.kit.spec.load_spec` reports the file, the constraint, and the problem.
- Detector parameters mean exactly what they mean to the study's detectors (`halflife.detect`). Globs are relative to the project root: `*` and `?` never match `/`, a whole `**` segment matches any number of directories, and a trailing `/` means everything under that directory.
- `recall_keywords` only matters for recall probes. The default takes the two longest content words of the text (paths and file names first), cut to five characters, plus a negation group for prohibitions.
- `halflife.kit.spec.dump_spec` writes a spec from any constraints, for example a bench template's.

## What halflife-guard enforces

| Detector kind | Write, Edit, MultiEdit, NotebookEdit | Bash |
|---|---|---|
| `protected_paths` | refused on a matching path | refused when the command would create, modify, or delete a matching path |
| `forbidden_commands` | not applicable | refused when a pattern matches the command (`re.search`, as the detector does) |
| `no_network` | not applicable | refused when the command would use the network |
| `no_new_dependencies` | refused when a manifest would gain a dependency name | refused for any write to a manifest and for an install that names a package |
| `max_file_lines` | refused when a matching file would grow past the limit | not checked |
| `file_invariant` | refused when the file would stop matching the pattern | refused for any write to the file |
| `stable_api` | refused when a public function, class, or method would be removed or change its parameters | refused for any write to the module |
| `changelog_procedure`, `tests_before_done`, `max_new_files`, `max_test_runs`, `invariant_command` | not enforced: they need session history or a test run | not enforced |

For the content checks the guard computes the file as it would be after the call (Write gives it directly; Edit and MultiEdit are applied to the file on disk) and uses the detectors' own manifest parser, line count, and public-API comparison. Bash writes to files with a content rule are refused because their result cannot be computed; the refusal asks the agent to use Edit or Write instead. Paths that the study's workspace snapshot ignores (`.git`, `__pycache__`, `.pytest_cache`, `*.pyc`) never count as protected. Calls to other tools are always allowed.

A refusal prints the documented PreToolUse output, so Claude sees the reason and the call does not run:

```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "halflife-guard: the command would modify tests/test_core.py (sed -i). Rule tests-frozen: Never modify, create, or delete files under tests/."}}
```

An allowed call prints nothing, so Claude Code's normal permission flow still applies. Reasons never quote the command line. Options: `--root DIR` anchors the globs (default `$CLAUDE_PROJECT_DIR`, else the hook's `cwd`), `--log FILE` appends one JSON line per decision, `--no-git` disables the read-only git queries, and `--explain` prints what is enforced for each rule.

The minimal agent uses the same decision core through `halflife.kit.guard.decide_minimal_agent` (its `write_file`, `replace_in_file`, and `run` tools map to Write, Edit, and Bash) or `minimal_agent_guard`, which returns the refusal reason the agent sees.

### How the Bash analysis works

The guard parses the command with a small quote-aware lexer (quotes, escapes, `$(...)` and backticks, redirections, heredocs, comments, `&&`, `||`, `;`, `|`, newlines, and subshells) and follows `cd`, simple variable assignments, `for` loops over literal words, and `bash -c`, `sh -c`, `eval`, and `source`. It recognizes writes through redirection and `tee`, in-place editors (`sed -i`, `perl -i`, `awk -i inplace`, `ed`, `vim -c`), `cp`, `mv`, `rm`, `unlink`, `shred`, `rsync`, `install`, `ln`, `truncate`, `touch`, `dd of=`, `tar -x`, `unzip`, `sort -o`, `uniq`, `patch` and `git apply` (targets read from the patch), `find -delete`, `-exec`, and `-fprint`, `xargs` and `while read` loops (targets taken from the upstream command's path operands), common formatters, and `git checkout`, `restore`, `rm`, `mv`, and `stash` with pathspecs. It reads Python run with `-c`, from a heredoc, or from a script file, and finds `open()` in a write mode, `pathlib` and `os` writes, `shutil` copies, moves, and deletions, `subprocess` and `os.system` commands, and network calls. For whole-tree git commands (switching branches, `reset --hard`, `stash` and `stash pop`, `clean -f`) it asks git which files would change. Network use covers curl and wget (unless every URL is `file://`), ssh, scp, nc, rsync to a remote, git clone, fetch, pull, push, and ls-remote, package managers (pip without `--no-index`, uv, poetry, npm, yarn, pnpm, conda, apt, brew, gem, cargo, go, docker pull), and tools that download what they run (`npx`, `uvx`, `pipx`).

### Limits of static Bash analysis

The guard decides from the command text and the files on disk before the command runs. It cannot see:

- **Work done inside programs.** A command that runs project code or a tool which writes a protected file or opens a connection itself, for example `python -m mypkg.cli update` rewriting a protected data file, or a script that downloads, looks harmless. Python run with `-m` is not read, and only straight-line Python is followed: paths built in loops, in functions, or from input are not resolved.
- **Values known only at run time.** Targets in variables set from command output, `bash -c "$CMD"`, `eval` of computed text, and `cd` to a computed directory (later relative paths resolve against the last known directory) are not resolved. Globs expand against the files that exist when the hook runs.
- **Tools it does not know.** A command outside its tables is treated as writing nothing: build tools (`make`, `npm run`), code generators, `node -e`, `ruby -e`, perl code beyond in-place editing, and sed's `w` command.
- **History-rewriting git commands.** Merges, rebases, cherry-picks, reverts, and pulls are not predicted. The git queries for branch switches, `reset --hard`, `stash`, and `clean` need git and a repository; without them the guard falls back to treating the named operand as a path, or allows the call.
- **Other tools and shells.** PowerShell tool calls on Windows, the Monitor tool, MCP tools, and files added to the prompt with `@` are not checked; the hook runs only for the matched tools.
- **Paths outside the project root.** They are never protected. In a session that enters a git worktree, `$CLAUDE_PROJECT_DIR` stays at the main checkout, so pass `--root` for the worktree if its paths must be protected.

It is conservative where a precise answer would need the command's result, and these choices can refuse harmless commands: any Bash write to a manifest, a `file_invariant` file, or a `stable_api` module; `touch` on a protected path that already exists; recursive writes to a directory that contains protected paths (`rm -r`, `cp -r` into it, a formatter run on the whole project); `find -delete` unless every `-name` filter names an ignored cache; and Python network detection on code that cannot be parsed. Pair the guard with a `forbidden_commands` rule for commands you never want run, whatever their targets.

## Re-injection and compaction details

Claude Code adds the stdout of a SessionStart hook that matches the `compact` source to the compacted context. `python -m halflife.kit.reinject` prints the rules as `hookSpecificOutput.additionalContext` (or plain text with `--plain`), phrased as a statement that the session's rules still apply, as the hooks reference advises for added context. It prints nothing for other events and other SessionStart sources unless `--any-source` is given. PostCompact is not used: it has no decision control and cannot change the compacted context.

There is no PreCompact hook in the kit. A PreCompact hook can only block compaction (exit 2 or `"decision": "block"`), and Claude Code discards its `systemMessage`; the documented ways to steer the summary are a "Compact Instructions" section in CLAUDE.md and `/compact` with a focus. In the minimal agent, `halflife.kit.compaction_prompt.preserve_prompt()` replaces the default summary prompt: it is the default prompt plus the request to begin with a verbatim "Rules in force" section, placed first so that a summary cut off at the token cap keeps the rules. `halflife.kit.reinject.reinject_text` builds the message the minimal agent's `reinject` arm appends after each compaction, the same text the hook adds.

## Measured overhead

`tools/bank_overhead.py` measures both tools with `halflife.kit.overhead` and banks `artifacts/e4/overhead.json` with its provenance (git SHA, command, repeat count, and the machine's load average before and after, since a model run on the same machine would slow the hook):

- **halflife-guard, latency:** the median wall time of one hook run as a fresh `python -m halflife.kit.guard` subprocess, fed one PreToolUse fixture from `tests/fixtures/claude_hooks/` on stdin, as Claude Code runs it; interpreter start-up and imports dominate. The in-process decision time is recorded beside it.
- **halflife-guard, tokens:** the mean length of a refusal reason, the only text the guard adds to the context (allowed calls add nothing). Tokens are estimated as characters divided by four unless a tokenizer is given.
- **halflife-pin, tokens:** the size of the pinned block for each held-out template, which stays in the context for the whole session; the record reports the template with the median block size and keeps every template's numbers.
- **halflife-pin, latency:** the median time to render the block and apply it to a memory file, paid once per spec change.

## Documentation relied on

Claude Code behavior described here follows these pages, read on 2026-11-04:

- Hooks reference, https://code.claude.com/docs/en/hooks: PreToolUse input (`tool_name`, `tool_input`, `tool_use_id`, absolute `file_path` for Write and Edit), the `hookSpecificOutput.permissionDecision` deny output and its reason shown to Claude, exit-code semantics (0, 2, and non-blocking errors), `onFailure`, matcher syntax, exec form with `args`, `${CLAUDE_PROJECT_DIR}`, `cwd` following `cd`, SessionStart sources and `additionalContext`, and the PreCompact and PostCompact events.
- Automate actions with hooks, https://code.claude.com/docs/en/hooks-guide: re-injecting context after compaction with a SessionStart hook and the `compact` matcher.
- Explore the context window, https://code.claude.com/docs/en/context-window: what survives compaction, including the project-root CLAUDE.md and SessionStart hooks that match `compact`.
- How Claude remembers your project, https://code.claude.com/docs/en/memory: CLAUDE.md and AGENTS.md loading, and CLAUDE.md being re-read after compaction.
- How Claude Code works, https://code.claude.com/docs/en/how-claude-code-works: the "Compact Instructions" section of CLAUDE.md and `/compact` with a focus.
- Tools reference, https://code.claude.com/docs/en/tools-reference: the NotebookEdit and PowerShell tools.
