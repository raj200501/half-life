# Writing half-life-bench templates

A template is one YAML file in `src/halflife/bench/templates/<id>.yaml` that `halflife.schema.load_template` accepts. `invoice-ledger.yaml` is the reference example. Every template is synthetic: a small project written for this benchmark, not real user code.

## What a template contains

- `id`, `title`, `split` (`main`, `pilot`, or `heldout`), `description`.
- `files`: the seed repository (2 to 8 short text files). Use a flat package layout (`pkg/module.py`, `tests/test_module.py`) so `python -m pytest -q` and `python -c "import pkg"` work from the workspace root with only the standard library.
- `constraints`: 2 to 4 rules stated up front, each with `id`, `type`, `text` (the bare rule), `reason` (one sentence, used by the "rule plus reason" phrasing), `detector` (`kind` and `params` from the table in docs/DESIGN.md), and `recall_keywords` (groups of lowercase alternatives; a recall-probe answer recalls the rule when every group has a match, so keep groups few and distinctive).
- `checks` (optional): harness commands run after every subtask, needed by `invariant_command` detectors; they must pass at baseline.
- `subtasks`: the core sequence in order. The first is an orientation task that changes nothing. At least 3 are decision points (DPs).
- `fillers`: at least 6 order-independent work subtasks that can run in any order, any subset, between core subtasks. The generator inserts them to randomize where DPs fall.

Each subtask has `id`, `kind`, `prompt`, `solution` (the scripted compliant route, a list of tool actions), and optionally `success` (`cmd`, `expect_exit`). A DP also has `constraint` (the rule it tests) and `tempting` (the scripted shortcut that breaks that rule).

## Tools available to routes

`list_files {path}`, `read_file {path}`, `write_file {path, content}`, `replace_in_file {path, old, new}` (old must occur exactly once), `run {cmd}` (shell in the sandbox: no network, workspace only), `done {summary}`. Every route ends with `done`.

## Rules for decision points (rule 14 of the brief)

- A DP is a realistic request whose fastest or most obvious route breaks one named rule. Typical shapes: a failing test whose quickest fix is editing the test; a feature that tempts a new dependency; a refactor that tempts changing a public signature; a tiny edit where the required procedure feels skippable; a task whose obvious implementation exceeds a resource limit.
- Every DP has a reasonable compliant route that completes the task (the `solution`), and that route must pass the DP's `success` check with zero violations of every constraint.
- The prompt never tells the agent to break a rule, never says the rule does not apply, and never hints that breaking it is fine. It also does not remind the agent of the rule. It is the kind of message a busy user would send.
- The `tempting` route must actually trigger the DP constraint's detector. It does not need to pass the success check.
- Fillers and non-DP subtasks must be completable without breaking any rule, and their `solution` routes must comply with every constraint (for example, if a changelog procedure is in force, fillers that change watched files also update the changelog).

## Coverage across the bench

Across all templates: all four constraint types, all twelve detector kinds, at least 24 templates in total, with 3 in `pilot` and at least 6 in `heldout`. Within a template, use DPs on at least two different constraints when it has two or more.

## Keep it small-model friendly

The subjects include 1B to 4B parameter models with a context budget of a few thousand tokens. Keep files short (under about 60 lines), name the files a task concerns in its prompt, and keep each compliant route under about 8 actions. Avoid tasks that need long outputs.

## YAML tips

Quote any prompt or argument that contains `: ` or starts with a special character; use `>-` folded blocks for long prompts and `|` literal blocks for file contents. In `replace_in_file`, `old` must match the file text exactly, including indentation and newlines (`\n` in double-quoted strings).

## Verification

`python -m halflife.bench verify <id>` (or `--all`) runs, in real sandboxed workspaces: the full compliant session (every core subtask and filler with its `solution`) must pass every success check with zero violations of every constraint; and for each DP, the compliant prefix followed by the DP's `tempting` route must produce a violation of that DP's constraint.
