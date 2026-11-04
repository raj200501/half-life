# half-life: how long do coding agents keep the rules they are given?

Technical report. Draft in progress: the method and experiment design are written; the abstract, results, and discussion are generated from banked artifacts once the runs finish (`make report`).

## Abstract

Pending results.

## 1. Introduction

Coding agents now work for hours on a single task. Users set rules early in a session ("never edit tests/", "no new dependencies", "keep the public API stable", "no network calls"), and an agent may compact its context several times before one of those rules matters. Whether the rules survive is a reliability question, and when the rules encode safety or review boundaries it is also a safety question.

This report describes half-life, a measurement of constraint persistence. It has four parts. half-life-bench is a set of long, multi-stage coding sessions on small synthetic projects, each with rules stated up front and later decision points where breaking a rule is the easiest way to finish a request. A harness controls session length, where and how the rules are stated, and when and how context is compacted, for a controlled minimal agent driven by local open-weight models and for real agent tools. Deterministic detectors decide from file hashes, commands, manifests, syntax trees, and sandbox logs whether a rule was broken. A survival analysis estimates the per-opportunity hazard of a violation as a function of how long ago the rule was stated, the causal effect of compaction, and the effect of mitigations that real tools can adopt today.

## 2. Related work

Pending verified citations (`make citecheck`).

## 3. Method

### 3.1 Sessions and decision points

A session template is a small synthetic Python project with two to four rules and an ordered list of requests. Rules come in four types: prohibitions (do not touch these paths, add dependencies, use the network, or run these commands), invariants (keep the public API, keep a harness check passing, keep a file property), procedures (log every change in the changelog, run the tests before finishing), and resource limits (at most N new files, at most N lines per module, at most N test runs per request).

Each template has at least three decision points. A decision point is a realistic request whose fastest or most obvious route breaks one named rule, for example a failing test whose quickest fix is editing the test, a feature that tempts a new dependency, or a refactor that tempts changing a public signature. Every decision point also has a reasonable compliant route. The request never mentions the rule and never suggests that it does not apply. Each template also has order-independent filler requests. The session generator inserts a random number of fillers before each core request, so the position of every decision point varies from session to session.

Each template ships with two scripted routes per decision point: the compliant route and the tempting route. A template is accepted only when, in real sandboxed workspaces, the full compliant session passes every success check with no detector firing (with fillers placed first and with fillers placed last), and every tempting route makes the detector for its rule fire.

### 3.2 The minimal agent

The controlled agent holds the scaffold constant across models. It has one system prompt, one JSON tool-call format enforced by a grammar, and six tools: `list_files`, `read_file`, `write_file`, `replace_in_file`, `run` (a shell command in the sandbox), and `done`. The harness sends one request at a time. A request ends when the agent calls `done` or after a fixed number of tool calls, and a session ends after its last request or at the session cap. An unparseable reply counts as a tool call.

Placement conditions decide where the rules are stated: in the system prompt; at the top of the first user message; in a pinned memory file that the harness loads at the start of the session and again after every compaction, as Claude Code does with CLAUDE.md; or in the first message plus periodic reminders appended to tool results. Phrasing conditions state the bare rule or the rule plus a one-sentence reason.

### 3.3 Compaction

The system prompt is never compacted. Compaction replaces the conversation after it. Strategies: a free-text summary written by the same model (`full_summary`), keeping only the most recent messages (`sliding_window`), a fixed JSON state filled by the model (`structured_state`), a summary with the rules appended verbatim by the harness (`constraint_pinned`), and a sham control. The sham runs the same summary but keeps the original message that stated the rules, so the rule text provably survives. Comparing `full_summary` with the sham separates losing the rule text from the disruption of compaction itself.

Compaction happens at randomized tool-call indices drawn independently of the decision-point positions, and also whenever the context exceeds a small token budget. Every event is logged with the full context before and after, the summary, token counts, and whether each rule's text survived verbatim.

### 3.4 Recall probes

Before each decision point and after each compaction, the harness forks the exact context and asks the fork to list every rule it was asked to follow. The answer never enters the main conversation. The main context is hashed before and after each probe, and every later main-run model call is checked for the probe question and for earlier probe answers. Answers are scored against each rule's keyword groups. The scorer is validated on a labeled set of synthetic answers that includes near misses.

### 3.5 Detection

Detectors read an action log that records, for every tool call, the call itself, its output, the SHA-256 of every workspace file after the call, and network attempts logged by the sandbox. File changes are never inferred from parsing commands: a `sed -i`, a scripted write, a `git checkout --`, or a `cp` over a protected file shows up as a hash change on that path. One deterministic detector exists for each rule kind: protected paths, forbidden commands, new dependencies (from package manifests), public API changes (from syntax-tree signatures), network attempts, changelog and test-before-done procedures, new-file, line, and test-run limits, failing harness checks, and file invariants.

The detectors were validated before any model run on a suite of seeded action logs with ground-truth labels. The seeded cases were written by a separate builder from the written definitions only, without access to the detector code, and their labels were committed before the detectors were scored against them.

### 3.6 Sandboxing

Every agent command, success check, and harness check runs in a disposable workspace inside a sandbox with no network: user, network, mount, and process namespaces, with every mount except the workspace and a per-run directory remounted read-only. The environment is an allowlist with all credentials removed, and Python commands run in a subject environment that contains only the standard library and pytest. Network attempts are logged by an in-process socket wrapper and by shims for common networked tools.

### 3.7 Statistics

The unit of analysis is an opportunity: a decision point that a session reached. Its position is the tool-call index at which the decision point starts, counted from the statement of the rules. The primary model is a discrete-time hazard model: a logistic regression of the violation indicator on position, compaction count, and rule type, fit to each session's opportunities up to and including its first violation. The half-life is the position at which the predicted hazard reaches one half, reported as "not reached within N tool calls" when it does not. Intervals come from a cluster bootstrap over sessions. Kaplan-Meier curves of time to first violation are shown as descriptive only, because they depend on how often decision points occur. The confirmatory hypotheses, sample sizes, exclusions, and stopping rules are preregistered in `prereg/` before any confirmatory data exists.

## 4. Experiments

- **E0, instrument validity.** Detector precision and recall on the seeded logs, with Wilson intervals; probe leakage across validation runs; completeness of compaction logs; agreement of the probe scorer with labels.
- **E1, controlled decay study (confirmatory).** The minimal agent with at least two open-weight model families that passed a pilot on templates E1 never uses, randomized decision-point positions, randomized compaction timing, and a randomized comparison of `full_summary` with the sham control.
- **E2, production agent (ecological arm).** Claude Code in headless mode at two model tiers on the same templates, with its own auto-compaction at the smallest supported window. Reported as ecological in every table.
- **E3, placement and phrasing.** The four placements crossed with the two phrasings.
- **E4, mitigations.** On held-out templates: the PreToolUse policy guard, the pinned memory file, a constraint-preserving compaction prompt, re-injection after compaction, and periodic self-checks, each reported with its cost in tokens, latency, and task success.

## 5. Results

Pending runs.

## 6. Limitations

Pending results. Known in advance: the subjects are small quantized models on CPU, the projects are synthetic and short, and the production arm depends on a machine where the Claude backend passes the billing guard.

## 7. Ethics and disclosure

The study uses only synthetic projects, mock tools, and disposable sandboxes with no network and no real credentials. Decision points never instruct an agent to break a rule, and no prompt search for rule-dropping behavior was performed. The mitigations are defensive.

## 8. Reproducibility

Every number in this report is produced by a committed script from a banked artifact under `artifacts/`, checked by `make claimcheck` and re-derived by `make freshness`. `make reproduce` rebuilds every figure and table, and `make demo` runs an offline end-to-end example in under three minutes.
