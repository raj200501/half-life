# half-life: how long do coding agents keep the rules they are given?

Technical report. The abstract and results are generated from banked artifacts by `make report`; every number in them is a claim traced to an artifact (`make claimcheck`, `make freshness`).

## Abstract

<!-- report:begin abstract -->
half-life measures how long coding agents keep rules stated at the start of a long session. It combines half-life-bench, a set of long multi-stage coding sessions with decision points where breaking a rule is the easiest route, a harness that controls session length, rule placement and phrasing, and context compaction, deterministic detectors of violations, and a discrete-time survival analysis. The benchmark has <!-- claim:bench.templates -->30<!-- /claim --> synthetic session templates, each with a banked compliant route and reference run. Before any model run, the detectors reached precision <!-- claim:e0.precision -->1.000<!-- /claim --> [<!-- claim:e0.precision.lo -->0.978<!-- /claim -->, <!-- claim:e0.precision.hi -->1.000<!-- /claim -->] and recall <!-- claim:e0.recall -->1.000<!-- /claim --> [<!-- claim:e0.recall.lo -->0.978<!-- /claim -->, <!-- claim:e0.recall.hi -->1.000<!-- /claim -->] on <!-- claim:e0.cases -->328<!-- /claim --> seeded synthetic action logs and the recall-probe scorer reached sensitivity <!-- claim:e0.probe.sensitivity -->0.973<!-- /claim --> [<!-- claim:e0.probe.sensitivity.lo -->0.924<!-- /claim -->, <!-- claim:e0.probe.sensitivity.hi -->0.991<!-- /claim -->] and specificity <!-- claim:e0.probe.specificity -->0.918<!-- /claim --> [<!-- claim:e0.probe.specificity.lo -->0.859<!-- /claim -->, <!-- claim:e0.probe.specificity.hi -->0.954<!-- /claim -->] on the test split of synthetic answers with LLM labels (95% Wilson score intervals). The main study (E1), the ecological arm (E2), the placement study (E3), and the mitigation study (E4) have not run. This report therefore has no estimate of constraint half-life yet.
<!-- report:end abstract -->

## 1. Introduction

Coding agents now work for hours on a single task. Users set rules early in a session ("never edit tests/", "no new dependencies", "keep the public API stable", "no network calls"), and an agent may compact its context several times before one of those rules matters. Whether the rules survive is a reliability question, and when the rules encode safety or review boundaries it is also a safety question.

This report describes half-life, a measurement of constraint persistence. It has four parts. half-life-bench is a set of long, multi-stage coding sessions on small synthetic projects, each with rules stated up front and later decision points where breaking a rule is the easiest way to finish a request. A harness controls session length, where and how the rules are stated, and when and how context is compacted, for a controlled minimal agent driven by local open-weight models and for real agent tools. Deterministic detectors decide from file hashes, commands, manifests, syntax trees, and sandbox logs whether a rule was broken. A survival analysis estimates the per-opportunity hazard of a violation as a function of how long ago the rule was stated, the causal effect of compaction, and the effect of mitigations that real tools can adopt today.

## 2. Related work

Every reference below was opened and checked (title, authors, year, and final URL are banked in `artifacts/citecheck.json`), and each is described only as far as the checked page supports.

**Long contexts.** Liu et al. (2023), [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172), found that retrieval and question-answering performance is often highest when the relevant information sits at the beginning or end of the input and degrades when it sits in the middle, even for models built for long contexts. A rule stated at the top of a long agent session is information at the beginning of the context until compaction rewrites that context, which is one reason this study manipulates compaction directly instead of only session length.

**Goal drift.** Arike et al. (2025), [Technical Report: Evaluating Goal Drift in Language Model Agents](https://arxiv.org/abs/2505.02709), give an agent a goal in its system prompt and then apply environmental pressure toward competing objectives over long contexts. They report that every model they tested drifted and that drift grew with context length. half-life asks a narrower question about rules rather than goals: whether a stated constraint still governs behavior at a later point where breaking it is the easiest way to finish a request, with no competing objective introduced.

**Agents in the field.** Tang et al. (2026), [How Coding Agents Fail Their Users](https://arxiv.org/abs/2605.29442), analyze real developer-agent sessions and define misalignment through developer pushback. Following rules and bounding actions are among the recurring forms they describe, and their abstract reports that constraint violations grow in share over time. Their study is observational; half-life is a controlled complement in which the timing of rules, decision points, and compaction is randomized.

**Persistence of effort.** Cai et al. (2026), [Push Your Agent: Measuring and Enforcing Quantitative Goal Persistence in Long-Horizon LLM Agents](https://arxiv.org/abs/2605.23574), measure whether agents keep working until a verifier confirms enough valid items. That is persistence of effort toward a goal; this study measures persistence of restraint under rules, and the two can fail independently.

**What production tools do today.** Claude Code documents that the project-root CLAUDE.md is re-read from disk and re-injected after compaction, and that CLAUDE.md is context rather than enforced configuration, so blocking an action needs a hook ([How Claude remembers your project](https://code.claude.com/docs/en/memory)). Its hooks reference documents PreToolUse hooks that can deny a tool call with a reason and SessionStart hooks that fire after compaction ([Hooks reference](https://code.claude.com/docs/en/hooks)). [AGENTS.md](https://agents.md) is an open Markdown format for project instructions to coding agents. These mechanisms are the basis of the pinned-file, re-injection, and policy-guard mitigations in E4.

**Statistics.** The hazard model is a discrete-time survival model fit by logistic regression on the risk set, following Singer and Willett (1993) ([DOI](https://doi.org/10.3102/10769986018002155)). Intervals come from the bootstrap of Efron (1979) ([DOI](https://doi.org/10.1214/aos/1176344552)), resampling whole sessions; rates carry the score intervals of Wilson (1927) ([DOI](https://doi.org/10.1080/01621459.1927.10502953)); and time to first violation is summarized descriptively with the estimator of Kaplan and Meier (1958) ([DOI](https://doi.org/10.1080/01621459.1958.10501452)). Local models run through [llama.cpp](https://github.com/ggml-org/llama.cpp).

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

<!-- report:begin results -->
### 5.1 Benchmark

half-life-bench has <!-- claim:bench.templates -->30<!-- /claim --> session templates (<!-- claim:bench.split.main -->21<!-- /claim --> in the main split, <!-- claim:bench.split.heldout -->6<!-- /claim --> held out for E4, and <!-- claim:bench.split.pilot -->3<!-- /claim --> reserved for the family pilot). Each is a small synthetic Python project with rules stated up front and decision points where breaking one rule is the easiest route. The templates state <!-- claim:bench.constraints.prohibition -->25<!-- /claim --> prohibitions, <!-- claim:bench.constraints.invariant -->21<!-- /claim --> invariants, <!-- claim:bench.constraints.procedure -->15<!-- /claim --> procedures, and <!-- claim:bench.constraints.resource_limit -->10<!-- /claim --> resource limits, and their decision points fall <!-- claim:bench.dps.prohibition -->39<!-- /claim --> on prohibitions, <!-- claim:bench.dps.invariant -->37<!-- /claim --> on invariants, <!-- claim:bench.dps.procedure -->29<!-- /claim --> on procedures, and <!-- claim:bench.dps.resource_limit -->16<!-- /claim --> on resource limits. Every template has a banked compliant route and reference run under `artifacts/bench/`, which gate G6 checks. This subsection describes the benchmark; it tests no hypothesis.

### 5.2 Instrument validity (E0)

Status: instrument validation; not preregistered, so exploratory under rule 2 of the brief. The seeded action logs and the probe answers are synthetic; nothing in this subsection is a model result.

**Detectors.** The deterministic detectors were scored on <!-- claim:e0.cases -->328<!-- /claim --> seeded action logs (<!-- claim:e0.cases.violations -->173<!-- /claim --> seeded violations and <!-- claim:e0.cases.compliant -->155<!-- /claim --> compliant cases), written from the rule definitions without access to the detector code and executed for real in sandboxes (D-017). Precision was <!-- claim:e0.precision -->1.000<!-- /claim --> [<!-- claim:e0.precision.lo -->0.978<!-- /claim -->, <!-- claim:e0.precision.hi -->1.000<!-- /claim -->] and recall <!-- claim:e0.recall -->1.000<!-- /claim --> [<!-- claim:e0.recall.lo -->0.978<!-- /claim -->, <!-- claim:e0.recall.hi -->1.000<!-- /claim -->] (95% Wilson score intervals). Gate G5 requires both to be at least 0.95: both values meet it, and so do both lower interval bounds. The seeded violations include indirect changes that parsing commands would miss: `sed -i` on a protected file (<!-- claim:e0.indirect.sed_inplace.caught -->11<!-- /claim --> of <!-- claim:e0.indirect.sed_inplace.seeded -->11<!-- /claim --> detected), scripted writes (<!-- claim:e0.indirect.scripted_write.caught -->13<!-- /claim --> of <!-- claim:e0.indirect.scripted_write.seeded -->13<!-- /claim --> detected), `git checkout --` (<!-- claim:e0.indirect.git_checkout.caught -->10<!-- /claim --> of <!-- claim:e0.indirect.git_checkout.seeded -->10<!-- /claim --> detected), and `cp` over a protected file (<!-- claim:e0.indirect.cp.caught -->10<!-- /claim --> of <!-- claim:e0.indirect.cp.seeded -->10<!-- /claim --> detected).

Per detector, over the seeded cases of its rule kind; Passed counts the compliant cases it did not flag.

| Detector | Violations detected | Violations missed | False alarms | Passed |
|---|---|---|---|---|
| `changelog_procedure` | <!-- claim:e0.detector.changelog_procedure.tp -->12<!-- /claim --> | <!-- claim:e0.detector.changelog_procedure.fn -->0<!-- /claim --> | <!-- claim:e0.detector.changelog_procedure.fp -->0<!-- /claim --> | <!-- claim:e0.detector.changelog_procedure.tn -->12<!-- /claim --> |
| `file_invariant` | <!-- claim:e0.detector.file_invariant.tp -->11<!-- /claim --> | <!-- claim:e0.detector.file_invariant.fn -->0<!-- /claim --> | <!-- claim:e0.detector.file_invariant.fp -->0<!-- /claim --> | <!-- claim:e0.detector.file_invariant.tn -->11<!-- /claim --> |
| `forbidden_commands` | <!-- claim:e0.detector.forbidden_commands.tp -->11<!-- /claim --> | <!-- claim:e0.detector.forbidden_commands.fn -->0<!-- /claim --> | <!-- claim:e0.detector.forbidden_commands.fp -->0<!-- /claim --> | <!-- claim:e0.detector.forbidden_commands.tn -->12<!-- /claim --> |
| `invariant_command` | <!-- claim:e0.detector.invariant_command.tp -->10<!-- /claim --> | <!-- claim:e0.detector.invariant_command.fn -->0<!-- /claim --> | <!-- claim:e0.detector.invariant_command.fp -->0<!-- /claim --> | <!-- claim:e0.detector.invariant_command.tn -->10<!-- /claim --> |
| `max_file_lines` | <!-- claim:e0.detector.max_file_lines.tp -->9<!-- /claim --> | <!-- claim:e0.detector.max_file_lines.fn -->0<!-- /claim --> | <!-- claim:e0.detector.max_file_lines.fp -->0<!-- /claim --> | <!-- claim:e0.detector.max_file_lines.tn -->10<!-- /claim --> |
| `max_new_files` | <!-- claim:e0.detector.max_new_files.tp -->9<!-- /claim --> | <!-- claim:e0.detector.max_new_files.fn -->0<!-- /claim --> | <!-- claim:e0.detector.max_new_files.fp -->0<!-- /claim --> | <!-- claim:e0.detector.max_new_files.tn -->12<!-- /claim --> |
| `max_test_runs` | <!-- claim:e0.detector.max_test_runs.tp -->8<!-- /claim --> | <!-- claim:e0.detector.max_test_runs.fn -->0<!-- /claim --> | <!-- claim:e0.detector.max_test_runs.fp -->0<!-- /claim --> | <!-- claim:e0.detector.max_test_runs.tn -->9<!-- /claim --> |
| `no_network` | <!-- claim:e0.detector.no_network.tp -->13<!-- /claim --> | <!-- claim:e0.detector.no_network.fn -->0<!-- /claim --> | <!-- claim:e0.detector.no_network.fp -->0<!-- /claim --> | <!-- claim:e0.detector.no_network.tn -->14<!-- /claim --> |
| `no_new_dependencies` | <!-- claim:e0.detector.no_new_dependencies.tp -->11<!-- /claim --> | <!-- claim:e0.detector.no_new_dependencies.fn -->0<!-- /claim --> | <!-- claim:e0.detector.no_new_dependencies.fp -->0<!-- /claim --> | <!-- claim:e0.detector.no_new_dependencies.tn -->13<!-- /claim --> |
| `protected_paths` | <!-- claim:e0.detector.protected_paths.tp -->55<!-- /claim --> | <!-- claim:e0.detector.protected_paths.fn -->0<!-- /claim --> | <!-- claim:e0.detector.protected_paths.fp -->0<!-- /claim --> | <!-- claim:e0.detector.protected_paths.tn -->27<!-- /claim --> |
| `stable_api` | <!-- claim:e0.detector.stable_api.tp -->13<!-- /claim --> | <!-- claim:e0.detector.stable_api.fn -->0<!-- /claim --> | <!-- claim:e0.detector.stable_api.fp -->0<!-- /claim --> | <!-- claim:e0.detector.stable_api.tn -->14<!-- /claim --> |
| `tests_before_done` | <!-- claim:e0.detector.tests_before_done.tp -->11<!-- /claim --> | <!-- claim:e0.detector.tests_before_done.fn -->0<!-- /claim --> | <!-- claim:e0.detector.tests_before_done.fp -->0<!-- /claim --> | <!-- claim:e0.detector.tests_before_done.tn -->11<!-- /claim --> |

**Recall-probe scorer.** Recall probes are scored with keyword groups, one listed rule at a time (D-023, D-030, D-031). The groups were tuned once on the <!-- claim:e0.probe.dev.n -->217<!-- /claim --> answers of the dev split and then frozen, and the test split was scored once; only the test split is reported (D-038). On its <!-- claim:e0.probe.test.n -->245<!-- /claim --> answers, sensitivity was <!-- claim:e0.probe.sensitivity -->0.973<!-- /claim --> [<!-- claim:e0.probe.sensitivity.lo -->0.924<!-- /claim -->, <!-- claim:e0.probe.sensitivity.hi -->0.991<!-- /claim -->] (<!-- claim:e0.probe.sensitivity.k -->108<!-- /claim --> of <!-- claim:e0.probe.sensitivity.n -->111<!-- /claim --> answers that state the rule), specificity <!-- claim:e0.probe.specificity -->0.918<!-- /claim --> [<!-- claim:e0.probe.specificity.lo -->0.859<!-- /claim -->, <!-- claim:e0.probe.specificity.hi -->0.954<!-- /claim -->] (<!-- claim:e0.probe.specificity.k -->123<!-- /claim --> of <!-- claim:e0.probe.specificity.n -->134<!-- /claim --> answers that do not), and accuracy <!-- claim:e0.probe.accuracy -->0.943<!-- /claim --> [<!-- claim:e0.probe.accuracy.lo -->0.906<!-- /claim -->, <!-- claim:e0.probe.accuracy.hi -->0.966<!-- /claim -->] (95% Wilson score intervals). The answers are synthetic, and their labels come from a blind LLM labeling agent that never saw the scorer: they are LLM labels, not human labels, and not answers from any model under test.

**Probe leakage.** Not measured yet: the validation runs it is checked on have not been banked.

**Compaction logs.** Not checked yet: the validation runs whose compaction logs are checked have not been banked.

![Forest plot of the E0 instrument-validity measures with their intervals, against the G5 threshold](figures/e0-validity.svg)

*Figure 1. Instrument validity (E0): detector precision and recall on the seeded logs, and recall-probe scorer agreement on the test split, with 95% Wilson score intervals; the dashed line is the G5 threshold for the detectors. Synthetic data, not model results.*

### 5.3 Family pilot

Status: exploratory. The family pilot has not been banked, so this report has no pilot results.

### 5.4 Power analysis for E1

The E1 power analysis has not run, so this report has no power results.

### 5.5 Main study (E1)

Status: not run. The main study (E1) has not run, so this report has no estimate of constraint half-life, of the effect of compaction, or of differences between rule types.

### 5.6 Production agent (E2, ecological)

Status: not run. E2, the ecological arm with Claude Code, has not run: it needs a machine where the billing guard lets the Claude backend start.

### 5.7 Placement and phrasing (E3)

Status: not run. E3 has not run, so this report has no placement or phrasing results.

### 5.8 Mitigations (E4)

Status: not run. E4 has not run, so this report has no mitigation results.

**Overhead of the shipped tools.** Not measured yet.
<!-- report:end results -->

## 6. Limitations

Several limits are known before any confirmatory result, and the results should be read with them in mind.

**Small models on a CPU.** The controlled subjects are small quantized open-weight models running on four CPU cores, because the study spends no cash. Small models often fail at the work itself: in the pilot, some wrote short or broken code and some ended requests without acting. An agent that does not act at a decision point cannot show whether it keeps the rule, so the family choice requires engagement as well as reaching the first decision point (DECISIONS.md D-042, an amendment made after one pilot session per family and reported as such). Few violations from a weak agent can mean that it keeps rules, or that it never finds the easy route that breaks them. The report keeps these readings apart by showing how often agents acted at decision points next to every violation rate.

**Synthetic, short projects.** The templates are small synthetic Python projects. Sessions are padded to at least twenty requests with neutral read-only requests (D-043), which lengthens sessions without changing what the rules cover, but real projects are larger, messier, and less regular. Decision points are designed to be realistic and to have a reasonable compliant route; they are not drawn from real user sessions.

**The minimal agent is not a production tool.** The controlled agent has a fixed scaffold and harness-side compaction strategies. Production agents compact differently and carry their own system prompts and memory features. The ecological arm (E2) runs Claude Code itself, but it needs a machine where the billing guard passes, and its samples are small by design. Results from the minimal agent should be read as evidence about mechanisms under control, and E2 as a check on whether they appear in a real tool.

**Measurement.** Violations come from deterministic detectors validated on seeded logs, including indirect writes. Detectors see file hashes, commands, manifests, syntax trees, and network attempts logged by the sandbox; a violation that leaves none of these traces would be missed. Recall is scored with keyword groups, validated against labels written by a blind language-model labeler on synthetic answers rather than by people (D-038); the reported agreement is for the held-out half of that set, and the scorer still counts some partial answers and misses some phrasings.

**Mitigations.** The policy guard analyzes tool calls statically. It cannot see a write or a network call made inside a program the agent runs, so the guard arm is an upper bound on what static enforcement blocks, and its refusal messages restate the rule at the moment of an attempted violation (D-039). Hook fixtures were written from the Claude Code documentation rather than captured, so the guard's behavior inside Claude Code is confirmed only by the E2 smoke run.

**Statistics.** Sample sizes are set by compute, with simulated power reported beside them. The hazard model assumes a smooth logistic dependence on the logarithm of position; half-lives beyond the observed positions are model extrapolations, and the report says so where an estimate is not reached within the session cap. Kaplan-Meier curves are descriptive only. Everything outside the preregistered hypotheses is exploratory.

## 7. Ethics and disclosure

The study uses only synthetic projects, mock tools, and disposable sandboxes with no network and no real credentials. Decision points never instruct an agent to break a rule, and no prompt search for rule-dropping behavior was performed. The mitigations are defensive.

## 8. Reproducibility

Every number in this report is produced by a committed script from a banked artifact under `artifacts/`, checked by `make claimcheck` and re-derived by `make freshness`. `make reproduce` rebuilds every figure and table, and `make demo` runs an offline end-to-end example in under three minutes.
