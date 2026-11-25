# half-life: one-page summary (draft for Raj to review; not sent)

**The problem.** Coding agents are handed rules at the start of a session: do not edit the tests, do not add dependencies, keep the public API, stay inside the workspace. Then they work for hundreds of tool calls, and their context is compacted along the way. Whether the rules still govern what the agent does an hour later is rarely measured.

**What half-life is.** A measurement instrument and a mitigation kit for rule persistence in coding agents.

- **half-life-bench:** thirty small synthetic Python projects, each with two or three rules of four kinds (prohibitions, invariants, procedures, resource limits) and decision points where breaking a rule is the shortcut and a reasonable compliant route exists. Every template ships a verified compliant route and a reference run, and carries a canary string so it can be kept out of training data.
- **A controlled harness:** sessions whose decision-point positions and compaction schedule are randomized, a sham compaction that keeps the rule text verbatim (so losing the text can be told apart from the disruption of compaction), sandboxed tool calls, and forked recall probes that never enter the main run.
- **Detectors that do not take the agent's word for it:** they decide from file hashes after every tool call, commands, manifests, syntax trees, and network logs, and catch indirect writes such as `sed -i`, `tee`, `cp`, and `git checkout --`. They were validated on seeded logs before any model ran; precision and recall, with intervals, are on the README's first screen.
- **A mitigation kit for Claude Code:** `halflife-guard`, a PreToolUse hook that refuses tool calls breaking a YAML rule spec; `halflife-pin`, which renders the same spec into CLAUDE.md or AGENTS.md; re-injection after compaction; and a rule-preserving compaction prompt. The overhead of the guard and the pin is measured.

**Why it is credible.** The study is preregistered: hypotheses, models, exclusions, sample size, and stopping rules were frozen and published before any main-study data, with the analysis code, plans, and model files pinned by hash and the run settings and sandbox fixed. Every number in the README and the report resolves to a banked artifact through a script, so none can be typed by hand. The project's fourteen ship gates are code with mutation tests, five independent adversarial reviews and two statistics audits ran before any confirmatory data, and every changed claim is logged publicly. The preregistration states in advance what the design cannot show.

**Status.** The instrument is validated, the benchmark is complete, and the main study (two local open-weight model families, ninety-six sessions) is preregistered with its first session run; it needs a few days of CPU to finish. The Claude Code arm has its driver, fixtures, and runbook ready and needs a machine where the Claude billing guard passes. There is no estimate of rule half-life yet; it will be reported whatever it shows, including a null.

**Why it matters.** As agents run longer and unattended, "does it still follow the rules?" becomes the question that decides whether teams can trust them with real repositories. half-life turns that question into a number with an interval, and its guard turns the rules into enforcement for the cases that matter most.

**Next.** Finish E1, run the Claude Code arm, then measure placement and phrasing of rules (E3) and the mitigations' effect and cost on held-out projects (E4).

Repository: https://github.com/raj200501/half-life (MIT).
