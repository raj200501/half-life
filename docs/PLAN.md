# Plan

Milestones mapped to the section 6 gates, with risks, environment-dependent fallbacks, and usage estimates. The environment facts come from `make doctor` on 2026-10-23 (see STATUS.md); re-run it whenever the machine changes and update this plan.

## Environment as of bootstrap

- Cloud container: 4 CPUs, about 15.7 GB RAM, no GPU, about 30 GB free disk (floor 8 GB).
- Docker CLI present, no daemon. No bubblewrap. `unshare -rn` (user plus network namespace) works.
- No Ollama, llama.cpp, or transformers installed yet.
- `claude` CLI 2.1.309 present, but the billing guard refuses it: the cloud harness sets `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2`. Claude cannot be a subject or a headless reviewer from this machine.
- No free-tier keys confirmed. `ALLOW_PUSH` not set (see DECISIONS.md D-001 for the push rule in this session).

## Milestones

| # | Milestone | Gates | Main work |
|---|---|---|---|
| M0 | Bootstrap | G2, G3 (partial), G14 | Scaffold, doctor, budget, secrets scan, ship check, state files. Done 2026-10-23. |
| M1 | Instruments | G1, G5 (detectors) | Session and action-log schema; filesystem hashing after every tool call; `detect` (paths, commands, manifests, AST API diff, sandbox network log); seeded logs with indirect violations (`sed -i`, scripted writes, `git checkout --`, `cp`); E0 precision and recall with Wilson intervals. Port `prereg.py`, `claimcheck.py`, `freshness.py`, and the gate mutation tests from raise-v1-public (Apache-2.0, with attribution). |
| M2 | Offline demo, v0.1 | G4 | `make demo` replays recorded fixtures through detect, compaction logs, probes, and stats, and matches `demo/golden_output.txt`; every number in it labeled synthetic. |
| M3 | Benchmark | G6 | `sessions` generator; at least 24 templates over prohibitions, invariants, procedures, and resource limits; a scripted compliant route and a reference run per template; 3 extra templates reserved for pilots and a held-out set for E4. |
| M4 | Local backend and minimal agent | G5 (probes, compaction) | Install a local runtime (llama.cpp or Ollama) within the disk floor; pin model, revision, quantization, and file hash; client wrapper (caps, ledger, content-hash cache, `HALFLIFE_NO_MODEL_CALLS`); `minimal-agent`; compaction strategies (full summary, sliding window, structured state, constraint-pinned) plus the sham control; forked recall probes with a leakage test. |
| M5 | Preregistration and E1, v0.5 | G7, G11 | Pilot candidate families on the 3 pilot templates (keep families reaching the first decision point in at least 80% of pilot sessions); power analysis under the caps; `prereg/0001-e1.md` frozen before data; E1 in the background, one model-heavy run at a time; `stats` (discrete-time hazard, half-life with bootstrap intervals, descriptive Kaplan-Meier). |
| M6 | Placement, phrasing, and ecological arms | G8 | E3 on local models (system prompt, first user message, pinned file, periodic reminder; bare rule versus rule plus reason). E2 needs a machine where the Claude backend passes the billing guard (Needs Raj). |
| M7 | Mitigations | G9 | `halflife-guard` (PreToolUse hook enforcing a YAML spec), `halflife-pin` (CLAUDE.md and AGENTS.md templates), constraint-preserving compaction prompt, re-injection hook; check current Claude Code hook events first; E4 on held-out templates with costs. |
| M8 | Claims, citations, report | G10, G13 | `claimcheck`, `freshness`, `citecheck` (citation-checker subagent banks `artifacts/citecheck.json`), `make report` figures, docs/REPORT.md (4,000 to 8,000 words), final README per section 11. |
| M9 | Review and release, v1.0 | G12, all | Reviewer subagent passes; final sequence in section 10.6. |

## Critical path

M1 and M4 gate everything empirical. E1 is the longest step (many CPU hours), so the order is: M1, M4 (minimal), M3 templates enough for the pilot, M5 preregistration, then E1 in the background while M2, M3, M7, and M8 proceed in the foreground.

## Risks and fallbacks

- **CPU-only inference is slow.** A 150-tool-call session with a small quantized model on 4 CPUs may take most of an hour. Fallbacks: the smallest model families that still reach decision points in the pilot, a short context budget (which also forces compaction), batched sessions in the background, and E3 and E4 sized from what E1 actually costs.
- **The container is ephemeral.** Background runs can die when the container is reclaimed. Every run writes resumable per-session results under `runs/<id>/` and banks completed sessions to `artifacts/` with a commit, so a lost container loses at most the sessions in flight.
- **No Docker daemon or bubblewrap.** The sandbox is `unshare -rn` (no network) plus a temporary workspace with a scrubbed environment. Write confinement through a mount namespace or an installed bubblewrap is a M1 task; until then only this project's own generated tasks run (DECISIONS.md D-003).
- **Claude backend unavailable here.** E2 and headless `make review` need Raj to run from a machine with a clean billing guard. The reviewer subagent covers G12 meanwhile. If E2 cannot run, G8 stays failing and is listed under Needs Raj.
- **Model downloads.** Network access to model hosts through the proxy is unverified; check in M4 and record the result. Respect the 8 GB disk floor.
- **Citations.** Some 2026 references may not open from this network; the citation-checker uses official arXiv copies and lists failures under Needs Raj.

## Usage estimates

- Claude: zero subject runs from this machine (backend refused). Reviews run through the subagent: about one per milestone, so about 10 in total, inside `claude_review_runs_per_day = 6`.
- Local models: E1 at 2 or more families and at least 40 sessions per family reaching a decision point, plus pilots, E3, and E4. Exact counts come from the power analysis in M5 and are recorded in the preregistration, not here.
- Cash: 0.
