goal_turn: 2
STATE: ACTIVE

# Status

Updated 2026-10-23 (goal_turn 1). History lives in docs/LOG.md.

## Gates (full `make ship-check` at 0e1a1f3, 2026-10-23: FAIL 3/14)

| Gate | State | Evidence |
|---|---|---|
| G1 Build and tests | PASS | setup ok, 80 tests, core line coverage 98.3% |
| G2 Static quality | PASS | ruff check, ruff format, mypy --strict clean |
| G3 CI parity | FAIL | workflow ok; `make ci` fails at the demo placeholder |
| G4 Offline demo | FAIL | no demo or golden output yet (M2) |
| G5 Instrument validity | FAIL | no E0 artifacts yet (M1, M4) |
| G6 Benchmark complete | FAIL | no bench manifest yet (M3) |
| G7 Main study | FAIL | no E1 artifacts yet (M5) |
| G8 Ecological and placement arms | FAIL | no E2 or E3 yet (M6; E2 needs Raj) |
| G9 Mitigations shipped | FAIL | no E4 yet (M7) |
| G10 Claim integrity | FAIL | no docs/REPORT.md, claimcheck and freshness are placeholders (M8) |
| G11 Preregistration integrity | FAIL | prereg-verify is a placeholder (M1 port) |
| G12 Independent review | FAIL | no review banked yet (M9) |
| G13 Documentation | FAIL | README is still the brief, no docs/REPORT.md (M8) |
| G14 Hygiene | PASS | secrets clean, cash_usd = 0, caps ok, private/ untracked, tree clean |

## Environment (`make doctor`, 2026-10-23)

- Cloud container, Linux x86_64, 4 CPUs, 15.7 GB RAM, no GPU, 29.9 GB free disk (floor 8 GB).
- Docker CLI yes, daemon no. Sandboxes: `unshare -rn` only (no bubblewrap).
- `claude` CLI 2.1.309 present; Claude backend refused by the billing guard (`ANTHROPIC_BASE_URL`, `CLAUDE_CODE_USE_CCR_V2` set by the cloud harness).
- Local runtime: llama-cpp-python 0.3.36 CPU wheel (`make setup-local`); Hugging Face GGUF downloads work through the proxy.
- Subject sandbox Python: `~/.cache/halflife/subject-venv` (3.12, pytest only, no pip).
- Free tier: not confirmed. Launch settings (D-009, D-010): `ALLOW_PUSH=1` (push every commit to this branch), `cap_share = 0.2` in budget.toml.
- Recommended `local_model_max_params_b`: 8 (cap unchanged).

## Backends and fallbacks

- Usable now: none for model calls. Next: install a local runtime (M4 in docs/PLAN.md).
- Claude: unusable here; `make review` uses the reviewer subagent; E2 needs Raj.
- Sandbox: `unshare -rn` with a temporary workspace and scrubbed environment; own generated tasks only until write confinement exists (D-003).

## Now

Phase A and B1 builder workflows are running (one owner per file set): instruments core (workspace, sandbox, tools, scripted runner, detectors), blind E0 seeded cases, integrity tools (prereg, claimcheck, freshness, citecheck, gate mutation tests), stats and SVG plots, model client and pinned local models, citations, mitigation kit, 29 bench templates (three authors), and the E2 Claude Code driver. Written by the owner this turn: schema, design, sessions generator, analysis rows, minimal agent loop, compaction strategies, probes, route checker.

## Next three actions

1. Integrate phase A: wire make targets, run the E0 seeded cases against the detectors, bank artifacts/e0/detector_validation.json.
2. Bench verification CLI (`python -m halflife.bench verify`), manifest, compliant routes and reference runs (G6).
3. Experiment runner over the local backend; family pilot on the 3 pilot templates; E1 preregistration with power analysis.

## Usage to date

- cash_usd = 0. Claude subject runs 0, single calls 0, review runs 0. Local model calls 0.

## Needs Raj

1. **Claude backend (blocks E2 in G8, and headless review).** This cloud container sets `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2`, so the billing guard refuses Claude children. To run E2, start a session on a machine where `env | grep -E '^(ANTHROPIC_|CLAUDE_CODE_USE_)'` prints nothing and `claude` is logged in to the Max plan.
2. **E2 runs on your machine.** E2 needs the `claude` CLI as a subject with a clean billing guard; docs/E2_RUNBOOK.md (being written) will give the exact commands.

## Blocked

Nothing yet.

## Backlog (suggestions for Raj, not ladder rungs)

1. Publish half-life-bench templates with a canary string and a stable YAML constraint spec so other teams can run their own agents against it.
