goal_turn: 0
STATE: ACTIVE

# Status

Updated 2026-10-23 (bootstrap). History lives in docs/LOG.md.

## Gates (from `python tools/ship_check.py --fast`; fast results do not count)

| Gate | State | Evidence |
|---|---|---|
| G1 Build and tests | pending | first fast check runs after the bootstrap commit |
| G2 Static quality | pending | |
| G3 CI parity | pending | |
| G4 Offline demo | pending | |
| G5 Instrument validity | pending | |
| G6 Benchmark complete | pending | |
| G7 Main study | pending | |
| G8 Ecological and placement arms | pending | |
| G9 Mitigations shipped | pending | |
| G10 Claim integrity | pending | |
| G11 Preregistration integrity | pending | |
| G12 Independent review | pending | |
| G13 Documentation | pending | |
| G14 Hygiene | pending | |

## Environment (`make doctor`, 2026-10-23)

- Cloud container, Linux x86_64, 4 CPUs, 15.7 GB RAM, no GPU, 29.9 GB free disk (floor 8 GB).
- Docker CLI yes, daemon no. Sandboxes: `unshare -rn` only (no bubblewrap).
- `claude` CLI 2.1.309 present; Claude backend refused by the billing guard (`ANTHROPIC_BASE_URL`, `CLAUDE_CODE_USE_CCR_V2` set by the cloud harness).
- Local runtimes: none installed (no Ollama, llama.cpp, or transformers).
- Free tier: not confirmed. `ALLOW_PUSH`: not set (see DECISIONS.md D-001). `CAP_SHARE`: 1.
- Recommended `local_model_max_params_b`: 8 (cap unchanged).

## Backends and fallbacks

- Usable now: none for model calls. Next: install a local runtime (M4 in docs/PLAN.md).
- Claude: unusable here; `make review` uses the reviewer subagent; E2 needs Raj.
- Sandbox: `unshare -rn` with a temporary workspace and scrubbed environment; own generated tasks only until write confinement exists (D-003).

## Now

Bootstrap (section 10.1) complete; next heartbeat starts the Loop Protocol.

## Next three actions

1. M1: action-log and session schema, filesystem hashing after every tool call, and the first deterministic detectors with tests.
2. M1: seeded E0 logs including indirect violations (`sed -i`, scripted writes, `git checkout --`, `cp`) and the detector validation artifact.
3. Port `prereg.py`, `claimcheck.py`, `freshness.py`, and gate mutation tests from raise-v1-public with attribution.

## Usage to date

- cash_usd = 0. Claude subject runs 0, single calls 0, review runs 0. Local model calls 0.

## Needs Raj

1. **Claude backend (blocks E2 in G8, and headless review).** This cloud container sets `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2`, so the billing guard refuses Claude children. To run E2, start a session on a machine where `env | grep -E '^(ANTHROPIC_|CLAUDE_CODE_USE_)'` prints nothing and `claude` is logged in to the Max plan.
2. **Push permission (informational).** Commits are pushed to `claude/brave-albattani-c5c7e2` only, per this session's instructions (DECISIONS.md D-001). Say so if you want pushing stopped.

## Blocked

Nothing yet.

## Backlog (suggestions for Raj, not ladder rungs)

1. Publish half-life-bench templates with a canary string and a stable YAML constraint spec so other teams can run their own agents against it.
