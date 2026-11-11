# Security

## Reporting a vulnerability

Please report security problems privately through GitHub's private vulnerability reporting on this repository ("Security" tab, "Report a vulnerability"). Do not open a public issue for a security problem.

## Scope and design

- The minimal agent's commands, and the harness's own checks, run in a disposable workspace inside the strongest sandbox `halflife.sandbox` finds on the machine: bubblewrap, or unprivileged user, network, mount, and PID namespaces (`unshare-mount`). Both block the network and confine writes to the workspace and a per-run directory. Where neither works, `unshare-net` blocks the network but does not confine writes, and `none` isolates nothing beyond in-process socket refusal and the tool shims, so the project's policy is not to run experiments under it (DECISIONS.md D-019); every run records its method (`sandbox_method`), so a run under `none` would be visible in its provenance. There is no Docker method. Claude Code subjects (E2) run with Claude Code's own sandbox required. Every subject starts from an allowlisted environment with no API keys or tokens.
- Subject agents act only on synthetic projects in disposable workspaces. Their file and shell tools are real inside that workspace; network-facing commands (`curl`, `wget`, `pip`, `npm`, and the network `git` subcommands) are shims that refuse and log. No real credentials, real systems, or harmful payloads are used.
- `halflife-guard` is a defensive policy hook. It is not a security boundary against a determined adversary, and the project builds nothing that helps an agent evade graders, monitors, or safety measures.
- Secrets live only in environment variables. `make secrets-scan` runs in CI and in the ship check. It fails on text in tracked files that matches known credential formats (Anthropic, OpenAI-style, OpenRouter, GitHub, Google, AWS, and Slack keys, and private-key blocks) or that contains the value of a secret-named environment variable. It scans the files at HEAD, not the git history.
