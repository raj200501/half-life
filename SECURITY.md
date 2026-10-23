# Security

## Reporting a vulnerability

Please report security problems privately through GitHub's private vulnerability reporting on this repository ("Security" tab, "Report a vulnerability"). Do not open a public issue for a security problem.

## Scope and design

- half-life runs subject agents only in disposable sandboxes: Docker with `--network none` when available, otherwise an OS sandbox that blocks network and confines writes to a temporary workspace, with credentials removed from the environment.
- All tools exposed to subject agents are mock tools. No real credentials, real systems, or harmful payloads are used.
- `halflife-guard` is a defensive policy hook. It is not a security boundary against a determined adversary, and the project builds nothing that helps an agent evade graders, monitors, or safety measures.
- Secrets live only in environment variables. `make secrets-scan` runs in CI and in the ship check and fails on any credential-shaped string or live secret value in tracked files.
