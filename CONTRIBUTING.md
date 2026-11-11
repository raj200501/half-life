# Contributing

## Setup

```bash
make setup     # uv sync --frozen
make doctor    # what this machine can run
make ci        # lint, typecheck, test, offline demo
```

## Rules for changes

- Python 3.11+, typed (`mypy --strict` clean), formatted with ruff, with docstrings and tests for every public function. Coverage of core modules stays at or above 85%.
- Every `type: ignore` carries a reason comment: `# type: ignore[code]  # reason`.
- No result number is typed by hand. Numbers in README.md, docs/REPORT.md, figures, and tables come from banked artifacts under `artifacts/` through committed scripts, and `make claimcheck` checks this.
- Confirmatory analyses follow a preregistration in `prereg/` committed before the data. Changes are new dated amendment files; frozen files are never edited.
- Synthetic, procedurally generated, and mock data is labeled as such everywhere.
- Null and negative results are reported with the same prominence as positive ones. Changed or withdrawn claims get a dated entry in CORRECTIONS.md.
- No secrets in files, logs, artifacts, or command lines.
- Plain writing: no hype words and no em dashes.

## Checks

`make ship-check` verifies all release gates on a fresh clone. `python tools/ship_check.py --fast` is a quicker check that does not count toward a release.
