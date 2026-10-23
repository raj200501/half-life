# half-life developer commands (section 10.1 step 7 of docs/BUILD_BRIEF.md).
# Every target exists from day one. Targets whose feature is not built yet call
# tools/not_ready.py, which exits non-zero, so no gate can pass on a placeholder.

SHELL := /bin/bash
.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run --frozen --quiet
PY ?= python3
COV_JSON ?= .cache/coverage.json
DEMO_OUT ?= .cache/demo_output.txt

TARGETS := setup doctor test lint typecheck ci demo prereg-verify claimcheck freshness \
	citecheck secrets-scan budget review report reproduce ship-check
.PHONY: help $(TARGETS)

help:
	@echo "targets: $(TARGETS)"

setup:
	$(UV) sync --frozen

doctor:
	@$(RUN) python -m halflife.doctor

test:
	@mkdir -p $(dir $(COV_JSON))
	$(RUN) pytest --cov=halflife --cov-report=term-missing \
		--cov-report=json:$(COV_JSON) --cov-fail-under=85

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

typecheck:
	$(RUN) mypy --strict src/halflife

ci: lint typecheck test demo

demo:
	@$(PY) tools/not_ready.py demo "offline demo from recorded fixtures, gate G4"

prereg-verify:
	@$(PY) tools/not_ready.py prereg-verify "preregistration chain, gate G11"

claimcheck:
	@$(PY) tools/not_ready.py claimcheck "result-number provenance gate, G10"

freshness:
	@$(PY) tools/not_ready.py freshness "re-derive every live number, G10"

citecheck:
	@$(PY) tools/not_ready.py citecheck "validate artifacts/citecheck.json, G13"

secrets-scan:
	@$(RUN) python -m halflife.secrets_scan .

budget:
	@$(RUN) python -m halflife.budget

review:
	@$(PY) tools/review.py

report:
	@$(PY) tools/not_ready.py report "figures and tables from banked artifacts"

reproduce:
	@$(PY) tools/not_ready.py reproduce "rebuild every figure from artifacts"

ship-check:
	@$(PY) tools/ship_check.py
