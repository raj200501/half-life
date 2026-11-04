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
.PHONY: help setup-local subject-env $(TARGETS)

help:
	@echo "targets: $(TARGETS)"

setup:
	$(UV) sync --frozen

setup-local:
	$(UV) sync --frozen --extra local

# Python environment for sandboxed subject commands: Python 3.12 and pytest only.
SUBJECT_VENV ?= $(HOME)/.cache/halflife/subject-venv
subject-env:
	$(UV) venv -q --allow-existing $(SUBJECT_VENV) --python 3.12
	$(UV) pip install -q --python $(SUBJECT_VENV)/bin/python "pytest==9.1.1"

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
	@mkdir -p $(dir $(DEMO_OUT))
	@$(RUN) python -m halflife.demo --out $(DEMO_OUT) > /dev/null
	@diff -u demo/golden_output.txt $(DEMO_OUT) && echo "DEMO: output matches demo/golden_output.txt"

prereg-verify:
	@$(RUN) python -m halflife.integrity.prereg

claimcheck:
	@$(RUN) python -m halflife.integrity.claimcheck

freshness:
	@$(RUN) python -m halflife.integrity.freshness

citecheck:
	@$(RUN) python -m halflife.integrity.citecheck

secrets-scan:
	@$(RUN) python -m halflife.secrets_scan .

budget:
	@$(RUN) python -m halflife.budget

review:
	@$(PY) tools/review.py

report:
	@$(RUN) python -m halflife.report

reproduce:
	@$(RUN) python -m halflife.report --reproduce

ship-check:
	@$(PY) tools/ship_check.py
