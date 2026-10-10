PY ?= .venv/Scripts/python.exe

.PHONY: help setup data eda smoke baseline tune-smoke tune benchmark stats explain test all clean

help:
	@echo "churn-boosting-benchmark: available targets"
	@echo "  setup       - create .venv and install pinned dependencies"
	@echo "  data        - load, clean and split data (train/test indices)"
	@echo "  eda         - run EDA and save figures"
	@echo "  smoke       - train default models on shared 5-fold CV (train only)"
	@echo "  baseline    - run baseline models with CV"
	@echo "  tune-smoke  - Optuna smoke run (5 trials per model)"
	@echo "  tune        - full Optuna run (100 trials per model)"
	@echo "  benchmark   - final benchmark on holdout test"
	@echo "  stats       - statistical significance tests"
	@echo "  explain     - SHAP explanations"
	@echo "  test        - run pytest"
	@echo "  all         - run full pipeline end to end"
	@echo "  clean       - remove generated artifacts"

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.data > reports/logs/data.log 2>&1
	$(PY) -c "import sys; sys.stdout.write(open('reports/logs/data.log', encoding='utf-8').read())"

eda:
	$(PY) -m src.eda > reports/logs/eda.log 2>&1
	$(PY) -c "import sys; sys.stdout.write(open('reports/logs/eda.log', encoding='utf-8').read())"

smoke:
	$(PY) -m src.pipelines > reports/logs/smoke.log 2>&1
	$(PY) -c "import sys; sys.stdout.write(open('reports/logs/smoke.log', encoding='utf-8').read())"

baseline:
	$(PY) -m src.baseline > reports/logs/baseline.log 2>&1
	$(PY) -c "import sys; sys.stdout.write(open('reports/logs/baseline.log', encoding='utf-8').read())"

tune-smoke:
	$(PY) -m src.tune --smoke

tune:
	$(PY) -m src.tune

benchmark:
	$(PY) -m src.evaluate

stats:
	$(PY) -m src.stats

explain:
	$(PY) -m src.explain

test:
	$(PY) -m pytest -q

all: data eda baseline tune benchmark stats explain test

clean:
	$(PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['reports/figures', 'reports/tables', 'reports/runs']]"
