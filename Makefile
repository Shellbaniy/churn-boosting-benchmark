PY ?= .venv/Scripts/python.exe
MODEL ?= xgb
VARIANT ?= A
TRIALS ?= 100
TAG ?=

.PHONY: help setup data eda smoke baseline tune-smoke tune tune-one benchmark stats explain test all clean

help:
	@echo "churn-boosting-benchmark: available targets"
	@echo "  setup       - create .venv and install pinned dependencies"
	@echo "  data        - load, clean and split data (train/test indices)"
	@echo "  eda         - run EDA and save figures"
	@echo "  smoke       - train default models on shared 5-fold CV (train only)"
	@echo "  baseline    - run baseline models with CV"
	@echo "  tune-smoke  - Optuna smoke run (5 trials per model)"
	@echo "  tune        - full Optuna run (100 trials per model); долгий, ~81 мин (SKIP_TUNE=1 пропускает)"
	@echo "  tune-one    - one study: make tune-one MODEL=xgb VARIANT=A TRIALS=100 [TAG=wide]"
	@echo "  benchmark   - final benchmark on holdout test"
	@echo "  stats       - statistical significance tests"
	@echo "  explain     - SHAP explanations"
	@echo "  test        - run pytest"
	@echo "  all         - full pipeline: data, eda, baseline, tune, benchmark, stats, explain, test"
	@echo "  clean       - remove generated artifacts"
	@echo ""
	@echo "Внимание: all включает долгий тюнинг. Сумма duration_sec по 13 study в"
	@echo "reports/tables/best_params.json = 4838.7 с (~81 мин), из них catboost_B 2561.8 с"
	@echo "и catboost_B_wide 1492.6 с. Быстрый прогон без тюнинга: make all SKIP_TUNE=1"
	@echo "(тогда тюнинг пропускается и берётся сохранённый reports/tables/best_params.json)."

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
	$(PY) -u -m src.tune --smoke > reports/logs/tune_smoke.log 2>&1
	$(PY) -c "import sys; sys.stdout.write(open('reports/logs/tune_smoke.log', encoding='utf-8').read())"

tune:
ifeq ($(SKIP_TUNE),1)
	@$(PY) -X utf8 -c "import pathlib, sys; p = pathlib.Path('reports/tables/best_params.json'); sys.exit(f'нет {p}: сначала выполните make tune') if not p.exists() else print(f'SKIP_TUNE=1: тюнинг пропущен, использую сохранённый {p} ({p.stat().st_size} байт)')"
else
	$(PY) -u -m src.tune > reports/logs/tune_full.log 2>&1
	$(PY) -c "import sys; sys.stdout.write(open('reports/logs/tune_full.log', encoding='utf-8').read())"
endif

tune-one:
	$(PY) -u -m src.tune --model $(MODEL) --variant $(VARIANT) --trials $(TRIALS) $(if $(TAG),--tag $(TAG))

benchmark:
	$(PY) -u -m src.evaluate > reports/logs/benchmark.log 2>&1
	$(PY) -c "import sys; sys.stdout.write(open('reports/logs/benchmark.log', encoding='utf-8').read())"

stats:
	$(PY) -u -m src.stats > reports/logs/stats.log 2>&1
	$(PY) -c "import sys; sys.stdout.write(open('reports/logs/stats.log', encoding='utf-8').read())"

explain:
	$(PY) -u -m src.explain > reports/logs/explain.log 2>&1
	$(PY) -c "import sys; sys.stdout.write(open('reports/logs/explain.log', encoding='utf-8').read())"

test:
	$(PY) -m pytest -q

all: data eda baseline tune benchmark stats explain test

clean:
	$(PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['reports/figures', 'reports/tables', 'reports/runs']]"
