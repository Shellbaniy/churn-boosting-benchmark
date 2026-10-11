@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set PY=.venv\Scripts\python.exe
set "SKIP_TUNE=%SKIP_TUNE: =%"

if "%~1"=="" goto help
if /I "%~1"=="help" goto help
if /I "%~1"=="setup" goto setup
if /I "%~1"=="data" goto data
if /I "%~1"=="eda" goto eda
if /I "%~1"=="smoke" goto smoke
if /I "%~1"=="baseline" goto baseline
if /I "%~1"=="tune-smoke" goto tune_smoke
if /I "%~1"=="tune" goto tune
if /I "%~1"=="tune-one" goto one
if /I "%~1"=="benchmark" goto benchmark
if /I "%~1"=="stats" goto stats
if /I "%~1"=="explain" goto explain
if /I "%~1"=="test" goto test
if /I "%~1"=="all" goto all
echo Unknown target: %~1
goto help

:help
echo churn-boosting-benchmark: available targets
echo   setup       - create .venv and install pinned dependencies
echo   data        - load, clean and split data
echo   eda         - run EDA and save figures
echo   smoke       - train default models on shared 5-fold CV (train only)
echo   baseline    - run baseline models with CV
echo   tune-smoke  - Optuna smoke run (5 trials per model)
echo   tune        - full Optuna run (100 trials per model); долгий, ~81 мин (SKIP_TUNE=1 пропускает)
echo   tune-one    - one study: make.bat tune-one xgb A 100 [tag]
echo   benchmark   - final benchmark on holdout test
echo   stats       - statistical significance tests
echo   explain     - SHAP explanations
echo   test        - run pytest
echo   all         - full pipeline: data, eda, baseline, tune, benchmark, stats, explain, test
echo   SKIP_TUNE=1 - set SKIP_TUNE=1 (or 'make.bat tune SKIP_TUNE=1') to skip tuning and reuse
echo                 the saved reports\tables\best_params.json
echo.
echo Внимание: all включает долгий тюнинг: сумма duration_sec по 13 study в
echo reports\tables\best_params.json = 4838.7 с (~81 мин), из них catboost_B 2561.8 с
echo и catboost_B_wide 1492.6 с. Быстрый прогон: set "SKIP_TUNE=1" ^&^& make.bat all
goto end

:setup
python -m venv .venv
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
goto end

:data
%PY% -m src.data > reports\logs\data.log 2>&1
type reports\logs\data.log
goto end

:eda
%PY% -m src.eda > reports\logs\eda.log 2>&1
type reports\logs\eda.log
goto end

:smoke
%PY% -m src.pipelines > reports\logs\smoke.log 2>&1
type reports\logs\smoke.log
goto end

:baseline
%PY% -m src.baseline > reports\logs\baseline.log 2>&1
type reports\logs\baseline.log
goto end

:tune_smoke
%PY% -u -m src.tune --smoke > reports\logs\tune_smoke.log 2>&1
type reports\logs\tune_smoke.log
goto end

:tune
if "%SKIP_TUNE%"=="1" goto tune_skip
if /I "%~2"=="SKIP_TUNE" if "%~3"=="1" goto tune_skip
%PY% -u -m src.tune > reports\logs\tune_full.log 2>&1
type reports\logs\tune_full.log
goto end

:tune_skip
echo SKIP_TUNE=1: тюнинг пропущен, используется сохранённый reports\tables\best_params.json
%PY% -X utf8 -c "import pathlib,sys; p=pathlib.Path('reports/tables/best_params.json'); sys.exit('нет ' + str(p) + ': сначала выполните make.bat tune') if not p.exists() else print('использую ' + str(p) + ' (' + str(p.stat().st_size) + ' байт)')"
goto end

:one
set TAGARG=
if not "%~5"=="" set TAGARG=--tag %~5
%PY% -u -m src.tune --model %~2 --variant %~3 --trials %~4 %TAGARG%
goto end

:benchmark
%PY% -u -m src.evaluate > reports\logs\benchmark.log 2>&1
type reports\logs\benchmark.log
goto end

:stats
%PY% -u -m src.stats > reports\logs\stats.log 2>&1
type reports\logs\stats.log
goto end

:explain
%PY% -u -m src.explain > reports\logs\explain.log 2>&1
type reports\logs\explain.log
goto end

:test
%PY% -m pytest -q
goto end

:all
call make.bat data
call make.bat eda
call make.bat baseline
call make.bat tune
call make.bat benchmark
call make.bat stats
call make.bat explain
call make.bat test
goto end

:end
endlocal
