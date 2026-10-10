@echo off
setlocal enabledelayedexpansion
set PY=.venv\Scripts\python.exe

if "%~1"=="" goto help
if /I "%~1"=="help" goto help
if /I "%~1"=="setup" goto setup
if /I "%~1"=="data" goto data
if /I "%~1"=="eda" goto eda
if /I "%~1"=="baseline" goto baseline
if /I "%~1"=="tune-smoke" goto tune_smoke
if /I "%~1"=="tune" goto tune
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
echo   baseline    - run baseline models with CV
echo   tune-smoke  - Optuna smoke run (5 trials per model)
echo   tune        - full Optuna run (100 trials per model)
echo   benchmark   - final benchmark on holdout test
echo   stats       - statistical significance tests
echo   explain     - SHAP explanations
echo   test        - run pytest
echo   all         - run full pipeline end to end
goto end

:setup
python -m venv .venv
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
goto end

:data
%PY% -m src.data
goto end

:eda
%PY% -m src.eda
goto end

:baseline
%PY% -m src.baseline
goto end

:tune_smoke
%PY% -m src.tune --smoke
goto end

:tune
%PY% -m src.tune
goto end

:benchmark
%PY% -m src.evaluate
goto end

:stats
%PY% -m src.stats
goto end

:explain
%PY% -m src.explain
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
