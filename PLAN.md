# PLAN.md — churn-boosting-benchmark

План работ. Выполняем **по одному этапу**: перечислить файлы → сделать → прогнать
критерий готовности → отметить галочку → стоп до «ок».

## Состояние репозитория (на старте)

- `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` — исходный датасет (7043 строки), целевая
  переменная `Churn`.
- `prompt.md` — ТЗ; `.gitignore` и `.clinerules` — служебные.
- Всё остальное (`src/`, `configs/`, `tests/`, `notebooks/`, `reports/`, `article/`, `Makefile`,
  `requirements.txt`, `README.md`) создаётся в этапе 0.
- Окружение: Python 3.13 в изолированном `.venv`; набор зависимостей — в `requirements.txt`.

## Зафиксированные решения

- **D1** Окружение: изолированный `.venv`, совместимый пиннинг под Python 3.13, проверка
  импорта всех библиотек. Блокирующее условие: если `catboost` или `shap` не ставятся на
  3.13 — остановиться и доложить.
- **D2** Бизнес-допущения: `retention_offer_cost = 50`, `customer_lifetime_value = 500`,
  `retention_success_rate = 0.3`; анализ чувствительности по сетке.
- **D3** Optuna: 100 trials/модель; обязательный smoke-прогон на 5 trials перед полным.
- **D4** Порог по бизнес-стоимости (F1 — побочная метрика), подбор на out-of-fold
  предсказаниях train, не на тесте.
- **D5** MLflow: SQLite backend `reports/mlflow.db`.
- **D6** Второй датасет отложен.

## Допущения

- Python 3.13 в `.venv`; `seed = 42` сквозной.
- Тест (80/20, стратифицированный) читается ровно один раз — на этапе 6; далее только
  сохранённые предсказания `reports/tables/test_predictions.parquet`.
- Early stopping внутри trial — на внутреннем разбиении train-фолда, независимо от
  фолда, по которому считается score.
- Если вывод команды не виден в терминале — писать результат в `reports/logs/*.log` и читать файл.
- Фридман не используем (один датасет). ROC-AUC на Telco ожидаем ~0.84–0.85.

## Хранение артефактов

- Итоговые результаты (метрики, таблицы, предсказания) — в `reports/tables/` (в git).
- Фигуры и графики — в `reports/figures/` (в git).
- Логи запусков (`reports/logs/*.log`) — диагностика, не результат: исключены через `.gitignore`.
- Артефакты MLflow (`reports/runs/`, `reports/mlflow.db`) — исключены через `.gitignore`.
- Промежуточный результат split (`data/processed/`) — исключён через `.gitignore`.

## Итоговое дерево репозитория

```
.
├── README.md
├── PLAN.md
├── prompt.md
├── Makefile
├── make.bat
├── requirements.txt
├── .gitignore
├── configs/        config.yaml, business.yaml, search_spaces.yaml
├── data/
│   ├── raw/        WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/  split_indices.npz
├── src/            config, utils, data, eda, pipelines, baseline, tune,
│                   evaluate, stats, explain
├── notebooks/      01_eda.ipynb, 02_results_figures.ipynb
├── reports/        figures/, tables/, logs/, runs/, mlflow.db
├── article/        article.md
└── tests/          conftest, test_no_leakage, test_data_shape, test_determinism
```

## Зависимости (requirements.txt)

Версии фиксируются фактически установленными в `.venv` под Python 3.13; блокирующая проверка
на этапе 0 — импорт `catboost` и `shap`.

- Данные и математика: `numpy`, `pandas`, `scipy`, `scikit-learn`
- Градиентный бустинг: `xgboost`, `lightgbm`, `catboost`
- Тюнинг и логирование: `optuna`, `mlflow`
- Интерпретация: `shap`
- Визуализация: `matplotlib`, `seaborn`
- Конфигурации: `pyyaml` — чтение `configs/*.yaml`
- Обмен данными: `pyarrow` — parquet (`reports/tables/test_predictions.parquet`)
- Тесты: `pytest`

## Этапы

### Этап 0 — Скелет репозитория и окружение
- [x] Готово
- **Цель:** воспроизводимая структура и зависимости.
- **Файлы:** `requirements.txt`, `Makefile`, `make.bat`, `.gitignore`, `README.md`,
  `PLAN.md`, `configs/{config,business,search_spaces}.yaml`, `src/{__init__,config,utils}.py`.
- **Входы:** ТЗ. **Выходы:** `.venv`, скелет каталогов, `reports/logs/stage0_imports.log`.
- **Готовность:** `.venv\Scripts\python.exe -c "import xgboost,lightgbm,catboost,optuna,shap,mlflow,pytest,matplotlib,seaborn,sklearn,pandas,numpy,scipy"` → OK;
  `make help` / `make.bat help` печатает цели. **Стоп при провале импорта `catboost`/`shap`.**
- **Риски:** несовместимость `catboost`/`shap` с Python 3.13 (блокирующий).

### Этап 1 — Данные: загрузка, очистка, split
- [x] Готово
- **Цель:** один детерминированный источник данных без утечки.
- **Файлы (создать):** `src/data.py`, `tests/test_data_shape.py`; (изменить) `Makefile`, `README.md`.
- **Входы:** `data/raw/*.csv`. **Выходы:** `data/processed/split_indices.npz`,
  `reports/tables/data_summary.csv`, `reports/logs/data.log`.
- **Логика:** drop `customerID`; `TotalCharges` → numeric; заполнить 0 при `tenure==0`;
  target `Churn` → 0/1; `train_test_split` 80/20 stratified seed=42, индексы сохраняются.
- **Готовность:** `make data` печатает формы train/test, долю оттока (~0.265), число
  пропущенных `TotalCharges` (ожидаем 11, все при `tenure==0`), дубликаты, удалённые колонки.
- **Риски:** не читать тест нигде, кроме этапа 6; dtype-нюансы pandas 3.

### Этап 2 — EDA и графики
- [ ] Готово
- **Цель:** обязательные графики и выводы.
- **Файлы:** `src/eda.py`, `notebooks/01_eda.ipynb`; **Выходы:** `reports/figures/eda_*.png`,
  `reports/tables/eda_*.csv`, `reports/logs/eda.log`.
- **Готовность:** `make eda` создаёт 5–7 фигур (дисбаланс; `TotalCharges` vs `tenure==0`;
  `Contract`; `tenure`; `InternetService`; `PaymentMethod`) и печатает выводы по accuracy и заполнению.
- **Риски:** не плодить однотипные графики.

### Этап 3 — Пайплайны A/B и фабрики моделей
- [ ] Готово
- **Цель:** препроцессинг строго внутри `Pipeline`/`ColumnTransformer`; нативные ветки.
- **Файлы:** `src/pipelines.py`, `tests/test_no_leakage.py`, `tests/test_determinism.py`,
  `tests/conftest.py`; (изменить) `Makefile`.
- **Логика:** Pipeline A — OHE + passthrough числовых (+ scaling для LogReg); Pipeline B —
  `cat_features` (CatBoost), `category` dtype (LightGBM), `enable_categorical=True` (XGBoost).
- **Готовность:** `make test` зелёный; smoke обучение дефолтов на CV печатает ROC-AUC.
- **Риски:** заполнение `TotalCharges` и scaling только внутри фолда; совместимость native dtype.

### Этап 4 — Baseline
- [ ] Готово
- **Цель:** опорные числа.
- **Файлы:** `src/baseline.py`; **Выходы:** `reports/tables/baseline.csv`, `reports/logs/baseline.log`.
- **Состав:** `DummyClassifier(prior)`; `LogisticRegression(OHE+scaling)`; xgb/lgbm/catboost с
  дефолтами в обоих пайплайнах; CV на train.
- **Готовность:** `make baseline` печатает таблицу метрик и времени; ROC-AUC ~0.82–0.85.
- **Риски:** результат > 0.85 → искать утечку.

### Этап 5 — Тюнинг Optuna
- [ ] Готово
- **Цель:** подбор гиперпараметров на train через CV.
- **Файлы:** `src/tune.py`; **Выходы:** `reports/tables/best_params.json`,
  `reports/figures/optuna_history_*.png`, `reports/figures/optuna_importance_*.png`,
  `reports/logs/tune_smoke.log`, `reports/logs/tune_full.log`, MLflow `reports/mlflow.db`.
- **Логика:** сначала `--smoke` (5 trials/модель), затем 100 trials/модель; TPE + MedianPruner;
  early stopping на внутреннем разбиении train-фолда; score — на отдельном валидационном фолде.
- **Готовность:** `make tune-smoke` печатает best ROC-AUC по 5 trials; `make tune` — по 100 и
  сохраняет history/importances.
- **Риски:** длительность; несовместимость пространств с версиями.

### Этап 6 — Финальный бенчмарк на отложенном тесте
- [ ] Готово
- **Цель:** единственная честная оценка на тесте.
- **Файлы:** `src/evaluate.py`; **Выходы:** `reports/tables/test_benchmark.csv`,
  `reports/tables/test_predictions.parquet` (кэш для этапа 7), `reports/figures/test_*.png`,
  `reports/logs/benchmark.log`.
- **Логика:** обучение на train; порог по max ожидаемой прибыли на **out-of-fold предсказаниях train**;
  метрики на тесте: ROC-AUC, PR-AUC, F1, recall, precision, Brier + calibration curve, время
  train/inference; `ExpectedProfit(t) = Σ_{p_i ≥ t}[p_i · CLV · success_rate − cost]`;
  sensitivity по сетке `retention_success_rate`.
- **Готовность:** `make benchmark` печатает таблицу метрик/времени и бизнес-таблицу; создан
  `test_predictions.parquet`. **Тест читается только здесь.**
- **Риски:** подбор порога на тесте (запрещено) → жёстко на train OOF.

### Этап 7 — Статистическая значимость
- [ ] Готово
- **Цель:** вывод «разница в пределах шума».
- **Файлы:** `src/stats.py`; **Выходы:** `reports/tables/stats_resampled_ttest.csv`,
  `stats_bootstrap_ci.csv`, `stats_bootstrap_diff.csv`, `stats_mcnemar_holm.csv`, `reports/logs/stats.log`.
- **Вход:** сохранённые `test_predictions.parquet` и результаты CV. **Тест заново не читается.**
- **Логика:** RepeatedStratifiedKFold 5×5 на train; corrected resampled t-test (Nadeau & Bengio);
  bootstrap-CI для ROC-AUC и bootstrap разницы между парами; McNemar на пороге + Holm; effect size.
  Фридман не используем.
- **Готовность:** `make stats` печатает p-value, скорректированные p-value, CI и текстовый вывод по парам.
- **Риски:** корректность поправки на зависимость фолдов; интерпретация узких CI.

### Этап 8 — SHAP и интерпретация
- [ ] Готово
- **Цель:** объяснимость и согласованность моделей.
- **Файлы:** `src/explain.py`; **Выходы:** `reports/figures/shap_beeswarm_*.png`, `shap_bar_*.png`,
  `shap_dependence_{tenure,MonthlyCharges,Contract}_*.png`, `shap_waterfall_case.png`,
  `reports/tables/feature_importance_spearman.csv`, `reports/logs/explain.log`.
- **Логика:** `TreeExplainer` на тесте; dependence plots; Спирмен между важностями моделей; оговорка
  о корреляции `TotalCharges`/`tenure`/`MonthlyCharges`.
- **Готовность:** `make explain` создаёт фигуры и таблицу корреляций, печатает вывод.
- **Риски:** native dtype vs матрица для SHAP.

### Этап 9 — Тесты (pytest) как gate
- [ ] Готово
- **Цель:** ≥3 теста (нет утечки, форма данных, детерминизм).
- **Файлы:** `tests/{conftest,test_no_leakage,test_data_shape,test_determinism}.py`.
- **Готовность:** `make test` → `pytest -q` все зелёные.
- **Риски:** детерминизм при многопоточности → фиксируем `n_jobs`/`random_state`.

### Этап 10 — Статья, README, финализация
- [ ] Готово
- **Цель:** академический текст только с реальными числами.
- **Файлы:** `article/article.md`, `README.md`, `PLAN.md`.
- **Готовность:** `make all` проходит end-to-end; числа совпадают с `reports/tables/*.csv`.
- **Риски:** расхождение текста и артефактов → все числа из CSV.

## Порядок зависимостей

```
0 (env/skeleton)
└─> 1 (data+split) ──> 2 (EDA)
└─> 3 (pipelines) ──> 4 (baseline) ──> 5 (tune) ──> 6 (benchmark) ──> 7 (stats)
                                                          └──────> 8 (SHAP) ──> 10 (article/README)
9 (tests) — инкрементально (1→shape, 3→leakage/determinism), gate после 3 и далее.
```

## Ключевые риски (сводно)

- **Блокирующий:** `catboost`/`shap` не ставятся на Python 3.13 → остановиться и доложить (этап 0).
- **Утечка данных:** препроцессинг только внутри `Pipeline`/`ColumnTransformer` и внутри фолдов.
- **Раннее касание теста:** тест читается ровно один раз на этапе 6; далее только
  сохранённый `reports/tables/test_predictions.parquet`.
- **Порог на тесте** подбирать запрещено: только на out-of-fold предсказаниях train (этап 6).
- **Переобучение при тюнинге:** 100 trials × 2 пайплайна × 3 модели; early stopping и score
  на независимых разбиениях train-фолда (этап 5).
- **Длительность Optuna** → обязательный smoke на 5 trials перед полным прогоном (этап 5).
- **Категориальные dtype** в native-пайплайне B могут расходиться между версиями библиотек (этап 3).
- **Отсутствие `make` в Windows** → `make.bat` и дублирующие команды в README (этап 0).

