# churn-boosting-benchmark

Сравнительный анализ алгоритмов градиентного бустинга XGBoost, LightGBM и CatBoost
на задаче прогнозирования оттока клиентов (Telco Customer Churn).

## Идея

На датасете такого размера (7043 строки) разница между boosting-алгоритмами, скорее
всего, находится в пределах статистического шума. Вывод строится на статистических
тестах, а не на разнице метрик в третьем знаке.

## Главные выводы

1. **По ROC-AUC бустинги не отличаются от logreg после поправки на множественные
   сравнения.** Bootstrap на тесте: xgb_A_tuned − logreg_tuned = +0.0064,
   CI [0.0003; 0.0125], p = 0.0412, p_holm = 0.1648 (`stats_bootstrap_diff.csv`);
   corrected resampled t-test по CV: p = 0.0189, p_holm = 0.1134
   (`stats_resampled_ttest.csv`).
2. **Между тремя бустингами разницы нет.** ROC-AUC на тесте: lgbm_B_tuned 0.8475,
   xgb_A_tuned 0.8474, catboost_A_tuned 0.8472, catboost_B_tuned 0.8463,
   lgbm_A_tuned 0.8459 (`test_benchmark.csv`); bootstrap p ≥ 0.2292
   (`stats_bootstrap_diff.csv`), McNemar p_holm ≥ 0.068 (`stats_mcnemar_holm.csv`).
3. **PR-AUC — единственная метрика с устойчивым перевесом бустингов:** 0.6675 у
   xgb_A_tuned против 0.6271 у logreg_tuned (`test_benchmark.csv`); разница +0.0391,
   CI [0.0171; 0.0609], p = 0.0008, p_holm = 0.0016 (`stats_bootstrap_diff.csv`).
   Вариант A против B разницы не даёт: p_holm ≥ 0.1094 (`stats_mcnemar_holm.csv`).
4. **Практическая разница мала, польза — в самом факте скоринга.** Прибыль на клиента
   при пороге из OOF: 11.92 у.е. у xgb_A_tuned, 10.72 у logreg_tuned, −10.18 у
   «контактировать всех» (`test_benchmark.csv`), CI bootstrap для xgb_A_tuned
   [9.51; 14.30] (`stats_bootstrap_ci.csv`). SHAP: у всех пяти объясняемых моделей в
   топ-3 входят Contract, tenure и InternetService или MonthlyCharges
   (`shap_top_features.csv`), ρ Спирмена между важностями 0.56–0.95
   (`shap_spearman.csv`).

## Структура репозитория

```
configs/           config.yaml, business.yaml, search_spaces.yaml
src/               config, utils, data, eda, pipelines, baseline, tune, evaluate, stats, explain
tests/             conftest.py + 7 файлов тестов (shape, determinism, leakage, evaluate, explain, stats, tuning)
notebooks/         01_eda.ipynb
reports/figures/   фигуры EDA, Optuna, бенчмарка, статистики и SHAP (в git)
reports/tables/    итоговые таблицы и предсказания (в git)
reports/logs/      логи прогонов (в .gitignore)
article/           текст статьи (пока пусто)
data/raw/          исходный CSV (в .gitignore)
data/processed/    split_indices.npz (в .gitignore)
Makefile, make.bat, requirements.txt, README.md, PLAN.md, prompt.md
```

## Установка

Python 3.13, изолированное окружение `.venv`, версии зависимостей зафиксированы в
`requirements.txt` (ключевые: `xgboost==3.4.1`, `lightgbm==4.7.0`, `catboost==1.2.10`,
`scikit-learn==1.9.1`, `shap==0.53.0`, `optuna==5.0.0`, `pandas==3.0.6`, `numpy==2.5.3`).

```bash
python -m venv .venv
# Windows
.venv/Scripts/python.exe -m pip install -r requirements.txt
# Linux/macOS
.venv/bin/python -m pip install -r requirements.txt
```

То же одной командой: `make setup` (Windows: `make.bat setup`).

## Данные

Используется Telco Customer Churn (Kaggle, набор `blastchar/telco-customer-churn`).
Скачанный CSV нужно положить в `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`; в
репозитории файла нет (`data/raw/` в `.gitignore`), условия использования — на странице
набора.

Что получается после `make data` (`reports/tables/data_summary.csv`, `reports/logs/data.log`):
7043 строки, 21 колонка на входе, после удаления `customerID` — 20 колонок и 19 признаков;
доля оттока 0.2654; 11 пропусков `TotalCharges`, все при `tenure == 0` (заполнены нулём);
22 полных дубликата (не удаляются); train 5634 строки / test 1409 строк, стратифицированный
split 80/20, `seed = 42`.

## Как воспроизвести

`make all` проходит шаги по порядку: `data → eda → baseline → tune → benchmark → stats →
explain → test`. Время по шагам (числа — из артефактов; у шагов без wall-clock в логе
время указано качественно):

| Шаг | Что делает | Время |
| --- | --- | --- |
| `make data` | загрузка, очистка, split | секунды (wall-clock в логе не пишется) |
| `make eda` | EDA и 6 фигур (`reports/logs/eda.log`: «Сохранено фигур: 6») | секунды (wall-clock в логе не пишется) |
| `make baseline` | 8 конфигураций на 5 фолдах | секунды: обучение 0.03–4.07 с/фолд (`baseline.log`) |
| `make tune` | Optuna: 13 study по 100 trials | **долго:** Σ `duration_sec` = 4838.7 с ≈ 81 мин (`best_params.json`) |
| `make benchmark` | финальный бенчмарк на тесте | обучение на train 0.0–46.9 с/модель, инференс 0.0116–0.0406 с (`test_benchmark.csv`) |
| `make stats` | CV + t-test, bootstrap, McNemar | CV-оценки переиспользуются из `stats_cv_scores.csv` (175 строк) |
| `make explain` | обучение 5 моделей + SHAP на тесте | обучение 0.1–1.5 с/модель, инференс 0.0118–0.0406 с (`test_benchmark.csv`) |
| `make test` | `pytest -q` | секунды |

Основное время съедает тюнинг, и почти всё — CatBoost с категориями:
`catboost_B` 2561.8 с и `catboost_B_wide` 1492.6 с из 4838.7 с (`best_params.json`).
Быстрый прогон без тюнинга (используется уже сохранённый `reports/tables/best_params.json`):

```bash
make all SKIP_TUNE=1                   # Linux/macOS
set "SKIP_TUNE=1" && make.bat all      # Windows (cmd)
make.bat all SKIP_TUNE=1               # Windows (cmd), вариант с аргументом
```

Windows: вместо `make` используйте `make.bat` (`make.bat help` печатает цели); на
Linux/macOS — `make help`. `SKIP_TUNE=1` работает и как переменная окружения, и как
аргумент (`make.bat tune SKIP_TUNE=1`). Отдельные шаги: `make data`, `make eda`,
`make baseline`, `make tune-smoke` (5 trials), `make tune`, `make benchmark`,
`make stats`, `make explain`, `make test`. Предупреждение о длительности есть в
`make help` / `make.bat help`.

## Результаты на тесте

Тест: 1409 строк, 374 клиента с оттоком, доля оттока 0.2654 (`test_benchmark.csv`).
Порог — из OOF-предсказаний train, не с теста. Все числа — из `reports/tables/test_benchmark.csv`:

| Модель (tuned) | ROC-AUC | PR-AUC | Brier | Порог | Прибыль на клиента, у.е. | Контактов |
| --- | --- | --- | --- | --- | --- | --- |
| xgb_A_tuned | 0.8474 | 0.6675 | 0.1349 | 0.36 | 11.92 | 474 |
| lgbm_B_tuned | 0.8475 | 0.6626 | 0.1350 | 0.32 | 11.99 | 508 |
| xgb_B_tuned | 0.8473 | 0.6667 | 0.1349 | 0.35 | 11.96 | 485 |
| catboost_A_tuned | 0.8472 | 0.6669 | 0.1350 | 0.36 | 11.46 | 457 |
| lgbm_A_tuned | 0.8459 | 0.6679 | 0.1355 | 0.37 | 11.89 | 460 |
| catboost_B_tuned | 0.8463 | 0.6678 | 0.1350 | 0.33 | 11.46 | 505 |
| logreg_tuned | 0.8410 | 0.6271 | 0.1387 | 0.32 | 10.72 | 511 |

Опорные точки из того же файла: `dummy` — ROC-AUC 0.5000, PR-AUC 0.2654, Brier 0.1950;
«не контактировать никого» — 0.0 у.е. на клиента; «контактировать всех» — −10.18 у.е.
Bootstrap-CI ROC-AUC на тесте у всех tuned-моделей лежит внутри [0.818; 0.868]
(`stats_bootstrap_ci.csv`), то есть наблюдаемые различия в 0.0001–0.0065 внутри шума.

## Статистическая значимость

Числа — из `reports/tables/stats_resampled_ttest.csv`, `stats_bootstrap_diff.csv`,
`stats_mcnemar_holm.csv`, `stats_bootstrap_ci.csv`, `stats_cv_scores.csv`.

- CV: RepeatedStratifiedKFold 5×5, 25 одинаковых разбиений для всех моделей, 175 строк оценок
  (7 моделей); ROC-AUC: 0.8489–0.8506 у бустингов против 0.8460 у logreg.
- Corrected resampled t-test (Nadeau & Bengio) по 15 парам, Holm внутри трёх семейств
  (бустинг vs logreg, вариант A vs B, алгоритм vs алгоритм): после поправки ни одна пара не
  значима по ROC-AUC (минимальный p_holm = 0.1134) и по PR-AUC (минимальный p_holm = 0.1673).
- Bootstrap на тесте: ROC-AUC-разница бустингов с logreg не значима после Holm (p_holm =
  0.1296–0.1896), PR-AUC-разница значима (p_holm = 0.0012–0.0032), прибыль на клиента
  значимо выше у xgboost/lightgbm (p_holm = 0.0144–0.0252), у catboost — нет (p_holm = 0.172).
- McNemar exact на пороге: бустинг vs logreg значимо у 5 пар из 6 (p_holm от 4.3e-05 до
  0.0117), кроме catboost_B (p_holm = 0.0722); между бустингами различий нет (p_holm ≥ 0.068).
- Вариант A против B (label/OHE против native categorical) не различается нигде:
  p_holm ≥ 0.952 в t-test, ≥ 0.109 в McNemar.
- Тест Фридмана не использовался: датасет один.

## Ограничения

- **Тюнинг и сравнение на одних и тех же данных.** Гиперпараметры подбирались на train
  (каждый study — 100 trials; `best_params.json`), там же считались и CV-оценки сравнения,
  поэтому итоговые метрики слегка оптимистичны. Nested CV не делали — это возможное улучшение.
- **25 перекрывающихся разбиений.** CV-оценки получены RepeatedStratifiedKFold 5×5
  (25 разбиений; `stats_cv_scores.csv`), они зависимы. Corrected resampled t-test учитывает
  это лишь частично, поэтому p-value по CV нельзя читать как по независимым выборкам.
- **Тест — одна выборка.** 1409 строк, 374 уходящих клиента. Bootstrap-CI ROC-AUC шириной
  около 0.043 (`stats_bootstrap_ci.csv`) перекрывает все наблюдаемые различия
  (0.0001–0.0065, `test_benchmark.csv`), поэтому «победитель по третьему знаку» не определён.
- **Бизнес-параметры условные.** `retention_offer_cost = 50`, `customer_lifetime_value = 500`,
  `retention_success_rate = 0.3` (`configs/business.yaml`) — это сценарные допущения, а не
  факты из данных. По сетке чувствительности (`business_sensitivity.csv`) прибыль на клиента
  падает до 0–3.7 у.е. при success_rate 0.1–0.2 и растёт до 21.4–21.5 у.е. при 0.4;
  при success_rate 0.1 порог уезжает к 0.85–0.99 и контактов почти нет.
- **SHAP и коррелированные признаки.** `TotalCharges ≈ tenure × MonthlyCharges`, поэтому SHAP
  делит вклад между этими признаками неоднозначно, и важность каждого по отдельности читается
  осторожно. SHAP описывает модель, а не причинные связи; значения — в лог-шансах, вклады
  OHE-столбцов суммированы в 19 исходных признаков (`shap_importance.csv`).
- **Тест читается один раз.** Ровно на этапе 6; далее используются сохранённые
  `test_predictions.parquet` и `thresholds.json`. Этап 8 обучает те же финальные конфигурации
  заново и сверяет предсказания с этим файлом (в прогоне — максимальное расхождение 0.0).

## Лицензия

Файла `LICENSE` в репозитории нет, лицензия не заявлена: проект учебный. Датасет
Telco Customer Churn скачивается с Kaggle, условия его использования — на странице набора.
